"""Segment deep dive: what one rating cell is made of, and whether it can be priced.

The mix page answers "which segments matter". This answers the next question: inside one segment,
how are premium, exposure, severity and frequency distributed, how does that compare with the rest
of the book, and how much more data would it take to reach the credibility target.

Everything is measured at policy-term grain (`project.policy_key`) — the unit that is priced. A
policy can straddle the segment boundary (several classes on one policy), so the scope is part of
the grain: its in-segment rows and its other rows are two different policy-term records.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gl_dq.checks._recon import pipeline_agg
from gl_dq.checks.distribution import inverse_transform, transform_sql
from gl_dq.core.results import _fmt

SEGMENT, REST = "this segment", "rest of book"
PROBS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
LOG_METHODS = {"Linear": "none", "Log (log10, drops ≤ 0)": "log10",
               "Log (log1p, keeps zeros)": "log1p", "Signed log (keeps negatives)": "signed_log"}

# metric -> (label, how to format one value); the keys are the aliases in _policy_grain.sql.j2
METRICS = {
    "premium": ("Premium per policy", "{:,.0f}"),
    "exposure": ("Exposure per policy", "{:,.0f}"),
    "severity": ("Severity (loss per claim)", "{:,.0f}"),
    "frequency": ("Frequency (claims per exposure)", "{:,.4f}"),
    "loss": ("Loss per policy", "{:,.0f}"),
    "claims": ("Claims per policy", "{:,.0f}"),
    "loss_ratio": ("Loss ratio", "{:.1%}"),
}
DEFAULT_METRICS = ["premium", "exposure", "severity", "frequency"]


# ---- scoping -------------------------------------------------------------------------
def _scalar(v):
    """A numpy scalar must become a python one: Dialect.lit uses repr(), and repr(np.int64(5))
    is "np.int64(5)" on numpy 2."""
    return v.item() if isinstance(v, np.generic) else v


def segment_where(schema, dialect, values: dict) -> str:
    """SQL predicate for one segment, e.g. `class_cd_std` = '16676' AND `trr_cd` = 12.

    Every identifier goes through schema.ref (the column whitelist); every value through
    dialect.lit. A null level becomes IS NULL rather than = NULL.
    """
    parts = []
    for col, v in values.items():
        ref = schema.ref(col)
        parts.append(f"{ref} IS NULL" if _fmt(v) == "<null>" else f"{ref} = {dialect.lit(_scalar(v))}")
    return " AND ".join(parts) or "1=1"


def _scope_expr(dialect, where: str) -> str:
    return f"CASE WHEN {where} THEN {dialect.lit(SEGMENT)} ELSE {dialect.lit(REST)} END"


def _base_pred(check, base: str | None) -> str:
    """Exposure is only additive within one exposure base, so it is summed for a single base."""
    if not base:
        return "1=1"
    return f"{check.schema.ref(check.project.measures.exposure_base)} = {check.ctx.dialect.lit(base)}"


def _sql_args(check, where: str, base: str | None, per: float) -> dict:
    s, m = check.schema, check.project.measures
    return dict(keys=s.validate(check.project.policy_key), scope_expr=_scope_expr(check.ctx.dialect, where),
                premium=s.ref(m.written_premium), loss=s.ref(m.loss), claims=s.ref(m.claim_count),
                expo=s.ref(m.exposure), base_pred=_base_pred(check, base), per=float(per), where=None)


# ---- compute -------------------------------------------------------------------------
def policy_stats(check, where: str, base: str | None, per: float) -> pd.DataFrame:
    """One row per metric x scope: counts, min/max/mean/sum and a percentile grid."""
    return check.query("deep dive: policy stats", check.ctx.render_sql(
        "segment_policy_stats.sql.j2", metrics=list(METRICS), probs=PROBS, **_sql_args(check, where, base, per)))


def bounds_for(stats: pd.DataFrame, metric: str, method: str) -> tuple[float, float] | None:
    """Shared histogram bounds for one metric: the p1-p99 range pooled over both scopes.

    Both scopes must share bins or the overlay compares nothing. log10 keeps only positive values,
    so its bounds come from the positive part of the distribution (pcts_pos).
    """
    rows = stats[stats["metric"] == metric]
    col = "pcts_pos" if method == "log10" else "pcts"
    lo, hi = [], []
    for _, r in rows.iterrows():
        p = r[col]
        if p is None or (np.ndim(p) == 0 and pd.isna(p)):
            continue
        p = [float(x) for x in p if x is not None and not pd.isna(x)]
        if len(p) < 2:
            continue
        lo.append(p[0])
        hi.append(p[-1])
    if not lo:
        return None
    a, b = min(lo), max(hi)
    if method == "log10" and a <= 0:
        return None
    a, b = (float(x) for x in transform_bounds(a, b, method))
    return (a, b) if math.isfinite(a) and math.isfinite(b) and b > a else None


def transform_bounds(lo: float, hi: float, method: str) -> tuple[float, float]:
    """The raw p1/p99 mapped onto the transformed axis (every transform is monotonic)."""
    if method == "log10":
        return math.log10(lo), math.log10(hi)
    if method == "log1p":
        return math.log1p(max(lo, -0.999999)), math.log1p(max(hi, -0.999999))
    if method == "signed_log":
        return (math.copysign(math.log1p(abs(lo)), lo), math.copysign(math.log1p(abs(hi)), hi))
    return lo, hi


def bucket_specs(check, stats: pd.DataFrame, metrics: list[str], method: str, bins: int):
    """(SQL bucket blocks, per-metric bin geometry) built from numeric literals only."""
    q, lit = check.ctx.dialect.quote, check.ctx.dialect.lit
    specs, geom = [], {}
    for metric in metrics:
        b = bounds_for(stats, metric, method)
        if b is None:
            continue
        lo, hi = b
        width = (hi - lo) / bins
        v, valid = transform_sql(q(metric), method)
        clipped = f"GREATEST(LEAST({v}, {lit(hi)}), {lit(lo)})"
        specs.append({
            "metric": metric,
            "expr": f"LEAST(GREATEST(CAST(FLOOR(({clipped} - {lit(lo)}) / {lit(width)}) AS INT), 0), {bins - 1})",
            "where": f"{q(metric)} IS NOT NULL AND {valid}",
        })
        geom[metric] = (lo, width)
    return specs, geom


def policy_hist(check, where: str, base: str | None, per: float, metrics: tuple[str, ...],
                method: str, bins: int) -> pd.DataFrame:
    """Histogram counts per metric x scope, on the transformed scale, clipped to p1-p99."""
    stats = policy_stats(check, where, base, per)
    specs, geom = bucket_specs(check, stats, list(metrics), method, bins)
    if not specs:
        return pd.DataFrame(columns=["metric", "scope", "bucket", "n", "left", "right", "center", "share"])
    df = check.query("deep dive: policy histogram", check.ctx.render_sql(
        "segment_policy_hist.sql.j2", buckets=specs, **_sql_args(check, where, base, per)))
    if df.empty:
        return df
    lo = df["metric"].map(lambda m: geom[m][0])
    width = df["metric"].map(lambda m: geom[m][1])
    df["left"] = lo + df["bucket"] * width
    df["right"] = df["left"] + width
    df["center"] = df["left"] + width / 2
    df["from"] = inverse_transform(df["left"].to_numpy(), method)
    df["to"] = inverse_transform(df["right"].to_numpy(), method)
    df["share"] = df["n"] / df.groupby(["metric", "scope"])["n"].transform("sum")
    return df.sort_values(["metric", "scope", "bucket"]).reset_index(drop=True)


def policy_trend(check, where: str, base: str | None, per: float, time_dim: str = "pol_yr") -> pd.DataFrame:
    """Premium, loss, claims and exposure by year, for the segment and for the rest of the book."""
    s, m = check.schema, check.project.measures
    if not s.has(time_dim):
        return pd.DataFrame()
    measures = {"premium": s.ref(m.written_premium), "loss": s.ref(m.loss), "claims": s.ref(m.claim_count),
                "exposure": f"CASE WHEN {_base_pred(check, base)} THEN {s.ref(m.exposure)} END"}
    frames = []
    for scope, clause in ((SEGMENT, where), (REST, f"NOT COALESCE(({where}), FALSE)")):
        df = pipeline_agg(check, [time_dim], measures, clause, label=f"deep dive: {scope} by {time_dim}")
        frames.append(df.assign(scope=scope))
    out = pd.concat(frames, ignore_index=True)
    for col in ("premium", "loss", "claims", "exposure"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["loss_ratio"] = np.where(out["premium"] > 0, out["loss"] / out["premium"], np.nan)
        out["severity"] = np.where(out["claims"] > 0, out["loss"] / out["claims"], np.nan)
        out["frequency"] = np.where(out["exposure"] > 0, out["claims"] / out["exposure"] * per, np.nan)
    return out.sort_values([time_dim, "scope"]).reset_index(drop=True)


def claims_for_z(z: float, full_credibility: float) -> float:
    """Inverse of the square-root rule: how many claims a target Z needs."""
    return float(min(max(z, 0.0), 1.0) ** 2 * full_credibility)


# ---- UI ------------------------------------------------------------------------------
def _money(v: float) -> str:
    """Short enough to fit a metric card: 571.5M, 12.3K, 940."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.1f}{suf}"
    return f"{v:,.0f}"


def _short(segment: str) -> str:
    """'class_cd_std=78838|trr_cd=0101' -> '78838 · 0101' (the dimensions are named by the picker)."""
    from gl_dq.core.results import parse_segment

    parts = parse_segment(str(segment))
    return " · ".join(parts.values()) if parts else str(segment)


def render_detail(check, st, mix_df: pd.DataFrame, dims: list[str], t) -> None:
    """The deep-dive tab: one segment, end to end."""
    import plotly.express as px

    from gl_dq.ui import state
    from gl_dq.ui.theme import CATEGORICAL, NEUTRAL, line, style

    if mix_df is None or mix_df.empty:
        st.info("Nothing to drill into yet — pick dimensions on the mix tab.")
        return
    cfg, m = check.cfg, check.project.measures
    colors = {SEGMENT: CATEGORICAL[0], REST: NEUTRAL}

    segs = [str(s) for s in mix_df["segment"]]
    focus = st.session_state.get("sm_focus")
    c1, c2, c3 = st.columns([3, 1.2, 1])
    seg = c1.selectbox("Segment", segs, index=segs.index(focus) if focus in segs else 0, key="sm_detail_seg",
                       format_func=_short, help="Select a row in the tables on the mix tab to land here")

    row = mix_df[mix_df["segment"].astype(str) == seg].iloc[0]
    values = {d: row[d] for d in dims if d in row.index}
    where = segment_where(check.schema, check.ctx.dialect, values)

    # the default exposure base is the one this segment actually writes, not the first one in the book
    in_seg = state.cached_method(check.name, cfg, "profile", dims=(m.exposure_base,), thresholds=t, where=where)
    bases = [str(b) for b in state.distinct_values(m.exposure_base) if b is not None]
    biggest = str(in_seg.iloc[0][m.exposure_base]) if not in_seg.empty else None
    base = c2.selectbox("Exposure base", bases, index=bases.index(biggest) if biggest in bases else 0,
                        key="sm_detail_base",
                        help=f"`{m.exposure}` is only additive within one `{m.exposure_base}`, so exposure and "
                             "frequency are measured for a single base") if bases else None
    per = c3.number_input("Claims per N exposure", 1.0, value=1000.0, step=1000.0, key="sm_detail_per")

    # ---- what this segment is ---------------------------------------------------------
    claims, premium = int(row["claims"]), float(row["premium"])
    z = float(row["z_claims"])
    k = st.columns(5)
    k[0].metric("Premium", _money(premium), f"{row['premium_share']:.2%} of book", delta_color="off",
                help=f"{premium:,.2f}")
    k[1].metric("Policies (records)", f"{int(row['records']):,}")
    k[2].metric("Claims", f"{claims:,}", f"{row['claims_share']:.2%} of book", delta_color="off")
    k[3].metric("Credibility Z", f"{z:.2f}", f"target {t.z_target:.2f}",
                delta_color="normal" if z >= t.z_target else "inverse")
    k[4].metric("Flag", row["flag"] or "—")

    need = claims_for_z(t.z_target, cfg.full_credibility)
    if claims >= need:
        st.success(f"**Credible enough to price.** {claims:,} claims is at or above the {need:,.0f} needed for "
                   f"Z = {t.z_target:.2f} (full credibility = {cfg.full_credibility:,.0f} claims).")
    else:
        st.warning(f"**Short of the credibility target.** {claims:,} claims vs {need:,.0f} needed for "
                   f"Z = {t.z_target:.2f} — {need - claims:,.0f} more claims, i.e. about "
                   f"{need / max(claims, 1):.1f}x the current claim volume. Rates here should be complemented "
                   f"from a broader class.")
    st.caption(f"Filter: `{where}`")

    # ---- how it is developing ---------------------------------------------------------
    time_dim = "pol_yr" if check.schema.has("pol_yr") else None
    if time_dim:
        tr = state.cached_method(check.name, cfg, "detail_trend", where=where, base=base, per=float(per))
        if not tr.empty:
            st.markdown(f"**How it is developing** — by `{time_dim}`")
            seg_tr = tr[tr["scope"] == SEGMENT]
            # two per row, paired by what they answer: how big, how bad, how it decomposes.
            # Sizes are the segment's own (the book is orders of magnitude bigger and would flatten
            # them); the ratios are unit-free, so those compare directly against the rest of the book.
            rows = [[("premium", "Premium", None, False),
                     ("exposure", f"Exposure ({base})" if base else "Exposure", None, False)],
                    [("loss", "Loss", None, False),
                     ("loss_ratio", "Loss ratio", ".0%", True)],
                    [("severity", "Severity (loss per claim)", None, True),
                     ("frequency", f"Frequency (per {per:,.0f})", None, True)]]
            first_compared = True
            for panels in rows:
                g = st.columns(2)
                for i, (col, label, tick, compare_scopes) in enumerate(panels):
                    sub = (tr if compare_scopes else seg_tr)
                    sub = sub[sub[col].notna()]
                    enc = dict(color="scope", color_discrete_map=colors,
                               category_orders={"scope": [SEGMENT, REST]}) if compare_scopes else \
                        dict(color_discrete_sequence=[colors[SEGMENT]])
                    fig = line(sub, x=time_dim, y=col, markers=True, **enc)
                    fig.update_yaxes(rangemode="tozero", **({"tickformat": tick} if tick else {}))
                    fig.update_layout(showlegend=compare_scopes and first_compared)
                    g[i].plotly_chart(style(fig, 260, label), use_container_width=True)
                    first_compared = first_compared and not compare_scopes
            st.caption("Premium, exposure and loss are this segment's own — the rest of the book is larger by orders "
                       "of magnitude and would flatten them. Loss ratio, severity and frequency are shown against "
                       "the rest of the book, which is only meaningful because they are rates.")

    # ---- how the metrics are distributed ----------------------------------------------
    st.markdown("**Distribution across policy terms**")
    d1, d2, d3 = st.columns([3, 1.5, 1])
    picked = d1.multiselect("Metrics", list(METRICS), default=DEFAULT_METRICS, key="sm_detail_metrics",
                            format_func=lambda k: METRICS[k][0])
    # insurance amounts are lognormal and their upper tail is long: linear bins hide the whole body
    scale = d2.selectbox("Scale", list(LOG_METHODS), index=1, key="sm_detail_scale")
    bins = int(d3.number_input("Bins", 5, 120, 40, 5, key="sm_detail_bins"))
    method = LOG_METHODS[scale]
    compare = st.toggle("Compare against the rest of the book", True, key="sm_detail_cmp")

    if picked:
        hist = state.cached_method(check.name, cfg, "detail_hist", where=where, base=base, per=float(per),
                                   metrics=tuple(picked), log_method=method, bins=bins)
        if hist.empty:
            st.info("No policy-level values to plot for these metrics on this scale.")
        else:
            view = hist if compare else hist[hist["scope"] == SEGMENT]
            cols = st.columns(2)
            for i, metric in enumerate([p for p in picked if p in set(hist["metric"])]):
                h = view[view["metric"] == metric]
                fig = px.bar(h, x="center", y="share", color="scope", barmode="overlay", opacity=0.65,
                             color_discrete_map=colors, category_orders={"scope": [SEGMENT, REST]},
                             hover_data={"n": ":,", "from": ":,.2f", "to": ":,.2f", "center": False},
                             labels={"center": "value" if method == "none" else f"{method}(value)",
                                     "share": "share of policies"})
                fig.update_layout(bargap=0.02, showlegend=i == 0)
                cols[i % 2].plotly_chart(style(fig, 300, METRICS[metric][0]), use_container_width=True)
            missing = [p for p in picked if p not in set(hist["metric"])]
            if missing:
                st.caption(f"Not plottable on this scale (no values in range): "
                           f"{', '.join(METRICS[x][0] for x in missing)}.")
            st.caption(f"Bins span the 1st to 99th percentile pooled over both scopes, so the two series are "
                       f"comparable; values outside are folded into the end bins."
                       + (" log10 drops values ≤ 0." if method == "log10" else ""))

    # ---- concentration and composition -------------------------------------------------
    stats = state.cached_method(check.name, cfg, "detail_stats", where=where, base=base, per=float(per))
    seg_stats = stats[stats["scope"] == SEGMENT].set_index("metric")
    if "loss" in seg_stats.index:
        r = seg_stats.loc["loss"]
        total, biggest = float(r["vsum"] or 0), float(r["vmax"] or 0)
        if total > 0:
            share = biggest / total
            msg = (f"The largest single policy carries **{share:.1%}** of this segment's loss "
                   f"({biggest:,.0f} of {total:,.0f}).")
            (st.warning if share > 0.25 else st.info)(
                msg + (" A severity that rests on one claim is not a severity — cap or exclude it before "
                       "fitting." if share > 0.25 else ""))

    others = [d for d in (cfg.dimensions + [check.project.src_col]) if d not in dims and check.schema.has(d)]
    if others:
        st.markdown("**What is inside this cell**")
        by = st.selectbox("Break down by", others, key="sm_detail_by",
                          help="A segment that is really one state or one source is not the segment you think")
        inner = state.cached_method(check.name, cfg, "profile", dims=(by,), thresholds=t, where=where)
        if inner.empty:
            st.caption("No rows.")
        else:
            show = inner.head(20)[[by, "premium", "premium_share", "records", "claims", "z_claims"]]
            st.dataframe(show, hide_index=True, use_container_width=True, column_config={
                "premium": st.column_config.NumberColumn(format="localized"),
                "premium_share": st.column_config.NumberColumn("share of segment", format="percent"),
                "records": st.column_config.NumberColumn(format="localized"),
                "claims": st.column_config.NumberColumn(format="localized"),
                "z_claims": st.column_config.NumberColumn("Z (claims)", format="%.2f")})
            top = inner.iloc[0]
            if float(top["premium_share"]) > 0.8:
                st.warning(f"`{by} = {_fmt(top[by])}` alone is {top['premium_share']:.0%} of this segment — it is "
                           f"effectively a single-{by} cell.")

    with st.expander("Policy-grain statistics"):
        cols = ["metric", "scope", "n_policies", "n_nonnull", "n_positive", "vmin", "vmean", "vmax", "vsum"]
        st.dataframe(stats[[c for c in cols if c in stats]], hide_index=True, use_container_width=True)
