"""Target analysis: how a modelling target moves with each rating variable, and where two of them
interact.

The question a pricing model asks first, answered before any model is fitted:

  * **univariate** - for one variable, the target by level with its weight and credibility, and the
    relativity against the portfolio average. This is the one-way relativity table.
  * **interaction** - for two variables, the target by level of A with one line per level of B.
    Parallel lines mean the two act independently, which is what a main-effects model assumes;
    lines that cross or fan out are an interaction that model would miss.

Each target is a ratio, and each is weighted by its own denominator - frequency by exposure,
severity by claims, loss ratio by premium - because that is the measure whose volume makes a cell's
estimate reliable. Frequency and loss cost divide by `expo_amt`, which is only additive inside one
exposure base, so those two are always measured for a single base.

Interaction strength is the weighted RMS of log(actual / expected), where expected is what the two
one-way relativities predict on their own. Read it as "a main-effects model is typically wrong by
this much in this pair" - the cells behind it are the candidate interaction terms.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from pydantic import BaseModel

from gl_dq.checks.base import Check, CheckResult
from gl_dq.checks.segment_mix import credibility
from gl_dq.core.binning import BinningSet, BinSpec
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import _fmt, grade

OTHER = "<other>"


class TargetSpec(BaseModel):
    label: str
    numerator: str  # a measure name: loss | claims
    denominator: str  # exposure | claims | premium
    fmt: str = "{:,.4f}"

    @property
    def needs_base(self) -> bool:
        """Exposure is only additive within one exposure base, so those targets scope to one."""
        return "exposure" in (self.numerator, self.denominator)


TARGETS: dict[str, TargetSpec] = {
    "frequency": TargetSpec(label="Frequency (claims per exposure)", numerator="claims",
                            denominator="exposure", fmt="{:,.4f}"),
    "severity": TargetSpec(label="Severity (loss per claim)", numerator="loss", denominator="claims",
                           fmt="{:,.0f}"),
    "loss_cost": TargetSpec(label="Loss cost (loss per exposure)", numerator="loss", denominator="exposure",
                            fmt="{:,.2f}"),
    "loss_ratio": TargetSpec(label="Loss ratio (loss per premium)", numerator="loss", denominator="premium",
                             fmt="{:.1%}"),
}

MEASURES = ("records", "premium", "loss", "claims", "exposure")


def _ratio(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def fold_levels(df: pd.DataFrame, col: str, keep: set, weight: str) -> pd.DataFrame:
    """Fold every level outside `keep` into <other>. Exact, because every measure is a sum."""
    if df.empty:
        return df
    out = df.copy()
    out[col] = np.where(out[col].isin(keep), out[col], OTHER)
    keys = [c for c in ("var_a", "level_a", "var_b", "level_b") if c in out]
    return out.groupby(keys, as_index=False, dropna=False)[list(MEASURES)].sum()


def interaction_strength(cells: pd.DataFrame, weight: str) -> float:
    """Weighted RMS of log(actual / expected) - how wrong a main-effects-only model is here."""
    ok = cells["lift"].notna() & (cells["lift"] > 0) & (cells[weight] > 0)
    if not ok.any():
        return float("nan")
    w, lift = cells.loc[ok, weight].to_numpy(float), cells.loc[ok, "lift"].to_numpy(float)
    return float(np.sqrt(np.average(np.log(lift) ** 2, weights=w)))


@register_check("target_analysis")
class TargetAnalysis(Check):
    title = "Target analysis"
    icon = "🎯"
    description = ("How frequency, severity, loss cost or loss ratio moves with each rating variable, and "
                   "which pairs of variables interact - the one-way relativities and interaction plots a "
                   "pricing model is built from.")
    default_order = 85

    class Config(CheckConfig):
        variables: list[str] = []
        default_target: str = "loss_ratio"
        max_levels: int = 12  # levels charted per variable; the rest fold into <other>
        min_claims: int = 10  # a cell below this is too thin to read as signal
        full_credibility: float = 1082.0
        spread_warn: float = 3.0  # max/min relativity across credible levels
        interaction_warn: float = 0.15  # weighted RMS log deviation from the main-effects prediction
        where: str | None = None

    # ---- compute -----------------------------------------------------------------------
    def profile(self, variables: tuple[str, ...], target: str, base: str | None = None,
                max_levels: int | None = None, binning: BinningSet | None = None) -> dict[str, pd.DataFrame]:
        """Total, one-ways and two-ways for one target, in a single query.

        `binning` replaces a variable's raw values with its bin label, in the grouping itself - so a
        re-binned view costs exactly the same one query.
        """
        spec, s, m = TARGETS[target], self.schema, self.project.measures
        variables = s.validate([v for v in dict.fromkeys(variables) if v])
        max_levels = max_levels or self.cfg.max_levels
        if not variables:
            return {k: pd.DataFrame() for k in ("total", "oneway", "pairs", "strength")}

        sets = [[]] + [[v] for v in variables] + [list(p) for p in itertools.combinations(variables, 2)]
        where = self._where(base)
        exprs = {v: self._expr(v, binning) for v in variables}
        raw = self.query(f"{target} by {', '.join(variables)}", self.ctx.render_sql(
            "target_sets.sql.j2", vars=variables, sets=sets, where=where, exprs=exprs,
            premium=s.ref(m.written_premium), loss=s.ref(m.loss), claims=s.ref(m.claim_count),
            exposure=s.ref(m.exposure)))
        out = self._shape(raw, variables, spec, max_levels)
        out["binning"] = pd.DataFrame([{"variable": v, "scheme": (b.name if (b := (binning or BinningSet()).get(v))
                                                                  and b.method != "categorical" else ""),
                                        "summary": b.summary() if b else "raw levels"} for v in variables])
        return out

    def _expr(self, variable: str, binning: BinningSet | None) -> str:
        """The grouping expression for a variable: the column, or the CASE that bins it."""
        spec = (binning or BinningSet()).get(variable)
        ref = self.schema.ref(variable)
        if spec is None or spec.method == "categorical" or not spec.resolved:
            return ref
        return spec.case_sql(ref, self.ctx.dialect.lit)

    def evaluate_binning(self, variable: str, target: str, base: str | None = None,
                         spec: BinSpec | None = None) -> dict:
        """How well one scheme separates the target - the number an experiment is judged on."""
        binning = BinningSet(specs=[spec]) if spec is not None else None
        t = self.profile((variable,), target, base=base, max_levels=200, binning=binning)
        part = t["oneway"]
        if part.empty:
            return {"n_bins": 0, "signal": float("nan"), "spread": float("nan"), "credible_weight": float("nan")}
        credible = part[part["claims"] >= self.cfg.min_claims]
        rel = credible["relativity"].replace([np.inf, -np.inf], np.nan)
        ok = rel.notna() & (rel > 0) & (credible["weight"] > 0)
        signal = (float(np.sqrt(np.average(np.log(rel[ok]) ** 2, weights=credible.loc[ok, "weight"])))
                  if ok.any() else float("nan"))
        return {"n_bins": int(len(part)),
                "n_credible": int(len(credible)),
                "signal": signal,
                "spread": float(rel[ok].max() / rel[ok].min()) if ok.sum() > 1 else float("nan"),
                "credible_weight": float(credible["weight"].sum() / part["weight"].sum())
                if part["weight"].sum() else float("nan")}

    def _where(self, base: str | None) -> str | None:
        parts = [self.cfg.where]
        if base:
            parts.append(f"{self.schema.ref(self.project.measures.exposure_base)} = {self.ctx.dialect.lit(base)}")
        return " AND ".join(f"({p})" for p in parts if p) or None

    def _shape(self, raw: pd.DataFrame, variables: list[str], spec: TargetSpec,
               max_levels: int) -> dict[str, pd.DataFrame]:
        for c in MEASURES:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
        grouped = {v: raw[f"g__{v}"] == 0 for v in variables}
        depth = sum(grouped.values())

        total = raw[depth == 0][list(MEASURES)].sum().to_frame().T
        total["value"] = _ratio(total[spec.numerator], total[spec.denominator])
        overall = float(total["value"].iloc[0]) if len(total) else float("nan")

        one = []
        for v in variables:
            rows = raw[grouped[v] & (depth == 1)]
            one.append(pd.DataFrame({"var_a": v, "level_a": [_fmt(x) for x in rows[v]],
                                     **{c: rows[c].to_numpy() for c in MEASURES}}))
        oneway = pd.concat(one, ignore_index=True) if one else pd.DataFrame()

        # keep the biggest levels by the target's own weight; the rest become <other>
        keep = {}
        for v in variables:
            part = oneway[oneway["var_a"] == v].sort_values(spec.denominator, ascending=False)
            keep[v] = set(part["level_a"].head(max_levels))
        oneway = pd.concat([fold_levels(oneway[oneway["var_a"] == v], "level_a", keep[v], spec.denominator)
                            .assign(var_a=v) for v in variables], ignore_index=True) if len(oneway) else oneway

        pairs = []
        for a, b in itertools.combinations(variables, 2):
            rows = raw[grouped[a] & grouped[b] & (depth == 2)]
            if rows.empty:
                continue
            cells = pd.DataFrame({"var_a": a, "level_a": [_fmt(x) for x in rows[a]],
                                  "var_b": b, "level_b": [_fmt(x) for x in rows[b]],
                                  **{c: rows[c].to_numpy() for c in MEASURES}})
            cells["level_a"] = np.where(cells["level_a"].isin(keep[a]), cells["level_a"], OTHER)
            cells["level_b"] = np.where(cells["level_b"].isin(keep[b]), cells["level_b"], OTHER)
            pairs.append(cells.groupby(["var_a", "level_a", "var_b", "level_b"], as_index=False)[list(MEASURES)].sum())
        pairs = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(
            columns=["var_a", "level_a", "var_b", "level_b", *MEASURES])

        for df in (oneway, pairs):
            if df.empty:
                continue
            df["weight"] = df[spec.denominator]
            df["value"] = _ratio(df[spec.numerator], df[spec.denominator])
            df["relativity"] = df["value"] / overall if overall and np.isfinite(overall) else np.nan
            df["z"] = credibility(df["claims"], self.cfg.full_credibility)
        if not oneway.empty:
            oneway["weight_share"] = oneway["weight"] / oneway.groupby("var_a")["weight"].transform("sum")

        # what a model with only main effects would predict for each cell
        strength = []
        if not pairs.empty:
            rel = oneway.set_index(["var_a", "level_a"])["relativity"]
            pairs["expected"] = [
                overall * rel.get((r.var_a, r.level_a), np.nan) * rel.get((r.var_b, r.level_b), np.nan)
                for r in pairs.itertuples()]
            pairs["lift"] = _ratio(pairs["value"], pairs["expected"])
            for (a, b), cells in pairs.groupby(["var_a", "var_b"]):
                credible = cells[cells["claims"] >= self.cfg.min_claims]
                strength.append({"var_a": a, "var_b": b, "pair": f"{a} × {b}",
                                 "strength": interaction_strength(credible, "weight"),
                                 "n_cells": len(cells), "n_credible": len(credible),
                                 "weight_credible": (credible["weight"].sum() / cells["weight"].sum()
                                                     if cells["weight"].sum() else np.nan)})
        strength = pd.DataFrame(strength).sort_values("strength", ascending=False, ignore_index=True) \
            if strength else pd.DataFrame(columns=["var_a", "var_b", "pair", "strength", "n_cells",
                                                   "n_credible", "weight_credible"])
        total["target"] = spec.label
        return {"total": total, "oneway": oneway, "pairs": pairs, "strength": strength}

    def findings_for(self, target: str, t: dict[str, pd.DataFrame]) -> list[dict]:
        cfg, rows = self.cfg, []
        oneway, strength = t["oneway"], t["strength"]
        if not oneway.empty:
            for v, part in oneway.groupby("var_a"):
                credible = part[part["claims"] >= cfg.min_claims]
                rel = credible["relativity"].replace([np.inf, -np.inf], np.nan).dropna()
                rel = rel[rel > 0]
                spread = float(rel.max() / rel.min()) if len(rel) > 1 else np.nan
                rows.append(dict(variable=v, item=target, segment="ALL", metric="target_spread",
                                 value=spread, threshold=cfg.spread_warn,
                                 status=grade(spread, cfg.spread_warn, None),
                                 detail=f"{target} varies {spread:,.1f}x across {len(rel)} credible levels"
                                        if np.isfinite(spread) else "not enough credible levels to compare"))
                thin = part[part["claims"] < cfg.min_claims]["weight"].sum() / (part["weight"].sum() or np.nan)
                rows.append(dict(variable=v, item=target, segment="ALL", metric="pct_weight_low_credibility",
                                 value=float(thin), threshold=0.25, status=grade(float(thin), 0.1, 0.25),
                                 detail=f"{thin:.1%} of weight sits in levels with fewer than "
                                        f"{cfg.min_claims} claims"))
        for r in strength.itertuples():
            if not np.isfinite(r.strength):
                continue
            rows.append(dict(variable=r.var_a, item=f"{target}: {r.pair}", segment="ALL",
                             metric="interaction_strength", value=float(r.strength),
                             threshold=cfg.interaction_warn,
                             status=grade(float(r.strength), cfg.interaction_warn, None),
                             detail=f"a main-effects model is typically {np.expm1(r.strength):.0%} out "
                                    f"across {r.n_credible} credible cells"))
        return rows

    def run(self) -> CheckResult:
        if not self.cfg.variables:
            self.note("No rating variables configured yet - add them under Settings.")
            return self.result([], {})
        target = self.cfg.default_target
        spec = TARGETS[target]
        base = self._default_base() if spec.needs_base else None
        t = self.profile(tuple(self.cfg.variables), target, base)
        tables = {f"{k}::{target}": v for k, v in t.items()}
        if base:
            self.note(f"{spec.label} is measured for exposure base **{base}** "
                      f"(`{self.project.measures.exposure}` is not additive across bases).")
        return self.result(self.findings_for(target, t), tables)

    def _default_base(self) -> str | None:
        """The exposure base carrying the most premium - the one worth defaulting to."""
        m = self.project.measures
        df = self.query("exposure bases", self.ctx.render_sql(
            "agg_by_dims.sql.j2", dims=[m.exposure_base], where=self.cfg.where,
            measures={"premium": self.schema.ref(m.written_premium)}))
        df = df[df[m.exposure_base].notna()]
        return None if df.empty else str(df.sort_values("premium", ascending=False).iloc[0][m.exposure_base])

    # ---- UI ----------------------------------------------------------------------------
    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        cols = [c for c in self.segment_options() if self.schema.has(c)] or self.schema.names()
        new.variables = st.multiselect("Rating variables", cols,
                                       default=[v for v in cfg.variables if v in cols], key="ta_vars")
        c1, c2, c3 = st.columns(3)
        new.default_target = c1.selectbox("Target for the refresh job", list(TARGETS),
                                          index=list(TARGETS).index(cfg.default_target),
                                          format_func=lambda k: TARGETS[k].label, key="ta_target")
        new.max_levels = int(c2.number_input("Levels per variable", 2, 50, cfg.max_levels, key="ta_lv"))
        new.min_claims = int(c3.number_input("Minimum claims for a credible cell", 0, 10000,
                                             cfg.min_claims, key="ta_mc"))
        w1, w2 = st.columns(2)
        new.spread_warn = w1.number_input("Flag a variable when levels vary more than", 1.0, 100.0,
                                          cfg.spread_warn, 0.5, key="ta_sw")
        new.interaction_warn = w2.number_input("Flag an interaction above", 0.0, 2.0,
                                               cfg.interaction_warn, 0.01, key="ta_iw")
        return new

    def render(self, result: CheckResult):
        import streamlit as st

        cfg = self.cfg
        opts = [c for c in self.segment_options() if self.schema.has(c)]
        if not opts:
            st.info("No rating variables available. Add `segment_candidates` to the profile.")
            return

        c1, c2, c3, c4 = st.columns([1.6, 1.2, 3, 1])
        target = c1.selectbox("Target", list(TARGETS), index=list(TARGETS).index(cfg.default_target),
                              format_func=lambda k: TARGETS[k].label, key="ta_pick_target")
        spec = TARGETS[target]
        base = None
        if spec.needs_base:
            from gl_dq.ui import state

            bases = [str(b) for b in state.distinct_values(self.project.measures.exposure_base) if b is not None]
            default = self._default_base()
            base = c2.selectbox("Exposure base", bases,
                                index=bases.index(default) if default in bases else 0, key="ta_base",
                                help=f"`{self.project.measures.exposure}` is in different units per "
                                     f"base, so it cannot be added across them") if bases else None
        else:
            c2.markdown("<div style='padding-top:1.9rem;color:#898781'>no exposure base needed</div>",
                        unsafe_allow_html=True)
        default_vars = [v for v in (cfg.variables or opts[:3]) if v in opts]
        variables = c3.multiselect("Rating variables", opts, default=default_vars, key="ta_pick_vars",
                                   help="Every pair of these is tested for interaction, in the same query")
        min_claims = int(c4.number_input("Min claims", 0, 10000, cfg.min_claims, key="ta_pick_mc",
                                         help="Cells below this are shown greyed out and excluded from "
                                              "the interaction measure"))
        if not variables:
            st.info("Pick at least one rating variable.")
            return

        from gl_dq.ui import state

        with st.spinner("Aggregating…"):
            t = state.cached_method(self.name, cfg, "profile", variables=tuple(variables), target=target,
                                    base=base, binning=state.binning_set(variables))
        if t["oneway"].empty:
            st.info("No rows for this selection.")
            return
        overall = float(t["total"]["value"].iloc[0])
        st.caption(f"Portfolio {spec.label.lower()}: **{spec.fmt.format(overall)}**"
                   + (f" · exposure base **{base}**" if base else "")
                   + f" · levels beyond the largest {cfg.max_levels} per variable are folded into `{OTHER}`")

        tab_uni, tab_int = st.tabs(["Univariate", "🔀 Interactions"])
        with tab_uni:
            self._render_univariate(st, t, variables, spec, overall, min_claims, target, base)
        with tab_int:
            self._render_interactions(st, t, variables, spec, overall, min_claims)

    # -- univariate ----------------------------------------------------------------------
    def _render_univariate(self, st, t, variables, spec, overall, min_claims, target, base):
        from plotly.subplots import make_subplots

        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import CATEGORICAL, NEUTRAL, STATUS, style

        c1, c2 = st.columns([2, 2])
        var = c1.selectbox("Variable", variables, key="ta_uni_var")
        weight_col = c2.selectbox(
            "Weight the bars by", list(MEASURES), index=list(MEASURES).index(spec.denominator),
            key="ta_uni_weight",
            help=f"Defaults to `{spec.denominator}`, this target's own denominator - the volume that makes "
                 "a level's estimate trustworthy. Any other measure is available for context.")

        from gl_dq.core.results import value_sort_key

        part = t["oneway"][t["oneway"]["var_a"] == var].copy()
        binned = self._binning_row(t, var)
        # a line is only readable along an ordered axis: bins in bin order, a numeric column in
        # numeric order (500K before 1M), and only a true category by size
        if binned or self.schema.is_numeric(var):
            part = part.sort_values("level_a", key=lambda c: c.map(value_sort_key))
        else:
            part = part.sort_values("weight", ascending=False)
        order = [str(x) for x in part["level_a"]]
        credible = (part["claims"] >= min_claims).to_numpy()

        # the target as a line, the weight as bars - stacked on a shared x rather than sharing one
        # plot, so two very different scales are never read off the same axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            row_heights=[0.62, 0.38])
        fig.add_scatter(x=part["level_a"], y=part["value"], row=1, col=1, mode="lines+markers",
                        name=spec.label, line=dict(color=CATEGORICAL[0], width=2),
                        marker=dict(size=np.where(credible, 10, 7),
                                    color=np.where(credible, CATEGORICAL[0], NEUTRAL),
                                    line=dict(width=1, color="white")),
                        customdata=np.stack([part["claims"], part["z"], part["relativity"]], axis=-1),
                        hovertemplate="%{x}<br>value %{y:,.4f}<br>claims %{customdata[0]:,.0f}"
                                      "<br>Z %{customdata[1]:.2f}<br>relativity %{customdata[2]:.2f}x<extra></extra>")
        fig.add_hline(y=overall, line_dash="dash", line_color=STATUS["warn"], row=1, col=1,
                      annotation_text=f"portfolio {spec.fmt.format(overall)}", annotation_position="top left")
        fig.add_bar(x=part["level_a"], y=part[weight_col], row=2, col=1, name=weight_col,
                    marker_color=CATEGORICAL[2],
                    hovertemplate="%{x}<br>" + weight_col + " %{y:,.0f}<extra></extra>")
        fig.update_xaxes(categoryorder="array", categoryarray=order, tickangle=-35, row=2, col=1)
        fig.update_yaxes(title_text=spec.label.split(" (")[0], row=1, col=1)
        fig.update_yaxes(title_text=weight_col, row=2, col=1)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 470, f"{spec.label} by {var}"), use_container_width=True)
        st.caption(f"Small grey markers are levels with fewer than {min_claims} claims - too thin to read as "
                   f"signal. The bars are `{weight_col}`: a level is only as trustworthy as the volume "
                   f"underneath it.")

        self._binning_controls(st, var, target, base, t)

        view = part.assign(level=part["level_a"])[
            ["level", "weight", "weight_share", "claims", "z", "value", "relativity", "records"]]
        status_table(view.rename(columns={"value": spec.label.split(" (")[0].lower()}),
                     percent_cols=["weight_share"],
                     number_formats={"relativity": "%.2f", "z": "%.2f"}, key=f"ta_uni_{var}")
        st.download_button("⬇️ CSV", part.to_csv(index=False), file_name=f"oneway_{var}.csv",
                           mime="text/csv", key=f"ta_dl_{var}")
        # findings for what is on screen now, not for whatever the last refresh ran
        live = pd.DataFrame(self.findings_for(target, t))
        if not live.empty:
            status_table(live[live["variable"] == var][["item", "metric", "value", "threshold",
                                                        "status", "detail"]], key=f"ta_find_{var}")

    @staticmethod
    def _binning_row(t, var) -> bool:
        b = t.get("binning")
        if b is None or b.empty:
            return False
        row = b[b["variable"] == var]
        return bool(len(row)) and bool(row.iloc[0]["scheme"])

    # -- binning: the experiment workbench ------------------------------------------------
    def _binning_controls(self, st, var, target, base, t):
        """Try a binning, see it immediately, save it with a name, compare it against the others."""
        from gl_dq.core.binning import METHOD_LABELS, BinSpec, save_binnings
        from gl_dq.ui import state

        if not self.schema.is_numeric(var):
            return
        lib = state.session_binnings()
        saved = lib.for_variable(var)
        current = state.current_binning(var)

        with st.expander(f"🧪 Binning for `{var}`" + (f" — **{current.name}**" if current else " — raw levels"),
                         expanded=False):
            st.caption("A GLM is fitted by trying a variable several ways. Each scheme below is saved with "
                       "its resolved cut points, so it can be reused on another target, compared against the "
                       "others, and handed to the modelling pipeline unchanged.")
            names = ["(raw levels)"] + [b.name for b in saved]
            if current is not None and current.name not in names:
                names.append(current.name)  # an unsaved draft from Preview
            pick = st.selectbox("Scheme", names,
                                index=names.index(current.name) if current and current.name in names else 0,
                                key=f"ta_bin_pick_{var}")
            if pick != (current.name if current else "(raw levels)"):
                state.set_binning(var, lib.get(var, pick) if pick != "(raw levels)" else None)
                st.rerun()

            st.markdown("**Try another**")
            c1, c2, c3 = st.columns([2, 1, 3])
            method = c1.selectbox("Method", [m for m in METHOD_LABELS if m != "categorical"],
                                  format_func=METHOD_LABELS.get, key=f"ta_bin_m_{var}")
            n_bins = int(c2.number_input("Bins", 2, 50, 5, key=f"ta_bin_n_{var}",
                                         disabled=method == "custom"))
            cuts_text = c3.text_input("Cut points (comma-separated)", key=f"ta_bin_c_{var}",
                                      placeholder="1000, 5000, 25000",
                                      disabled=method != "custom",
                                      help="Interior edges: 3 cuts make 4 bins")
            try:
                cuts = [float(x.strip()) for x in cuts_text.split(",") if x.strip()]
            except ValueError:
                st.error("Cut points must be numbers.")
                return
            draft = None
            try:
                draft = BinSpec(variable=var, name="(draft)", method=method, bins=n_bins, cuts=cuts)
            except Exception as e:  # noqa: BLE001  (a custom scheme with no cuts yet)
                st.caption(str(e))

            b1, b2, b3 = st.columns([1, 1.4, 3])
            if draft is not None and b1.button("👁 Preview", key=f"ta_bin_prev_{var}"):
                state.set_binning(var, state.cached_method(self.name, self.cfg, "resolve_binning",
                                                           spec=draft, base=base))
                st.rerun()
            name = b3.text_input("Save as", key=f"ta_bin_name_{var}", placeholder="quintiles",
                                 label_visibility="collapsed")
            live = state.current_binning(var)
            if b2.button("💾 Save scheme", key=f"ta_bin_save_{var}", type="primary",
                         disabled=not (name and live)):
                spec = live.model_copy(update={"name": name}).stamped(state.current_user())
                state.set_session_binnings(lib.put(spec))
                save_binnings(self.ctx.config_store, state.session_binnings())
                state.set_binning(var, spec)
                st.toast(f"Saved binning {var}/{name}", icon="🧪")
                st.rerun()
            if live is not None:
                st.caption(f"In play: `{live.summary()}`"
                           + (f" · {live.author} on {live.created[:10]}" if live.created else " · unsaved draft"))

            if saved:
                st.markdown("**Experiments** — every saved scheme for this variable, scored on the "
                            f"current target ({TARGETS[target].label.split(' (')[0].lower()})")
                rows = []
                for b in [None] + saved:
                    m = state.cached_method(self.name, self.cfg, "evaluate_binning", variable=var,
                                            target=target, base=base, spec=b)
                    rows.append({"scheme": b.name if b else "(raw levels)",
                                 "method": b.method if b else "categorical",
                                 "cuts": b.summary() if b else "", **m,
                                 "why": b.description if b else "", "author": b.author if b else "",
                                 "created": (b.created or "")[:10] if b else ""})
                rank = pd.DataFrame(rows).sort_values("signal", ascending=False, ignore_index=True)
                from gl_dq.ui.components import status_table

                status_table(rank, percent_cols=["credible_weight"],
                             number_formats={"signal": "%.3f", "spread": "%.2f"}, key=f"ta_bin_rank_{var}")
                st.caption("`signal` is how far the bins separate the target, weighted and counting only "
                           "credible bins - higher separates more. Read it with `credible_weight` and "
                           "`n_bins`: more bins almost always raise signal, and the thin ones are noise.")

                adopt = st.selectbox("Adopt for modelling", [b.name for b in saved], key=f"ta_bin_adopt_{var}")
                why = st.text_input("Why this one", key=f"ta_bin_why_{var}",
                                    placeholder="Separates loss ratio, every bin credible")
                if st.button("➕ Add to recommended preprocessing", key=f"ta_bin_pp_{var}"):
                    self._adopt_binning(st, lib.get(var, adopt), why)

    def resolve_binning(self, spec: BinSpec, base: str | None = None) -> BinSpec:
        """Cut points from the data - computed once, then frozen into the scheme."""
        from gl_dq.core.binning import resolve

        return resolve(self, spec, self._where(base))

    def _adopt_binning(self, st, spec, why: str):
        """A chosen scheme becomes a preprocessing step, cuts and all, for the modelling pipeline."""
        from gl_dq.core.knowledge import PreprocessingStep
        from gl_dq.ui import state

        if spec is None:
            return
        wf = self.ctx.workflow
        step = PreprocessingStep(
            op="bin", params={"method": spec.method, "n_bins": spec.n_bins,
                              "cuts": ", ".join(str(c) for c in spec.cuts)},
            rationale=(why or spec.description or f"binning scheme '{spec.name}'"),
            author=state.current_user())
        target_stage = next((s.key for s in wf.stages if s.requires_preprocessing), None)
        self.ctx.knowledge.add_preprocessing_step(spec.variable, step, state.current_user(),
                                                  set_status=target_stage)
        st.toast(f"{spec.variable}: binning '{spec.name}' added to recommended preprocessing", icon="🧰")

    # -- interactions --------------------------------------------------------------------
    def _render_interactions(self, st, t, variables, spec, overall, min_claims):
        import plotly.express as px

        from gl_dq.core.results import sort_segments
        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import DIVERGING, entity_colors, line, style

        strength = t["strength"]
        if strength.empty:
            st.info("Pick at least two rating variables to look for interactions.")
            return
        st.markdown("**Which pairs interact?** — how far each pair departs from what the two one-way "
                    "relativities predict on their own.")
        rank = strength.assign(typical_error=np.expm1(strength["strength"]))[
            ["pair", "typical_error", "strength", "n_credible", "n_cells", "weight_credible"]]
        status_table(rank, percent_cols=["typical_error", "weight_credible"],
                     number_formats={"strength": "%.3f"}, key="ta_rank")
        st.caption("`typical_error` is how wrong a main-effects-only model is in that pair, weighted by "
                   f"`{spec.denominator}` and counting only cells with at least {min_claims} claims. "
                   "Large and credible means a candidate interaction term.")

        pair = st.selectbox("Pair", list(strength["pair"]), key="ta_pair")
        row = strength[strength["pair"] == pair].iloc[0]
        a, b = row["var_a"], row["var_b"]
        cells = t["pairs"].query("var_a == @a and var_b == @b").copy()
        flip = st.toggle(f"Swap: lines by {a} instead of {b}", False, key="ta_flip")
        x_col, color_col = ("level_b", "level_a") if flip else ("level_a", "level_b")
        x_var, color_var = (b, a) if flip else (a, b)

        shown = cells[cells["claims"] >= min_claims].copy()
        if shown.empty:
            st.warning(f"No cell in {pair} has {min_claims} claims. Lower the threshold, or accept that "
                       "this pair cannot be read.")
            return
        top = (shown.groupby(color_col)["weight"].sum().sort_values(ascending=False).head(8).index)
        shown = shown[shown[color_col].isin(top)]
        shown[x_col] = shown[x_col].astype(str)
        shown[color_col] = shown[color_col].astype(str)
        orders = {x_col: sort_segments(shown[x_col]), color_col: sort_segments(shown[color_col])}

        fig = line(shown.sort_values([color_col, x_col]), x=x_col, y="value", color=color_col, markers=True,
                   category_orders=orders, color_discrete_map=entity_colors(shown[color_col], None),
                   labels={x_col: x_var, "value": spec.label.split(" (")[0], color_col: color_var},
                   hover_data={"claims": ":,", "weight": ":,.0f", "lift": ":.2f"})
        fig.add_hline(y=overall, line_dash="dash", line_color="#898781")
        st.plotly_chart(style(fig, 400, f"{spec.label} by {x_var}, one line per {color_var}"),
                        use_container_width=True)
        st.caption("Parallel lines mean the two variables act independently - a main-effects model is "
                   "enough. Lines that cross or fan out are the interaction.")

        piv = shown.pivot_table(index=color_col, columns=x_col, values="lift", aggfunc="mean")
        piv = piv.reindex(index=sort_segments(piv.index), columns=sort_segments(piv.columns))
        if piv.notna().any().any():
            lim = float(np.nanmax(np.abs(np.log(piv.to_numpy(dtype=float)))))
            lim = min(max(lim, 0.1), 1.6)
            fig = px.imshow(np.log(piv.to_numpy(dtype=float)), x=list(piv.columns), y=list(piv.index),
                            color_continuous_scale=DIVERGING, zmin=-lim, zmax=lim, aspect="auto",
                            labels={"x": x_var, "y": color_var, "color": "log(actual / expected)"})
            text = np.vectorize(lambda v: "" if not np.isfinite(v) else f"{v:.2f}x")(piv.to_numpy(dtype=float))
            fig.update_traces(text=text, texttemplate="%{text}", textfont=dict(size=11),
                              hovertemplate=f"{color_var} %{{y}}<br>{x_var} %{{x}}<br>%{{text}} of expected<extra></extra>")
            fig.update_coloraxes(colorbar=dict(title=dict(text=""), tickvals=[-lim, 0, lim],
                                               ticktext=[f"{np.exp(-lim):.2f}x", "1.00x", f"{np.exp(lim):.2f}x"]))
            fig.update_xaxes(type="category", side="top")
            st.plotly_chart(style(fig, max(240, 44 * piv.shape[0] + 120)), use_container_width=True)
            st.caption("Actual over what the two one-way relativities predict. 1.00x means the main effects "
                       "explain the cell; away from 1.00x is the interaction, and the sign says which way.")
        st.download_button("⬇️ CSV", cells.to_csv(index=False), file_name=f"interaction_{a}_{b}.csv",
                           mime="text/csv", key="ta_dl_pair")

