"""Distribution profiles for a user-maintained list of variables, at user-chosen levels.

Numeric: summary stats, a configurable percentile grid, histogram with log transforms,
share of zero and negative values, outlier ratio (max / p99), PSI across a dimension.
Categorical: top-N shares with an <other> bucket, PSI across a dimension, new categories.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import grade, segment_key

LogMethod = Literal["none", "log1p", "log10", "signed_log"]
PRESET_BINS = [4, 10, 20, 50, 100]


class LogScale(BaseModel):
    method: LogMethod = "none"  # transform applied to values before histogram binning
    y: bool = False  # log count axis


class PsiRule(BaseModel):
    across: str  # dimension compared, e.g. pol_yr: each year vs the other years in the same partition
    bins: int = 10
    warn: float = 0.10
    fail: float = 0.25


class VarSpec(BaseModel):
    name: str
    type: Literal["auto", "numeric", "categorical"] = "auto"
    group_by: list[str] = ["src"]
    percentile_bins: int | list[float] = 20
    log_scale: LogScale = LogScale()
    hist_bins: int = 40
    clip: tuple[float, float] = (0.001, 0.999)
    allow_negative: bool = True
    negative_fail: float = 0.001
    outlier_ratio: float | None = None  # flag when max / p99 exceeds this
    psi: PsiRule | None = None
    top_n: int = 25
    max_categories: int = 500
    new_category_min_share: float = 0.005

    @field_validator("percentile_bins")
    @classmethod
    def _bins(cls, v):
        if isinstance(v, int):
            if not 2 <= v <= 1000:
                raise ValueError("percentile_bins must be between 2 and 1000")
        elif not v or any(not 0 <= p <= 1 for p in v):
            raise ValueError("custom percentiles must be within [0, 1]")
        return v


def percentile_probs(bins: int | list[float]) -> list[float]:
    """20 -> [0, .05, ..., 1]; a custom list is sorted, de-duplicated and always includes 0.99."""
    if isinstance(bins, int):
        probs = [round(i / bins, 6) for i in range(bins + 1)]
    else:
        probs = sorted({round(float(p), 6) for p in bins} | {0.0, 1.0})
    if 0.99 not in probs:
        probs = sorted(set(probs) | {0.99})
    return probs


def transform_sql(x: str, method: str) -> tuple[str, str]:
    """(transformed expression, validity predicate) for a log method."""
    if method == "log1p":
        return f"LN(1 + {x})", f"{x} > -1"
    if method == "log10":
        return f"LOG10({x})", f"{x} > 0"
    if method == "signed_log":
        return f"SIGN({x}) * LN(1 + ABS({x}))", "1=1"
    return x, "1=1"


def inverse_transform(v: np.ndarray, method: str) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if method == "log1p":
        return np.expm1(v)
    if method == "log10":
        return np.power(10.0, v)
    if method == "signed_log":
        return np.sign(v) * np.expm1(np.abs(v))
    return v


def psi(actual: np.ndarray, expected: np.ndarray, eps: float = 1e-4) -> float:
    a = np.clip(np.asarray(actual, float), eps, None)
    e = np.clip(np.asarray(expected, float), eps, None)
    a, e = a / a.sum(), e / e.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _sort_key(v):
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


@register_check("distribution")
class Distribution(Check):
    title = "Distributions"
    icon = "📊"
    description = ("Profiles the listed variables at the chosen levels. Add or remove variables and levels, choose the "
                   "number of percentile bins and log scales, and flag shifts with PSI.")
    default_order = 30

    class Config(CheckConfig):
        variables: list[VarSpec] = []

    # ---- helpers -------------------------------------------------------------
    def kind(self, spec: VarSpec) -> str:
        if spec.type != "auto":
            return spec.type
        return "numeric" if self.schema.is_amount(spec.name) or (
            self.schema.is_numeric(spec.name) and not spec.name.endswith(("_cd", "_id", "_yr"))) else "categorical"

    def _seg(self, row, group_by) -> str:
        return segment_key({g: row[g] for g in group_by})

    def _psi_findings(self, spec: VarSpec, counts: pd.DataFrame, key: str) -> tuple[list[dict], pd.DataFrame]:
        """counts: group_by cols + key + n.

        Each segment is compared with every other value of `across` in the same partition (the
        other group_by dims); the reported PSI is the median of those pairwise PSIs, so a single
        broken segment does not raise the PSI of all its neighbours.
        """
        rule = spec.psi
        if rule is None or rule.across not in spec.group_by:
            return [], pd.DataFrame()
        others = [g for g in spec.group_by if g != rule.across]
        wide = counts.pivot_table(index=spec.group_by, columns=key, values="n", aggfunc="sum", fill_value=0)
        idx_frame = wide.index.to_frame(index=False).astype(str)
        part = idx_frame[others].agg("|".join, axis=1) if others else pd.Series("", index=idx_frame.index)
        findings, rows = [], []
        mat = wide.to_numpy(dtype=float)
        for i in range(len(wide)):
            peers = [j for j in np.flatnonzero(part.to_numpy() == part.iloc[i]) if j != i and mat[j].sum() > 0]
            if not peers or mat[i].sum() == 0:
                continue
            pair = [psi(mat[i], mat[j]) for j in peers]
            value = float(np.median(pair))
            status = grade(value, rule.warn, rule.fail)
            seg_vals = dict(zip(spec.group_by, wide.index[i] if isinstance(wide.index[i], tuple) else (wide.index[i],)))
            seg = segment_key(seg_vals)
            rows.append({"segment": seg, "psi": value, "max_pairwise_psi": float(np.max(pair)), "status": status,
                         "n": int(mat[i].sum())})
            findings.append(dict(variable=spec.name, item=f"across {rule.across}", segment=seg, metric="psi",
                                 value=value, threshold=rule.fail, status=status,
                                 detail=f"median PSI vs {len(peers)} other {rule.across} values"
                                        + (f" within {', '.join(others)}" if others else "")))
        return findings, pd.DataFrame(rows)

    # ---- numeric -------------------------------------------------------------
    def profile_numeric(self, spec: VarSpec) -> tuple[list[dict], dict[str, pd.DataFrame]]:
        s, lit = self.schema, self.ctx.dialect.lit
        s.validate([spec.name] + spec.group_by)
        x = s.ref(spec.name)
        probs = percentile_probs(spec.percentile_bins)
        df = self.query(f"{spec.name} stats", self.ctx.render_sql(
            "dist_numeric_stats.sql.j2", x=x, group_by=spec.group_by, probs=probs))
        findings, stats, pct_rows = [], [], []
        for _, r in df.iterrows():
            seg = self._seg(r, spec.group_by)
            pcts = list(r["pcts"]) if r["pcts"] is not None else [None] * len(probs)
            p99 = pcts[probs.index(0.99)]
            nn = int(r["n_nonnull"] or 0)
            pct_neg = (r["n_negative"] or 0) / nn if nn else None
            ratio = (r["max"] / p99) if p99 and p99 > 0 and r["max"] is not None else None
            stats.append({"segment": seg, "n_rows": int(r["n_rows"]), "n_nonnull": nn,
                          "pct_zero": (r["n_zero"] or 0) / nn if nn else None, "pct_negative": pct_neg,
                          "min": r["min"], "mean": r["mean"], "p50": pcts[probs.index(0.5)] if 0.5 in probs else None,
                          "p99": p99, "max": r["max"], "std": r["std"], "max_to_p99": ratio})
            pct_rows += [{"segment": seg, "percentile": p, "value": v} for p, v in zip(probs, pcts)]
            if not spec.allow_negative and pct_neg is not None:
                findings.append(dict(variable=spec.name, item="negative values", segment=seg, metric="pct_negative",
                                     value=pct_neg, threshold=spec.negative_fail,
                                     status=grade(pct_neg, 0.0, spec.negative_fail),
                                     detail=f"{int(r['n_negative'] or 0):,} negative of {nn:,}"))
            if spec.outlier_ratio and ratio is not None:
                findings.append(dict(variable=spec.name, item="outliers", segment=seg, metric="max_to_p99",
                                     value=ratio, threshold=spec.outlier_ratio,
                                     status=grade(ratio, spec.outlier_ratio, spec.outlier_ratio * 10),
                                     detail=f"max {r['max']:,.2f} vs p99 {p99:,.2f}"))
        tables = {"stats": pd.DataFrame(stats), "percentiles": pd.DataFrame(pct_rows)}

        if spec.psi and spec.psi.across in spec.group_by:
            edge_probs = [i / spec.psi.bins for i in range(1, spec.psi.bins)]
            lim = self.query(f"{spec.name} psi edges", self.ctx.render_sql(
                "dist_limits.sql.j2", v=x, probs=edge_probs, where=f"{x} IS NOT NULL")).iloc[0]
            edges = sorted({float(e) for e in (lim["limits"] if lim["limits"] is not None else [])})
            if edges:
                case = "CASE " + " ".join(f"WHEN {x} <= {lit(e)} THEN {i}" for i, e in enumerate(edges)) + f" ELSE {len(edges)} END"
                counts = self.query(f"{spec.name} psi buckets", self.ctx.render_sql(
                    "dist_bucket_counts.sql.j2", group_by=spec.group_by, bucket_expr=case, where=f"{x} IS NOT NULL"))
                f, psi_tbl = self._psi_findings(spec, counts, "bucket")
                findings += f
                tables["psi"] = psi_tbl
        return findings, tables

    def histogram(self, spec: VarSpec) -> pd.DataFrame:
        """Histogram on the transformed scale, clipped to the configured percentiles (UI only)."""
        s, lit = self.schema, self.ctx.dialect.lit
        x = s.ref(spec.name)
        v, valid = transform_sql(x, spec.log_scale.method)
        where = f"{x} IS NOT NULL AND {valid}"
        lim = self.query(f"{spec.name} histogram limits", self.ctx.render_sql(
            "dist_limits.sql.j2", v=v, probs=list(spec.clip), where=where)).iloc[0]
        if not lim["n_valid"]:
            return pd.DataFrame()
        lo, hi = (float(t) for t in lim["limits"])
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = float(lim["vmin"]), float(lim["vmax"]) + 1e-9
        bins = spec.hist_bins
        width = (hi - lo) / bins
        bucket = (f"LEAST(GREATEST(CAST(FLOOR((GREATEST(LEAST({v}, {lit(hi)}), {lit(lo)}) - {lit(lo)}) / {lit(width)}) AS INT), 0), {bins - 1})")
        counts = self.query(f"{spec.name} histogram", self.ctx.render_sql(
            "dist_bucket_counts.sql.j2", group_by=spec.group_by, bucket_expr=bucket, where=where))
        if counts.empty:
            return counts
        counts["segment"] = [self._seg(r, spec.group_by) for _, r in counts.iterrows()]
        counts["left"] = lo + counts["bucket"] * width
        counts["right"] = counts["left"] + width
        counts["center"] = counts["left"] + width / 2
        counts["range"] = [f"{a:,.2f} to {b:,.2f}" for a, b in zip(inverse_transform(counts["left"], spec.log_scale.method),
                                                                   inverse_transform(counts["right"], spec.log_scale.method))]
        counts["share"] = counts["n"] / counts.groupby("segment")["n"].transform("sum")
        total = self.query(f"{spec.name} excluded", f"SELECT COUNT(*) AS n FROM {self.project.table} "
                                                     f"WHERE {x} IS NOT NULL AND NOT ({valid})").iloc[0]["n"]
        counts.attrs.update(excluded=int(total or 0), lo=lo, hi=hi, method=spec.log_scale.method)
        return counts.sort_values(["segment", "bucket"])

    # ---- categorical -----------------------------------------------------------
    def profile_categorical(self, spec: VarSpec) -> tuple[list[dict], dict[str, pd.DataFrame]]:
        self.schema.validate([spec.name] + spec.group_by)
        df = self.query(f"{spec.name} categories", self.ctx.render_sql(
            "dist_categorical.sql.j2", x=self.schema.ref(spec.name), group_by=spec.group_by,
            max_categories=spec.max_categories))
        if df.empty:
            return [], {}
        df["segment"] = [self._seg(r, spec.group_by) for _, r in df.iterrows()]
        df["share"] = df["n"] / df.groupby("segment")["n"].transform("sum")
        findings, tables = [], {}
        seg_n = df.groupby("segment")["n"].sum()
        n_cat = df[~df["category"].isin(["<null>", "<other>"])].groupby("segment")["category"].nunique()
        tables["summary"] = pd.DataFrame({"n_rows": seg_n, "n_categories": n_cat}).reset_index()
        top = df.sort_values("n", ascending=False).groupby("segment").head(spec.top_n)
        tables["top"] = top[["segment", "category", "n", "share"]].sort_values(["segment", "n"], ascending=[True, False])

        f, psi_tbl = self._psi_findings(spec, df, "category")
        findings += f
        if not psi_tbl.empty:
            tables["psi"] = psi_tbl

        if spec.psi and spec.psi.across in spec.group_by:
            across = spec.psi.across
            others = [g for g in spec.group_by if g != across]
            new_rows = []
            part_key = df[others].astype(str).agg("|".join, axis=1) if others else pd.Series("", index=df.index)
            for _, part in df.assign(_p=part_key).groupby("_p"):
                order = sorted(part[across].unique(), key=_sort_key)
                seen: set = set()
                for i, val in enumerate(order):
                    cur = part[part[across] == val]
                    cats = set(cur.loc[cur["n"] > 0, "category"]) - {"<null>", "<other>"}
                    if i > 0:
                        new = cur[cur["category"].isin(cats - seen) & (cur["share"] >= spec.new_category_min_share)]
                        seg = cur["segment"].iloc[0]
                        status = "warn" if len(new) else "pass"
                        findings.append(dict(variable=spec.name, item=f"new vs earlier {across}", segment=seg,
                                             metric="new_categories", value=len(new), threshold=0, status=status,
                                             detail=", ".join(f"{c} ({s:.1%})" for c, s in zip(new["category"], new["share"]))[:500]))
                        new_rows += [{"segment": seg, "category": c, "share": s} for c, s in zip(new["category"], new["share"])]
                    seen |= cats
            tables["new_categories"] = pd.DataFrame(new_rows, columns=["segment", "category", "share"])
        return findings, tables

    # ---- run -----------------------------------------------------------------
    def profile(self, spec: VarSpec) -> tuple[list[dict], dict[str, pd.DataFrame]]:
        return self.profile_numeric(spec) if self.kind(spec) == "numeric" else self.profile_categorical(spec)

    def run(self) -> CheckResult:
        findings, tables = [], {}
        for spec in self.cfg.variables:
            f, t = self.profile(spec)
            if not f:  # still register the variable as profiled
                f = [dict(variable=spec.name, item="profiled", segment="ALL", metric="profiled", value=None,
                          threshold=None, status="info", detail=f"levels: {', '.join(spec.group_by) or 'ALL'}")]
            findings += f
            tables.update({f"{spec.name}::{k}": v for k, v in t.items()})
        return self.result(findings, tables)

    # ---- UI ------------------------------------------------------------------
    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        seg_opts = self.segment_options()
        st.markdown("**Variables to profile**: add or delete rows. Levels are comma-separated. Percentile bins take a "
                    "number (e.g. 20) or a list of percentiles (e.g. `0.01, 0.05, 0.5, 0.95, 0.99`).")
        rows = [{"variable": v.name, "type": v.type, "levels": ", ".join(v.group_by),
                 "percentile_bins": v.percentile_bins if isinstance(v.percentile_bins, int) else ", ".join(map(str, v.percentile_bins)),
                 "log_method": v.log_scale.method, "log_y": v.log_scale.y, "hist_bins": v.hist_bins,
                 "psi_across": v.psi.across if v.psi else None, "allow_negative": v.allow_negative,
                 "outlier_ratio": v.outlier_ratio, "top_n": v.top_n} for v in cfg.variables]
        cols = ["variable", "type", "levels", "percentile_bins", "log_method", "log_y", "hist_bins", "psi_across",
                "allow_negative", "outlier_ratio", "top_n"]
        edited = st.data_editor(
            pd.DataFrame(rows, columns=cols).astype({"percentile_bins": "string", "levels": "string"}),
            num_rows="dynamic", use_container_width=True, key="dist_vars",
            column_config={
                "variable": st.column_config.SelectboxColumn(options=self.schema.names(), required=True),
                "type": st.column_config.SelectboxColumn(options=["auto", "numeric", "categorical"], default="auto"),
                "levels": st.column_config.TextColumn(help=f"Options: {', '.join(seg_opts)}", default="src"),
                "percentile_bins": st.column_config.TextColumn(default="20"),
                "log_method": st.column_config.SelectboxColumn(options=list(LogMethod.__args__), default="none"),
                "log_y": st.column_config.CheckboxColumn(default=False),
                "hist_bins": st.column_config.NumberColumn(min_value=5, max_value=200, default=40),
                "psi_across": st.column_config.SelectboxColumn(options=seg_opts),
                "allow_negative": st.column_config.CheckboxColumn(default=True),
                "outlier_ratio": st.column_config.NumberColumn(min_value=1.0),
                "top_n": st.column_config.NumberColumn(min_value=1, max_value=500, default=25),
            })
        old = {v.name: v for v in cfg.variables}
        variables = []
        for r in edited.to_dict("records"):
            if not r.get("variable"):
                continue
            na = lambda x: x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA  # noqa: E731
            base = old.get(r["variable"], VarSpec(name=r["variable"]))
            levels = [g.strip() for g in (r.get("levels") or "").split(",") if g.strip()]
            bins_txt = str(r.get("percentile_bins") or "20").strip()
            bins = int(bins_txt) if bins_txt.isdigit() else [float(p) for p in bins_txt.split(",") if p.strip()]
            across = None if na(r.get("psi_across")) else r["psi_across"]
            psi_rule = (base.psi.model_copy(update={"across": across}) if base.psi else PsiRule(across=across)) if across else None
            variables.append(base.model_copy(update={
                "type": r.get("type") or "auto", "group_by": levels, "percentile_bins": bins,
                "log_scale": LogScale(method=r.get("log_method") or "none", y=bool(r.get("log_y"))),
                "hist_bins": int(r["hist_bins"]) if not na(r.get("hist_bins")) else 40,
                "psi": psi_rule, "allow_negative": bool(r.get("allow_negative", True)),
                "outlier_ratio": None if na(r.get("outlier_ratio")) else float(r["outlier_ratio"]),
                "top_n": int(r["top_n"]) if not na(r.get("top_n")) else 25}))
        new.variables = variables
        return new

    def render(self, result):
        import streamlit as st

        from gl_dq.ui import state
        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import MAX_SERIES

        if not self.cfg.variables:
            st.info("No variables configured. Add them under ⚙️ Settings.")
            return
        names = [v.name for v in self.cfg.variables]
        c1, c2 = st.columns([2, 3])
        name = c1.selectbox("Variable", names, key="dist_var")
        base = next(v for v in self.cfg.variables if v.name == name)
        kind = self.kind(base)
        c2.caption(f"Type: **{kind}** · configured levels: **{', '.join(base.group_by) or 'ALL'}**")

        # interactive overrides for this variable (not saved unless requested)
        with st.container(border=True):
            seg_opts = self.segment_options()
            k1, k2, k3, k4 = st.columns([3, 2, 2, 2])
            levels = k1.multiselect("Levels", seg_opts, default=[g for g in base.group_by if g in seg_opts], key=f"dl_{name}")
            spec = base.model_copy(update={"group_by": levels})
            if kind == "numeric":
                preset = base.percentile_bins if isinstance(base.percentile_bins, int) and base.percentile_bins in PRESET_BINS else "custom"
                choice = k2.selectbox("Percentile bins", PRESET_BINS + ["custom"],
                                      index=(PRESET_BINS + ["custom"]).index(preset), key=f"dpb_{name}")
                if choice == "custom":
                    default_txt = ", ".join(map(str, base.percentile_bins)) if isinstance(base.percentile_bins, list) else "0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99"
                    txt = st.text_input("Custom percentiles (0–1, comma-separated)", default_txt, key=f"dpc_{name}")
                    try:
                        bins = [float(p) for p in txt.split(",") if p.strip()]
                    except ValueError:
                        st.error("Could not parse percentiles")
                        bins = base.percentile_bins
                else:
                    bins = int(choice)
                method = k3.selectbox("Log transform (x)", list(LogMethod.__args__),
                                      index=list(LogMethod.__args__).index(base.log_scale.method), key=f"dlm_{name}",
                                      help="log1p handles zeros, signed_log handles negatives, log10 drops values ≤ 0")
                hist_bins = k4.number_input("Histogram bins", 5, 200, base.hist_bins, key=f"dhb_{name}")
                log_y = st.toggle("Log scale on the count axis", base.log_scale.y, key=f"dly_{name}")
                spec = spec.model_copy(update={"percentile_bins": bins, "hist_bins": int(hist_bins),
                                               "log_scale": LogScale(method=method, y=log_y)})
            else:
                top_n = k2.number_input("Top N", 1, 500, base.top_n, key=f"dtn_{name}")
                spec = spec.model_copy(update={"top_n": int(top_n)})
            if spec.psi and spec.psi.across not in spec.group_by:
                st.caption(f"PSI needs '{spec.psi.across}' in the levels.")
            if spec != base and st.button(f"💾 Keep these settings for {name}", key=f"dsave_{name}"):
                cfg = self.cfg.model_copy(deep=True)
                cfg.variables = [spec if v.name == name else v for v in cfg.variables]
                state.set_session_config(self.name, cfg)
                st.rerun()

        f, tables = state.cached_method(self.name, self.cfg, "profile", spec=spec)
        flagged = [x for x in f if x["status"] in ("warn", "fail")]
        if flagged:
            st.warning(f"{len(flagged)} flagged finding(s) for {name}")
        if kind == "numeric":
            all_segs = list(tables["stats"].sort_values("n_rows", ascending=False)["segment"])
            chosen = all_segs
            if len(all_segs) > MAX_SERIES and len(spec.group_by) > 2:
                st.caption(f"{len(all_segs)} segments at this level — charts show the segments you pick "
                           "(the tables below cover all of them).")
                chosen = st.multiselect("Segments to chart", all_segs, default=all_segs[:MAX_SERIES],
                                        key=f"dseg_{name}") or all_segs[:MAX_SERIES]
            self._render_numeric(spec, tables, state, chosen)
        else:
            self._render_categorical(spec, tables)
        if "psi" in tables and not tables["psi"].empty:
            with st.expander(f"PSI across {spec.psi.across}", expanded=bool(flagged)):
                status_table(tables["psi"], number_formats={"psi": "%.4f"})

    def _render_numeric(self, spec, tables, state, chosen=None):
        import plotly.express as px
        import streamlit as st

        from gl_dq.core.results import parse_segment
        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import MAX_SERIES, line, series_encoding, style

        stats = tables["stats"].sort_values("segment")
        keep = None if chosen is None or len(chosen) == len(stats) else set(chosen)
        status_table(stats, percent_cols=["pct_zero", "pct_negative"],
                     number_formats={c: "%.2f" for c in ["min", "mean", "p50", "p99", "max", "std", "max_to_p99"]})
        tab_h, tab_p = st.tabs(["Histogram", "Percentiles"])
        with tab_h:
            hist = state.cached_method(self.name, self.cfg, "histogram", spec=spec)
            if keep is not None:
                hist = hist[hist["segment"].isin(keep)]
            if hist.empty:
                st.info("No valid values to plot.")
            else:
                axis = {"none": "value", "log1p": "ln(1 + x)", "log10": "log10(x)",
                        "signed_log": "sign(x)·ln(1 + |x|)"}[spec.log_scale.method]
                enc = series_encoding(hist, "segment", self.project.sources, spec.group_by)
                n_panels = hist[enc["facet_col"]].nunique() if "facet_col" in enc else 1
                # step lines stay readable with several overlaid segments (overlaid bars do not)
                fig = line(hist, x="center", y="share", line_shape="hvh",
                              hover_data={"range": True, "n": ":,", "center": False, "share": ":.2%", "segment": True},
                              labels={"center": axis, "share": "share of segment"}, log_y=spec.log_scale.y, **enc)
                fig.update_traces(line=dict(width=1.5))
                style(fig, height=380 if n_panels == 1 else 260 * ((n_panels + 2) // 3),
                      title=f"{spec.name}: share of segment by value (hover shows the original range)")
                fig.update_xaxes(title_text=None, showticklabels=True)
                st.caption(f"x axis: {axis}")
                fig.update_layout(bargap=0.02)
                st.plotly_chart(fig, use_container_width=True)
                notes = [f"clipped to percentiles {spec.clip[0]:.3%}–{spec.clip[1]:.3%} (edge bins include the tails)"]
                if hist.attrs.get("excluded"):
                    notes.append(f"{hist.attrs['excluded']:,} values outside the domain of {spec.log_scale.method} were excluded")
                st.caption(" · ".join(notes))
        with tab_p:
            pct = tables["percentiles"]
            wide = pct.pivot(index="percentile", columns="segment", values="value")
            if keep is not None:
                pct = pct[pct["segment"].isin(keep)]
            pct = pct.join(pd.DataFrame([parse_segment(s) for s in pct["segment"]], index=pct.index))
            enc = series_encoding(pct, "segment", self.project.sources, spec.group_by)
            n_panels = pct[enc["facet_col"]].nunique() if "facet_col" in enc else 1
            fig = line(pct, x="percentile", y="value", markers=True, log_y=spec.log_scale.method != "none",
                          hover_data={"segment": True}, **enc)
            fig.update_xaxes(tickformat=".0%")
            style(fig, height=360 if n_panels == 1 else 240 * ((n_panels + 2) // 3))
            st.plotly_chart(fig, use_container_width=True)
            if spec.log_scale.method != "none" and (pct["value"] <= 0).any():
                st.caption("Log y axis: values ≤ 0 are not shown on the chart (see the table).")
            st.dataframe(wide.style.format("{:,.2f}"), use_container_width=True)

    def _render_categorical(self, spec, tables):
        import plotly.express as px
        import streamlit as st

        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import SEQ_SCALE, series_encoding, style

        if not tables:
            st.info("No data.")
            return
        st.dataframe(tables["summary"], hide_index=True, use_container_width=True)
        top = tables["top"]
        n_seg, n_cat = top["segment"].nunique(), top["category"].nunique()
        if n_seg <= 4:
            fig = px.bar(top, x="share", y="category", barmode="group", orientation="h",
                         hover_data={"n": ":,", "share": ":.2%"}, **series_encoding(top, "segment", self.project.sources))
            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            style(fig, height=max(320, 22 * n_cat * max(1, n_seg // 2) + 60))
        else:  # many segments: share heatmap (category x segment) reads better than dozens of bars
            heat = top.pivot_table(index="category", columns="segment", values="share", aggfunc="sum").fillna(0)
            heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
            fig = px.imshow(heat, text_auto=".1%" if heat.shape[1] <= 12 else False, aspect="auto",
                            color_continuous_scale=SEQ_SCALE, zmin=0, labels={"color": "share"})
            style(fig, height=max(320, 22 * len(heat) + 120))
        st.plotly_chart(fig, use_container_width=True)
        if "new_categories" in tables and not tables["new_categories"].empty:
            st.markdown("**New categories** (absent from all earlier values of the PSI dimension)")
            status_table(tables["new_categories"], percent_cols=["share"])
