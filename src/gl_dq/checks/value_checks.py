"""Are the values themselves sane?

Categorical: the three sources should describe the same book with the same vocabulary, so a value
used by one source and not the others is usually a mapping difference rather than a real difference.
Numeric: values should sit inside a plausible range, keep their sign, and mean the same thing in
every source (a median ten times larger in one source is a unit problem, not a risk difference).

Complements missing_rate (absence) and distribution (shape).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import grade, segment_key


class CategoricalRule(BaseModel):
    max_values: int = 2000  # above this the value set is reported but not compared
    warn_unique: float = 0.0  # warn when any value is used by one source only
    warn_rows: float = 0.005  # share of rows carrying a source-unique value
    fail_rows: float = 0.05


class MedianRatio(BaseModel):
    warn: float = 10.0  # max/min of the per-source medians: catches cents vs dollars
    fail: float = 50.0


class NumericRule(BaseModel):
    min: float | None = None  # plausible bounds; None = unbounded on that side
    max: float | None = None
    allow_negative: bool = True
    allow_zero: bool = True
    discrete: bool = False  # a ladder (deductibles, limits): report the share on its most common value
    top_value_warn: float | None = None
    # compare the median across sources? Turn off where the grain genuinely differs (e.g. exposure is
    # per location in one source and per policy in another), so real structure is not read as a bug.
    compare_medians: bool = True
    median_ratio: MedianRatio | None = None  # per-column override of the check-wide thresholds
    out_of_bounds_warn: float = 0.0
    out_of_bounds_fail: float = 0.001
    sign_warn: float = 0.0
    sign_fail: float = 0.01


@register_check("value_checks")
class ValueChecks(Check):
    title = "Values check"
    icon = "🔤"
    description = ("Are values consistent across the sources and plausible in themselves? Flags values used by "
                   "one source only, values outside their expected range, unexpected signs, sentinel spikes, "
                   "and medians that differ by source in a way that looks like a unit error.")
    default_order = 27

    class Config(CheckConfig):
        categorical: dict[str, CategoricalRule] = {}
        numeric: dict[str, NumericRule] = {}
        median_ratio: MedianRatio = MedianRatio()

    # ---- categorical ---------------------------------------------------------------
    def value_matrix(self, column: str) -> pd.DataFrame:
        """value x source counts for one column."""
        self.schema.validate([column])
        return self.query(f"{column} values", self.ctx.render_sql(
            "value_categorical.sql.j2", x=self.schema.ref(column), src_col=self.project.src_col))

    def categorical_findings(self, column: str, rule: CategoricalRule, counts: pd.DataFrame):
        src_col = self.project.src_col
        findings: list[dict] = []
        if counts.empty:
            return findings, pd.DataFrame()
        wide = counts.pivot_table(index="value", columns=src_col, values="n", aggfunc="sum", fill_value=0)
        sources = [s for s in wide.columns]
        n_values = len(wide)
        findings.append(dict(variable=column, item="inventory", segment="ALL", metric="n_values",
                             value=n_values, threshold=None, status="info",
                             detail=f"{n_values:,} distinct values across {len(sources)} sources"))
        if n_values > rule.max_values:
            self.note(f"{column}: {n_values:,} distinct values — too many to compare across sources "
                      f"(raise max_values in the config to force it).")
            return findings, wide

        used_by = (wide > 0).sum(axis=1)
        for src in sources:
            unique_vals = wide.index[(wide[src] > 0) & (used_by == 1)]
            n_rows_src = wide[src].sum()
            rows_unique = int(wide.loc[unique_vals, src].sum()) if len(unique_vals) else 0
            share = rows_unique / n_rows_src if n_rows_src else 0.0
            seg = segment_key({src_col: src})
            findings.append(dict(
                variable=column, item="used by one source only", segment=seg, metric="source_unique_values",
                value=len(unique_vals), threshold=rule.warn_unique,
                status=grade(len(unique_vals), rule.warn_unique, None),
                detail=", ".join(map(str, unique_vals[:10])) + ("…" if len(unique_vals) > 10 else "")))
            findings.append(dict(
                variable=column, item="used by one source only", segment=seg, metric="pct_rows_source_unique",
                value=share, threshold=rule.fail_rows, status=grade(share, rule.warn_rows, rule.fail_rows),
                detail=f"{rows_unique:,} of {int(n_rows_src):,} rows"))
        return findings, wide

    # ---- numeric ---------------------------------------------------------------
    def numeric_profile(self, column: str, rule: NumericRule) -> pd.DataFrame:
        self.schema.validate([column])
        df = self.query(f"{column} plausibility", self.ctx.render_sql(
            "value_numeric.sql.j2", x=self.schema.ref(column), src_col=self.project.src_col,
            lo=rule.min, hi=rule.max))
        df["median"] = [None if v is None or (isinstance(v, float) and np.isnan(v)) else float(np.ravel(v)[0])
                        for v in df["median"]]
        return df

    def numeric_findings(self, column: str, rule: NumericRule, df: pd.DataFrame,
                         value_counts: pd.DataFrame | None):
        src_col = self.project.src_col
        findings: list[dict] = []
        for _, r in df.iterrows():
            nn = int(r["n_nonnull"] or 0)
            seg = segment_key({src_col: r[src_col]})
            if not nn:
                continue
            if rule.min is not None or rule.max is not None:
                out = (int(r["n_below"] or 0) + int(r["n_above"] or 0)) / nn
                bounds = f"[{rule.min if rule.min is not None else '-inf'}, {rule.max if rule.max is not None else 'inf'}]"
                findings.append(dict(variable=column, item=f"outside {bounds}", segment=seg,
                                     metric="pct_out_of_bounds", value=out, threshold=rule.out_of_bounds_fail,
                                     status=grade(out, rule.out_of_bounds_warn, rule.out_of_bounds_fail),
                                     detail=f"{int(r['n_below'] or 0):,} below, {int(r['n_above'] or 0):,} above "
                                            f"of {nn:,} · observed {r['min']:,.0f} to {r['max']:,.0f}"))
            if not rule.allow_negative:
                neg = int(r["n_negative"] or 0) / nn
                findings.append(dict(variable=column, item="negative values", segment=seg, metric="pct_negative",
                                     value=neg, threshold=rule.sign_fail,
                                     status=grade(neg, rule.sign_warn, rule.sign_fail),
                                     detail=f"{int(r['n_negative'] or 0):,} of {nn:,}"))
            if not rule.allow_zero:
                zero = int(r["n_zero"] or 0) / nn
                findings.append(dict(variable=column, item="zero values", segment=seg, metric="pct_zero",
                                     value=zero, threshold=rule.sign_fail,
                                     status=grade(zero, rule.sign_warn, rule.sign_fail),
                                     detail=f"{int(r['n_zero'] or 0):,} of {nn:,}"))
            if rule.discrete and rule.top_value_warn is not None and value_counts is not None \
                    and r[src_col] in value_counts.columns:
                col = value_counts[r[src_col]]
                total = col.sum()
                if total:
                    share = float(col.max() / total)
                    findings.append(dict(variable=column, item=f"most common value: {col.idxmax()}", segment=seg,
                                         metric="top_value_share", value=share, threshold=rule.top_value_warn,
                                         status=grade(share, rule.top_value_warn, None),
                                         detail=f"{int(col.max()):,} of {int(total):,} rows"))
        medians = [m for m in df["median"] if m not in (None, 0) and not pd.isna(m)]
        thresholds = rule.median_ratio or self.cfg.median_ratio
        if rule.compare_medians and len(medians) > 1:
            ratio = max(medians) / min(medians)
            findings.append(dict(variable=column, item="median by source", segment="ALL",
                                 metric="median_ratio_across_src", value=ratio,
                                 threshold=thresholds.fail,
                                 status=grade(ratio, thresholds.warn, thresholds.fail),
                                 detail=" · ".join(f"{s}: {m:,.2f}" for s, m in zip(df[src_col], df["median"])
                                                   if m is not None and not pd.isna(m))))
        return findings

    # ---- run ---------------------------------------------------------------------
    def run(self) -> CheckResult:
        findings, tables = [], {}
        matrices: dict[str, pd.DataFrame] = {}
        for column, rule in self.cfg.categorical.items():
            counts = self.value_matrix(column)
            f, wide = self.categorical_findings(column, rule, counts)
            findings += f
            if not wide.empty:
                matrices[column] = wide
                tables[f"values::{column}"] = wide.reset_index()
        for column, rule in self.cfg.numeric.items():
            df = self.numeric_profile(column, rule)
            tables[f"numeric::{column}"] = df
            findings += self.numeric_findings(column, rule, df, matrices.get(column))
        if not self.cfg.categorical and not self.cfg.numeric:
            self.note("No columns configured yet — add them under ⚙️ Settings.")
        return self.result(findings, tables)

    # ---- UI ---------------------------------------------------------------------
    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        cols = self.schema.names(include_derived=False)
        st.markdown("**Categorical columns** — compared across sources")
        picked = st.multiselect("Columns", cols, default=list(cfg.categorical), key="vc_cat")
        new.categorical = {c: cfg.categorical.get(c, CategoricalRule()) for c in picked}
        st.markdown("**Numeric columns** — plausible range and sign. Bounds live in the YAML config, "
                    "so they are reviewed like code.")
        rows = [{"column": c, "min": r.min, "max": r.max, "allow_negative": r.allow_negative,
                 "allow_zero": r.allow_zero, "discrete": r.discrete, "top_value_warn": r.top_value_warn}
                for c, r in cfg.numeric.items()]
        edited = st.data_editor(
            pd.DataFrame(rows, columns=["column", "min", "max", "allow_negative", "allow_zero", "discrete",
                                        "top_value_warn"]),
            num_rows="dynamic", use_container_width=True, key="vc_num",
            column_config={"column": st.column_config.SelectboxColumn(options=cols, required=True)})
        numeric = {}
        for r in edited.to_dict("records"):
            if not r.get("column"):
                continue
            base = cfg.numeric.get(r["column"], NumericRule())
            na = lambda v: v is None or (isinstance(v, float) and np.isnan(v))  # noqa: E731
            numeric[r["column"]] = base.model_copy(update={
                "min": None if na(r.get("min")) else float(r["min"]),
                "max": None if na(r.get("max")) else float(r["max"]),
                "allow_negative": bool(r.get("allow_negative", True)),
                "allow_zero": bool(r.get("allow_zero", True)),
                "discrete": bool(r.get("discrete", False)),
                "top_value_warn": None if na(r.get("top_value_warn")) else float(r["top_value_warn"])})
        new.numeric = numeric
        c1, c2 = st.columns(2)
        new.median_ratio = MedianRatio(
            warn=c1.number_input("Median ratio warn", 1.0, value=cfg.median_ratio.warn, key="vc_mrw"),
            fail=c2.number_input("Median ratio fail", 1.0, value=cfg.median_ratio.fail, key="vc_mrf"))
        return new

    def render(self, result: CheckResult):
        import plotly.express as px
        import streamlit as st

        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import SEQ_SCALE, style

        f = result.findings
        tab_c, tab_n = st.tabs(["Across sources (categorical)", "Plausibility (numeric)"])
        with tab_c:
            cats = list(self.cfg.categorical)
            if not cats:
                st.info("No categorical columns configured.")
            else:
                # lead with the columns that actually differ across sources
                unique_by_col = (f[f["metric"] == "source_unique_values"].groupby("variable")["value"].sum()
                                 if not f.empty else pd.Series(dtype=float))
                cats = sorted(cats, key=lambda c: -float(unique_by_col.get(c, 0)))
                col = st.selectbox("Column", cats, key="vc_col",
                                   format_func=lambda c: f"{c}  ⚠ {int(unique_by_col.get(c, 0))}"
                                   if unique_by_col.get(c, 0) else c)
                wide = result.tables.get(f"values::{col}")
                if wide is None or wide.empty:
                    st.info("No values found.")
                else:
                    wide = wide.set_index("value")
                    share = wide / wide.sum()
                    used_by = (wide > 0).sum(axis=1)
                    only = wide[used_by == 1]
                    st.markdown(f"**{len(wide):,} values** · **{len(only)}** used by one source only")
                    if len(only):
                        rows = [{"value": v, "source": only.columns[list(only.loc[v] > 0).index(True)],
                                 "rows": int(only.loc[v].max())} for v in only.index]
                        status_table(pd.DataFrame(rows).sort_values("rows", ascending=False))
                    top = share.loc[share.max(axis=1).sort_values(ascending=False).index[:40]]
                    fig = px.imshow(top, text_auto=".1%" if len(top) <= 20 else False, aspect="auto",
                                    color_continuous_scale=SEQ_SCALE, zmin=0,
                                    labels={"color": "share of source"})
                    st.plotly_chart(style(fig, height=max(300, 20 * len(top) + 120),
                                          title=f"{col}: share of each source's rows"),
                                    use_container_width=True)
        with tab_n:
            if not self.cfg.numeric:
                st.info("No numeric columns configured.")
            else:
                num = f[f["metric"].isin(["pct_out_of_bounds", "pct_negative", "pct_zero", "top_value_share",
                                          "median_ratio_across_src"])]
                status_table(num[["variable", "segment", "metric", "value", "status", "detail"]],
                             percent_cols=["value"])
                col = st.selectbox("Column detail", list(self.cfg.numeric), key="vc_ncol")
                st.dataframe(result.tables.get(f"numeric::{col}", pd.DataFrame()), hide_index=True,
                             use_container_width=True)
