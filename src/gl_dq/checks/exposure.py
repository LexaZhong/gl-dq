"""Exposure summary by exposure base (expn_bs_std), plus exposure sanity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gl_dq.checks._recon import pipeline_agg
from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import grade, segment_key


@register_check("exposure")
class Exposure(Check):
    title = "Exposure summary"
    icon = "📐"
    description = ("Exposure totals and premium per exposure unit by exposure base. Flags negative exposure, zero "
                   "exposure with positive premium, and class codes mapped to more than one exposure base.")
    default_order = 60

    class Config(CheckConfig):
        summary_by: list[str] = ["src", "pol_yr"]  # always also split by exposure base
        anomaly_by: list[str] = ["src"]
        class_col: str = "class_cd_std"
        exclude_bases: list[str] = ["", "UNK"]  # placeholder bases ignored in the class consistency check
        negative: dict[str, float] = {"warn": 0.0, "fail": 0.001}
        zero_expo_pos_prem: dict[str, float] = {"warn": 0.0, "fail": 0.001}
        multi_base_classes: dict[str, float] = {"warn": 0, "fail": 5}

    def summary(self, by: list[str]) -> pd.DataFrame:
        m, s = self.project.measures, self.schema
        dims = list(dict.fromkeys([m.exposure_base] + by))
        df = pipeline_agg(self, dims, {"exposure": s.ref(m.exposure), "premium": s.ref(m.written_premium),
                                       "rows": "1"}, label="exposure by base")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["premium_per_expo"] = np.where(df["exposure"] > 0, df["premium"] / df["exposure"], np.nan)
        return df.sort_values(dims).reset_index(drop=True)

    def run(self) -> CheckResult:
        m, s, cfg = self.project.measures, self.schema, self.cfg
        s.validate(cfg.anomaly_by + cfg.summary_by + [cfg.class_col])
        findings = []
        anom = self.query("anomalies", self.ctx.render_sql(
            "exposure_anomalies.sql.j2", group_by=cfg.anomaly_by, expo=s.ref(m.exposure), prem=s.ref(m.written_premium)))
        for _, r in anom.iterrows():
            seg = segment_key({g: r[g] for g in cfg.anomaly_by})
            n = int(r["n_rows"])
            for metric, col, th, var in [("pct_negative_expo", "n_negative_expo", cfg.negative, m.exposure),
                                         ("pct_zero_expo_pos_prem", "n_zero_expo_pos_prem", cfg.zero_expo_pos_prem, m.exposure)]:
                val = (r[col] or 0) / n if n else 0.0
                findings.append(dict(variable=var, item=metric, segment=seg, metric=metric, value=val,
                                     threshold=th["fail"], status=grade(val, th["warn"], th["fail"]),
                                     detail=f"{int(r[col] or 0):,} of {n:,} rows"))
        anom["segment"] = [segment_key({g: r[g] for g in cfg.anomaly_by}) for _, r in anom.iterrows()]

        classes = self.query("class to base consistency", self.ctx.render_sql(
            "exposure_class_bases.sql.j2", class_col=cfg.class_col, **{"class": s.ref(cfg.class_col)},
            base=s.ref(m.exposure_base), exclude=cfg.exclude_bases))
        n_multi = len(classes)
        th = cfg.multi_base_classes
        findings.append(dict(variable=m.exposure_base, item=f"{cfg.class_col} with >1 base", segment="ALL",
                             metric="n_classes_multi_base", value=n_multi, threshold=th["fail"],
                             status=grade(n_multi, th["warn"], th["fail"]),
                             detail=", ".join(classes[cfg.class_col].astype(str).head(20))))
        return self.result(findings, {"summary": self.summary(cfg.summary_by), "anomalies": anom,
                                      "multi_base_classes": classes})

    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        opts = self.segment_options()
        new.summary_by = st.multiselect("Summary levels (in addition to exposure base)", opts,
                                        default=[g for g in cfg.summary_by if g in opts], key="ex_sum")
        new.anomaly_by = st.multiselect("Anomaly check levels", opts, default=[g for g in cfg.anomaly_by if g in opts],
                                        key="ex_anom")
        txt = st.text_input("Placeholder bases to ignore (comma-separated)", ", ".join(cfg.exclude_bases), key="ex_excl")
        new.exclude_bases = [t.strip() for t in txt.split(",")]
        return new

    def render(self, result):
        import plotly.express as px
        import streamlit as st

        from gl_dq.ui.components import status_table

        m = self.project.measures
        base = m.exposure_base
        summ = result.tables["summary"]
        by = self.cfg.summary_by
        st.markdown(f"**Exposure by `{base}`**: exposures are only comparable within a base.")
        bases = sorted(summ[base].dropna().astype(str).unique())
        sel = st.selectbox("Exposure base", bases, key="ex_base") if bases else None
        view = (summ[summ[base].astype(str) == str(sel)] if sel else summ).copy()
        if by:
            from gl_dq.ui.theme import line, ordered_categories, series_encoding, style

            x = by[-1]
            enc = series_encoding(view, by[0], self.project.sources, by) if len(by) > 1 else dict(color_discrete_sequence=["#2a78d6"])
            # series_encoding casts the level columns to strings, and plotly then orders an axis by
            # first appearance: a source that starts writing late would put 2021 before 2018
            orders = ordered_categories(view, enc, x)
            c1, c2 = st.columns(2)
            fig = px.bar(view, x=x, y="exposure", barmode="group", category_orders=orders, **enc)
            c1.plotly_chart(style(fig, 320, f"{m.exposure}: {sel}"), use_container_width=True)
            fig = line(view, x=x, y="premium_per_expo", markers=True, category_orders=orders, **enc)
            fig.update_yaxes(rangemode="tozero")
            c2.plotly_chart(style(fig, 320, f"Premium per exposure unit: {sel}"), use_container_width=True)
        with st.expander("Exposure summary table", expanded=not by):
            st.dataframe(summ, hide_index=True, use_container_width=True, column_config={
                "exposure": st.column_config.NumberColumn(format="localized"),
                "premium": st.column_config.NumberColumn(format="localized"),
                "premium_per_expo": st.column_config.NumberColumn(format="%.3f")})
        st.markdown("**Anomalies**")
        f = result.findings
        status_table(f[f["metric"] != "n_classes_multi_base"][["variable", "segment", "metric", "value", "status", "detail"]],
                     percent_cols=["value"])
        mb = result.tables["multi_base_classes"]
        st.markdown(f"**`{self.cfg.class_col}` codes mapped to more than one `{base}`**: {len(mb)}")
        if not mb.empty:
            st.dataframe(mb, hide_index=True, use_container_width=True)
