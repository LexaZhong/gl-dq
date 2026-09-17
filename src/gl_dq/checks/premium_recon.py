"""Premium reconciliation: pipeline written premium vs pricing study source of truth, by src x coverage."""
from __future__ import annotations

from gl_dq.checks._recon import (Tolerance, pipeline_agg, reconcile, recon_findings, render_recon, sot_agg,
                                 sot_columns)
from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check


@register_check("premium_recon")
class PremiumRecon(Check):
    title = "Premium reconciliation"
    icon = "💰"
    description = ("Written premium from the pipeline vs the pricing study (source of truth) at the chosen grain. "
                   "A segment fails when both the absolute and relative differences exceed tolerance.")
    default_order = 40

    class Config(CheckConfig):
        measure: str | None = None  # default: project measures.written_premium
        dims: list[str] = ["src", "covg_type_desc"]
        sot_query: str = "sql/sot_premium.sql"
        sot_measure: str = "wrtn_prm"
        dim_map: dict[str, str] = {}  # pipeline dim -> SOT column name, when they differ
        tolerance: Tolerance = Tolerance(abs=1000, pct=0.005)
        where: str | None = None  # optional config-authored pipeline filter

    @property
    def measure(self) -> str:
        return self.cfg.measure or self.project.measures.written_premium

    def run(self) -> CheckResult:
        sot_sql = self.ctx.render_user_sql(self.cfg.sot_query)
        dims = self.cfg.dims
        m = self.schema.ref(self.measure)
        pipe = pipeline_agg(self, dims, {"premium": m}, self.cfg.where)
        sot = sot_agg(self, sot_sql, dims, self.cfg.dim_map, {"premium": self.cfg.sot_measure})
        rec = reconcile(pipe, sot, dims, "premium", self.cfg.tolerance)
        return self.result(recon_findings(rec, self.measure, "written_premium", self.cfg.tolerance), {"recon": rec})

    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        try:
            sot_cols = sot_columns(self, self.ctx.render_user_sql(cfg.sot_query))
        except Exception as e:  # noqa: BLE001
            st.error(f"SOT query failed: {e}")
            sot_cols = []
        available = [d for d in self.segment_options() if cfg.dim_map.get(d, d) in sot_cols]
        new.dims = st.multiselect("Grain (dimensions present in both pipeline and SOT)", available,
                                  default=[d for d in cfg.dims if d in available], key="pr_dims")
        c1, c2 = st.columns(2)
        new.tolerance = Tolerance(
            abs=c1.number_input("Absolute tolerance", 0.0, value=float(cfg.tolerance.abs), step=100.0, key="pr_abs"),
            pct=c2.number_input("Relative tolerance", 0.0, 1.0, float(cfg.tolerance.pct), 0.001, format="%.3f", key="pr_pct"))
        st.caption(f"SOT query: `{cfg.sot_query}` · SOT columns: {', '.join(sot_cols) or 'n/a'}")
        return new

    def render(self, result):
        import streamlit as st

        render_recon(st, result.tables["recon"], self.cfg.dims, "Written premium", "prem")
