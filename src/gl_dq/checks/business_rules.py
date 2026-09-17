"""Business / validity rules: config-authored SQL predicates evaluated per segment."""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import grade, segment_key


class Rule(BaseModel):
    id: str
    description: str = ""
    variables: list[str] = []  # tracked variables this rule documents (first = primary)
    violation: str  # SQL predicate that is TRUE for a bad row
    applies_when: str | None = None
    warn: float = 0.0
    fail: float = 0.001


@register_check("business_rules")
class BusinessRules(Check):
    title = "Business rules"
    icon = "📏"
    description = ("Validity rules written as SQL predicates in config: date order, event date inside the policy "
                   "period, mutually exclusive deductibles, and so on. Shows the violation rate per segment.")
    default_order = 25

    class Config(CheckConfig):
        group_by: list[str] = ["src"]
        rules: list[Rule] = []

    def run(self) -> CheckResult:
        cfg = self.cfg
        self.schema.validate(cfg.group_by)
        for r in cfg.rules:
            self.schema.validate(r.variables)
        if not cfg.rules:
            return self.result([], {"detail": pd.DataFrame()})
        df = self.query("rules", self.ctx.render_sql("business_rules.sql.j2", group_by=cfg.group_by, rules=cfg.rules))
        findings, detail = [], []
        for _, row in df.iterrows():
            seg = segment_key({g: row[g] for g in cfg.group_by})
            for i, r in enumerate(cfg.rules):
                n_app, n_bad = int(row[f"a__{i}"] or 0), int(row[f"v__{i}"] or 0)
                if not n_app:
                    continue
                rate = n_bad / n_app
                status = grade(rate, r.warn, r.fail)
                var = r.variables[0] if r.variables else r.id
                findings.append(dict(variable=var, item=r.id, segment=seg, metric="violation_rate", value=rate,
                                     threshold=r.fail, status=status, detail=f"{n_bad:,} of {n_app:,} rows · {r.description}"))
                detail.append({"rule": r.id, "segment": seg, "status": status, "violation_rate": rate,
                               "n_violations": n_bad, "n_applicable": n_app, "description": r.description,
                               "violation": r.violation})
        return self.result(findings, {"detail": pd.DataFrame(detail)})

    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        opts = self.segment_options()
        new.group_by = st.multiselect("Level", opts, default=[g for g in cfg.group_by if g in opts], key="br_lvl")
        st.caption("Rules are SQL predicates, so add or change them in `config/checks/business_rules.yaml` (or with "
                   "the gl-dq-configure skill). They are reviewed like code.")
        rows = pd.DataFrame([{"id": r.id, "warn": r.warn, "fail": r.fail, "description": r.description,
                              "violation": r.violation} for r in cfg.rules])
        edited = st.data_editor(rows, use_container_width=True, key="br_rules", disabled=["id", "description", "violation"],
                                column_config={"warn": st.column_config.NumberColumn(format="%.4f"),
                                               "fail": st.column_config.NumberColumn(format="%.4f")})
        th = {r["id"]: r for r in edited.to_dict("records")}
        new.rules = [r.model_copy(update={"warn": float(th[r.id]["warn"]), "fail": float(th[r.id]["fail"])})
                     if r.id in th else r for r in cfg.rules]
        return new

    def render(self, result):
        import plotly.express as px
        import streamlit as st

        from gl_dq.ui.components import status_table

        d = result.tables.get("detail", pd.DataFrame())
        if d.empty:
            st.info("No rules configured.")
            return
        piv = d.pivot_table(index="rule", columns="segment", values="violation_rate", aggfunc="max")
        from gl_dq.ui.theme import SEQ_SCALE

        fig = px.imshow(piv, text_auto=".2%", aspect="auto", color_continuous_scale=SEQ_SCALE, zmin=0,
                        labels={"color": "violation rate"})
        fig.update_layout(height=max(260, 40 * len(piv) + 100), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        status_table(d, percent_cols=["violation_rate"])
