"""Missing-rate check for every column, with per-variable thresholds, levels and missing definitions."""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import grade, segment_key


class VarRule(BaseModel):
    warn: float | None = None
    fail: float | None = None
    group_by: list[str] | None = None
    treat_as_missing: list[str | float | int] | None = None
    applies_when: str | None = None  # config-authored SQL predicate; rate is computed over these rows only


@register_check("missing_rate")
class MissingRate(Check):
    title = "Missing rate"
    icon = "🔲"
    description = ("Share of missing values per variable and segment. 'Missing' means null, plus blank strings and any "
                   "configured sentinel values. `applies_when` limits the denominator, e.g. evt_dt only where claims > 0.")
    default_order = 20

    class Config(CheckConfig):
        default: VarRule = VarRule(warn=0.05, fail=0.20, group_by=["src"])
        blank_strings_are_missing: bool = True
        variables: dict[str, VarRule] = {}
        include: list[str] = []  # empty = every table column
        exclude: list[str] = []

    def rule(self, var: str) -> VarRule:
        d = self.cfg.default
        r = self.cfg.variables.get(var, VarRule())
        return VarRule(
            warn=r.warn if r.warn is not None else d.warn,
            fail=r.fail if r.fail is not None else d.fail,
            group_by=r.group_by if r.group_by is not None else (d.group_by or []),
            treat_as_missing=r.treat_as_missing if r.treat_as_missing is not None else (d.treat_as_missing or []),
            applies_when=r.applies_when or d.applies_when,
        )

    def variables(self) -> list[str]:
        names = self.cfg.include or self.schema.names(include_derived=False)
        return [n for n in self.schema.validate(names) if n not in self.cfg.exclude]

    def _missing_expr(self, var: str, rule: VarRule) -> str:
        ref, lit = self.schema.ref(var), self.ctx.dialect.lit
        parts = [f"{ref} IS NULL"]
        if self.schema.is_string(var):
            sentinels = [str(s) for s in rule.treat_as_missing]
            if self.cfg.blank_strings_are_missing:
                sentinels = [""] + [s for s in sentinels if s != ""]
            if sentinels:
                parts.append(f"TRIM({ref}) IN ({', '.join(lit(s) for s in sentinels)})")
        elif self.schema.is_numeric(var) and rule.treat_as_missing:
            nums = [float(s) for s in rule.treat_as_missing]
            parts.append(f"{ref} IN ({', '.join(lit(n) for n in nums)})")
        return " OR ".join(parts)

    def run(self) -> CheckResult:
        by_level: dict[tuple, list[tuple[str, VarRule]]] = {}
        for var in self.variables():
            rule = self.rule(var)
            self.schema.validate(rule.group_by)
            by_level.setdefault(tuple(rule.group_by), []).append((var, rule))

        findings, detail = [], []
        for level, items in by_level.items():
            spec = [{"applies": r.applies_when or "1=1", "missing": self._missing_expr(v, r)} for v, r in items]
            df = self.query(f"level: {', '.join(level) or 'ALL'}",
                            self.ctx.render_sql("missing_rate.sql.j2", group_by=list(level), variables=spec))
            for _, row in df.iterrows():
                seg = segment_key({g: row[g] for g in level})
                for i, (var, rule) in enumerate(items):
                    n_app, n_miss = int(row[f"a__{i}"] or 0), int(row[f"m__{i}"] or 0)
                    if not n_app:
                        continue  # variable does not apply to this segment
                    rate = n_miss / n_app
                    status = grade(rate, rule.warn, rule.fail)
                    findings.append(dict(variable=var, item=rule.applies_when, segment=seg, metric="missing_rate",
                                         value=rate, threshold=rule.fail, status=status,
                                         detail=f"{n_miss:,} of {n_app:,} rows"))
                    detail.append({"variable": var, "level": ", ".join(level) or "ALL", "segment": seg,
                                   "status": status, "missing_rate": rate, "n_missing": n_miss,
                                   "n_applicable": n_app, "n_rows": int(row["n_rows"]),
                                   "warn": rule.warn, "fail": rule.fail, "applies_when": rule.applies_when})
        return self.result(findings, {"detail": pd.DataFrame(detail)})

    # ---- UI ------------------------------------------------------------------
    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        seg_opts = self.segment_options()
        st.markdown("**Default rule** (applies to every variable without an override)")
        c1, c2, c3 = st.columns([1, 1, 2])
        new.default.warn = c1.number_input("Warn if rate >", 0.0, 1.0, float(cfg.default.warn or 0), 0.01, key="mr_dw")
        new.default.fail = c2.number_input("Fail if rate >", 0.0, 1.0, float(cfg.default.fail or 0), 0.01, key="mr_df")
        new.default.group_by = c3.multiselect("Default level", seg_opts, default=[g for g in cfg.default.group_by or [] if g in seg_opts], key="mr_dg")
        new.blank_strings_are_missing = st.checkbox("Blank strings count as missing", cfg.blank_strings_are_missing, key="mr_blank")

        st.markdown("**Per-variable overrides**: edit, add or delete rows. Leave a cell empty to use the default. "
                    "Level and sentinels are comma-separated.")
        rows = [{"variable": v, "warn": r.warn, "fail": r.fail,
                 "level": ", ".join(r.group_by) if r.group_by is not None else None,
                 "treat_as_missing": ", ".join(map(str, r.treat_as_missing)) if r.treat_as_missing else None,
                 "applies_when": r.applies_when} for v, r in cfg.variables.items()]
        edited = st.data_editor(
            pd.DataFrame(rows, columns=["variable", "warn", "fail", "level", "treat_as_missing", "applies_when"]),
            num_rows="dynamic", use_container_width=True, key="mr_vars",
            column_config={
                "variable": st.column_config.SelectboxColumn(options=self.schema.names(include_derived=False), required=True),
                "warn": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, format="%.3f"),
                "fail": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, format="%.3f"),
                "applies_when": st.column_config.TextColumn(disabled=True, help="Edit in YAML (config-authored SQL)"),
            })
        variables = {}
        for r in edited.to_dict("records"):
            if not r.get("variable"):
                continue
            nn = lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) or x == "" else x  # noqa: E731
            level = nn(r.get("level"))
            tam = nn(r.get("treat_as_missing"))
            variables[r["variable"]] = VarRule(
                warn=nn(r.get("warn")), fail=nn(r.get("fail")),
                group_by=[g.strip() for g in level.split(",") if g.strip()] if level is not None else None,
                treat_as_missing=[t.strip() for t in tam.split(",")] if tam is not None else None,
                applies_when=cfg.variables.get(r["variable"], VarRule()).applies_when)
        new.variables = variables
        return new

    def render(self, result):
        import plotly.express as px
        import streamlit as st

        from gl_dq.ui.components import status_table

        df = result.tables.get("detail", pd.DataFrame())
        if df.empty:
            st.info("No variables in scope.")
            return
        c1, c2 = st.columns([2, 1])
        only_flagged = c2.toggle("Only flagged (warn/fail)", value=False, key="mr_flagged")
        view = df[df["status"].isin(["warn", "fail"])] if only_flagged else df
        levels = sorted(view["level"].unique())
        level = c1.selectbox("Level", levels, key="mr_level") if len(levels) > 1 else (levels[0] if levels else None)
        view = view[view["level"] == level] if level else view
        if view.empty:
            st.success("Nothing flagged at this level.")
            return
        pivot = view.pivot_table(index="variable", columns="segment", values="missing_rate", aggfunc="max")
        from gl_dq.ui.theme import SEQ_SCALE

        fig = px.imshow(pivot, text_auto=".1%", aspect="auto", color_continuous_scale=SEQ_SCALE, zmin=0,
                        labels={"color": "missing rate"})
        fig.update_layout(height=max(300, 22 * len(pivot) + 120), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        status_table(view.drop(columns=["level"]), percent_cols=["missing_rate", "warn", "fail"])
