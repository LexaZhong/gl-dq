"""Level check: is the candidate key unique within each source and across the whole table?"""
from __future__ import annotations

import pandas as pd

from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import grade, segment_key

ALL = "_all"


@register_check("key_uniqueness")
class KeyUniqueness(Check):
    title = "Key uniqueness"
    icon = "🔑"
    description = ("Checks each source's candidate key (and the whole table's key): row count vs distinct keys, "
                   "duplicate rows, nulls in key columns. The key suggestion runs a greedy search over columns.")
    default_order = 10

    class Config(CheckConfig):
        candidate_keys: dict[str, list[str]] = {}  # source -> key columns; "_all" = whole table
        warn_dup_rate: float = 0.0
        fail_dup_rate: float = 0.001
        sample_limit: int = 50
        suggest_max_cols: int = 8
        suggest_exclude_amounts: bool = True

    def _where(self, src: str) -> str | None:
        if src == ALL:
            return None
        return f"{self.schema.ref(self.project.src_col)} = {self.ctx.dialect.lit(src)}"

    def run(self) -> CheckResult:
        rows, summary, samples = [], [], {}
        for src, keys in self.cfg.candidate_keys.items():
            self.schema.validate(keys)
            seg = segment_key({} if src == ALL else {self.project.src_col: src})
            where = self._where(src)
            r = self.query(f"key {src}", self.ctx.render_sql("key_uniqueness.sql.j2", keys=keys, where=where)).iloc[0]
            n = int(r["n_rows"] or 0)
            dup_rate = (r["n_dup_rows"] or 0) / n if n else 0.0
            nulls = {k: int(r[f"null__{k}"] or 0) for k in keys}
            status = grade(dup_rate, self.cfg.warn_dup_rate, self.cfg.fail_dup_rate)
            summary.append({
                "source": src, "status": status, "key": ", ".join(keys), "n_rows": n,
                "n_distinct_keys": int(r["n_distinct_keys"] or 0), "n_dup_rows": int(r["n_dup_rows"] or 0),
                "n_dup_groups": int(r["n_dup_groups"] or 0), "dup_rate": dup_rate,
                "max_rows_per_key": int(r["max_rows_per_key"] or 0),
                "key_cols_with_nulls": ", ".join(f"{k} ({v:,})" for k, v in nulls.items() if v) or "none",
            })
            rows.append(dict(variable="_key", item=", ".join(keys), segment=seg, metric="dup_rate", value=dup_rate,
                             threshold=self.cfg.fail_dup_rate, status=status,
                             detail=f"{int(r['n_dup_rows'] or 0):,} duplicate rows in {int(r['n_dup_groups'] or 0):,} keys"))
            for k, v in nulls.items():
                if v:
                    rows.append(dict(variable=k, item="key column", segment=seg, metric="key_null_rows", value=v,
                                     threshold=None, status="info", detail=f"null in key column for {v:,} rows"))
            if r["n_dup_rows"]:
                samples[src] = self.query(f"duplicates {src}", self.ctx.render_sql(
                    "key_duplicates.sql.j2", keys=keys, where=where, limit=self.cfg.sample_limit))
        if not self.cfg.candidate_keys:
            self.note("No candidate keys configured yet. Use 'Suggest a key' below or add candidate_keys to the config.")
        tables = {"summary": pd.DataFrame(summary)}
        tables.update({f"duplicates: {k}": v for k, v in samples.items()})
        return self.result(rows, tables)

    # ---- key discovery -----------------------------------------------------
    def suggest_key(self, src: str, candidates: list[str] | None = None, start: list[str] | None = None) -> pd.DataFrame:
        """Greedy forward search: add the column that most increases distinct key count until unique."""
        cols = candidates or [c for c in self.schema.names(include_derived=False)
                              if not (self.cfg.suggest_exclude_amounts and self.schema.is_amount(c))]
        chosen = list(start or [])
        steps = []
        where = self._where(src)
        for _ in range(self.cfg.suggest_max_cols):
            options = [c for c in cols if c not in chosen]
            if not options:
                break
            cand = {f"opt{i}": chosen + [c] for i, c in enumerate(options)}
            exprs = {k: [self.schema.ref(c) for c in v] for k, v in cand.items()}
            r = self.query(f"suggest {src} step {len(chosen)}", self.ctx.render_sql(
                "key_suggest.sql.j2", candidates=exprs, where=where)).iloc[0]
            n = int(r["n_rows"])
            best = max(cand, key=lambda k: r[k])
            chosen = cand[best]
            steps.append({"step": len(chosen), "added": chosen[-1], "key": ", ".join(chosen),
                          "distinct_keys": int(r[best]), "n_rows": n, "coverage": r[best] / n if n else 1.0})
            if r[best] == n:
                break
        return pd.DataFrame(steps)

    # ---- UI ------------------------------------------------------------------
    def settings_ui(self, cfg):
        import streamlit as st

        cols = self.schema.names(include_derived=False)
        new = cfg.model_copy(deep=True)
        c1, c2 = st.columns(2)
        new.warn_dup_rate = c1.number_input("Warn if duplicate rate >", value=float(cfg.warn_dup_rate), format="%.4f", step=0.0001)
        new.fail_dup_rate = c2.number_input("Fail if duplicate rate >", value=float(cfg.fail_dup_rate), format="%.4f", step=0.0001)
        keys = {}
        for src in self.project.sources + [ALL]:
            label = "Whole table (_all)" if src == ALL else f"{src} key"
            keys[src] = st.multiselect(label, cols, default=[k for k in cfg.candidate_keys.get(src, []) if k in cols],
                                       key=f"key_{src}")
        new.candidate_keys = {k: v for k, v in keys.items() if v}
        return new

    def render(self, result):
        import streamlit as st

        from gl_dq.ui.components import status_table

        status_table(result.tables.get("summary", pd.DataFrame()),
                     percent_cols=["dup_rate"], height=None)
        for label, df in result.tables.items():
            if label.startswith("duplicates"):
                with st.expander(f"Sample {label} (top {self.cfg.sample_limit})"):
                    st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("🔍 Suggest a key (greedy search)"):
            c1, c2 = st.columns([1, 3])
            src = c1.selectbox("Level", self.project.sources + [ALL], key="suggest_src")
            start = c2.multiselect("Start from", self.schema.names(include_derived=False),
                                   default=[k for k in self.project.policy_key
                                            if src == ALL or k != self.project.src_col],
                                   key=f"suggest_start_{src}")
            if st.button("Run search", key="suggest_go"):
                with st.spinner("Testing column combinations…"):
                    steps = self.suggest_key(src, start=start)
                st.dataframe(steps, use_container_width=True, hide_index=True,
                             column_config={"coverage": st.column_config.ProgressColumn(
                                 "distinct / rows", min_value=0.0, max_value=1.0, format="%.4f")})
                if not steps.empty and steps.iloc[-1]["coverage"] >= 1:
                    st.success(f"Unique key found: {steps.iloc[-1]['key']}")
                else:
                    st.warning("No unique key found within the column limit. The data may contain exact duplicates.")
