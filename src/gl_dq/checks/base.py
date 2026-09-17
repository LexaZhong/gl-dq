"""Check base class. A check = Config model + run() (pure, no Streamlit) + optional UI hooks.

Minimal plugin:

    @register_check("my_check")
    class MyCheck(Check):
        title = "My check"
        icon = "🧪"
        default_order = 90

        class Config(CheckConfig):
            group_by: list[str] = ["src"]

        def run(self) -> CheckResult:
            sql = self.ctx.render_sql("my_check.sql.j2", group_by=self.cfg.group_by)
            df = self.query("main", sql)
            rows = [...]  # finding dicts (see gl_dq.core.results.FINDING_COLS)
            return self.result(findings=rows, tables={"main": df})

`settings_ui(cfg) -> cfg` and `render(result)` are called by the dashboard; both import
streamlit lazily so run() stays usable from jobs and tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from gl_dq.core.config import CheckConfig
from gl_dq.core.results import findings_frame


@dataclass
class CheckResult:
    check: str
    findings: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    sql: dict[str, str] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


class Check:
    name: ClassVar[str] = ""
    title: ClassVar[str] = ""
    icon: ClassVar[str] = "🧪"
    description: ClassVar[str] = ""
    default_order: ClassVar[int] = 100

    class Config(CheckConfig):
        pass

    def __init__(self, ctx, cfg: CheckConfig):
        self.ctx = ctx
        self.cfg = cfg
        self._sql: dict[str, str] = {}
        self._messages: list[str] = []

    # ---- helpers -----------------------------------------------------------
    @property
    def schema(self):
        return self.ctx.schema

    @property
    def project(self):
        return self.ctx.project

    def query(self, label: str, sql: str) -> pd.DataFrame:
        self._sql[label] = sql
        return self.ctx.db.query(sql)

    def note(self, msg: str) -> None:
        self._messages.append(msg)

    def result(self, findings: list[dict], tables: dict[str, pd.DataFrame] | None = None) -> CheckResult:
        for f in findings:
            f.setdefault("check", self.name)
        return CheckResult(self.name, findings_frame(findings), tables or {}, dict(self._sql), list(self._messages))

    def segment_options(self) -> list[str]:
        cands = self.project.segment_candidates or [self.project.src_col]
        return [c for c in cands if self.schema.has(c)]

    # ---- to implement ----------------------------------------------------------
    def run(self) -> CheckResult:
        raise NotImplementedError

    def settings_ui(self, cfg):
        """Streamlit widgets that return a modified copy of cfg (default: no settings)."""
        return cfg

    def render(self, result: CheckResult) -> None:
        """Streamlit rendering of the result (default: tables)."""
        import streamlit as st

        for label, df in result.tables.items():
            st.subheader(label)
            st.dataframe(df, use_container_width=True, hide_index=True)
