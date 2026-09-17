"""Context: everything a check needs (project, db, schema, SQL rendering, config/knowledge stores)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
import yaml

from gl_dq import PACKAGE_ROOT, REPO_ROOT
from gl_dq.core.config import ProjectConfig, deep_merge, dump_yaml, load_project
from gl_dq.core.db import Database, make_database
from gl_dq.core.knowledge import KnowledgeStore
from gl_dq.core.registry import discover
from gl_dq.core.results import DeltaResults, ParquetResults, ResultsStore
from gl_dq.core.schema import TableSchema
from gl_dq.core.storage import Storage, make_storage
from gl_dq.core.workflow import Workflow, load_workflow


@dataclass
class Context:
    profile: str
    project: ProjectConfig
    db: Database
    schema: TableSchema
    config_store: Storage
    knowledge: KnowledgeStore
    results: ResultsStore
    checks: dict[str, type] = field(default_factory=dict)
    workflow: Workflow = field(default_factory=Workflow)

    def __post_init__(self):
        self._jinja = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(PACKAGE_ROOT / "sql")),
            undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)

    # ---- SQL ---------------------------------------------------------------
    @property
    def dialect(self):
        return self.db.dialect

    def render_sql(self, template: str, **kw) -> str:
        """Render a packaged SQL template with column helpers available as c(), sel(), lit()."""
        s, d = self.schema, self.dialect
        return self._jinja.get_template(template).render(
            table=self.project.table, c=s.ref, sel=s.select_as, q=d.quote, lit=d.lit, dialect=d, **kw)

    def render_user_sql(self, path: str) -> str:
        """Render a user-supplied SQL file from the config store (e.g. a source-of-truth query)."""
        text = self.config_store.read_text(path)
        if text is None:
            raise FileNotFoundError(f"{path} not found in {self.config_store}")
        return jinja2.Template(text, undefined=jinja2.StrictUndefined).render(
            table=self.project.table, **self.project.sql_vars)

    # ---- check configs -----------------------------------------------------
    def check_config(self, name: str):
        cls = self.checks[name]
        text = self.config_store.read_text(f"checks/{name}.yaml")
        raw = yaml.safe_load(text) if text else {}
        raw = deep_merge(raw or {}, self.project.check_overrides.get(name, {}))
        return cls.Config.model_validate(raw)

    def save_check_config(self, name: str, cfg) -> None:
        self.config_store.write_text(f"checks/{name}.yaml", dump_yaml(cfg.model_dump(mode="json")))

    def make_check(self, name: str, cfg=None):
        cls = self.checks[name]
        return cls(self, cfg if cfg is not None else self.check_config(name))

    def enabled_checks(self) -> list[str]:
        out = []
        for name in self.checks:
            cfg = self.check_config(name)
            if cfg.enabled:
                out.append((cfg.order, name))
        return [n for _, n in sorted(out)]


def load_context(profile: str | None = None) -> Context:
    profile = profile or os.environ.get("DQ_PROFILE", "synthetic")
    project = load_project(profile)
    db = make_database(project)
    schema = TableSchema(project.table, db.describe(project.table), project.derived_columns, db.dialect)
    config_store = make_storage(project.config_dir, REPO_ROOT)
    knowledge = KnowledgeStore(make_storage(project.knowledge_dir, REPO_ROOT))
    if project.results.type == "delta":
        results = DeltaResults(db, project.results.table)
    else:
        p = Path(project.results.path)
        results = ParquetResults(p if p.is_absolute() else REPO_ROOT / p)
    checks = discover(project.plugins)
    return Context(profile, project, db, schema, config_store, knowledge, results, checks, load_workflow(config_store))
