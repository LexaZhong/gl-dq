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
from gl_dq.core.filters import FilterSet, load_filters, save_filters
from gl_dq.core.knowledge import KnowledgeStore
from gl_dq.core.registry import discover
from gl_dq.core.results import DeltaResults, ParquetResults, ResultsStore
from gl_dq.core.schema import TableSchema
from gl_dq.core.storage import Storage, make_storage
from gl_dq.core.transforms import TransformLibrary, load_transforms, save_transforms, sql_expr
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
    filters: FilterSet = field(default_factory=FilterSet)
    transforms: TransformLibrary = field(default_factory=TransformLibrary)

    def __post_init__(self):
        self._jinja = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(PACKAGE_ROOT / "sql")),
            undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)

    # ---- SQL ---------------------------------------------------------------
    @property
    def dialect(self):
        return self.db.dialect

    @property
    def table_expr(self) -> str:
        """The table every query reads: the real table, or a subquery that cleans and filters it.

        `project.table` stays the real name (DESCRIBE, error messages, the preprocessing spec);
        this is what `{{ table }}` renders to. Every packaged template uses `FROM {{ table }}`
        bare, with no alias and no alias-qualified columns, so a subquery drops straight in.

        Transforms sit inside the filters, so a rule like "US states only" is written against the
        standardized value rather than whichever spelling the source happened to use.
        """
        source = self._transformed_source()
        where = self.filters.where(self)
        return source if not where else f"(SELECT * FROM {source} WHERE {where}) AS gl"

    def _transformed_source(self) -> str:
        """The table with every active column transform applied, or just the table."""
        lib = self.transforms
        if not lib.apply_to_dashboard or not lib.active():
            return self.project.table
        by_col = {t.column: t for t in lib.active() if self.schema.has(t.column)}
        cols = [f"{sql_expr(self.schema, self.dialect, by_col[c])} AS {self.dialect.quote(c)}"
                if c in by_col else self.dialect.quote(c) for c in self.schema.columns]
        return f"(SELECT {', '.join(cols)} FROM {self.project.table}) AS tx"

    def render_sql(self, template: str, **kw) -> str:
        """Render a packaged SQL template with column helpers available as c(), sel(), lit()."""
        s, d = self.schema, self.dialect
        return self._jinja.get_template(template).render(
            table=self.table_expr, raw_table=self.project.table,
            c=s.ref, sel=s.select_as, q=d.quote, lit=d.lit, dialect=d, **kw)

    def render_user_sql(self, path: str) -> str:
        """Render a user-supplied SQL file from the config store (e.g. a source-of-truth query).

        `{{ table }}` is the filtered population, so a source of truth restricted to the policies
        in gl_master is restricted to the same policies the dashboard shows; `{{ raw_table }}` is
        the unfiltered table for a query that deliberately wants everything.
        """
        text = self.config_store.read_text(path)
        if text is None:
            raise FileNotFoundError(f"{path} not found in {self.config_store}")
        return jinja2.Template(text, undefined=jinja2.StrictUndefined).render(
            table=self.table_expr, raw_table=self.project.table, **self.project.sql_vars)

    # ---- check configs -----------------------------------------------------
    def check_config(self, name: str):
        cls = self.checks[name]
        text = self.config_store.read_text(f"checks/{name}.yaml")
        raw = yaml.safe_load(text) if text else {}
        raw = deep_merge(raw or {}, self.project.check_overrides.get(name, {}))
        return cls.Config.model_validate(raw)

    def save_transforms(self, lib: TransformLibrary) -> None:
        """Write config/transforms.json - the mappings the modelling pipeline will read."""
        save_transforms(self.config_store, lib)
        self.transforms = lib

    def save_filters(self, fs: FilterSet) -> None:
        """Write config/filters.yaml, so every session and the refresh job see the same rules."""
        save_filters(self.config_store, fs)
        self.filters = fs

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

    def checks_by_category(self) -> dict[str, list[str]]:
        """Enabled checks grouped into sidebar sections, each in `order`, sections first-seen first."""
        groups: dict[str, list[str]] = {}
        for name in self.enabled_checks():
            groups.setdefault(self.check_config(name).category, []).append(name)
        return groups


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
        results = ParquetResults(make_storage(project.results.path, REPO_ROOT))
    checks = discover(project.plugins)
    return Context(profile, project, db, schema, config_store, knowledge, results, checks,
                   load_workflow(config_store), load_filters(config_store),
                   load_transforms(config_store))
