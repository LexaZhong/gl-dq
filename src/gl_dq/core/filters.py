"""Global data filters: named rules that restrict the population every query sees.

One rule, defined once, applies to every check, every page and the refresh job — instead of the
same exclusion being copied into nine per-check `where:` configs where nobody can see it.

A rule is either structured (column + operator + values, built through the column whitelist and
`Dialect.lit`, so it is always safe) or a raw SQL predicate for the rules no picker can express,
e.g. "drop ProductsCompletedOps rows for ids that write no products/completed-ops business". A raw
predicate is trusted config SQL, at the same level as `derived_columns.expr` and business rules;
`validate()` runs it against the table before it can be saved.

A row is KEPT only when the predicate is true: a NULL predicate excludes, which is what
"exclude non-US locations" has to do for a row with no state.
"""
from __future__ import annotations

import hashlib
import re
from typing import Literal

import jinja2
import yaml
from pydantic import BaseModel, field_validator, model_validator

NULL_LABEL = "<null>"

Op = Literal["in", "not_in", "is_null", "not_null", "eq", "ne", "gt", "gte", "lt", "lte"]
OP_LABELS = {
    "in": "is one of", "not_in": "is not one of", "is_null": "is null", "not_null": "is not null",
    "eq": "=", "ne": "≠", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤",
}
COMPARISONS = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
KEY_RE = re.compile(r"^[a-z0-9_]+$")


class Filter(BaseModel):
    key: str  # stable id; part of the fingerprint stored with every run
    label: str = ""
    description: str = ""  # why this rule exists - shown next to it and kept with the findings
    enabled: bool = False
    profiles: list[str] = []  # empty = every profile (one filters.yaml serves prod and synthetic)
    column: str | None = None
    op: Op = "in"
    values: list[str] = []
    expr: str | None = None  # raw predicate; may use {{ raw_table }} to reference the table itself

    @field_validator("key")
    @classmethod
    def _key(cls, v):
        if not KEY_RE.match(v or ""):
            raise ValueError(f"filter key {v!r} must be lowercase letters, digits and underscores")
        return v

    @model_validator(mode="after")
    def _one_of(self):
        if bool(self.column) == bool(self.expr):
            raise ValueError(f"filter {self.key}: set either `column` (with `op`) or `expr`, not both")
        if self.column and self.op in ("in", "not_in") and not self.values:
            raise ValueError(f"filter {self.key}: `{self.op}` needs at least one value")
        if self.column and self.op in COMPARISONS and len(self.values) != 1:
            raise ValueError(f"filter {self.key}: `{self.op}` needs exactly one value")
        return self

    @property
    def title(self) -> str:
        return self.label or self.key

    def applies_to(self, profile: str) -> bool:
        return not self.profiles or profile in self.profiles

    def summary(self) -> str:
        """Human-readable shape of the rule, for lists and logs."""
        if self.expr:
            return "SQL"
        vals = ", ".join(self.values[:4]) + ("…" if len(self.values) > 4 else "")
        return f"{self.column} {OP_LABELS[self.op]}" + (f" {vals}" if self.values else "")


def _literal(dialect, column_type: str, value: str):
    """A UI value is a string; compare numbers as numbers so `expo_amt > 0` is not a text compare."""
    from gl_dq.core.schema import NUMERIC

    if column_type.startswith(NUMERIC):
        try:
            return dialect.lit(float(value) if "." in value else int(value))
        except ValueError:
            pass
    return dialect.lit(value)


def predicate(ctx, f: Filter) -> str:
    """SQL that is TRUE for the rows this rule keeps."""
    if f.expr:
        # same variables a source-of-truth query gets, so a rule can name a reference list
        # (a CSV of ids, a lookup table) that is spelled differently in each profile
        return jinja2.Template(f.expr, undefined=jinja2.StrictUndefined).render(
            raw_table=ctx.project.table, table=ctx.project.table, **ctx.project.sql_vars).strip()
    col = ctx.schema.validate([f.column])[0]
    ref, typ, d = ctx.schema.ref(col), ctx.schema.type_of(col), ctx.dialect
    if f.op == "is_null":
        return f"{ref} IS NULL"
    if f.op == "not_null":
        return f"{ref} IS NOT NULL"
    if f.op in COMPARISONS:
        return f"{ref} {COMPARISONS[f.op]} {_literal(d, typ, f.values[0])}"
    chosen = [v for v in f.values if v != NULL_LABEL]
    has_null = NULL_LABEL in f.values
    inside = f"CAST({ref} AS STRING) IN ({', '.join(d.lit(v) for v in chosen)})" if chosen else None
    if f.op == "in":
        parts = [p for p in (inside, f"{ref} IS NULL" if has_null else None) if p]
        return f"({' OR '.join(parts)})" if len(parts) > 1 else parts[0]
    parts = [p for p in (f"NOT ({inside})" if inside else None,
                         f"{ref} IS NOT NULL" if has_null else None) if p]
    return f"({' AND '.join(parts)})" if len(parts) > 1 else parts[0]


class FilterSet(BaseModel):
    filters: list[Filter] = []

    @model_validator(mode="after")
    def _keys(self):
        keys = [f.key for f in self.filters]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate filter keys")
        return self

    def get(self, key: str) -> Filter | None:
        return next((f for f in self.filters if f.key == key), None)

    def active(self, profile: str | None = None) -> list[Filter]:
        return [f for f in self.filters if f.enabled and (profile is None or f.applies_to(profile))]

    def where(self, ctx) -> str | None:
        """AND of the active rules, or None when nothing is active (then nothing is wrapped)."""
        parts = [f"COALESCE(({predicate(ctx, f)}), FALSE)" for f in self.active(ctx.profile)]
        return " AND ".join(parts) if parts else None

    def fingerprint(self, ctx=None) -> str:
        """'exclude_zero_exposure+us_only#a1b2c3' - stored with every run and used as a cache key."""
        act = self.active(ctx.profile if ctx is not None else None)
        if not act:
            return ""
        keys = sorted(f.key for f in act)
        blob = "|".join(f"{f.key}={f.expr or (f.column, f.op, tuple(f.values))}" for f in sorted(act, key=lambda x: x.key))
        return "+".join(keys) + "#" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:6]

    def with_all_disabled(self) -> FilterSet:
        return FilterSet(filters=[f.model_copy(update={"enabled": False}) for f in self.filters])


def validate(ctx, f: Filter) -> None:
    """Raise if the rule does not compile against the real table (unknown column, bad SQL).

    LIMIT 0 so this binds and type-checks the predicate without scanning the table.
    """
    ctx.db.query(f"SELECT * FROM {ctx.project.table} WHERE ({predicate(ctx, f)}) LIMIT 0")


def load_filters(config_store) -> FilterSet:
    text = config_store.read_text("filters.yaml")
    return FilterSet.model_validate(yaml.safe_load(text) or {}) if text else FilterSet()


def save_filters(config_store, fs: FilterSet) -> None:
    from gl_dq.core.config import dump_yaml

    config_store.write_text("filters.yaml", dump_yaml(fs.model_dump(mode="json")))
