"""Render every check's SQL with the Databricks dialect (no warehouse): no DuckDB-only syntax may leak."""
import copy
import re

import pandas as pd
import pytest

from gl_dq.core.db import DatabricksDialect
from gl_dq.core.schema import TableSchema

DUCKDB_ONLY = [r"quantile_cont\(", r"\brow\(", r"string_agg\(", r'"[a-z_]+"']


class RecordingDB:
    """Captures SQL and returns empty frames; enough to exercise SQL generation paths."""

    dialect = DatabricksDialect()

    def __init__(self, real):
        self.real, self.sql = real, []

    def query(self, sql):
        self.sql.append(sql)
        return self.real.query(_to_duckdb(sql))  # execute the equivalent so code paths continue

    def describe(self, table):
        return self.real.describe(table)


def _to_duckdb(sql: str) -> str:
    sql = re.sub(r"`([^`]+)`", r'"\1"', sql)
    sql = re.sub(r"percentile_approx\(([^,]+), array\(([^)]*)\), 10000\)", r"quantile_cont(\1, [\2])", sql)
    sql = sql.replace("struct(", "row(")
    sql = re.sub(r"array_join\(array_sort\(collect_set\(([^)]+)\)\), ', '\)", r"string_agg(DISTINCT \1, ', ')", sql)
    return sql


@pytest.fixture(scope="module")
def dbx_ctx(ctx_injected):
    ctx = copy.copy(ctx_injected)
    ctx.db = RecordingDB(ctx_injected.db)
    ctx.schema = TableSchema(ctx.schema.table, ctx.schema.columns, ctx.schema.derived, ctx.db.dialect)
    ctx.__post_init__()
    return ctx


def test_databricks_sql_has_no_duckdb_syntax(dbx_ctx):
    for name in dbx_ctx.enabled_checks():
        chk = dbx_ctx.make_check(name)
        chk.run()
        if name == "key_uniqueness":
            chk.suggest_key("CMQ")
        if name == "distribution":
            for spec in chk.cfg.variables:
                if chk.kind(spec) == "numeric":
                    chk.histogram(spec)
    assert dbx_ctx.db.sql
    for sql in dbx_ctx.db.sql:
        body = re.sub(r"'[^']*'", "''", sql)  # ignore string literals
        for pat in DUCKDB_ONLY:
            assert not re.search(pat, body), f"{pat} in:\n{sql}"
