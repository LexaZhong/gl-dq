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


def _sweep(dbx_ctx):
    for name in dbx_ctx.enabled_checks():
        chk = dbx_ctx.make_check(name)
        chk.run()
        if name == "key_uniqueness":
            chk.suggest_key("CMQ")
        if name == "distribution":
            for spec in chk.cfg.variables:
                if chk.kind(spec) == "numeric":
                    chk.histogram(spec)
        if name == "segment_mix":  # the deep dive is render-only, so run() does not reach its SQL
            from gl_dq.checks._segment_detail import segment_where

            where = segment_where(chk.schema, chk.ctx.dialect, {dbx_ctx.project.src_col: "BOP"})
            chk.detail_stats(where=where, base="SALES", per=1000.0)
            chk.detail_hist(where=where, base="SALES", per=1000.0, metrics=("premium", "severity"),
                            log_method="log10", bins=10)
            chk.detail_trend(where=where, base="SALES", per=1000.0)
    assert dbx_ctx.db.sql
    for sql in dbx_ctx.db.sql:
        body = re.sub(r"--[^\n]*", "", sql)  # ignore SQL comments (the SOT files document examples there)
        body = re.sub(r"'[^']*'", "''", body)  # and string literals
        for pat in DUCKDB_ONLY:
            assert not re.search(pat, body), f"{pat} in:\n{sql}"


def test_databricks_sql_has_no_duckdb_syntax(dbx_ctx):
    _sweep(dbx_ctx)


def test_databricks_sql_survives_a_global_filter(dbx_ctx):
    """Every template reads `FROM {{ table }}`, which a global filter turns into a subquery."""
    from dataclasses import replace

    from gl_dq.core.filters import Filter, FilterSet

    fs = FilterSet(filters=[Filter(key="bop_only", enabled=True, column="src", op="in", values=["BOP"]),
                            Filter(key="no_zero_expo", enabled=True, column="expo_amt", op="gt", values=["0"])])
    ctx = replace(dbx_ctx, filters=fs)
    ctx.db.sql.clear()
    _sweep(ctx)
    assert any("AS gl" in sql for sql in ctx.db.sql), "the filtered subquery never reached the SQL"
