"""Query backends (DuckDB locally, Databricks SQL warehouse in prod) and SQL dialect helpers."""
from __future__ import annotations

import os
import re
import threading
from abc import ABC, abstractmethod

import pandas as pd

from gl_dq.core.config import expand_env


class Dialect:
    name = "base"

    def quote(self, ident: str) -> str:
        raise NotImplementedError

    def percentiles(self, expr: str, probs: list[float]) -> str:
        raise NotImplementedError

    def row(self, exprs: list[str]) -> str:
        raise NotImplementedError

    def distinct_list(self, expr: str) -> str:
        raise NotImplementedError

    @staticmethod
    def lit(value) -> str:
        """SQL literal for a config/UI supplied value."""
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return repr(value)
        return "'" + str(value).replace("'", "''") + "'"


class DuckDBDialect(Dialect):
    name = "duckdb"

    def quote(self, ident):
        return '"' + ident.replace('"', '""') + '"'

    def percentiles(self, expr, probs):
        return f"quantile_cont({expr}, [{', '.join(repr(float(p)) for p in probs)}])"

    def row(self, exprs):
        return f"row({', '.join(exprs)})"

    def distinct_list(self, expr):
        return f"string_agg(DISTINCT {expr}, ', ')"


class DatabricksDialect(Dialect):
    name = "databricks"

    def quote(self, ident):
        return "`" + ident.replace("`", "``") + "`"

    def percentiles(self, expr, probs):
        return f"percentile_approx({expr}, array({', '.join(repr(float(p)) for p in probs)}), 10000)"

    def row(self, exprs):
        return f"struct({', '.join(exprs)})"

    def distinct_list(self, expr):
        return f"array_join(array_sort(collect_set({expr})), ', ')"


class Database(ABC):
    dialect: Dialect

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame: ...

    @abstractmethod
    def describe(self, table: str) -> dict[str, str]:
        """column name -> lower-case type name."""


class DuckDBDatabase(Database):
    dialect = DuckDBDialect()

    def __init__(self, path: str):
        import duckdb

        self.path = path
        self._con = duckdb.connect(path, read_only=True)
        self._lock = threading.Lock()

    def query(self, sql):
        with self._lock:
            cur = self._con.cursor()
        try:
            return cur.execute(sql).df()
        finally:
            cur.close()

    def describe(self, table):
        df = self.query(f"DESCRIBE {table}")
        return {r.column_name: r.column_type.lower() for r in df.itertuples()}


class DatabricksDatabase(Database):
    """SQL warehouse via databricks-sql-connector.

    Auth: inside Databricks Apps (and with a configured CLI profile) the SDK's unified
    auth is used; DATABRICKS_HOST/DATABRICKS_TOKEN also work.
    """

    dialect = DatabricksDialect()

    def __init__(self, warehouse_id: str):
        from databricks import sql
        from databricks.sdk.core import Config

        cfg = Config()
        self._conn = sql.connect(
            server_hostname=cfg.host.replace("https://", ""),
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            credentials_provider=lambda: cfg.authenticate,
        )

    def query(self, sql):
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall_arrow().to_pandas()

    def describe(self, table):
        df = self.query(f"DESCRIBE TABLE {table}")
        out = {}
        for r in df.itertuples():
            name = r.col_name
            if not name or name.startswith("#"):
                break  # partition / metadata section
            out[name] = r.data_type.lower()
        return out


def make_database(project) -> Database:
    if project.backend == "duckdb":
        from gl_dq import REPO_ROOT

        path = expand_env(project.duckdb_path or "")
        if not os.path.isabs(path):
            path = str(REPO_ROOT / path)
        return DuckDBDatabase(path)
    if project.backend == "databricks":
        wid = expand_env(project.warehouse_id or "")
        if not wid or wid.startswith("$"):
            raise RuntimeError("warehouse_id is not set (DATABRICKS_WAREHOUSE_ID)")
        return DatabricksDatabase(wid)
    raise ValueError(f"unknown backend {project.backend!r}")


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_identifier(name: str) -> bool:
    return bool(_IDENT.match(name))
