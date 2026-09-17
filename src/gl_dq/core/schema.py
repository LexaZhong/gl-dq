"""Column whitelist: every column reference in generated SQL goes through TableSchema."""
from __future__ import annotations

from gl_dq.core.db import Dialect

NUMERIC = ("int", "bigint", "smallint", "tinyint", "double", "float", "decimal", "real", "hugeint", "long", "numeric")
AMOUNT = ("double", "float", "decimal", "real", "numeric")
DATE = ("date", "timestamp")


class UnknownColumnError(ValueError):
    pass


class TableSchema:
    def __init__(self, table: str, columns: dict[str, str], derived: dict, dialect: Dialect):
        self.table = table
        self.columns = columns  # name -> type
        self.derived = derived  # name -> DerivedColumn
        self.dialect = dialect

    # ---- lookups -------------------------------------------------------
    def names(self, include_derived: bool = True) -> list[str]:
        return list(self.columns) + (list(self.derived) if include_derived else [])

    def has(self, name: str) -> bool:
        return name in self.columns or name in self.derived

    def type_of(self, name: str) -> str:
        if name in self.columns:
            return self.columns[name]
        if name in self.derived:
            return self.derived[name].type
        raise UnknownColumnError(f"unknown column {name!r} (not in {self.table} or derived_columns)")

    def is_numeric(self, name: str) -> bool:
        return self.type_of(name).startswith(NUMERIC)

    def is_amount(self, name: str) -> bool:
        return self.type_of(name).startswith(AMOUNT)

    def is_date(self, name: str) -> bool:
        return self.type_of(name).startswith(DATE)

    def is_string(self, name: str) -> bool:
        t = self.type_of(name)
        return t.startswith(("varchar", "string", "char", "text"))

    # ---- SQL fragments -------------------------------------------------
    def ref(self, name: str) -> str:
        """SQL expression for a whitelisted column or configured derived column."""
        if name in self.columns:
            return self.dialect.quote(name)
        if name in self.derived:
            return f"({self.derived[name].expr})"
        raise UnknownColumnError(f"unknown column {name!r} (not in {self.table} or derived_columns)")

    def select_as(self, name: str) -> str:
        return f"{self.ref(name)} AS {self.dialect.quote(name)}"

    def validate(self, names) -> list[str]:
        bad = [n for n in names if not self.has(n)]
        if bad:
            raise UnknownColumnError(f"unknown column(s) {bad} for {self.table}")
        return list(names)
