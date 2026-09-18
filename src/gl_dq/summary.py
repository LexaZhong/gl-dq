"""Portfolio summary: record count, policy count and written premium at any grouping level.

A "policy" is one row per distinct `project.policy_key` (by default pol_num + pol_eff_dt + pol_exp_dt),
counted with COUNT(DISTINCT ...) so it is correct at every level - policy counts of sub-groups never
add up to the total, because one policy spans several coverages and rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


NULL_LABEL = "<null>"


def distinct_values(ctx, column: str, limit: int = 1000) -> list[str]:
    """Values of a column, as strings, for a filter picker. Nulls appear as '<null>'."""
    ref = ctx.schema.ref(ctx.schema.validate([column])[0])
    df = ctx.db.query(f"SELECT DISTINCT CAST({ref} AS STRING) AS v FROM {ctx.project.table} "
                      f"ORDER BY 1 LIMIT {int(limit)}")
    values = [NULL_LABEL if pd.isna(v) else str(v) for v in df["v"]]
    return sorted(values, key=lambda v: (v == NULL_LABEL, v))


def filter_clause(ctx, filters: dict[str, list[str]] | None) -> str | None:
    """SQL for {column: [values]} - empty or missing means 'all'. Values are escaped literals."""
    lit, parts = ctx.dialect.lit, []
    for column, values in (filters or {}).items():
        values = [v for v in (values or [])]
        if not values:
            continue  # no selection = no restriction
        ref = ctx.schema.ref(ctx.schema.validate([column])[0])
        chosen = [v for v in values if v != NULL_LABEL]
        clause = f"CAST({ref} AS STRING) IN ({', '.join(lit(v) for v in chosen)})" if chosen else None
        if NULL_LABEL in values:
            null_clause = f"{ref} IS NULL"
            clause = f"({clause} OR {null_clause})" if clause else null_clause
        parts.append(clause)
    return " AND ".join(f"({p})" for p in parts) if parts else None


def summarize(ctx, dims: list[str] | None = None, where: str | None = None) -> pd.DataFrame:
    dims = ctx.schema.validate(list(dims or []))
    sql = ctx.render_sql("summary.sql.j2", dims=dims,
                         policy_key=[ctx.schema.ref(c) for c in ctx.schema.validate(ctx.project.policy_key)],
                         premium=ctx.schema.ref(ctx.project.measures.written_premium), where=where)
    df = ctx.db.query(sql)
    for c in ("records", "policies"):
        df[c] = df[c].astype("int64")
    df["premium"] = df["premium"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["premium_per_policy"] = np.where(df["policies"] > 0, df["premium"] / df["policies"], np.nan)
        df["records_per_policy"] = np.where(df["policies"] > 0, df["records"] / df["policies"], np.nan)
    if dims:
        total = df["premium"].sum()
        df["premium_share"] = df["premium"] / total if total else np.nan
        df = df.sort_values("premium", ascending=False).reset_index(drop=True)
    return df


def summary_sql(ctx, dims: list[str] | None = None, where: str | None = None) -> str:
    return ctx.render_sql("summary.sql.j2", dims=ctx.schema.validate(list(dims or [])),
                          policy_key=[ctx.schema.ref(c) for c in ctx.project.policy_key],
                          premium=ctx.schema.ref(ctx.project.measures.written_premium), where=where)
