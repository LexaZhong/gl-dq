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
    df = ctx.db.query(f"SELECT DISTINCT CAST({ref} AS STRING) AS v FROM {ctx.table_expr} "
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


def filter_impact(ctx, fs=None) -> pd.DataFrame:
    """How much data each global filter removes, plus the combined total, in one pass.

    Measured against the unfiltered table: one row per active rule (what it removes on its own)
    and a TOTAL row (what they remove together, counting overlaps once).
    """
    from gl_dq.core.filters import predicate

    fs = ctx.filters if fs is None else fs
    # every rule defined for this profile, enabled or not: seeing what a rule *would* cost is how
    # you decide whether to turn it on
    rules = [f for f in fs.filters if f.applies_to(ctx.profile)]
    active = fs.active(ctx.profile)
    if not rules:
        return pd.DataFrame(columns=["key", "label", "enabled", "rows_removed", "pct_rows",
                                     "premium_removed", "pct_premium"])
    specs = [{"alias": f"f{i}", "predicate": predicate(ctx, f)} for i, f in enumerate(rules)]
    combined = " AND ".join(f"COALESCE(({predicate(ctx, f)}), FALSE)" for f in active) or "1 = 1"
    r = ctx.db.query(ctx.render_sql("filter_impact.sql.j2", filters=specs, combined=combined,
                                    premium=ctx.schema.ref(ctx.project.measures.written_premium))).iloc[0]
    rows_total, prem_total = float(r["rows_total"] or 0), float(r["premium_total"] or 0)
    out = [{"key": f.key, "label": f.title, "enabled": f.enabled,
            "rows_removed": int(r[f"rows_{s['alias']}"] or 0),
            "premium_removed": float(r[f"premium_{s['alias']}"] or 0)}
           for f, s in zip(rules, specs)]
    out.append({"key": "TOTAL", "label": f"All {len(active)} active filter(s) together", "enabled": True,
                "rows_removed": int(r["rows_removed_all"] or 0),
                "premium_removed": float(r["premium_removed_all"] or 0)})
    df = pd.DataFrame(out)
    df["pct_rows"] = df["rows_removed"] / rows_total if rows_total else np.nan
    df["pct_premium"] = df["premium_removed"] / prem_total if prem_total else np.nan
    df = df[["key", "label", "enabled", "rows_removed", "pct_rows", "premium_removed", "pct_premium"]]
    df.attrs.update(rows_total=int(rows_total), premium_total=prem_total)  # set last: slicing drops attrs
    return df


def summary_sql(ctx, dims: list[str] | None = None, where: str | None = None) -> str:
    return ctx.render_sql("summary.sql.j2", dims=ctx.schema.validate(list(dims or [])),
                          policy_key=[ctx.schema.ref(c) for c in ctx.project.policy_key],
                          premium=ctx.schema.ref(ctx.project.measures.written_premium), where=where)
