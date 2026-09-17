"""Check a source-of-truth query before wiring it into the dashboard.

    python jobs/validate_sot.py --profile prod --check premium_recon
    python jobs/validate_sot.py --profile prod --check loss_recon --print-sql   # paste-able comparison SQL

Reports: the columns the query returns, whether every reconciliation dimension is present (after
dim_map), whether the measure columns exist, the grain (duplicate rows per dimension combination are
summed, nulls in dimensions are flagged), the totals on both sides, and the biggest differences.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gl_dq.checks._recon import pipeline_agg, reconcile, sot_agg  # noqa: E402
from gl_dq.core.context import load_context  # noqa: E402

CHECKS = ("premium_recon", "loss_recon")


def measures_of(ctx, check: str, cfg) -> dict[str, tuple[str, str]]:
    """alias -> (pipeline column, SOT column)."""
    m = ctx.project.measures
    if check == "premium_recon":
        return {"premium": (cfg.measure or m.written_premium, cfg.sot_measure)}
    return {"loss": (m.loss, cfg.sot_loss_col), "claim_count": (m.claim_count, cfg.sot_claim_count_col)}


def dims_of(ctx, check: str, cfg) -> list[str]:
    return list(cfg.dims) if check == "premium_recon" else list(dict.fromkeys(list(cfg.segments) + [cfg.time_dim]))


def validate(ctx, check: str, print_sql: bool = False) -> list[str]:
    cfg = ctx.check_config(check)
    chk = ctx.make_check(check, cfg)
    dims, measures = dims_of(ctx, check, cfg), measures_of(ctx, check, cfg)
    dim_map = {d: cfg.dim_map.get(d, d) for d in dims}
    problems: list[str] = []
    print(f"\n=== {check} · {cfg.sot_query}")

    sot_sql = ctx.render_user_sql(cfg.sot_query)
    raw = ctx.db.query(ctx.render_sql("sot_columns.sql.j2", sot_sql=sot_sql))
    cols = list(raw.columns)
    print(f"returns {len(cols)} columns: {', '.join(cols)}")

    for dim, sot_col in dim_map.items():
        if sot_col in cols:
            print(f"  ok    dimension {dim}" + (f" <- {sot_col}" if sot_col != dim else ""))
        else:
            near = [c for c in cols if c.lower() == sot_col.lower() or sot_col.lower() in c.lower()]
            hint = f" (did you mean {near[0]}? set dim_map: {{{dim}: {near[0]}}})" if near else \
                   f" (add it to the query, drop {dim} from the grain, or map it with dim_map)"
            print(f"  MISS  dimension {dim}: no column {sot_col!r}{hint}")
            problems.append(f"{check}: dimension {dim} missing from the source of truth")
    for alias, (_, sot_col) in measures.items():
        ok = sot_col in cols
        print(("  ok    " if ok else "  MISS  ") + f"measure {alias} <- {sot_col}")
        if not ok:
            problems.append(f"{check}: measure column {sot_col!r} not in the source of truth")
    if problems:
        return problems

    present = [d for d in dims if dim_map[d] in cols]
    sot = sot_agg(chk, sot_sql, present, cfg.dim_map, {a: c for a, (_, c) in measures.items()})
    rows = ctx.db.query(f"SELECT COUNT(*) AS n FROM (\n{sot_sql}\n) s").iloc[0]["n"]
    print(f"grain: {int(rows):,} rows -> {len(sot):,} distinct {', '.join(present)} combinations"
          + (" (duplicates are summed)" if int(rows) != len(sot) else ""))
    for d in present:
        n_null = int(sot[d].isna().sum())
        if n_null:
            print(f"  warn  {n_null} rows have a null {d}: they will not match the pipeline")
            problems.append(f"{check}: null values in source-of-truth dimension {d}")

    pipe = pipeline_agg(chk, present, {a: ctx.schema.ref(p) for a, (p, _) in measures.items()},
                        getattr(cfg, "where", None))
    for alias in measures:
        tol = cfg.tolerance if check == "premium_recon" else (
            cfg.tolerance_loss if alias == "loss" else cfg.tolerance_claims)
        rec = reconcile(pipe, sot, present, alias, tol)
        p_tot, s_tot = rec["pipeline"].sum(), rec["sot"].sum()
        diff = p_tot - s_tot
        print(f"\n{alias}: pipeline {p_tot:,.0f} · source of truth {s_tot:,.0f} · diff {diff:,.0f}"
              + (f" ({diff / s_tot:+.2%})" if s_tot else ""))
        only = rec[rec["presence"] != "both"]
        if len(only):
            print(f"  {len(only)} segment(s) on one side only: "
                  + ", ".join(f"{r.segment} [{r.presence}]" for r in only.head(5).itertuples()))
        bad = rec[rec["status"] == "fail"]
        print(f"  {len(bad)} of {len(rec)} segments outside tolerance (abs {tol.abs:,.0f}, pct {tol.pct:.2%})")
        if len(bad):
            with pd.option_context("display.width", 200, "display.max_colwidth", 60):
                print(bad[["segment", "pipeline", "sot", "diff", "pct_diff"]].head(10).to_string(index=False))

    if print_sql:
        print(f"\n--- paste-able comparison SQL for {check} ---")
        print(ctx.render_sql(
            "recon_compare.sql.j2", dims=present, measures=list(measures),
            pipeline_sql=ctx.render_sql("agg_by_dims.sql.j2", dims=present,
                                        measures={a: ctx.schema.ref(p) for a, (p, _) in measures.items()},
                                        where=getattr(cfg, "where", None)),
            sot_sql=ctx.render_sql("agg_sot.sql.j2", sot_sql=sot_sql,
                                   dims={d: cfg.dim_map.get(d, d) for d in present},
                                   measures={a: c for a, (_, c) in measures.items()})))
    return problems


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=None)
    p.add_argument("--check", choices=[*CHECKS, "both"], default="both")
    p.add_argument("--print-sql", action="store_true", help="print a standalone comparison query")
    p.add_argument("--catalog"), p.add_argument("--schema"), p.add_argument("--warehouse-id")
    args = p.parse_args(argv)
    for env, val in [("DQ_CATALOG", args.catalog), ("DQ_SCHEMA", args.schema),
                     ("DATABRICKS_WAREHOUSE_ID", args.warehouse_id)]:
        if val:
            os.environ[env] = val
    ctx = load_context(args.profile)
    problems = []
    for check in (CHECKS if args.check == "both" else [args.check]):
        problems += validate(ctx, check, args.print_sql)
    print()
    if problems:
        for p_ in problems:
            print(f"  - {p_}")
        sys.exit(1)
    print("Source-of-truth queries look usable.")


if __name__ == "__main__":
    main()
