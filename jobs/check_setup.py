"""Preflight: does this profile actually match the table? Run before deploying to Databricks.

    DATABRICKS_WAREHOUSE_ID=<id> python jobs/check_setup.py --profile prod --catalog <cat> --schema <schema>
    python jobs/check_setup.py --profile synthetic          # same checks against the local synthetic data

Verifies the table is readable, the global filters compile, every configured column exists
(measures, derived columns, policy key, segment candidates, and every column named in the check
configs) and the source values match. Prints what to fix; exits non-zero if anything is missing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gl_dq.core.context import load_context  # noqa: E402

OK, BAD = "  ok   ", "  MISS "


def configured_columns(ctx) -> dict[str, list[str]]:
    """Every column each check config refers to, so a rename is caught before the dashboard breaks."""
    out: dict[str, list[str]] = {}
    for name in ctx.checks:
        try:
            cfg = ctx.check_config(name)
        except Exception as e:  # noqa: BLE001
            out[f"{name} (config error)"] = [f"!! {e}"]
            continue
        out[name] = sorted({c for c in _column_fields(cfg.model_dump()) if c})
    return out


def _column_fields(d: dict) -> list[str]:
    """Column names anywhere in one check config, nested sections included (e.g. `analytics`)."""
    cols: list[str] = []
    for field in ("group_by", "dims", "segments", "summary_by", "anomaly_by", "include", "exclude",
                  "dimensions"):  # dimensions: segment_mix rating dimensions
        cols += [c for c in (d.get(field) or []) if isinstance(c, str)]
    for field in ("categorical", "numeric"):  # value_checks: one entry per column
        cols += [c for c in (d.get(field) or {}) if isinstance(c, str)]
    cols += [d[f] for f in ("default_dimension", "time_dim", "lr_basis", "class_col") if isinstance(d.get(f), str)]
    for k in (d.get("candidate_keys") or {}).values():
        cols += k
    cols += list((d.get("variables") or {}).keys()) if isinstance(d.get("variables"), dict) else \
        [v.get("name") for v in (d.get("variables") or []) if isinstance(v, dict)]
    for r in (d.get("rules") or []):
        cols += r.get("variables") or []
    for key, value in d.items():  # a nested section (e.g. a check's `analytics` block)
        if isinstance(value, dict) and key not in ("categorical", "numeric", "candidate_keys", "variables"):
            cols += _column_fields(value)
    return cols


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=None)
    p.add_argument("--catalog"), p.add_argument("--schema"), p.add_argument("--warehouse-id")
    args = p.parse_args(argv)
    for env, val in [("DQ_CATALOG", args.catalog), ("DQ_SCHEMA", args.schema),
                     ("DATABRICKS_WAREHOUSE_ID", args.warehouse_id)]:
        if val:
            os.environ[env] = val

    ctx = load_context(args.profile)
    print(f"profile {ctx.profile} | backend {ctx.project.backend} | table {ctx.project.table}")
    problems: list[str] = []

    n = ctx.db.query(f"SELECT COUNT(*) AS n FROM {ctx.project.table}").iloc[0]["n"]
    print(f"  ok    table readable: {int(n):,} rows, {len(ctx.schema.columns)} columns")

    # global filters: they restrict every query below, so validate them before anything else runs
    from gl_dq.core.filters import validate as validate_filter

    active = ctx.filters.active(ctx.profile)
    for f in active:
        try:
            validate_filter(ctx, f)
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} filter {f.key}: {str(e)[:160]}")
            problems.append(f"filter {f.key}: {str(e)[:160]}")
            ctx.filters = ctx.filters.with_all_disabled()  # keep the rest of the preflight usable
            break
    if active and ctx.filters.active(ctx.profile):
        kept = ctx.db.query(f"SELECT COUNT(*) AS n FROM {ctx.table_expr}").iloc[0]["n"]
        print(f"  ok    global filters ({', '.join(f.key for f in active)}): "
              f"{int(n):,} rows -> {int(kept):,} ({(int(n) - int(kept)) / int(n):.1%} removed)")
    else:
        print("  ok    global filters: none active")

    m = ctx.project.measures.model_dump()
    groups = {"measures": list(m.values()), "policy_key": ctx.project.policy_key,
              "segment_candidates": ctx.project.segment_candidates,
              "derived_columns": list(ctx.project.derived_columns)}
    groups.update(configured_columns(ctx))
    for label, cols in groups.items():
        missing = []
        for c in cols:
            if c.startswith("!!"):
                missing.append(c)
            elif not ctx.schema.has(c):
                missing.append(c)
        status = BAD if missing else OK
        print(f"{status} {label}: {len(cols)} referenced" + (f" | MISSING {', '.join(missing)}" if missing else ""))
        if missing:
            problems.append(f"{label}: {', '.join(missing)}")

    try:  # derived columns must also *evaluate* (e.g. year(pol_eff_dt) needs pol_eff_dt to be a date)
        from gl_dq.summary import summary_sql

        ctx.db.query(summary_sql(ctx, list(ctx.project.derived_columns)))
        print(f"  ok    derived columns evaluate: {', '.join(ctx.project.derived_columns) or 'none'}")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} derived columns failed: {e}")
        problems.append(f"derived columns: {e}")

    # compared on the filtered population, which is what every check sees
    found = set(ctx.db.query(
        f"SELECT DISTINCT {ctx.schema.ref(ctx.project.src_col)} AS s FROM {ctx.table_expr}")["s"].dropna())
    expected = set(ctx.project.sources)
    note = " (after global filters)" if ctx.filters.active(ctx.profile) else ""
    print((BAD if expected != found else OK) + f" sources: configured {sorted(expected)}, in data{note} {sorted(found)}")
    if expected != found:
        problems.append(f"sources differ: only in config {sorted(expected - found)}, only in data {sorted(found - expected)}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) to fix in config/profiles/{ctx.profile}.yaml or the check configs:")
        for p_ in problems:
            print(f"  - {p_}")
        sys.exit(1)
    print("All good. Next: python jobs/refresh.py --profile " + ctx.profile)


if __name__ == "__main__":
    main()
