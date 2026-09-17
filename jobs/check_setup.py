"""Preflight: does this profile actually match the table? Run before deploying to Databricks.

    DATABRICKS_WAREHOUSE_ID=<id> python jobs/check_setup.py --profile prod --catalog <cat> --schema <schema>
    python jobs/check_setup.py --profile synthetic          # same checks against the local synthetic data

Verifies the table is readable, every configured column exists (measures, derived columns, policy key,
segment candidates, and every column named in the check configs), the source values match, and the
source-of-truth queries run. Prints what to fix; exits non-zero if anything is missing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gl_dq.core.context import load_context  # noqa: E402
from gl_dq.core.schema import UnknownColumnError  # noqa: E402

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
        cols: list[str] = []
        d = cfg.model_dump()
        for field in ("group_by", "dims", "segments", "summary_by", "anomaly_by", "include", "exclude"):
            cols += [c for c in (d.get(field) or []) if isinstance(c, str)]
        for k in (d.get("candidate_keys") or {}).values():
            cols += k
        cols += list((d.get("variables") or {}).keys()) if isinstance(d.get("variables"), dict) else \
            [v.get("name") for v in (d.get("variables") or []) if isinstance(v, dict)]
        cols += [d.get("class_col")] if d.get("class_col") else []
        for r in (d.get("rules") or []):
            cols += r.get("variables") or []
        out[name] = sorted({c for c in cols if c})
    return out


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
    print(f"profile {ctx.profile} · backend {ctx.project.backend} · table {ctx.project.table}")
    problems: list[str] = []

    n = ctx.db.query(f"SELECT COUNT(*) AS n FROM {ctx.project.table}").iloc[0]["n"]
    print(f"  ok    table readable: {int(n):,} rows, {len(ctx.schema.columns)} columns")

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
        print(f"{status} {label}: {len(cols)} referenced" + (f" · MISSING {', '.join(missing)}" if missing else ""))
        if missing:
            problems.append(f"{label}: {', '.join(missing)}")

    try:  # derived columns must also *evaluate* (e.g. year(evt_dt) needs evt_dt to be a date)
        ctx.db.query(ctx.render_sql("summary.sql.j2", dims=list(ctx.project.derived_columns),
                                    policy_key=[ctx.schema.ref(c) for c in ctx.project.policy_key],
                                    premium=ctx.schema.ref(ctx.project.measures.written_premium), where=None))
        print(f"  ok    derived columns evaluate: {', '.join(ctx.project.derived_columns) or 'none'}")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} derived columns failed: {e}")
        problems.append(f"derived columns: {e}")

    found = set(ctx.db.query(
        f"SELECT DISTINCT {ctx.schema.ref(ctx.project.src_col)} AS s FROM {ctx.project.table}")["s"].dropna())
    expected = set(ctx.project.sources)
    print((BAD if expected != found else OK) + f" sources: configured {sorted(expected)}, in data {sorted(found)}")
    if expected != found:
        problems.append(f"sources differ: only in config {sorted(expected - found)}, only in data {sorted(found - expected)}")

    for check, field in [("premium_recon", "sot_query"), ("loss_recon", "sot_query")]:
        try:
            path = getattr(ctx.check_config(check), field)
            sql = ctx.render_user_sql(path)
            cols = list(ctx.db.query(ctx.render_sql("sot_columns.sql.j2", sot_sql=sql)).columns)
            print(f"  ok    {check} source of truth ({path}): {', '.join(cols)}")
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} {check} source of truth failed: {str(e)[:160]}")
            problems.append(f"{check} source of truth: {str(e)[:160]}")

    print()
    if problems:
        print(f"{len(problems)} problem(s) to fix in config/profiles/{ctx.profile}.yaml or the check configs:")
        for p_ in problems:
            print(f"  - {p_}")
        sys.exit(1)
    print("All good. Next: python jobs/refresh.py --profile " + ctx.profile)


if __name__ == "__main__":
    main()
