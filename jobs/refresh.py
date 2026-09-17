"""Refresh job: run all enabled checks for a profile and append the findings to the results store.

Local:       python jobs/refresh.py --profile synthetic
Databricks:  spark_python_task in databricks.yml (parameters: --profile prod)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gl_dq.core.context import load_context  # noqa: E402
from gl_dq.runner import refresh  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=None, help="profile name in config/profiles (default: $DQ_PROFILE or synthetic)")
    p.add_argument("--checks", nargs="*", help="subset of checks to run")
    p.add_argument("--catalog", help="sets DQ_CATALOG (prod profile)")
    p.add_argument("--schema", help="sets DQ_SCHEMA (prod profile)")
    p.add_argument("--warehouse-id", help="sets DATABRICKS_WAREHOUSE_ID (prod profile)")
    args = p.parse_args(argv)
    for env, val in [("DQ_CATALOG", args.catalog), ("DQ_SCHEMA", args.schema), ("DATABRICKS_WAREHOUSE_ID", args.warehouse_id)]:
        if val:
            os.environ[env] = val
    ctx = load_context(args.profile)
    run_id, findings, errors = refresh(ctx, args.checks)
    if errors:
        print(f"{len(errors)} check(s) errored: {', '.join(errors)}", file=sys.stderr)
        for name, tb in errors.items():
            print(f"--- {name}\n{tb}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
