"""Write a parquet extract of gl_master to a volume folder.

Run in a Databricks notebook or job (needs Spark):

    python jobs/export_extract.py --profile workspace                  # -> <volume>/extract/...
    python jobs/export_extract.py --profile workspace --out /Volumes/.../extract --sample 100000

The extract is what `DQ_PROFILE=parquet` reads, so the dashboard can then run from a laptop, a job
or an app with no warehouse and no cluster.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gl_dq.core.context import load_context  # noqa: E402

def export(ctx, out: str, sample: int | None = None, apply_filters: bool = False) -> dict[str, str]:
    """Extract the table. Raw by default: the global filters then apply on top of the parquet, so a
    filter can be toggled off locally without re-exporting. `apply_filters` bakes them in."""
    spark = ctx.db.spark  # spark backend only: the extract is written by the cluster
    written = {}
    source = ctx.table_expr if apply_filters else ctx.project.table
    sql = f"SELECT * FROM {source}" + (f" LIMIT {int(sample)}" if sample else "")
    print(f"gl_master <- {sql}")
    spark.sql(sql).write.mode("overwrite").parquet(f"{out}/gl_master")
    written["gl_master"] = f"{out}/gl_master"
    return written


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="workspace")
    p.add_argument("--out", help="destination folder (default: <volume>/extract)")
    p.add_argument("--sample", type=int, help="export only N rows of gl_master (for a quick share)")
    p.add_argument("--filter", action="store_true", dest="apply_filters",
                   help="apply config/filters.yaml at export time (default: extract every row and let "
                        "the dashboard apply the filters, so they stay togglable)")
    args = p.parse_args(argv)
    ctx = load_context(args.profile)
    volume = os.environ.get("DQ_VOLUME_DIR") or str(Path(ctx.project.knowledge_dir).parent)
    out = (args.out or f"{volume}/extract").rstrip("/")
    written = export(ctx, out, args.sample, args.apply_filters)
    if args.apply_filters:
        print(f"filters baked in: {', '.join(f.key for f in ctx.filters.active(ctx.profile)) or 'none'}")
    print("\nwrote:")
    for name, path in written.items():
        print(f"  {name:<12} {path}")
    print("\nthen, anywhere:\n  DQ_PROFILE=parquet DQ_PARQUET_TABLE=" + written["gl_master"] + " streamlit run app/app.py")


if __name__ == "__main__":
    main()
