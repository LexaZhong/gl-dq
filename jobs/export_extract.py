"""Write a parquet extract of gl_master (and the pricing-study queries) to a volume folder.

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

SOT = [("premium_recon", "sot_premium"), ("loss_recon", "sot_loss")]


def export(ctx, out: str, sample: int | None = None, skip_sot: bool = False) -> dict[str, str]:
    spark = ctx.db.spark  # spark backend only: the extract is written by the cluster
    written = {}
    sql = f"SELECT * FROM {ctx.project.table}" + (f" LIMIT {int(sample)}" if sample else "")
    print(f"gl_master <- {sql}")
    spark.sql(sql).write.mode("overwrite").parquet(f"{out}/gl_master")
    written["gl_master"] = f"{out}/gl_master"
    if skip_sot:
        return written
    for check, name in SOT:
        try:
            cfg = ctx.check_config(check)
            print(f"{name} <- {cfg.sot_query}")
            spark.sql(ctx.render_user_sql(cfg.sot_query)).write.mode("overwrite").parquet(f"{out}/{name}")
            written[name] = f"{out}/{name}"
        except Exception as e:  # noqa: BLE001  a missing study is not fatal: the rest still works
            print(f"  skipped {name}: {str(e)[:200]}")
    return written


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="workspace")
    p.add_argument("--out", help="destination folder (default: <volume>/extract)")
    p.add_argument("--sample", type=int, help="export only N rows of gl_master (for a quick share)")
    p.add_argument("--skip-sot", action="store_true", help="export gl_master only")
    args = p.parse_args(argv)
    ctx = load_context(args.profile)
    volume = os.environ.get("DQ_VOLUME_DIR") or str(Path(ctx.project.knowledge_dir).parent)
    out = (args.out or f"{volume}/extract").rstrip("/")
    written = export(ctx, out, args.sample, args.skip_sot)
    print("\nwrote:")
    for name, path in written.items():
        print(f"  {name:<12} {path}")
    print("\nthen, anywhere:\n  DQ_PROFILE=parquet DQ_PARQUET_TABLE=" + written["gl_master"] + " streamlit run app/app.py")


if __name__ == "__main__":
    main()
