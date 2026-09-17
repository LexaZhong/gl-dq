"""Seed the UC Volume with the repo's default check configs and SOT query templates.

    python jobs/seed_volume.py --catalog main --schema pricing            # skips files that exist
    python jobs/seed_volume.py --catalog main --schema pricing --overwrite

Uses Databricks unified auth (CLI profile / DATABRICKS_HOST + token). Existing Volume files are
kept unless --overwrite, because the Volume copy is the live config that people edit from the app.
Remember to adjust column references in SQL predicates (applies_when / rules) for gl_master.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gl_dq.core.storage import VolumeStorage  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--volume", default="gl_dq")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    store = VolumeStorage(f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/config")
    for f in sorted((ROOT / "config").rglob("*")):
        if not f.is_file() or "profiles" in f.parts:
            continue
        rel = f.relative_to(ROOT / "config").as_posix()
        if store.version(rel) and not args.overwrite:
            print(f"skip   {rel} (exists)")
            continue
        store.write_text(rel, f.read_text())
        print(f"upload {rel}")


if __name__ == "__main__":
    main()
