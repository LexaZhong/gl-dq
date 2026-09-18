"""Copy the repo's check configs and source-of-truth SQL into the profile's config folder.

    python jobs/seed_volume.py --profile prod                 # skips files that already exist
    python jobs/seed_volume.py --profile prod --overwrite
    python jobs/seed_volume.py --path /Volumes/<cat>/<vol>/.../config

The destination is the profile's `config_dir` (a UC volume path in prod). Existing files are kept
unless --overwrite, because that copy is the live config people edit from the app.
Uses Databricks unified auth (CLI profile, or DATABRICKS_HOST + DATABRICKS_TOKEN).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gl_dq import REPO_ROOT  # noqa: E402
from gl_dq.core.config import load_project  # noqa: E402
from gl_dq.core.storage import make_storage  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="prod", help="profile whose config_dir is the destination")
    p.add_argument("--path", help="destination folder, overriding the profile's config_dir")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    dest = args.path or load_project(args.profile).config_dir
    print(f"destination: {dest}")
    store = make_storage(dest, REPO_ROOT)
    for f in sorted((ROOT / "config").rglob("*")):
        if not f.is_file() or "profiles" in f.parts:
            continue
        rel = f.relative_to(ROOT / "config").as_posix()
        if store.version(rel) and not args.overwrite:
            print(f"skip   {rel} (exists)")
            continue
        store.write_text(rel, f.read_text(encoding="utf-8"))
        print(f"upload {rel}")


if __name__ == "__main__":
    main()
