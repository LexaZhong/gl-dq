import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _generate(out: Path, inject: bool):
    from synthetic import generate

    args = ["--out", str(out), "--seed", "42"] + ([] if inject else ["--no-inject"])
    generate.main(args)
    return out / "gl_synth.duckdb"


@pytest.fixture(scope="session")
def data_dirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("synth")
    return {"injected": _generate(base / "injected", True), "clean": _generate(base / "clean", False)}


def _context(db_path: Path, tmp: Path):
    from gl_dq.core.context import load_context

    os.environ["DQ_DUCKDB_PATH"] = str(db_path)
    os.environ["DQ_KNOWLEDGE_DIR"] = str(tmp / "knowledge")
    os.environ["DQ_RESULTS_DIR"] = str(tmp / "results")
    return load_context("synthetic")


@pytest.fixture(scope="session")
def ctx_injected(data_dirs, tmp_path_factory):
    return _context(data_dirs["injected"], tmp_path_factory.mktemp("ctx_injected"))


@pytest.fixture(scope="session")
def ctx_clean(data_dirs, tmp_path_factory):
    return _context(data_dirs["clean"], tmp_path_factory.mktemp("ctx_clean"))


@pytest.fixture(scope="session")
def findings_injected(ctx_injected) -> pd.DataFrame:
    from gl_dq.runner import run_all

    findings, errors = run_all(ctx_injected, log=lambda *_: None)
    assert not errors, errors
    return findings


@pytest.fixture(scope="session")
def findings_clean(ctx_clean) -> pd.DataFrame:
    from gl_dq.runner import run_all

    findings, errors = run_all(ctx_clean, log=lambda *_: None)
    assert not errors, errors
    return findings
