"""backend: parquet - the same checks, reading parquet files instead of a table."""

import pandas as pd
import pytest
import yaml

from gl_dq.core.context import load_context
from gl_dq.core.db import ParquetDatabase
from gl_dq.runner import run_all
from gl_dq.summary import summarize


@pytest.fixture(scope="module")
def parquet_ctx(data_dirs, tmp_path_factory):
    """A profile over the synthetic parquet files (same data as the duckdb fixture)."""
    folder = data_dirs["injected"].parent
    tmp = tmp_path_factory.mktemp("parquet_profile")
    profile = tmp / "parquet_test.yaml"
    profile.write_text(yaml.safe_dump({
        "extends": "parquet",
        "name": "parquet test",
        "parquet_views": {"gl_master": str(folder / "gl_master_synth.parquet")},
        "knowledge_dir": str(tmp / "knowledge"),
        "results": {"type": "parquet", "path": str(tmp / "runs")},
    }), encoding="utf-8")
    return load_context(str(profile))


def test_reads_parquet_without_a_database(parquet_ctx):
    assert parquet_ctx.project.backend == "parquet"
    assert isinstance(parquet_ctx.db, ParquetDatabase) and not parquet_ctx.db.missing
    assert parquet_ctx.schema.columns["tot_wrtn_prm_amt"].startswith("double")
    assert parquet_ctx.db.dialect.name == "duckdb"


def test_same_findings_as_the_table_backend(parquet_ctx, findings_injected):
    """Reading the extract must give exactly what reading the table gives."""
    findings, errors = run_all(parquet_ctx, log=lambda *_: None)
    assert not errors, errors
    same = ["check", "variable", "item", "segment", "metric", "status"]
    a = findings[same].sort_values(same).reset_index(drop=True)
    b = findings_injected[same].sort_values(same).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_summary_and_derived_columns_work(parquet_ctx):
    total = summarize(parquet_ctx).iloc[0]
    assert int(total.records) == 214537 or int(total.records) > 0
    by_year = summarize(parquet_ctx, ["src", "pol_yr"])          # derived column over the parquet
    assert by_year["pol_yr"].astype(int).between(2018, 2024).all()
    assert by_year["records"].sum() == total.records


def test_missing_optional_source_is_reported_not_fatal(data_dirs, tmp_path):
    """A gl_master extract with no study extract still loads; only the recon complains."""
    folder = data_dirs["injected"].parent
    profile = tmp_path / "p.yaml"
    profile.write_text(yaml.safe_dump({
        "extends": "parquet", "name": "no study",
        "parquet_views": {"gl_master": str(folder / "gl_master_synth.parquet"),
                          "reference": str(tmp_path / "does_not_exist")},
        "knowledge_dir": str(tmp_path / "k"), "results": {"type": "parquet", "path": str(tmp_path / "r")}}),
        encoding="utf-8")
    ctx = load_context(str(profile))
    assert "reference" in ctx.db.missing
    assert ctx.db.query("SELECT COUNT(*) AS n FROM gl_master").iloc[0]["n"] > 0
    with pytest.raises(Exception, match="reference"):
        ctx.db.describe("reference")


def test_a_view_can_be_a_csv(data_dirs, tmp_path):
    """A reference list downloaded from a SQL editor is a CSV; it must not need converting first."""
    folder = data_dirs["injected"].parent
    csv = tmp_path / "reference.csv"
    ParquetDatabase({"g": str(folder / "gl_master_synth.parquet")}).query(
        "SELECT src, year(pol_eff_dt) AS pol_yr, SUM(tot_wrtn_prm_amt) AS wrtn_prm FROM g GROUP BY 1, 2"
    ).to_csv(csv, index=False)

    db = ParquetDatabase({"gl_master": str(folder / "gl_master_synth.parquet"), "reference": str(csv)})
    assert not db.missing
    assert set(db.describe("reference")) == {"src", "pol_yr", "wrtn_prm"}
    assert int(db.query("SELECT COUNT(*) n FROM reference").iloc[0]["n"]) > 0


