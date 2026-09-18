"""Preflight must pass on a matching profile and fail loudly on a mismatched one."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))

import check_setup  # noqa: E402


def _profile(tmp_path, **changes):
    raw = yaml.safe_load((ROOT / "config" / "profiles" / "synthetic.yaml").read_text())
    raw.update(changes)
    p = tmp_path / "test_profile.yaml"
    p.write_text(yaml.safe_dump(raw))
    return str(p)


def test_passes_on_the_synthetic_profile(ctx_injected, tmp_path, capsys):
    check_setup.main(["--profile", _profile(tmp_path)])
    out = capsys.readouterr().out
    assert "MISS" not in out and "All good" in out


def test_reports_a_renamed_measure(ctx_injected, tmp_path, capsys):
    prof = _profile(tmp_path, measures={"written_premium": "tot_wrtn_prm_amt", "loss": "allocation",
                                        "claim_count": "claim_cnt",  # renamed in the table
                                        "exposure": "expo_amt", "exposure_base": "expn_bs"})
    with pytest.raises(SystemExit) as e:
        check_setup.main(["--profile", prof])
    assert e.value.code == 1
    assert "measures: claim_cnt" in capsys.readouterr().out


def test_reports_unexpected_sources(ctx_injected, tmp_path, capsys):
    with pytest.raises(SystemExit):
        check_setup.main(["--profile", _profile(tmp_path, sources=["BOP", "BMQ", "CMQ", "PERSONAL"])])
    assert "only in config ['PERSONAL']" in capsys.readouterr().out


def test_reports_a_broken_source_of_truth_query(ctx_injected, tmp_path, capsys):
    prof = _profile(tmp_path, sql_vars={"sot_premium_table": "no_such_table", "sot_loss_table": "sot_loss_synth"})
    with pytest.raises(SystemExit):
        check_setup.main(["--profile", prof])
    assert "premium_recon source of truth failed" in capsys.readouterr().out


def test_spark_backend_uses_the_session(monkeypatch):
    """SparkDatabase must send SQL to spark.sql and parse DESCRIBE TABLE output."""
    import pandas as pd

    from gl_dq.core.db import SparkDatabase

    class FakeDF:
        def __init__(self, df):
            self._df = df

        def toPandas(self):  # noqa: N802 (pyspark API)
            return self._df

    seen = []

    class FakeSpark:
        def sql(self, q):
            seen.append(q)
            if q.startswith("DESCRIBE TABLE"):
                return FakeDF(pd.DataFrame({"col_name": ["src", "expo_amt", "", "# Partitioning", "Not partitioned"],
                                            "data_type": ["STRING", "DOUBLE", "", "", ""]}))
            return FakeDF(pd.DataFrame({"n": [1]}))

    db = SparkDatabase(FakeSpark())
    assert db.dialect.name == "databricks"
    assert db.describe("cat.sch.gl_master") == {"src": "string", "expo_amt": "double"}
    assert db.query("SELECT 1 AS n")["n"].tolist() == [1]
    assert seen == ["DESCRIBE TABLE cat.sch.gl_master", "SELECT 1 AS n"]


def test_export_extract_writes_the_expected_paths(ctx_injected, monkeypatch, capsys):
    """The extract layout must match what the parquet profile reads back."""
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs"))
    import export_extract

    calls = []

    class FakeWriter:
        def mode(self, m):
            return self

        def parquet(self, path):
            calls.append(path)

    class FakeDF:
        write = FakeWriter()

    class FakeSpark:
        def sql(self, q):
            calls.append(q.split("\n")[0][:60])
            return FakeDF()

    monkeypatch.setattr(ctx_injected, "db", type("D", (), {"spark": FakeSpark()})(), raising=False)
    written = export_extract.export(ctx_injected, "/Volumes/vol/GL/gl_master_cleaning/extract", sample=1000)
    assert written["gl_master"].endswith("/extract/gl_master")
    assert set(written) == {"gl_master", "sot_premium", "sot_loss"}
    assert any(c.endswith("/extract/sot_premium") for c in calls)
    assert any(c.endswith("/extract/sot_loss") for c in calls)
    assert any("LIMIT 1000" in c for c in calls)
