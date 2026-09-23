"""Preflight must pass on a matching profile and fail loudly on a mismatched one."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))

import check_setup  # noqa: E402


def _profile(tmp_path, **changes):
    raw = yaml.safe_load((ROOT / "config" / "profiles" / "synthetic.yaml").read_text(encoding="utf-8"))
    raw.update(changes)
    p = tmp_path / "test_profile.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(p)


def test_passes_on_the_synthetic_profile(ctx_injected, tmp_path, capsys):
    check_setup.main(["--profile", _profile(tmp_path)])
    out = capsys.readouterr().out
    assert "MISS" not in out and "All good" in out


def test_reports_a_renamed_measure(ctx_injected, tmp_path, capsys):
    prof = _profile(tmp_path, measures={"written_premium": "tot_wrtn_prm_amt", "loss": "allocation",
                                        "claim_count": "claim_cnt_v2",  # renamed in the table
                                        "exposure": "expo_amt", "exposure_base": "expn_bs_std"})
    with pytest.raises(SystemExit) as e:
        check_setup.main(["--profile", prof])
    assert e.value.code == 1
    assert "measures: claim_cnt_v2" in capsys.readouterr().out


def test_reports_unexpected_sources(ctx_injected, tmp_path, capsys):
    with pytest.raises(SystemExit):
        check_setup.main(["--profile", _profile(tmp_path, sources=["BOP", "BMQ", "CMQ", "PERSONAL"])])
    assert "only in config ['PERSONAL']" in capsys.readouterr().out


def test_reports_a_broken_global_filter(ctx_injected, tmp_path, capsys):
    """A filter that does not compile must be named by preflight, not break every page later."""
    prof = _profile(tmp_path, config_dir=str(tmp_path / "cfg"))
    cfg = tmp_path / "cfg"
    (cfg / "checks").mkdir(parents=True)
    for f in (ROOT / "config" / "checks").glob("*.yaml"):
        (cfg / "checks" / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    (cfg / "workflow.yaml").write_text((ROOT / "config" / "workflow.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (cfg / "filters.yaml").write_text(
        'filters:\n  - key: broken\n    enabled: true\n    exclude_when: "no_such_column = 1"\n',
        encoding="utf-8")
    with pytest.raises(SystemExit):
        check_setup.main(["--profile", prof])
    assert "filter broken" in capsys.readouterr().out


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
    assert written == {"gl_master": "/Volumes/vol/GL/gl_master_cleaning/extract/gl_master"}
    assert any("LIMIT 1000" in c for c in calls)


def test_file_io_and_console_output_are_portable():
    """Windows defaults to cp1252: every file read must name UTF-8, and job output must be ASCII."""
    import re

    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "src").rglob("*.py")) + sorted((root / "jobs").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"\.read_text\(\s*\)", text):
            line = text[:m.start()].count("\n") + 1
            raise AssertionError(f"{path.name}:{line} reads a file with the platform encoding; pass encoding='utf-8'")
    storage = (root / "src" / "gl_dq" / "core" / "storage.py").read_text(encoding="utf-8")
    for call in re.findall(r"\.(?:read_text|write_text)\([^)]*\)", storage):
        assert "encoding" in call, f"storage.py: {call} must name an encoding"
    for path in sorted((root / "jobs").glob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "print(" in line:
                assert line.isascii(), f"{path.name}:{i} prints non-ASCII, which a cp1252 console cannot encode"


def test_non_ascii_round_trips_through_storage(tmp_path):
    from gl_dq.core.knowledge import KnowledgeStore, Note
    from gl_dq.core.storage import LocalStorage

    store = KnowledgeStore(LocalStorage(tmp_path))
    store.add_note("expn_bs_std", Note(author="actuaire@co.com", text="Prime non alignée — écart de 3 % · 100°"))
    assert "écart de 3 %" in store.get("expn_bs_std")[0].notes[0].text
    assert "écart" in (tmp_path / "variables" / "expn_bs_std.yaml").read_text(encoding="utf-8")


def test_every_column_a_check_config_names_is_checked(ctx_injected):
    """Preflight is how a column rename gets caught, so it has to look at every config field that
    holds a column name - including the ones added later (segment_mix dimensions, value_checks)."""
    cols = check_setup.configured_columns(ctx_injected)
    assert set(ctx_injected.check_config("segment_mix").dimensions) <= set(cols["segment_mix"])
    vc = ctx_injected.check_config("value_checks")
    assert set(vc.categorical) | set(vc.numeric) <= set(cols["value_checks"])
    assert ctx_injected.check_config("loss_summary").analytics.lr_basis in cols["loss_summary"]
    for name, names in cols.items():
        for c in names:
            assert ctx_injected.schema.has(c), f"{name} references {c}, which is not in the table"
