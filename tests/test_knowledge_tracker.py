from pathlib import Path

import pandas as pd
import pytest
import yaml

from gl_dq.core.knowledge import (CheckSnapshot, ConflictError, KnowledgeStore, Note, PreprocessingStep,
                                  VariableRecord, export_markdown, preprocessing_spec)
from gl_dq.core.storage import LocalStorage
from gl_dq.core.workflow import Workflow, parse_mapping
from gl_dq.tracker import build_tracker, is_reopened, snapshots_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore(LocalStorage(tmp_path))


@pytest.fixture
def wf():
    return Workflow()


def test_parquet_results_round_trip_through_storage(tmp_path):
    """Run history must work on a volume too, where files go through the Files API, not the filesystem."""
    import pandas as pd

    from gl_dq.core.results import ParquetResults

    store = ParquetResults(LocalStorage(tmp_path))
    assert store.load().empty and store.latest().empty
    findings = pd.DataFrame([dict(check="missing_rate", variable="a", item=None, segment="src=BOP", metric="m",
                                  value=0.3, threshold=0.2, status="fail", detail="3 of 10")])
    store.append(findings, "run1", "2026-09-17T10:00:00Z", "prod")
    store.append(findings.assign(status="pass", value=0.0), "run2", "2026-09-17T11:00:00Z", "prod")
    assert sorted(p.name for p in (tmp_path / "runs").glob("*.parquet")) == \
        ["findings_run1.parquet", "findings_run2.parquet"]
    runs = store.runs()
    assert list(runs["run_id"]) == ["run1", "run2"] and list(runs["n_fail"]) == [1, 0]
    assert store.latest()["status"].tolist() == ["pass"]
    assert store.latest(1)["status"].tolist() == ["fail"]


def test_workflow_yaml_matches_and_validates():
    wf = Workflow.model_validate(yaml.safe_load((ROOT / "config" / "workflow.yaml").read_text(encoding="utf-8")))
    assert wf.keys() == Workflow().keys()
    assert {"resolved", "no_issue", "preprocess_in_modeling", "wont_fix"} == wf.done_keys()


def test_handoff_path_with_and_without_signoff(wf):
    path = ["not_started", "investigating", "actuary_review", "with_de", "ds_validation", "resolved"]
    assert [s.key for s in wf.flow()] == path
    assert wf.next_stage("ds_validation").key == "resolved"
    signoff = wf.model_copy(update={"require_actuary_signoff": True})
    assert signoff.next_stage("ds_validation").key == "actuary_signoff"
    assert signoff.next_stage("actuary_signoff").key == "resolved"
    assert wf.next_stage("resolved") is None and wf.next_stage("preprocess_in_modeling") is None
    assert wf.waiting_on("with_de") == "Data engineer" and wf.waiting_on("resolved") is None


def test_status_log_assignees_notes(store, wf):
    store.update("expn_bs", "ds@co.com", workflow=wf, status="investigating", assignees={"ds": "ds@co.com"},
                 note=Note(author="ds@co.com", text="UNK = unmapped legacy BMQ base", src="BMQ", tags=["mapping"]))
    store.update("expn_bs", "act@co.com", workflow=wf, status="actuary_review", assignees={"actuary": "act@co.com"})
    rec = store.update("expn_bs", "ds@co.com", workflow=wf, status="with_de", assignees={"de": "de@co.com", "ds": ""})
    assert rec.status == "with_de" and rec.assignees == {"actuary": "act@co.com", "de": "de@co.com"}
    assert [c.to_status for c in rec.status_log] == ["investigating", "actuary_review", "with_de"]
    assert rec.status_log[-1].from_status == "actuary_review"
    assert len(store.history("expn_bs")) == 3
    with pytest.raises(ValueError):
        store.update("expn_bs", "x", workflow=wf, status="not_a_stage")


def test_closing_stores_snapshots(store, wf):
    snaps = {"missing_rate": CheckSnapshot(judged_status="fail", judged_value=0.02)}
    store.update("a", "u", workflow=wf, status="investigating", snapshots=snaps)
    assert store.get("a")[0].checks == {}  # not closed yet
    store.update("a", "u", workflow=wf, status="no_issue", snapshots=snaps)
    assert store.get("a")[0].checks["missing_rate"].judged_value == 0.02


def test_preprocessing_steps_and_spec(store, wf):
    step = PreprocessingStep(op="map_values", params={"mapping": parse_mapping("UNK=null, X=Y")}, sources=["BMQ"],
                             rationale="UNK is an unmapped legacy base", author="ds")
    rec = store.add_preprocessing_step("expn_bs", step, "ds", set_status="preprocess_in_modeling")
    assert rec.status == "preprocess_in_modeling" and rec.status_log[-1].comment
    store.add_preprocessing_step("expn_bs", PreprocessingStep(op="impute", params={"strategy": "mode"}), "ds")
    store.update("loc_zipcd", "ds", workflow=wf, status="preprocess_in_modeling")  # no steps yet
    store.update("pol_num", "ds", workflow=wf, status="resolved")
    spec = preprocessing_spec(store.all(), wf, "gl_master")
    assert set(spec["variables"]) == {"expn_bs", "loc_zipcd"}
    steps = spec["variables"]["expn_bs"]["steps"]
    assert [s["op"] for s in steps] == ["map_values", "impute"] and steps[0]["params"]["mapping"] == {"UNK": None, "X": "Y"}
    assert steps[1]["sources"] == "all" and spec["variables"]["loc_zipcd"]["steps"] == []
    store.remove_preprocessing_step("expn_bs", 0, "ds")
    assert [s.op for s in store.get("expn_bs")[0].preprocessing] == ["impute"]
    md = export_markdown(store.all(), wf)
    assert "Recommended preprocessing" in md and "`impute`" in md


def test_conflict_detected(store, wf):
    store.add_note("x", Note(author="a", text="first"))
    rec, ver = store.get("x")
    store.add_note("x", Note(author="b", text="someone else"))  # concurrent edit
    with pytest.raises(ConflictError):
        store.update("x", "a", workflow=wf, status="resolved", expected_version=ver)


def test_legacy_status_mapping():
    assert VariableRecord(variable="v", status="accepted_as_is").status == "no_issue"


def test_reopen_rules():
    assert is_reopened(True, "pass", 0.0, "fail", 0.1)
    assert not is_reopened(True, "warn", 0.010, "warn", 0.0105)  # small drift
    assert is_reopened(True, "warn", 0.010, "warn", 0.02)  # materially worse
    assert not is_reopened(False, "pass", 0.0, "fail", 1.0)  # not closed
    assert not is_reopened(True, "fail", 0.1, "pass", 0.0)


def test_tracker(store, wf):
    before = pd.DataFrame([
        dict(check="missing_rate", variable="a", segment="src=BOP", metric="m", value=0.0, status="pass"),
        dict(check="missing_rate", variable="b", segment="src=BOP", metric="m", value=0.1, status="fail"),
    ])
    store.update("a", "u", workflow=wf, status="resolved", snapshots=snapshots_for("a", before))
    store.update("b", "u", workflow=wf, status="with_de", assignees={"de": "de@co.com"})
    after = before.assign(value=[0.3, 0.1], status=["fail", "fail"])
    var, long = build_tracker(["a", "b", "c"], after, store.all(), ["missing_rate"], wf)
    by = var.set_index("variable")
    assert by.loc["a", "reopened"] and not by.loc["a", "done"]
    assert by.loc["b", "waiting_on"] == "Data engineer" and by.loc["b", "assignee"] == "de@co.com"
    assert by.loc["b", "days_in_status"] is not None and not by.loc["c", "done"]
