"""Headless smoke test of every dashboard page with streamlit.testing.AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
PAGES = ["summary", "tracker", "preprocessing", "knowledge", "key_uniqueness", "missing_rate", "distribution",
         "exposure", "business_rules", "value_checks", "segment_mix", "target_analysis"]


@pytest.fixture(scope="module")
def refreshed(ctx_injected):
    from gl_dq.runner import refresh

    refresh(ctx_injected, log=lambda *_: None)
    return ctx_injected


def _open(page, monkeypatch):
    monkeypatch.setenv("DQ_START_PAGE", page)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(refreshed, page, monkeypatch):
    at = _open(page, monkeypatch)
    assert at.title, "page rendered no title"
    if page not in ("summary", "tracker", "preprocessing", "knowledge"):
        # app.py falls back to the summary page for an unknown name, which would let a deleted
        # page keep "passing" this test
        assert at.title[0].value != "📋 Portfolio summary", f"{page} fell back to the summary page"


def test_every_page_in_the_list_exists(ctx_injected):
    overview = {"summary", "tracker", "preprocessing", "knowledge"}
    assert set(PAGES) - overview == set(ctx_injected.enabled_checks())


def test_status_note_and_assignees_saved_from_check_page(refreshed, monkeypatch):
    monkeypatch.setenv("DQ_USER", "ds@test.com")
    at = _open("missing_rate", monkeypatch)
    var = at.selectbox(key="np_var_missing_rate").value
    at.selectbox(key=f"np_status_missing_rate_{var}").set_value("actuary_review").run()
    at.text_input(key=f"np_as_missing_rate_{var}_actuary").input("actuary@test.com")
    at.text_area[0].input("Blank values are legacy records; confirm treatment with pricing.")
    next(b for b in at.button if b.label == "Save").click().run()
    assert not at.exception, [e.value for e in at.exception]
    rec, _ = refreshed.knowledge.get(var)
    assert rec.status == "actuary_review" and rec.assignees["actuary"] == "actuary@test.com"
    assert rec.notes[-1].author == "ds@test.com" and rec.status_log[-1].to_status == "actuary_review"


def test_next_stage_button(refreshed, monkeypatch):
    at = _open("key_uniqueness", monkeypatch)
    var = at.selectbox(key="np_var_key_uniqueness").value
    before = refreshed.knowledge.get(var)[0].status
    nxt = refreshed.workflow.next_stage(before)
    at.button(key=f"np_next_key_uniqueness_{var}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert refreshed.knowledge.get(var)[0].status == nxt.key


def test_preprocess_status_shows_section_and_adds_step(refreshed, monkeypatch):
    at = _open("distribution", monkeypatch)
    var = at.selectbox(key="np_var_distribution").value
    key = f"distribution_{var}"
    assert not any("Recommended preprocessing" in m.value for m in at.markdown)
    at.selectbox(key=f"np_status_distribution_{var}").set_value("preprocess_in_modeling").run()
    assert any("Recommended preprocessing" in m.value for m in at.markdown)
    at.selectbox(key=f"pp_op_{key}").set_value("cap").run()
    at.number_input(key=f"pp_{key}_cap_upper_pct").set_value(0.995)
    at.multiselect(key=f"pp_src_{key}").set_value(["BMQ"])
    at.text_input(key=f"pp_why_{key}").input("Extreme premium outliers")
    at.button(key=f"pp_add_{key}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    rec, _ = refreshed.knowledge.get(var)
    assert rec.status == "preprocess_in_modeling"
    assert rec.preprocessing[-1].op == "cap" and rec.preprocessing[-1].params["upper_pct"] == 0.995
    assert rec.preprocessing[-1].sources == ["BMQ"]


def test_no_view_as_selector(refreshed, monkeypatch):
    at = _open("tracker", monkeypatch)
    assert not any(s.label == "View as" for s in at.selectbox)


def test_distribution_controls(refreshed, monkeypatch):
    at = _open("distribution", monkeypatch)
    name = at.selectbox(key="dist_var").value
    at.selectbox(key=f"dpb_{name}").set_value(50).run()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key=f"dlm_{name}").set_value("log10").run()
    at.toggle(key=f"dly_{name}").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key=f"dpb_{name}").set_value("custom").run()
    at.text_input(key=f"dpc_{name}").input("0.1, 0.5, 0.9").run()
    assert not at.exception, [e.value for e in at.exception]
    # switch to a categorical variable
    at.selectbox(key="dist_var").set_value("class_cd_std").run()
    assert not at.exception, [e.value for e in at.exception]


def test_summary_is_the_landing_page(refreshed, monkeypatch):
    import os

    os.environ.pop("DQ_START_PAGE", None)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("📋 Portfolio summary")
    assert {"Records", "Policies", "Written premium", "Loss", "Claims", "Loss ratio"} == \
        {m.label for m in at.metric}
    assert at.multiselect(key="sum_dims").value == ["src", "covg_type_desc"]


def test_summary_dimensions_can_be_changed(refreshed, monkeypatch):
    at = _open("summary", monkeypatch)
    at.multiselect(key="sum_dims").set_value(["src", "pol_yr", "loc_st_abbr"]).run()
    assert not at.exception, [e.value for e in at.exception]
    at.multiselect(key="sum_dims").set_value([]).run()
    assert not at.exception, [e.value for e in at.exception]


def test_tracker_kpis(refreshed, monkeypatch):
    at = _open("tracker", monkeypatch)
    labels = [m.label for m in at.metric]
    assert {"Columns", "Closed", "🔁 Re-opened", "With data engineer", "With actuary"} <= set(labels)


def test_running_with_plain_python_explains_itself():
    """`python app/app.py` must say how to start it, not warn about a missing cache runtime."""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, APP], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "streamlit run app/app.py" in (r.stderr + r.stdout)


def test_summary_filters_narrow_the_page(refreshed, monkeypatch):
    at = _open("summary", monkeypatch)
    records = lambda a: next(m.value for m in a.metric if m.label == "Records")  # noqa: E731
    before = records(at)
    at.multiselect(key="sum_f_src").set_value(["BOP"]).run()
    assert not at.exception, [e.value for e in at.exception]
    after = records(at)
    assert after != before and int(after.replace(",", "")) < int(before.replace(",", ""))
    assert any("Filtered to" in m.value for m in at.caption)
    # the filter reaches the tables, not just the KPIs
    table = next(df for df in at.dataframe if "src" in getattr(df.value, "columns", []))
    assert set(table.value["src"]) == {"BOP"}
    at.multiselect(key="sum_f_src").set_value([]).run()
    assert records(at) == before


def test_summary_filter_options_follow_the_dimensions(refreshed, monkeypatch):
    at = _open("summary", monkeypatch)
    page_filter = "Filter {} (this page only)".format
    assert {page_filter("src"), page_filter("covg_type_desc")} <= {m.label for m in at.multiselect}
    at.multiselect(key="sum_dims").set_value(["loc_st_abbr"]).run()
    labels = {m.label for m in at.multiselect}
    assert page_filter("loc_st_abbr") in labels and page_filter("covg_type_desc") not in labels
