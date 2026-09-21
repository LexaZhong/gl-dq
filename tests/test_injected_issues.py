"""Acceptance: every injected issue is flagged; a clean dataset flags nothing."""
from pathlib import Path

import pytest
import yaml

from gl_dq.core.results import parse_segment

ISSUES = yaml.safe_load((Path(__file__).resolve().parents[1] / "synthetic" / "injected_issues.yaml").read_text(encoding="utf-8"))["issues"]


def matching(findings, expect):
    f = findings[findings["check"] == expect["check"]]
    for key in ("variable", "item", "metric"):
        if key in expect:
            f = f[f[key] == expect[key]]
    want = {k: str(v) for k, v in (expect.get("segment") or {}).items()}
    if want:
        f = f[[all(parse_segment(s).get(k) == v for k, v in want.items()) for s in f["segment"]]]
    return f


@pytest.mark.parametrize("issue", ISSUES, ids=[i["id"] for i in ISSUES])
def test_injected_issue_is_flagged(findings_injected, issue):
    f = matching(findings_injected, issue["expect"])
    assert len(f), f"no findings match {issue['expect']}"
    assert f["status"].isin(["warn", "fail"]).any(), f"{issue['id']} not flagged:\n{f.to_string()}"


def test_clean_data_flags_nothing(findings_clean, ctx_clean):
    """Data-quality checks must be silent on clean data.

    Portfolio analysis modules are excluded: concentration and thin rating cells are properties of
    the book itself, so they are flagged on clean data by design.
    """
    quality = ctx_clean.checks_by_category().get("Checks", [])
    flagged = findings_clean[findings_clean["status"].isin(["warn", "fail"])
                             & findings_clean["check"].isin(quality)]
    assert flagged.empty, flagged.to_string()


def test_every_enabled_check_produced_findings(findings_injected, ctx_injected):
    assert set(ctx_injected.enabled_checks()) <= set(findings_injected["check"])
