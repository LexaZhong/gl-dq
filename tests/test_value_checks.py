"""Values check: consistency across sources, and numeric plausibility."""
import pandas as pd
import pytest

from gl_dq.checks.value_checks import CategoricalRule, NumericRule


def _finding(findings, **kw):
    f = findings
    for k, v in kw.items():
        f = f[f[k] == v]
    return f


def test_source_unique_values_are_flagged(ctx_injected):
    """A limit only BMQ uses is a mapping difference, not a real one."""
    res = ctx_injected.make_check("value_checks").run()
    f = res.findings
    unique = _finding(f, variable="each_occ_lmt_amt", metric="source_unique_values", segment="src=BMQ")
    assert len(unique) == 1 and unique.iloc[0]["value"] == 1
    assert "10000000" in unique.iloc[0]["detail"]
    assert unique.iloc[0]["status"] == "warn"
    # the other two sources use only shared values
    for src in ("BOP", "CMQ"):
        other = _finding(f, variable="each_occ_lmt_amt", metric="source_unique_values", segment=f"src={src}")
        assert other.iloc[0]["value"] == 0 and other.iloc[0]["status"] == "pass"
    rows = _finding(f, variable="each_occ_lmt_amt", metric="pct_rows_source_unique", segment="src=BMQ")
    assert 0 < rows.iloc[0]["value"] < 0.05


def test_out_of_bounds_catches_the_sentinel(ctx_injected):
    res = ctx_injected.make_check("value_checks").run()
    bad = _finding(res.findings, variable="bi_ded_amt", metric="pct_out_of_bounds", segment="src=BOP")
    assert bad.iloc[0]["status"] == "fail" and "99,999" in bad.iloc[0]["detail"]


def test_clean_data_is_silent(ctx_clean):
    res = ctx_clean.make_check("value_checks").run()
    assert not res.findings["status"].isin(["warn", "fail"]).any(), \
        res.findings[res.findings["status"].isin(["warn", "fail"])].to_string()
    assert (res.findings["status"] == "info").any()      # the value inventory is still reported


def test_median_ratio_detects_a_unit_error(ctx_injected):
    """BMQ 2019 premium is recorded in cents, which shows up as a median far from the others."""
    chk = ctx_injected.make_check("value_checks")
    rule = NumericRule(compare_medians=True, median_ratio=None)
    df = chk.numeric_profile("tot_wrtn_prm_amt", rule)
    findings = chk.numeric_findings("tot_wrtn_prm_amt", rule, df, None)
    ratio = [f for f in findings if f["metric"] == "median_ratio_across_src"]
    assert ratio and ratio[0]["value"] > 1
    assert df["median"].notna().all()


def test_comparison_can_be_switched_off_per_column(ctx_injected):
    """Exposure grain differs by source, so its medians must not be compared."""
    chk = ctx_injected.make_check("value_checks")
    rule = NumericRule(compare_medians=False)
    findings = chk.numeric_findings("expo_amt", rule, chk.numeric_profile("expo_amt", rule), None)
    assert not [f for f in findings if f["metric"] == "median_ratio_across_src"]
    assert ctx_injected.check_config("value_checks").numeric["expo_amt"].compare_medians is False


def test_high_cardinality_column_is_reported_not_compared(ctx_injected):
    chk = ctx_injected.make_check("value_checks")
    counts = chk.value_matrix("pol_num")
    findings, wide = chk.categorical_findings("pol_num", CategoricalRule(max_values=10), counts)
    assert [f["metric"] for f in findings] == ["n_values"]          # no comparison attempted
    assert findings[0]["status"] == "info" and findings[0]["value"] > 10
    assert any("too many to compare" in m for m in chk._messages)


def test_top_value_share_uses_the_value_matrix(ctx_injected):
    chk = ctx_injected.make_check("value_checks")
    rule = NumericRule(discrete=True, top_value_warn=0.2)
    counts = chk.value_matrix("bi_ded_amt")
    _, wide = chk.categorical_findings("bi_ded_amt", CategoricalRule(), counts)
    findings = chk.numeric_findings("bi_ded_amt", rule, chk.numeric_profile("bi_ded_amt", rule), wide)
    shares = [f for f in findings if f["metric"] == "top_value_share"]
    assert shares and all(0 < f["value"] <= 1 for f in shares)
