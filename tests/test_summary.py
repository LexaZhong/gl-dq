"""Portfolio summary: record count, distinct policy terms and premium."""
import pandas as pd
import pytest

from gl_dq.core.schema import UnknownColumnError
from gl_dq.summary import summarize


def test_totals_match_the_table(ctx_injected):
    t = summarize(ctx_injected).iloc[0]
    raw = ctx_injected.db.query("""
        SELECT COUNT(*) AS records,
               COUNT(DISTINCT (pol_num, pol_eff_dt, pol_exp_dt)) AS policies,
               SUM(tot_wrtn_prm_amt) AS premium
        FROM gl_master_synth""").iloc[0]
    assert int(t.records) == int(raw.records)
    assert int(t.policies) == int(raw.policies)
    assert t.premium == pytest.approx(float(raw.premium))
    assert t.premium_per_policy == pytest.approx(t.premium / t.policies)


def test_policy_count_uses_the_configured_policy_key(ctx_injected):
    assert ctx_injected.project.policy_key == ["pol_num", "pol_eff_dt", "pol_exp_dt"]
    t = summarize(ctx_injected).iloc[0]
    # a policy term spans many rows (coverages, locations, transactions) but is counted once
    assert int(t.policies) < int(t.records)
    assert int(t.policies) == ctx_injected.db.query(
        "SELECT COUNT(*) n FROM (SELECT DISTINCT pol_num, pol_eff_dt, pol_exp_dt FROM gl_master_synth)").iloc[0]["n"]


def test_grouped_summary(ctx_injected):
    by_src = summarize(ctx_injected, ["src"])
    assert set(by_src["src"]) == set(ctx_injected.project.sources)
    total = summarize(ctx_injected).iloc[0]
    assert by_src["records"].sum() == total.records
    assert by_src["premium"].sum() == pytest.approx(total.premium)
    assert by_src["premium_share"].sum() == pytest.approx(1.0)
    # policies are distinct counts: sub-group counts do not have to add up to the total
    assert by_src["policies"].sum() >= total.policies

    by_two = summarize(ctx_injected, ["src", "covg_type_desc"])
    assert len(by_two) > len(by_src)
    assert by_two.groupby("src")["records"].sum().sort_index().equals(
        by_src.set_index("src")["records"].sort_index())
    assert by_two["premium"].is_monotonic_decreasing  # sorted by premium


def test_unknown_dimension_rejected(ctx_injected):
    with pytest.raises(UnknownColumnError):
        summarize(ctx_injected, ["nope"])


def test_distinct_values_for_the_filter_picker(ctx_injected):
    from gl_dq.summary import distinct_values

    assert distinct_values(ctx_injected, "src") == ["BMQ", "BOP", "CMQ"]
    covg = distinct_values(ctx_injected, "covg_type_desc")
    assert "Liquor Liability" in covg and covg == sorted(covg)
    # a column with nulls offers them last, as an explicit choice
    classes = distinct_values(ctx_injected, "class_cd_std")
    assert classes[-1] == "<null>" and len(classes) > 10


def test_filter_clause_sql(ctx_injected):
    from gl_dq.summary import filter_clause

    assert filter_clause(ctx_injected, None) is None
    assert filter_clause(ctx_injected, {"src": []}) is None                       # empty = all
    one = filter_clause(ctx_injected, {"src": ["BOP"]})
    assert one == """(CAST("src" AS STRING) IN ('BOP'))"""
    two = filter_clause(ctx_injected, {"src": ["BOP", "BMQ"], "covg_type_desc": ["Liquor Liability"]})
    assert two.count(" AND ") == 1 and "'BMQ'" in two and "'Liquor Liability'" in two
    nulls = filter_clause(ctx_injected, {"class_cd_std": ["<null>"]})
    assert nulls == """("class_cd_std" IS NULL)"""
    both = filter_clause(ctx_injected, {"class_cd_std": ["10010", "<null>"]})
    assert "IS NULL" in both and "'10010'" in both and " OR " in both


def test_filter_clause_escapes_values(ctx_injected):
    """Picker values come from the data and may contain quotes."""
    from gl_dq.summary import filter_clause, summarize

    clause = filter_clause(ctx_injected, {"covg_type_desc": ["O'Brien's ' cover"]})
    assert "''Brien''s '' cover" in clause
    assert summarize(ctx_injected, ["src"], clause).empty                          # runs, matches nothing

    with pytest.raises(UnknownColumnError):
        filter_clause(ctx_injected, {"not_a_column": ["x"]})


def test_filtered_summary_matches_the_filter(ctx_injected):
    from gl_dq.summary import filter_clause, summarize

    where = filter_clause(ctx_injected, {"src": ["BOP"], "covg_type_desc": ["Premises/Operations"]})
    filtered = summarize(ctx_injected, ["src", "covg_type_desc"], where)
    assert list(filtered["src"]) == ["BOP"] and list(filtered["covg_type_desc"]) == ["Premises/Operations"]

    unfiltered = summarize(ctx_injected, ["src", "covg_type_desc"])
    row = unfiltered[(unfiltered["src"] == "BOP") & (unfiltered["covg_type_desc"] == "Premises/Operations")].iloc[0]
    assert int(filtered.iloc[0]["records"]) == int(row["records"])
    assert int(filtered.iloc[0]["policies"]) == int(row["policies"])       # distinct count, not a sum
    assert filtered.iloc[0]["premium_share"] == pytest.approx(1.0)         # shares are of the filtered slice

    totals = summarize(ctx_injected, [], where).iloc[0]
    assert int(totals.records) == int(row["records"]) and int(totals.records) < int(summarize(ctx_injected).iloc[0].records)
