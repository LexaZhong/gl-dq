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
