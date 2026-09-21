"""Segment deep dive: scoping SQL, policy-grain stats/histograms, and the credibility arithmetic."""
import numpy as np
import pandas as pd
import pytest

from gl_dq.checks._segment_detail import (
    METRICS,
    REST,
    SEGMENT,
    bounds_for,
    bucket_specs,
    claims_for_z,
    policy_hist,
    policy_stats,
    policy_trend,
    segment_where,
)
from gl_dq.checks.segment_mix import credibility
from gl_dq.core.results import split_segment_columns
from gl_dq.core.schema import UnknownColumnError


# ---- split_segment_columns -----------------------------------------------------------
def test_split_makes_one_column_per_dimension():
    df = pd.DataFrame({"segment": ["src=BOP|pol_yr=2019", "src=BMQ|pol_yr=2020"], "value": [1, 2]})
    out = split_segment_columns(df)
    assert list(out.columns) == ["src", "pol_yr", "value"]  # inserted where `segment` was
    assert list(out["src"]) == ["BOP", "BMQ"] and list(out["pol_yr"]) == ["2019", "2020"]


def test_split_handles_ungrouped_and_mixed_levels():
    df = pd.DataFrame({"segment": ["ALL", "src=BOP", "src=BOP|pol_yr=2019"], "value": [1, 2, 3]})
    out = split_segment_columns(df)
    assert list(out["src"]) == ["(all)", "BOP", "BOP"]
    assert list(out["pol_yr"]) == ["(all)", "", "2019"]  # a coarser row simply has no value


def test_split_leaves_frames_alone_when_there_is_nothing_to_split():
    every_row_ungrouped = pd.DataFrame({"segment": ["ALL", "ALL"], "value": [1, 2]})
    pd.testing.assert_frame_equal(split_segment_columns(every_row_ungrouped), every_row_ungrouped)
    no_segment = pd.DataFrame({"variable": ["a"], "value": [1]})
    pd.testing.assert_frame_equal(split_segment_columns(no_segment), no_segment)
    pd.testing.assert_frame_equal(split_segment_columns(pd.DataFrame()), pd.DataFrame())
    wide = pd.DataFrame({"segment": ["|".join(f"d{i}=v" for i in range(9))]})
    pd.testing.assert_frame_equal(split_segment_columns(wide), wide)  # over max_dims


def test_split_does_not_duplicate_columns_that_are_already_there():
    """Reconciliation tables carry the dimension columns as well as the label."""
    df = pd.DataFrame({"src": ["BOP"], "segment": ["src=BOP"], "value": [1]})
    out = split_segment_columns(df)
    assert list(out.columns) == ["src", "value"] and list(out["src"]) == ["BOP"]


# ---- scoping -------------------------------------------------------------------------
def test_segment_where_quotes_values_and_whitelists_columns(ctx_injected):
    s, d = ctx_injected.schema, ctx_injected.dialect
    assert segment_where(s, d, {"src": "BOP"}) == '"src" = \'BOP\''
    assert segment_where(s, d, {"src": "BOP", "pol_yr": 2019}) == '"src" = \'BOP\' AND (year(pol_eff_dt)) = 2019'
    assert segment_where(s, d, {"class1_cd": None}).endswith("IS NULL")
    assert segment_where(s, d, {"class1_cd": "<null>"}).endswith("IS NULL")
    assert segment_where(s, d, {"src": "O'Brien"}) == '"src" = \'O\'\'Brien\''
    assert segment_where(s, d, {"each_occ_lmt_amt": np.int64(1000000)}) == '"each_occ_lmt_amt" = 1000000'
    with pytest.raises(UnknownColumnError):
        segment_where(s, d, {"not_a_column": 1})


def test_segment_where_selects_exactly_the_segment(ctx_injected):
    chk = ctx_injected.make_check("segment_mix")
    mix = chk.profile(["src"])
    where = segment_where(chk.schema, chk.ctx.dialect, {"src": "BMQ"})
    got = ctx_injected.db.query(f"SELECT COUNT(*) n, SUM(tot_wrtn_prm_amt) p FROM gl_master_synth WHERE {where}").iloc[0]
    row = mix[mix["src"] == "BMQ"].iloc[0]
    assert int(got["n"]) == int(row["records"])
    assert float(got["p"]) == pytest.approx(float(row["premium"]))


# ---- policy-grain compute ------------------------------------------------------------
@pytest.fixture(scope="module")
def detail(ctx_injected):
    chk = ctx_injected.make_check("segment_mix")
    where = segment_where(chk.schema, chk.ctx.dialect, {"src": "BOP"})
    base = ctx_injected.db.query(
        "SELECT expn_bs FROM gl_master_synth GROUP BY 1 ORDER BY SUM(tot_wrtn_prm_amt) DESC LIMIT 1").iloc[0]["expn_bs"]
    return chk, where, str(base)


def test_policy_stats_covers_every_metric_and_both_scopes(detail):
    chk, where, base = detail
    stats = policy_stats(chk, where, base, 1000.0)
    assert set(stats["metric"]) == set(METRICS)
    assert set(stats["scope"]) == {SEGMENT, REST}
    prem = stats[(stats["metric"] == "premium") & (stats["scope"] == SEGMENT)].iloc[0]
    total = chk.ctx.db.query(
        f"SELECT SUM(tot_wrtn_prm_amt) p, COUNT(*) n FROM gl_master_synth WHERE {where}").iloc[0]
    assert float(prem["vsum"]) == pytest.approx(float(total["p"]))  # policy grain re-sums to the same premium
    assert 0 < int(prem["n_policies"]) < int(total["n"])  # many rows per policy term


def test_policy_stats_severity_is_loss_over_claims(detail):
    chk, where, base = detail
    stats = policy_stats(chk, where, base, 1000.0).set_index(["metric", "scope"])
    loss = float(stats.loc[("loss", SEGMENT), "vsum"])
    claims = float(stats.loc[("claims", SEGMENT), "vsum"])
    mean_sev = float(stats.loc[("severity", SEGMENT), "vmean"])
    assert loss > 0 and claims > 0
    # the mean of per-policy severities is not loss/claims, but must sit in the same order of magnitude
    assert 0.05 < mean_sev / (loss / claims) < 20


@pytest.mark.parametrize("method", ["none", "log10", "log1p"])
def test_histogram_shares_sum_to_one_per_scope(detail, method):
    chk, where, base = detail
    h = policy_hist(chk, where, base, 1000.0, ("premium", "severity"), method, 25)
    assert not h.empty
    for (metric, scope), part in h.groupby(["metric", "scope"]):
        assert part["share"].sum() == pytest.approx(1.0), (metric, scope)
        assert part["bucket"].between(0, 24).all()
        assert part["left"].is_monotonic_increasing


def test_both_scopes_share_the_same_bins(detail):
    """An overlay only means something if the two series are binned identically."""
    chk, where, base = detail
    h = policy_hist(chk, where, base, 1000.0, ("premium",), "none", 20)
    edges = {scope: sorted(part["left"].round(6)) for scope, part in h.groupby("scope")}
    common = set(edges[SEGMENT]) & set(edges[REST])
    assert common and set(edges[SEGMENT]) <= set(edges[SEGMENT]) | set(edges[REST])
    seg, rest = h[h["scope"] == SEGMENT], h[h["scope"] == REST]
    for b in set(seg["bucket"]) & set(rest["bucket"]):
        assert seg[seg["bucket"] == b]["left"].iloc[0] == pytest.approx(rest[rest["bucket"] == b]["left"].iloc[0])


def test_log_bounds_come_from_the_positive_part(detail):
    chk, where, base = detail
    stats = policy_stats(chk, where, base, 1000.0)
    lo, hi = bounds_for(stats, "premium", "log10")
    assert hi > lo and np.isfinite([lo, hi]).all()  # log10 bounds, so ~1 to ~6 for dollar premiums
    specs, geom = bucket_specs(chk, stats, ["premium"], "log10", 10)
    assert specs and "LOG10" in specs[0]["expr"] and "> 0" in specs[0]["where"]
    assert geom["premium"][1] > 0


def test_metric_with_no_usable_values_is_dropped_not_crashed(detail):
    chk, where, base = detail
    stats = policy_stats(chk, where, base, 1000.0).copy()
    stats.loc[stats["metric"] == "premium", ["pcts", "pcts_pos"]] = None
    assert bounds_for(stats, "premium", "none") is None
    specs, geom = bucket_specs(chk, stats, ["premium", "severity"], "none", 10)
    assert [s["metric"] for s in specs] == ["severity"]


def test_trend_splits_the_book_without_losing_rows(detail):
    chk, where, base = detail
    tr = policy_trend(chk, where, base, 1000.0)
    assert set(tr["scope"]) == {SEGMENT, REST}
    total = float(chk.ctx.db.query("SELECT SUM(tot_wrtn_prm_amt) p FROM gl_master_synth").iloc[0]["p"])
    assert tr["premium"].sum() == pytest.approx(total)  # segment + rest = the whole book
    assert (tr.groupby("scope")["pol_yr"].nunique() > 1).all()


# ---- credibility ---------------------------------------------------------------------
def test_claims_for_z_inverts_the_square_root_rule():
    for z in (0.25, 0.5, 0.8, 1.0):
        assert credibility(claims_for_z(z, 1082.0), 1082.0) == pytest.approx(z)
    assert claims_for_z(0.5, 1082.0) == pytest.approx(270.5)
    assert claims_for_z(1.5, 1082.0) == 1082.0  # Z is capped at 1
