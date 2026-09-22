"""Segment mix & credibility: Pareto shares and the square-root credibility rule."""
import numpy as np
import pytest

from gl_dq.checks.segment_mix import Thresholds, credibility


def test_credibility_square_root_rule():
    assert credibility(1082) == pytest.approx(1.0)          # full credibility standard
    assert credibility(270.5) == pytest.approx(0.5)         # a quarter of the claims -> half the weight
    assert credibility(5000) == 1.0                         # capped at 1
    assert credibility(0) == 0.0
    assert credibility(108.2) == pytest.approx(0.316, abs=1e-3)
    # a different standard rescales it
    assert credibility(500, full_credibility=500) == pytest.approx(1.0)


def test_shares_and_cumulative_share(ctx_injected):
    chk = ctx_injected.make_check("segment_mix")
    df = chk.profile(["class_cd_std"])
    assert not df.empty
    for col in ("premium", "records", "claims"):
        assert df[f"{col}_share"].sum() == pytest.approx(1.0)
    assert df["premium_share"].is_monotonic_decreasing          # sorted biggest first
    assert df["cum_premium_share"].is_monotonic_increasing
    assert df["cum_premium_share"].iloc[-1] == pytest.approx(1.0)
    assert (df["z_claims"] <= 1).all() and (df["z_records"] <= 1).all()
    np.testing.assert_allclose(df["z_claims"], credibility(df["claims"]))


def test_totals_match_the_table(ctx_injected):
    chk = ctx_injected.make_check("segment_mix")
    df = chk.profile(["class_cd_std"])
    raw = ctx_injected.db.query(
        "SELECT COUNT(*) r, SUM(tot_wrtn_prm_amt) p, SUM(claim_ant) c FROM gl_master_synth").iloc[0]
    assert df["records"].sum() == int(raw["r"])
    assert df["premium"].sum() == pytest.approx(float(raw["p"]))
    assert df["claims"].sum() == int(raw["c"])


def test_large_and_thin_flags(ctx_injected):
    chk = ctx_injected.make_check("segment_mix")
    t = Thresholds(large_share=0.05, material_share=0.005, z_target=0.5)
    df = chk.profile(["class_cd_std"], t)
    large, thin = df[df["flag"] == "large"], df[df["flag"] == "thin"]
    assert (large["premium_share"] > t.large_share).all()
    assert (thin["premium_share"] >= t.material_share).all() and (thin["z_claims"] < t.z_target).all()
    # a segment cannot be both, and an immaterial segment is never called thin
    assert set(large.index) & set(thin.index) == set()
    tiny = df[df["premium_share"] < t.material_share]
    assert (tiny["flag"] == "").all()


def test_thresholds_change_the_flags(ctx_injected):
    chk = ctx_injected.make_check("segment_mix")
    strict = chk.profile(["class_cd_std"], Thresholds(z_target=0.9, material_share=0.001))
    lenient = chk.profile(["class_cd_std"], Thresholds(z_target=0.1, material_share=0.05))
    assert (strict["flag"] == "thin").sum() > (lenient["flag"] == "thin").sum()


def test_two_way_combination_and_source_split(ctx_injected):
    chk = ctx_injected.make_check("segment_mix")
    one = chk.profile(["class_cd_std"])
    two = chk.profile(["class_cd_std", "trr_cd"])
    assert len(two) > len(one)                                  # cells multiply
    assert two["segment"].str.contains(r"class_cd_std=.*\|trr_cd=").all()
    assert two["premium"].sum() == pytest.approx(one["premium"].sum())
    by_src = chk.profile(["src", "class_cd_std"])
    assert by_src["segment"].str.startswith("src=").all()


def test_findings_feed_the_tracker(ctx_injected):
    res = ctx_injected.make_check("segment_mix").run()
    f = res.findings
    assert set(f["variable"]) <= set(ctx_injected.schema.names())   # tracker joins on real columns
    metrics = set(f[f["variable"] == "class_cd_std"]["metric"])
    assert {"n_segments", "top_segment_share", "pct_premium_below_credibility",
            "n_thin_material_segments"} == metrics
    below = f[(f["variable"] == "class_cd_std") & (f["metric"] == "pct_premium_below_credibility")].iloc[0]
    assert 0 <= below["value"] <= 1
