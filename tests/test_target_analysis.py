"""Target analysis: one-way relativities, and the interaction measure against a main-effects model."""
import numpy as np
import pandas as pd
import pytest

from gl_dq.checks.target_analysis import OTHER, TARGETS, interaction_strength


@pytest.fixture(scope="module")
def chk(ctx_injected):
    return ctx_injected.make_check("target_analysis")


VARS = ("src", "covg_type_desc", "expn_bs_std")


def test_every_target_knows_its_weight_and_whether_it_needs_a_base():
    assert TARGETS["loss_ratio"].denominator == "premium" and not TARGETS["loss_ratio"].needs_base
    assert TARGETS["severity"].denominator == "claims" and not TARGETS["severity"].needs_base
    # exposure is not additive across bases, so these two must be scoped to one
    assert TARGETS["frequency"].needs_base and TARGETS["loss_cost"].needs_base


def test_one_query_returns_the_total_every_one_way_and_every_pair(ctx_injected):
    chk = ctx_injected.make_check("target_analysis")  # fresh: _sql accumulates per instance
    t = chk.profile(VARS, "loss_ratio")
    assert len(chk._sql) == 1, "the grouping sets must come back in a single pass"
    assert set(t["oneway"]["var_a"]) == set(VARS)
    assert set(t["strength"]["pair"]) == {"src × covg_type_desc", "src × expn_bs_std",
                                          "covg_type_desc × expn_bs_std"}


def test_one_way_totals_reconcile_with_the_portfolio(chk, ctx_injected):
    t = chk.profile(VARS, "loss_ratio")
    total = t["total"].iloc[0]
    raw = ctx_injected.db.query(
        "SELECT SUM(tot_wrtn_prm_amt) p, SUM(allocation) l, SUM(claim_cnt) c FROM gl_master_synth").iloc[0]
    assert float(total["premium"]) == pytest.approx(float(raw["p"]))
    assert float(total["value"]) == pytest.approx(float(raw["l"]) / float(raw["p"]))
    for v in VARS:  # each one-way is a partition of the same book
        part = t["oneway"][t["oneway"]["var_a"] == v]
        assert part["premium"].sum() == pytest.approx(float(raw["p"]))
        assert part["claims"].sum() == pytest.approx(float(raw["c"]))


def test_relativity_is_the_level_against_the_portfolio(chk):
    t = chk.profile(("src",), "loss_ratio")
    overall = float(t["total"]["value"].iloc[0])
    part = t["oneway"]
    np.testing.assert_allclose(part["relativity"], part["value"] / overall)
    # weighted by premium, the relativities average back to 1
    assert np.average(part["relativity"], weights=part["weight"]) == pytest.approx(1.0)


def test_each_target_uses_its_own_numerator_and_denominator(chk):
    for name, spec in TARGETS.items():
        base = "Sales" if spec.needs_base else None
        row = chk.profile(("src",), name, base=base)["oneway"].iloc[0]
        assert row["value"] == pytest.approx(row[spec.numerator] / row[spec.denominator])
        assert row["weight"] == pytest.approx(row[spec.denominator])


def test_exposure_targets_are_scoped_to_one_base(ctx_injected):
    chk = ctx_injected.make_check("target_analysis")
    t = chk.profile(("src",), "frequency", base="Sales")
    sql = next(iter(chk._sql.values()))
    assert "expn_bs_std" in sql and "'Sales'" in sql
    scoped = ctx_injected.db.query(
        "SELECT SUM(claim_cnt) c, SUM(expo_amt) e FROM gl_master_synth WHERE expn_bs_std = 'Sales'").iloc[0]
    assert float(t["total"]["value"].iloc[0]) == pytest.approx(float(scoped["c"]) / float(scoped["e"]))


def test_levels_beyond_the_cap_fold_into_other(chk):
    t = chk.profile(("covg_type_desc",), "loss_ratio", max_levels=2)
    part = t["oneway"]
    assert OTHER in set(part["level_a"]) and len(part) == 3
    full = chk.profile(("covg_type_desc",), "loss_ratio", max_levels=99)["oneway"]
    # folding is summing, so the book is unchanged
    assert part["premium"].sum() == pytest.approx(full["premium"].sum())
    assert part["claims"].sum() == pytest.approx(full["claims"].sum())


def test_lift_is_actual_over_the_main_effects_prediction(chk):
    t = chk.profile(("src", "covg_type_desc"), "loss_ratio")
    overall = float(t["total"]["value"].iloc[0])
    rel = t["oneway"].set_index(["var_a", "level_a"])["relativity"]
    r = t["pairs"].iloc[0]
    expected = overall * rel[("src", r["level_a"])] * rel[("covg_type_desc", r["level_b"])]
    assert r["expected"] == pytest.approx(expected)
    assert r["lift"] == pytest.approx(r["value"] / expected)


def test_no_interaction_scores_zero():
    """A perfectly multiplicative table has nothing a main-effects model would miss."""
    cells = pd.DataFrame({"lift": [1.0, 1.0, 1.0, 1.0], "weight": [10.0, 20.0, 30.0, 40.0]})
    assert interaction_strength(cells, "weight") == pytest.approx(0.0)
    crossed = pd.DataFrame({"lift": [1.5, 1 / 1.5], "weight": [1.0, 1.0]})
    assert interaction_strength(crossed, "weight") == pytest.approx(np.log(1.5))
    assert np.isnan(interaction_strength(pd.DataFrame({"lift": [np.nan], "weight": [1.0]}), "weight"))


def test_strength_ranks_a_real_interaction_above_a_flat_one(chk):
    t = chk.profile(VARS, "loss_ratio")
    s = t["strength"]
    assert s["strength"].is_monotonic_decreasing                  # ranked, strongest first
    assert (s["n_credible"] <= s["n_cells"]).all()
    assert s["strength"].between(0, 5).all()


def test_findings_name_the_variable_and_the_pair(chk):
    t = chk.profile(VARS, "loss_ratio")
    f = pd.DataFrame(chk.findings_for("loss_ratio", t))
    assert set(f[f["metric"] == "target_spread"]["variable"]) == set(VARS)
    inter = f[f["metric"] == "interaction_strength"]
    assert len(inter) == 3 and inter["item"].str.contains("×").all()
    assert f["status"].isin(["pass", "warn", "fail", "info"]).all()


def test_run_uses_the_configured_target_and_notes_the_base(ctx_injected):
    chk = ctx_injected.make_check("target_analysis")
    cfg = chk.cfg.model_copy(update={"default_target": "frequency"})
    chk = type(chk)(ctx_injected, cfg)
    res = chk.run()
    assert any("exposure base" in m for m in res.messages)
    assert not res.findings.empty and "oneway::frequency" in res.tables
