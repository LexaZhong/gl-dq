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


# ---- binning schemes -------------------------------------------------------------------
from gl_dq.core.binning import BinningLibrary, BinningSet, BinSpec, load_binnings, resolve, save_binnings  # noqa: E402


def test_bin_labels_sort_on_an_axis_and_cover_the_line():
    s = BinSpec(variable="x", name="n", method="custom", cuts=[1000, 5000, 25000])
    assert s.n_bins == 4
    assert s.labels() == ["01. < 1K", "02. 1K to 5K", "03. 5K to 25K", "04. >= 25K"]
    assert sorted(s.labels()) == s.labels(), "zero-padded so a text axis keeps bin order"
    sql = s.case_sql('"x"', lambda v: repr(v) if isinstance(v, float) else f"'{v}'")
    assert sql.startswith("CASE WHEN \"x\" IS NULL THEN '<null>'") and sql.count("WHEN") == 4


def test_cut_points_are_sorted_and_deduplicated():
    s = BinSpec(variable="x", name="n", method="custom", cuts=[5000, 1000, 5000])
    assert s.cuts == [1000.0, 5000.0]


def test_quantiles_of_a_discrete_column_collapse_to_what_is_achievable(chk):
    """Most policies sit on the same limit, so the 20th and 40th percentile are the same number.
    Keeping both would make a bin that can never be reached."""
    s = resolve(chk, BinSpec(variable="each_occ_lmt_amt", name="q", method="quantile", bins=5))
    assert s.cuts == sorted(set(s.cuts)) and len(s.cuts) < 4
    assert s.n_bins == len(s.cuts) + 1
    assert all(a < b for a, b in zip(s.labels(), s.labels()[1:]))


def test_binning_changes_the_grouping_not_the_book(chk):
    s = resolve(chk, BinSpec(variable="each_occ_lmt_amt", name="q", method="quantile", bins=5))
    binned = chk.profile(("each_occ_lmt_amt",), "loss_ratio", binning=BinningSet(specs=[s]))["oneway"]
    raw = chk.profile(("each_occ_lmt_amt",), "loss_ratio")["oneway"]
    assert set(binned["level_a"]) == set(s.labels())
    assert len(binned) < len(raw)
    for c in ("premium", "loss", "claims", "records"):  # the same book, cut differently
        assert binned[c].sum() == pytest.approx(raw[c].sum())


def test_equal_width_and_custom_bins(chk):
    eq = resolve(chk, BinSpec(variable="expo_amt", name="w", method="equal_width", bins=4))
    assert len(eq.cuts) == 3 and all(a < b for a, b in zip(eq.cuts, eq.cuts[1:]))
    gaps = np.diff([eq.cuts[0] - (eq.cuts[1] - eq.cuts[0]), *eq.cuts])
    assert np.allclose(gaps, gaps[0])  # equal width by construction
    custom = resolve(chk, BinSpec(variable="expo_amt", name="c", method="custom", cuts=[100, 1000]))
    assert custom.cuts == [100.0, 1000.0]  # custom cuts are never re-derived


def test_evaluate_scores_a_scheme_for_the_current_target(chk):
    s = resolve(chk, BinSpec(variable="each_occ_lmt_amt", name="q", method="quantile", bins=5))
    raw = chk.evaluate_binning("each_occ_lmt_amt", "loss_ratio")
    binned = chk.evaluate_binning("each_occ_lmt_amt", "loss_ratio", spec=s)
    for m in (raw, binned):
        assert m["n_bins"] > 0 and 0 <= m["credible_weight"] <= 1
        assert np.isfinite(m["signal"]) and m["signal"] >= 0
    # collapsing a column that already separates the target loses signal, and the table must say so
    assert binned["n_bins"] < raw["n_bins"] and binned["signal"] < raw["signal"]


def test_a_scheme_is_frozen_when_saved(ctx_injected, tmp_path, chk):
    """A quantile scheme re-derived against changed data is a different scheme; the cuts travel."""
    from gl_dq.core.storage import make_storage

    spec = resolve(chk, BinSpec(variable="each_occ_lmt_amt", name="quintiles", method="quantile",
                                bins=5, description="first try")).stamped("ds@test.com")
    assert spec.author == "ds@test.com" and spec.created
    store = make_storage(str(tmp_path), tmp_path)
    save_binnings(store, BinningLibrary().put(spec))
    back = load_binnings(store).get("each_occ_lmt_amt", "quintiles")
    assert back.cuts == spec.cuts and back.method == "quantile" and back.description == "first try"


def test_library_keeps_one_scheme_per_name():
    lib = BinningLibrary().put(BinSpec(variable="x", name="a", method="custom", cuts=[1]))
    lib = lib.put(BinSpec(variable="x", name="b", method="custom", cuts=[2]))
    lib = lib.put(BinSpec(variable="x", name="a", method="custom", cuts=[9]))  # replaces, not duplicates
    assert [b.name for b in lib.for_variable("x")] == ["b", "a"]
    assert lib.get("x", "a").cuts == [9.0]
    assert lib.drop("x", "a").for_variable("x") == [BinSpec(variable="x", name="b", method="custom", cuts=[2])]
    with pytest.raises(ValueError):
        BinningLibrary(binnings=[BinSpec(variable="x", name="a", method="custom", cuts=[1]),
                                 BinSpec(variable="x", name="a", method="custom", cuts=[2])])


def test_shipped_binning_library_is_empty_and_valid(ctx_injected):
    assert load_binnings(ctx_injected.config_store).binnings == []
