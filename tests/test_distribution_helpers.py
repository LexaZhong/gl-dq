import numpy as np
import pytest

from gl_dq.checks.distribution import LogScale, VarSpec, inverse_transform, percentile_probs, psi


def test_percentile_probs_int_and_custom():
    p = percentile_probs(20)
    assert p[0] == 0 and p[-1] == 1 and len(p) == 22 and 0.99 in p  # 21-point grid + p99
    p10 = percentile_probs(10)
    assert 0.99 in p10 and len(p10) == 12
    assert percentile_probs([0.9, 0.5, 0.5]) == [0.0, 0.5, 0.9, 0.99, 1.0]


def test_invalid_bins():
    with pytest.raises(ValueError):
        VarSpec(name="x", percentile_bins=1)
    with pytest.raises(ValueError):
        VarSpec(name="x", percentile_bins=[1.5])


@pytest.mark.parametrize("method,values", [("log1p", [0.0, 5.0, 1e6]), ("signed_log", [-1e5, -1.0, 0.0, 3.0]),
                                           ("log10", [0.1, 10.0]), ("none", [-3.0, 3.0])])
def test_inverse_transform_roundtrip(method, values):
    x = np.array(values)
    fwd = {"log1p": np.log1p, "log10": np.log10, "signed_log": lambda v: np.sign(v) * np.log1p(np.abs(v)),
           "none": lambda v: v}[method](x)
    np.testing.assert_allclose(inverse_transform(fwd, method), x, rtol=1e-9, atol=1e-9)


def test_log_histogram_handles_zero_and_negative(ctx_injected):
    chk = ctx_injected.make_check("distribution")
    spec = VarSpec(name="expo_amt", group_by=["src"], log_scale={"method": "log10"}, hist_bins=20)
    h = chk.histogram(spec)
    assert h.attrs["excluded"] > 0  # zeros and negatives are outside log10's domain
    assert h["bucket"].between(0, 19).all()
    h2 = chk.histogram(spec.model_copy(update={"log_scale": LogScale(method="signed_log")}))
    assert h2.attrs["excluded"] == 0


def test_psi():
    assert psi(np.array([10, 10]), np.array([10, 10])) == pytest.approx(0)
    assert psi(np.array([90, 10]), np.array([10, 90])) > 1


def test_segments_with_numeric_parts_sort_numerically():
    """'pol_yr=10' must not land before 'pol_yr=9' in tables or on an axis."""
    from gl_dq.core.results import segment_sort_key, sort_segments

    segs = ["src=BOP|pol_yr=2019", "src=BOP|pol_yr=9", "src=BOP|pol_yr=10", "src=BMQ|pol_yr=2019",
            "src=BOP|pol_yr=<null>", "ALL"]
    assert sort_segments(segs) == ["ALL", "src=BMQ|pol_yr=2019", "src=BOP|pol_yr=9", "src=BOP|pol_yr=10",
                                   "src=BOP|pol_yr=2019", "src=BOP|pol_yr=<null>"]
    # the same key orders plain chart categories (colour / facet values)
    assert sorted(["2019", "9", "10", "BOP"], key=segment_sort_key) == ["9", "10", "2019", "BOP"]


def test_numeric_level_is_ordered_in_tables_and_charts(ctx_injected):
    from gl_dq.checks.distribution import VarSpec
    from gl_dq.core.results import sort_segments
    from gl_dq.ui.theme import ordered_categories, series_encoding

    chk = ctx_injected.make_check("distribution")
    spec = VarSpec(name="tot_wrtn_prm_amt", group_by=["pol_yr"], percentile_bins=4)
    _, tables = chk.profile(spec)
    stats = tables["stats"]
    years = [int(s.split("=")[1]) for s in stats.sort_values("segment", key=lambda c: c.map(
        __import__("gl_dq.core.results", fromlist=["segment_sort_key"]).segment_sort_key))["segment"]]
    assert years == sorted(years)

    hist = chk.histogram(spec)
    enc = series_encoding(hist, "segment", ctx_injected.project.sources, spec.group_by)
    orders = ordered_categories(hist, enc)
    assert orders, "charts must pin an explicit category order"
    for col, order in orders.items():
        assert order == sort_segments(hist[col])


def test_all_null_segment_does_not_crash(ctx_injected):
    """A segment with no values at all must be reported, not raise 'NAType' object is not iterable."""
    import pandas as pd

    from gl_dq.checks.distribution import VarSpec

    chk = ctx_injected.make_check("distribution")
    # rsk_itm_id is populated for BMQ only, so BOP and CMQ are entirely null at this level
    spec = VarSpec(name="rsk_itm_id", type="numeric", group_by=["src"], percentile_bins=4)

    # precondition: this is exactly the shape that used to break (percentiles come back as pd.NA)
    raw = ctx_injected.db.query(ctx_injected.render_sql(
        "dist_numeric_stats.sql.j2", x=ctx_injected.schema.ref("rsk_itm_id"), group_by=["src"],
        probs=[0.0, 0.5, 0.99, 1.0]))
    assert any(v is pd.NA for v in raw["pcts"]), "fixture no longer produces an all-null segment"

    findings, tables = chk.profile(spec)          # must not raise
    stats = tables["stats"].set_index("segment")
    assert stats.loc["src=BOP", "n_nonnull"] == 0
    assert pd.isna(stats.loc["src=BOP", "max_to_p99"]) and pd.isna(stats.loc["src=BOP", "p99"])
    assert stats.loc["src=BMQ", "n_nonnull"] > 0
    empty_findings = [f for f in findings if f["segment"] == "src=BOP" and f["metric"] == "n_nonnull"]
    assert empty_findings and empty_findings[0]["status"] == "info"
    assert chk.histogram(spec) is not None        # the chart path is NA-safe too


def test_empty_segment_reports_info_not_statistics(ctx_injected):
    import numpy as np
    import pandas as pd

    from gl_dq.checks.distribution import Distribution, VarSpec, _as_list, _isna

    assert _as_list(pd.NA, 3) == [None, None, None]     # the exact value that raised
    assert _as_list(np.array([1.0, 2.0]), 2) == [1.0, 2.0]
    assert _isna(pd.NA) and _isna(float("nan")) and _isna(None) and not _isna(0.0)

    chk = ctx_injected.make_check("distribution")
    # a variable that is null for a whole source: csl_ded_amt is null wherever bi/pd are set
    findings, tables = chk.profile(VarSpec(name="csl_ded_amt", group_by=["src", "tx_type_nm"], percentile_bins=4))
    empty = tables["stats"][tables["stats"]["n_nonnull"] == 0]
    if not empty.empty:
        assert empty["max_to_p99"].isna().all()
        info = [f for f in findings if f["metric"] == "n_nonnull"]
        assert info and all(f["status"] == "info" for f in info)
