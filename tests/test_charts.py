"""Charts must never fall back to WebGL: it is unavailable in some locked-down browsers."""
import pandas as pd
import plotly.express as px

from gl_dq.checks.distribution import LogScale, VarSpec
from gl_dq.ui.theme import limit_series, line, series_encoding


def _big():
    return pd.DataFrame({"x": list(range(2000)), "y": range(2000), "s": ["a", "b"] * 1000})


def test_line_helper_is_svg_even_for_many_points():
    assert {t.type for t in px.line(_big(), x="x", y="y", color="s").data} == {"scattergl"}  # plotly default
    assert {t.type for t in line(_big(), x="x", y="y", color="s").data} == {"scatter"}


def test_distribution_histogram_with_many_segments_has_no_webgl(ctx_injected):
    chk = ctx_injected.make_check("distribution")
    spec = VarSpec(name="tot_wrtn_prm_amt", group_by=["src", "loc_st_abbr"],
                   log_scale=LogScale(method="signed_log"), hist_bins=60)
    hist = chk.histogram(spec)
    enc = series_encoding(hist, "segment", ctx_injected.project.sources, spec.group_by)
    assert len(hist) > 1000  # would trigger plotly's WebGL fallback
    fig = line(hist, x="center", y="share", line_shape="hvh", **enc)
    assert {t.type for t in fig.data} == {"scatter"}
    _, tables = chk.profile(spec)
    assert {t.type for t in line(tables["percentiles"], x="percentile", y="value", color="segment").data} == {"scatter"}


def _grid(dims: dict):
    import itertools
    rows = [dict(zip(dims, vals)) for vals in itertools.product(*dims.values())]
    df = pd.DataFrame(rows)
    df["segment"] = df.apply(lambda r: "|".join(f"{k}={r[k]}" for k in dims), axis=1)
    df["n"] = 1
    return df


def test_facet_split_only_when_it_encodes_every_dimension():
    two = _grid({"src": ["BOP", "BMQ", "CMQ"], "pol_yr": list(range(2018, 2025))})  # 21 segments
    enc = series_encoding(two, "segment", ["BOP", "BMQ", "CMQ"], ["src", "pol_yr"])
    assert {enc["facet_col"], enc["color"]} == {"src", "pol_yr"}  # panels + colors cover both dims

    three = _grid({"src": ["BOP", "BMQ", "CMQ"], "pol_yr": list(range(2018, 2025)),
                   "loc_st_abbr": ["CA", "TX", "NY", "FL"]})
    enc3 = series_encoding(three, "segment", ["BOP", "BMQ", "CMQ"], ["src", "pol_yr", "loc_st_abbr"])
    # must not facet by one dimension and colour by another: the third would be silently merged
    assert enc3.get("facet_col") not in ("src", "pol_yr", "loc_st_abbr")
    assert enc3.get("color") in (None, "segment")


def test_limit_series_keeps_largest():
    df = pd.DataFrame({"segment": list("aabbbcccc"), "w": [5, 5, 1, 1, 1, 2, 2, 2, 2]})
    kept, dropped = limit_series(df, "segment", max_series=2, weight="w")
    assert set(kept["segment"]) == {"a", "c"} and dropped == ["b"]


def test_colours_do_not_repaint_when_values_are_filtered_out():
    """Color follows the entity: filtering must not shift the survivors onto other hues."""
    from gl_dq.ui.theme import entity_colors

    domain = ["Liquor Liability", "Medical Payments", "Personal & Advertising Injury",
              "Premises/Operations", "Products/Completed Ops"]
    full = entity_colors(domain, domain)
    subset = entity_colors(["Premises/Operations", "Liquor Liability"], domain)
    assert subset["Premises/Operations"] == full["Premises/Operations"]
    assert subset["Liquor Liability"] == full["Liquor Liability"]
    assert len(set(full.values())) == len(domain)          # distinct hues
    # without a domain the assignment follows whatever happens to be present (the bug this guards)
    assert entity_colors(["Premises/Operations", "Liquor Liability"])["Premises/Operations"] != full["Premises/Operations"]


def test_axis_stays_chronological_when_a_series_starts_late():
    """plotly orders a category axis by first appearance, and series_encoding casts levels to text:
    a source that only starts writing in 2021 would otherwise put 2021 before 2018."""
    import plotly.express as px

    from gl_dq.ui.theme import ordered_categories, series_encoding

    view = pd.DataFrame({"src": ["BMQ"] * 4 + ["BOP"] * 7,
                         "pol_yr": [2021, 2022, 2023, 2024] + [2018, 2019, 2020, 2021, 2022, 2023, 2024],
                         "exposure": range(11)})
    enc = series_encoding(view, "src", ["BOP", "BMQ", "CMQ"], ["src", "pol_yr"])

    bad = px.bar(view, x="pol_yr", y="exposure", barmode="group", **enc)
    assert _x_order(bad) == ["2021", "2022", "2023", "2024", "2018", "2019", "2020"]  # the bug

    good = px.bar(view, x="pol_yr", y="exposure", barmode="group",
                  category_orders=ordered_categories(view, enc, "pol_yr"), **enc)
    assert good.layout.xaxis.categoryarray == ("2018", "2019", "2020", "2021", "2022", "2023", "2024")
    assert _x_order(good) == ["2021", "2022", "2023", "2024", "2018", "2019", "2020"]  # trace data unchanged


def _x_order(fig):
    return list(dict.fromkeys(str(x) for tr in fig.data for x in tr.x))


def test_exposure_charts_order_their_levels(ctx_injected):
    """The rendered exposure charts must pin the axis order, not rely on the frame."""
    import inspect

    from gl_dq.checks.exposure import Exposure

    src = inspect.getsource(Exposure.render)
    assert src.count("category_orders=orders") == 2, "both exposure charts need a pinned axis order"
