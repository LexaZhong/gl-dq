"""Plotly chart theme and color helpers (validated reference palette, see dataviz skill)."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# categorical slots, fixed order (validated adjacent CVD dE >= 8 on light surface)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue, light -> dark
# diverging: blue (below) <- neutral gray -> red (above)
DIVERGING = [[0.0, "#104281"], [0.25, "#5598e7"], [0.5, "#f0efec"], [0.75, "#ef8a89"], [1.0, "#a3262a"]]
STATUS = {"pass": "#0ca30c", "warn": "#fab219", "serious": "#ec835a", "fail": "#d03b3b"}
NEUTRAL = "#c3c2b7"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MAX_SERIES = 8

pio.templates["gl_dq"] = go.layout.Template(layout=dict(
    colorway=CATEGORICAL,
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_2, size=12),
    title=dict(font=dict(color=INK, size=14), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="outside", zeroline=False,
               title=dict(font=dict(color=MUTED))),
    yaxis=dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zeroline=True, zerolinecolor=AXIS,
               title=dict(font=dict(color=MUTED))),
    colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]),
    legend=dict(title=dict(text=""), orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2)),
    hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
    bargap=0.15, bargroupgap=0.05,
))
pio.templates["gl_dq"].data.scatter = [go.Scatter(line=dict(width=2), marker=dict(size=8))]
pio.templates.default = "plotly_white+gl_dq"

SEQ_SCALE = [[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]


def entity_colors(values, known: list[str] | None = None) -> dict:
    """Stable color per entity: known entities (e.g. sources) keep their slot regardless of filtering.

    Segment labels like "src=BOP" match the known entity "BOP".
    """
    from gl_dq.core.results import segment_sort_key

    known = list(known or [])
    out, extra = {}, len(known)
    for v in sorted(set(map(str, values)), key=segment_sort_key):
        key = v.split("=", 1)[-1] if "|" not in v else v
        if key in known:
            out[v] = CATEGORICAL[known.index(key) % len(CATEGORICAL)]
        elif extra < MAX_SERIES:
            out[v] = CATEGORICAL[extra]
            extra += 1
    return out


def series_encoding(df, col: str, known: list[str] | None = None, facet_candidates: list[str] | None = None) -> dict:
    """plotly-express kwargs for a series column: color when <= 8 series, else small multiples.

    A facet split is only used when the panels plus the colors encode EVERY dimension (exactly two
    candidate dimensions); otherwise a dimension would be silently collapsed and unrelated rows
    would be drawn as one series. With more dimensions than that, the caller must reduce the
    series first (see `limit_series` or the segment picker on the distribution page).
    """
    for c in [col] + list(facet_candidates or []):  # numeric ids (e.g. pol_yr) are categories, not a color ramp
        if c in df and df[c].dtype.kind in "iuf":
            df[c] = df[c].astype("Int64").astype(str) if df[c].dtype.kind in "iu" else df[c].astype(str)
    n = df[col].nunique() if col in df else 0
    if n <= MAX_SERIES:
        return dict(color=col, color_discrete_map=entity_colors(df[col].astype(str), known))
    cands = list(facet_candidates or [])
    if len(cands) == 2 and all(c in df for c in cands):
        for f, color in (cands, cands[::-1]):
            if df[color].nunique() <= MAX_SERIES and df[f].nunique() <= 12:
                return dict(facet_col=f, facet_col_wrap=3, color=color,
                            color_discrete_map=entity_colors(df[color].astype(str), known))
    return dict(facet_col=col, facet_col_wrap=4)


def ordered_categories(df, enc: dict, *extra: str) -> dict:
    """`category_orders` for the columns an encoding uses, in natural order.

    Sorting the dataframe is not enough: plotly express orders discrete colours and facets by
    first appearance, and `series_encoding` casts those columns to strings.
    """
    from gl_dq.core.results import sort_segments

    cols = [enc.get("color"), enc.get("facet_col"), *extra]
    return {c: sort_segments(df[c]) for c in cols if c and c in df}


def limit_series(df, col: str, max_series: int = MAX_SERIES, weight: str | None = None):
    """Keep the biggest `max_series` values of `col`; returns (df, dropped names)."""
    if col not in df or df[col].nunique() <= max_series:
        return df, []
    size = df.groupby(col)[weight].sum() if weight else df[col].value_counts()
    keep = list(size.sort_values(ascending=False).head(max_series).index)
    return df[df[col].isin(keep)], [v for v in df[col].unique() if v not in keep]


def line(df, **kwargs):
    """px.line that always renders as SVG.

    Plotly switches to WebGL (scattergl) above ~1000 points, which fails outright in browsers
    where WebGL is unavailable or blocked ("WebGL is not supported by your browser"). Our charts
    are small enough that SVG is fine, and it also keeps them printable.
    """
    kwargs.setdefault("render_mode", "svg")
    return px.line(df, **kwargs)


def style(fig, height: int = 360, title: str | None = None):
    names = [getattr(t, "name", "") for t in fig.data if getattr(t, "showlegend", None) is not False and getattr(t, "name", "")]
    has_legend = bool(names)
    has_facets = len(fig.layout.annotations) > 0
    # a legend above the plot collides with panel titles, and wraps over the chart title when the
    # entries are many or long (e.g. full segment labels) - put those below the plot instead
    legend_below = has_legend and (has_facets or len(set(names)) > 4 or max(len(n) for n in names) > 24)
    top = (34 if title else 8) + (30 if has_legend and not legend_below else 0) + (22 if has_facets else 0)
    bottom = (80 if not has_facets else 90) if legend_below else 0
    fig.update_layout(height=height + top + bottom, margin=dict(l=0, r=0, t=top, b=bottom), legend_title_text="")
    if legend_below:
        fig.update_layout(legend=dict(y=-0.22 if not has_facets else -0.14, yanchor="top"))
    if title:
        fig.update_layout(title=dict(text=title, y=1.0, yanchor="top", yref="container", pad=dict(t=6)))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(color=INK_2)))
    return fig
