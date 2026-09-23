"""One aggregate over the table, grouped by whitelisted dimensions.

The building block behind the exposure summary, the loss analytics and the segment pages: a check
names its dimensions and its measures, and the warehouse does the grouping.
"""
from __future__ import annotations

import pandas as pd


def agg_by_dims(check, dims: list[str], measures: dict[str, str], where: str | None = None,
                label: str = "aggregate") -> pd.DataFrame:
    """SUM(<measure>) per dimension combination. `dims` are validated against the column whitelist."""
    check.schema.validate(dims)
    return check.query(label, check.ctx.render_sql("agg_by_dims.sql.j2", dims=dims, measures=measures, where=where))


def compact(v: float) -> str:
    """Short money for a metric card: 2.5B, 571.5M, 12.3K."""
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.0f}"
