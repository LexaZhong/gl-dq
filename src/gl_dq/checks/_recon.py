"""Shared reconciliation logic: pipeline aggregates vs a source-of-truth (SOT) query."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from gl_dq.core.results import _fmt, segment_key, sort_segments


class Tolerance(BaseModel):
    abs: float = 1000.0  # absolute difference allowed
    pct: float = 0.005  # relative difference allowed (vs SOT)


def sot_columns(check, sot_sql: str) -> list[str]:
    df = check.query("SOT columns", check.ctx.render_sql("sot_columns.sql.j2", sot_sql=sot_sql))
    return list(df.columns)


def pipeline_agg(check, dims: list[str], measures: dict[str, str], where: str | None = None, label="pipeline"):
    check.schema.validate(dims)
    return check.query(label, check.ctx.render_sql("agg_by_dims.sql.j2", dims=dims, measures=measures, where=where))


def sot_agg(check, sot_sql: str, dims: list[str], dim_map: dict[str, str], measures: dict[str, str], label="SOT"):
    return check.query(label, check.ctx.render_sql(
        "agg_sot.sql.j2", sot_sql=sot_sql, dims={d: dim_map.get(d, d) for d in dims}, measures=measures))


def reconcile(pipe: pd.DataFrame, sot: pd.DataFrame, dims: list[str], measure: str,
              tol: Tolerance) -> pd.DataFrame:
    """Outer-join on dims (compared as strings) and grade each row for one measure."""
    p = pipe[dims + [measure]].copy()
    s = sot[dims + [measure]].copy()
    for df in (p, s):
        for d in dims:
            df[d] = df[d].map(_fmt)
    p = p.groupby(dims, as_index=False)[measure].sum() if dims else p
    s = s.groupby(dims, as_index=False)[measure].sum() if dims else s
    if dims:
        m = p.merge(s, on=dims, how="outer", suffixes=("_pipeline", "_sot"), indicator=True)
    else:
        m = pd.concat([p.add_suffix("_pipeline"), s.add_suffix("_sot")], axis=1).assign(_merge="both")
    m = m.rename(columns={f"{measure}_pipeline": "pipeline", f"{measure}_sot": "sot"})
    m["pipeline"] = pd.to_numeric(m["pipeline"], errors="coerce")
    m["sot"] = pd.to_numeric(m["sot"], errors="coerce")
    m["diff"] = m["pipeline"].fillna(0) - m["sot"].fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m["pct_diff"] = np.where(m["sot"].fillna(0) != 0, m["diff"] / m["sot"].abs(),
                                 np.where(m["diff"] != 0, np.sign(m["diff"]), 0.0))
    m["presence"] = m["_merge"].map({"both": "both", "left_only": "pipeline only", "right_only": "SOT only"})
    breach = (m["diff"].abs() > tol.abs) & (m["pct_diff"].abs() > tol.pct)
    m["status"] = np.where(breach, "fail", "pass")
    m["segment"] = [segment_key({d: r[d] for d in dims}) for _, r in m.iterrows()]
    cols = dims + ["segment", "status", "presence", "pipeline", "sot", "diff", "pct_diff"]
    return m[cols].sort_values(["status", "segment"], ascending=[True, True]).reset_index(drop=True)


def recon_findings(rec: pd.DataFrame, variable: str, item: str, tol: Tolerance) -> list[dict]:
    return [dict(variable=variable, item=item, segment=r["segment"], metric="pct_diff", value=float(r["pct_diff"]),
                 threshold=tol.pct, status=r["status"],
                 detail=f"pipeline {r['pipeline']:,.0f} vs SOT {r['sot']:,.0f} ({r['presence']})"
                 .replace("nan", "missing"))
            for _, r in rec.iterrows()]


def _compact(v: float) -> str:
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.0f}"


def render_recon(st, rec: pd.DataFrame, dims: list[str], title: str, key: str):
    import plotly.express as px

    from gl_dq.ui.components import status_table
    from gl_dq.ui.theme import DIVERGING, style

    if rec.empty:
        st.info("Nothing to reconcile.")
        return
    n_fail = int((rec["status"] == "fail").sum())
    tot_p, tot_s = rec["pipeline"].sum(), rec["sot"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{title}: pipeline", _compact(tot_p), help=f"{tot_p:,.2f}")
    c2.metric("Source of truth", _compact(tot_s), help=f"{tot_s:,.2f}")
    c3.metric("Difference", _compact(tot_p - tot_s), f"{(tot_p - tot_s) / tot_s:+.2%} vs SOT" if tot_s else None,
              delta_color="off", help=f"{tot_p - tot_s:,.2f}")
    c4.metric("Segments out of tolerance", f"{n_fail} / {len(rec)}")

    # diverging heatmap of % difference: rows = all dims but the last, columns = last dim
    col = dims[-1] if dims else "segment"
    rows = dims[:-1]
    h = rec.copy()
    h["row"] = h[rows].astype(str).agg(" · ".join, axis=1) if rows else title
    h["label"] = [("no SOT" if pr == "pipeline only" else "no pipeline" if pr == "SOT only" else f"{v:+.1%}")
                  + (" ⚠" if st_ == "fail" else "")
                  for v, pr, st_ in zip(h["pct_diff"], h["presence"], h["status"])]
    z = h.pivot(index="row", columns=col, values="pct_diff")
    # pivot orders columns as text: 2018..2024 survives that, 1000000 vs 500000 does not
    z = z[sort_segments(z.columns)]
    text = h.pivot(index="row", columns=col, values="label").reindex_like(z).fillna("")
    both = h.loc[h["presence"] == "both", "pct_diff"].abs()
    lim = max(0.05, float(both.quantile(0.9)) if len(both) else 0.05)
    lim = min(lim, 0.5)
    fig = px.imshow(z, aspect="auto", color_continuous_scale=DIVERGING, zmin=-lim, zmax=lim,
                    labels={"color": "pipeline vs SOT", "x": col, "y": " · ".join(rows)})
    fig.update_traces(text=text.to_numpy(), texttemplate="%{text}", textfont=dict(size=11),
                      hovertemplate=f"{' · '.join(rows) or ''} %{{y}}<br>{col} %{{x}}<br>%{{text}}<extra></extra>")
    fig.update_coloraxes(colorbar=dict(tickformat="+.0%", title=dict(text="")))
    fig.update_xaxes(type="category", side="top")
    st.plotly_chart(style(fig, height=max(220, 44 * z.shape[0] + 110)), use_container_width=True, key=f"{key}_chart")
    st.caption(f"Color is clipped at ±{lim:.0%} so one extreme segment does not wash out the rest (the labels show "
               f"exact values). Blue means pipeline below SOT, red means above. ⚠ marks segments outside tolerance.")
    only = st.toggle("Only out-of-tolerance", value=n_fail > 0, key=f"{key}_only")
    status_table(rec[rec["status"] == "fail"] if only else rec, percent_cols=["pct_diff"],
                 number_formats={"pipeline": "%,.0f", "sot": "%,.0f", "diff": "%,.0f"})
