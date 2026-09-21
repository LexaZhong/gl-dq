"""Which rating segments drive the book, and which are too thin to be credible.

Two questions an actuary asks before pricing:
  - where is the premium concentrated? (a Pareto over the ISO rating dimensions)
  - which segments carry real premium but not enough claims to support a rate?

Exposure is deliberately never summed across segments: expo_amt is in different units per exposure
base (sales, payroll, area, units, admissions), so premium, record and claim shares are what compare.
Credibility uses the square-root rule Z = min(1, sqrt(n / full_credibility)), with the classic 1082
claim standard as the default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from gl_dq.checks import _segment_detail as _detail
from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import grade, segment_key

OTHER = "<other>"


def credibility(n, full_credibility: float = 1082.0) -> float:
    """Square-root (limited fluctuation) credibility: Z = min(1, sqrt(n / N_full))."""
    n = np.asarray(n, dtype=float)
    return np.minimum(1.0, np.sqrt(np.maximum(n, 0.0) / float(full_credibility)))


class Thresholds(BaseModel):
    large_share: float = 0.05  # premium share above which a segment drives the book
    material_share: float = 0.005  # premium share worth worrying about at all
    z_target: float = 0.5  # credibility a material segment should reach
    thin_premium_warn: float = 0.20  # share of premium sitting below z_target
    thin_premium_fail: float = 0.40


@register_check("segment_mix")
class SegmentMix(Check):
    title = "Segment mix & credibility"
    icon = "🧩"
    description = ("Where the premium is concentrated across the ISO rating dimensions, and which segments "
                   "carry material premium on too few claims to be credible (Z = min(1, √(n/1082))).")
    default_order = 80

    class Config(CheckConfig):
        dimensions: list[str] = []
        default_dimension: str | None = None
        split_by_source: bool = False
        full_credibility: float = 1082.0
        top_n: int = 25  # rows charted; a 2-way combination folds the rest into <other>
        thresholds: Thresholds = Thresholds()

    # ---- compute -----------------------------------------------------------------
    def profile(self, dims: list[str], thresholds: Thresholds | None = None,
                where: str | None = None) -> pd.DataFrame:
        """One row per segment with shares, cumulative premium share, credibility and flags."""
        t = thresholds or self.cfg.thresholds
        m = self.project.measures
        dims = self.schema.validate([d for d in dims if d])
        if not dims:
            return pd.DataFrame()
        df = self.query(f"segments: {', '.join(dims)}", self.ctx.render_sql(
            "segment_mix.sql.j2", dims=dims, premium=self.schema.ref(m.written_premium),
            claims=self.schema.ref(m.claim_count), where=where))
        df["records"] = df["records"].astype("int64")
        df["claims"] = df["claims"].fillna(0).astype("int64")
        df["premium"] = df["premium"].astype(float).fillna(0.0)
        df["segment"] = [segment_key({d: r[d] for d in dims}) for _, r in df.iterrows()]

        totals = {"premium": df["premium"].sum(), "records": df["records"].sum(), "claims": df["claims"].sum()}
        for col in ("premium", "records", "claims"):
            df[f"{col}_share"] = df[col] / totals[col] if totals[col] else np.nan
        df = df.sort_values("premium", ascending=False).reset_index(drop=True)
        df["cum_premium_share"] = df["premium_share"].cumsum()
        df["z_claims"] = credibility(df["claims"], self.cfg.full_credibility)
        df["z_records"] = credibility(df["records"], self.cfg.full_credibility)
        df["flag"] = np.where(df["premium_share"] > t.large_share, "large",
                              np.where((df["premium_share"] >= t.material_share) & (df["z_claims"] < t.z_target),
                                       "thin", ""))
        return df

    def findings_for(self, dims: list[str], df: pd.DataFrame, thresholds: Thresholds | None = None) -> list[dict]:
        t = thresholds or self.cfg.thresholds
        if df.empty:
            return []
        variable, item = dims[0], " x ".join(dims)
        thin = df[df["flag"] == "thin"]
        below = df[df["z_claims"] < t.z_target]["premium_share"].sum()
        top = df.iloc[0]
        hhi = float((df["premium_share"] ** 2).sum())
        n_for_half = int((df["cum_premium_share"] < 0.5).sum() + 1)
        return [
            dict(variable=variable, item=item, segment="ALL", metric="n_segments", value=len(df),
                 threshold=None, status="info",
                 detail=f"{n_for_half} segment(s) make up half the premium · HHI {hhi:.3f}"),
            dict(variable=variable, item=item, segment=str(top["segment"]), metric="top_segment_share",
                 value=float(top["premium_share"]), threshold=t.large_share,
                 status=grade(float(top["premium_share"]), t.large_share, None),
                 detail=f"largest segment: {top['premium']:,.0f} premium, {int(top['claims']):,} claims, "
                        f"Z={top['z_claims']:.2f}"),
            dict(variable=variable, item=item, segment="ALL", metric="pct_premium_below_credibility",
                 value=float(below), threshold=t.thin_premium_fail,
                 status=grade(float(below), t.thin_premium_warn, t.thin_premium_fail),
                 detail=f"{below:.1%} of premium sits in segments with Z < {t.z_target:.2f}"),
            dict(variable=variable, item=item, segment="ALL", metric="n_thin_material_segments",
                 value=len(thin), threshold=0, status="warn" if len(thin) else "pass",
                 detail=(", ".join(str(s) for s in thin["segment"].head(8)) + ("…" if len(thin) > 8 else ""))
                 or "every material segment reaches the credibility target"),
        ]

    # ---- deep dive (UI only; thin wrappers so ui.state can cache them per check) --------
    def detail_stats(self, where: str, base: str | None, per: float) -> pd.DataFrame:
        return _detail.policy_stats(self, where, base, per)

    def detail_hist(self, where: str, base: str | None, per: float, metrics: tuple[str, ...],
                    log_method: str, bins: int) -> pd.DataFrame:
        # `log_method`, not `method`: ui.state.cached_method takes the method name as `method`
        return _detail.policy_hist(self, where, base, per, metrics, log_method, bins)

    def detail_trend(self, where: str, base: str | None, per: float) -> pd.DataFrame:
        return _detail.policy_trend(self, where, base, per)

    def run(self) -> CheckResult:
        findings, tables = [], {}
        for dim in self.cfg.dimensions:
            dims = [self.project.src_col, dim] if self.cfg.split_by_source else [dim]
            df = self.profile(dims)
            findings += self.findings_for(dims, df)
            tables[f"mix::{dim}"] = df
        if not self.cfg.dimensions:
            self.note("No rating dimensions configured yet — add them under ⚙️ Settings.")
        return self.result(findings, tables)

    # ---- UI --------------------------------------------------------------------
    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        cols = self.schema.names(include_derived=False)
        new.dimensions = st.multiselect("Rating dimensions", cols,
                                        default=[d for d in cfg.dimensions if d in cols], key="sm_dims")
        c1, c2 = st.columns(2)
        new.full_credibility = c1.number_input("Full credibility (claims)", 1.0, value=float(cfg.full_credibility),
                                               step=100.0, key="sm_fc",
                                               help="1082 claims is the classic ±5% at 90% standard")
        new.top_n = int(c2.number_input("Rows charted", 5, 200, cfg.top_n, key="sm_topn"))
        new.split_by_source = st.checkbox("Split every dimension by source", cfg.split_by_source, key="sm_split")
        t1, t2, t3 = st.columns(3)
        new.thresholds = cfg.thresholds.model_copy(update={
            "large_share": t1.number_input("Large segment: premium share >", 0.0, 1.0,
                                           cfg.thresholds.large_share, 0.01, key="sm_large"),
            "material_share": t2.number_input("Material: premium share ≥", 0.0, 1.0,
                                              cfg.thresholds.material_share, 0.001, format="%.3f", key="sm_mat"),
            "z_target": t3.number_input("Credibility target Z", 0.0, 1.0, cfg.thresholds.z_target, 0.05,
                                        key="sm_z")})
        return new

    def render(self, result: CheckResult):
        import streamlit as st

        cfg = self.cfg
        if not cfg.dimensions:
            st.info("No rating dimensions configured. Add them under ⚙️ Settings.")
            return
        tab_mix, tab_detail = st.tabs(["Mix & credibility", "🔬 Segment deep dive"])
        with tab_mix:
            df, dims, t = self._render_mix(st, result, cfg)
        with tab_detail:
            _detail.render_detail(self, st, df, dims, t)

    def _render_mix(self, st, result, cfg):
        """The Pareto tab. Returns (profile frame, dimensions, thresholds) for the deep-dive tab."""
        import plotly.express as px

        from gl_dq.ui import state
        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import CATEGORICAL, STATUS, style

        # one control for slicing: any number of dimensions, source included
        opts = list(dict.fromkeys([d for d in cfg.dimensions if self.schema.has(d)]
                                  + [c for c in self.segment_options() if c not in cfg.dimensions]))
        default = [cfg.default_dimension if cfg.default_dimension in opts else opts[0]] if opts else []
        dims = st.multiselect("Slice by", opts, default=default, max_selections=3, key="sm_dims_pick",
                              help="Combine up to three dimensions. Every extra dimension multiplies the number of "
                                   "cells, and thins the claims in each one.")
        if not dims:
            st.info("Pick at least one dimension to slice by.")
            return pd.DataFrame(), [], cfg.thresholds
        s1, s2, s3 = st.columns(3)
        t = cfg.thresholds.model_copy(update={
            "large_share": s1.slider("Large: premium share >", 0.0, 0.5, cfg.thresholds.large_share, 0.005),
            "material_share": s2.slider("Material: premium share ≥", 0.0, 0.05, cfg.thresholds.material_share, 0.001,
                                        format="%.3f"),
            "z_target": s3.slider("Credibility target Z", 0.0, 1.0, cfg.thresholds.z_target, 0.05)})

        df = state.cached_method(self.name, cfg, "profile", dims=tuple(dims), thresholds=t)
        if df.empty:
            st.info("No rows.")
            return df, dims, t

        thin = df[df["flag"] == "thin"]
        large = df[df["flag"] == "large"]
        below = df[df["z_claims"] < t.z_target]["premium_share"].sum()
        k = st.columns(4)
        k[0].metric("Segments", f"{len(df):,}")
        k[1].metric("Driving the book", len(large), help=f"premium share above {t.large_share:.1%}")
        k[2].metric("Thin but material", len(thin), help=f"premium ≥ {t.material_share:.1%} with Z < {t.z_target}")
        k[3].metric("Premium below target Z", f"{below:.1%}")
        n_half = int((df["cum_premium_share"] < 0.5).sum() + 1)
        st.caption(f"**{n_half}** of {len(df):,} segments make up half the premium · "
                   f"credibility Z = min(1, √(n / {cfg.full_credibility:,.0f} claims))")

        top = df.head(cfg.top_n).copy()
        top["band"] = np.where(top["z_claims"] >= t.z_target, f"Z ≥ {t.z_target:.2f}", f"Z < {t.z_target:.2f}")
        # the dimensions are named by the picker and the axis title, so the ticks carry values only
        top["label"] = [_detail._short(s) for s in top["segment"]]
        fig = px.bar(top, x="label", y="premium_share", color="band",
                     color_discrete_map={f"Z ≥ {t.z_target:.2f}": CATEGORICAL[0], f"Z < {t.z_target:.2f}": STATUS["warn"]},
                     hover_data={"premium": ":,.0f", "records": ":,", "claims": ":,", "z_claims": ":.2f",
                                 "cum_premium_share": ":.1%"},
                     labels={"premium_share": "share of premium", "label": " × ".join(dims)},
                     # without this the axis is ordered trace by trace, i.e. grouped by credibility
                     # band, and the Pareto no longer descends
                     category_orders={"label": list(top["label"])})
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(tickangle=-40, type="category")
        sel = st.plotly_chart(style(fig, 380, f"Top {len(top)} segments by premium ({' × '.join(dims)})"),
                              use_container_width=True, on_select="rerun", key="sm_pareto")
        to_segment = dict(zip(top["label"], top["segment"]))
        picked = [to_segment.get(str(p.get("x"))) for p in (sel or {}).get("selection", {}).get("points", [])]
        if any(picked):
            st.session_state["sm_focus"] = str(next(p for p in picked if p))

        st.caption("Click a bar, or select a row below, to open that segment in the 🔬 deep-dive tab.")
        if len(thin):
            st.markdown(f"**Thin but material** — premium share ≥ {t.material_share:.1%} with Z < {t.z_target:.2f}")
            _table(st, thin.sort_values("premium", ascending=False), dims, key="sm_thin")
        else:
            st.success("Every material segment reaches the credibility target.")
        st.markdown("**All segments**")
        _table(st, df, dims, key="sm_all")
        st.download_button("⬇️ CSV", df.drop(columns=["segment"]).to_csv(index=False),
                           file_name=f"segment_mix_{'_'.join(dims)}.csv", mime="text/csv", key="sm_dl")
        status_table(result.findings[result.findings["variable"] == dims[0]][
            ["item", "segment", "metric", "value", "status", "detail"]], key="sm_find")
        return df, dims, t


def _table(st, df: pd.DataFrame, dims: list[str], key: str):
    """The segment table, one column per rating dimension. Selecting a row focuses the deep dive."""
    cols = [c for c in list(dims) + ["premium", "premium_share", "cum_premium_share", "records", "records_share",
                                     "claims", "claims_share", "z_claims", "z_records", "flag"] if c in df]
    sel = st.dataframe(df[cols], hide_index=True, use_container_width=True, key=key,
                       on_select="rerun", selection_mode="single-row", column_config={
        "premium": st.column_config.NumberColumn(format="localized"),
        "records": st.column_config.NumberColumn(format="localized"),
        "claims": st.column_config.NumberColumn(format="localized"),
        "premium_share": st.column_config.NumberColumn(format="percent"),
        "records_share": st.column_config.NumberColumn(format="percent"),
        "claims_share": st.column_config.NumberColumn(format="percent"),
        "cum_premium_share": st.column_config.NumberColumn(format="percent"),
        "z_claims": st.column_config.ProgressColumn("Z (claims)", min_value=0.0, max_value=1.0, format="%.2f"),
        "z_records": st.column_config.NumberColumn("Z (records)", format="%.2f")})
    rows = (sel or {}).get("selection", {}).get("rows", [])
    if rows:
        st.session_state["sm_focus"] = str(df.iloc[rows[0]]["segment"])
