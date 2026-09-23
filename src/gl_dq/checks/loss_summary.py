"""Loss summary: severity, frequency and loss ratio by segment, and the policy-level LR spread.

Frequency is always split by exposure base: `expo_amt` is in different units per base, so claims
per exposure unit cannot be added across them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from gl_dq.checks._agg import agg_by_dims
from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import grade, segment_key


class Analytics(BaseModel):
    segments: list[str] = ["src"]
    lr_basis: str = "pol_yr"  # written premium is policy-year based, so LR defaults to policy year
    frequency_per: float = 1000.0  # claims per N exposure units
    lr_cap: float = 5.0
    lr_bins: int = 50


@register_check("loss_summary")
class LossSummary(Check):
    title = "Loss summary"
    icon = "📉"
    description = ("Loss ratio, severity and claim frequency by segment, plus the spread of policy-level loss "
                   "ratios. Frequency is split by exposure base, which cannot be added together.")
    default_order = 70

    class Config(CheckConfig):
        where: str | None = None  # restrict the population this page describes
        analytics: Analytics = Analytics()
        loss_ratio_warn: float = 0.8  # flagged on the tracker so a segment's LR is reviewable
        loss_ratio_fail: float = 1.0

    @property
    def m(self):
        return self.project.measures

    # ---- compute -------------------------------------------------------------------
    def analytics(self, a: Analytics) -> dict[str, pd.DataFrame]:
        s, m = self.schema, self.m
        segs = s.validate(a.segments)
        out = {}
        lr_dims = list(dict.fromkeys(segs + [a.lr_basis]))
        df = agg_by_dims(self, lr_dims, {"premium": s.ref(m.written_premium), "loss": s.ref(m.loss),
                                         "claims": s.ref(m.claim_count)}, self.cfg.where,
                         label="loss ratio / severity")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["loss_ratio"] = np.where(df["premium"] > 0, df["loss"] / df["premium"], np.nan)
            df["severity"] = np.where(df["claims"] > 0, df["loss"] / df["claims"], np.nan)
        out["lr_severity"] = df.sort_values(lr_dims)

        f_dims = list(dict.fromkeys(segs + [m.exposure_base]))
        fq = agg_by_dims(self, f_dims, {"claims": s.ref(m.claim_count), "exposure": s.ref(m.exposure)},
                         self.cfg.where, label="frequency")
        with np.errstate(divide="ignore", invalid="ignore"):
            fq["frequency"] = np.where(fq["exposure"] > 0, fq["claims"] / fq["exposure"] * a.frequency_per, np.nan)
        out["frequency"] = fq.sort_values(f_dims)

        keys = list(dict.fromkeys(self.project.policy_key + segs))
        hist = self.query("policy loss ratio distribution", self.ctx.render_sql(
            "policy_lr_hist.sql.j2", keys=keys, segments=segs, premium=s.ref(m.written_premium), loss=s.ref(m.loss),
            cap=float(a.lr_cap), bins=int(a.lr_bins)))
        hist["segment"] = [segment_key({g: r[g] for g in segs}) for _, r in hist.iterrows()]
        width = a.lr_cap / a.lr_bins
        hist["lr_from"] = np.where(hist["bucket"] >= 0, hist["bucket"] * width, np.nan)
        hist["label"] = np.select([hist["bucket"] == -2, hist["bucket"] == -1],
                                  ["premium ≤ 0", "zero loss"], default="")
        out["policy_lr_hist"] = hist
        return out

    def findings_for(self, tables: dict[str, pd.DataFrame], a: Analytics) -> list[dict]:
        """One finding per segment x year, so a loss ratio out of range reaches the tracker."""
        cfg, rows = self.cfg, []
        lr = tables["lr_severity"]
        dims = [c for c in lr.columns if c not in ("premium", "loss", "claims", "loss_ratio", "severity")]
        for _, r in lr.iterrows():
            if not np.isfinite(r["loss_ratio"] or np.nan):
                continue
            value = float(r["loss_ratio"])
            rows.append(dict(variable=self.m.loss, item="loss ratio", segment=segment_key({d: r[d] for d in dims}),
                             metric="loss_ratio", value=value, threshold=cfg.loss_ratio_fail,
                             status=grade(value, cfg.loss_ratio_warn, cfg.loss_ratio_fail),
                             detail=f"loss {r['loss']:,.0f} on premium {r['premium']:,.0f}"
                                    + (f", {int(r['claims']):,} claims" if pd.notna(r["claims"]) else "")))
        return rows

    def run(self) -> CheckResult:
        tables = self.analytics(self.cfg.analytics)
        return self.result(self.findings_for(tables, self.cfg.analytics), tables)

    # ---- UI ---------------------------------------------------------------------------
    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        opts = self.segment_options()
        a = cfg.analytics
        c1, c2, c3 = st.columns([3, 1, 1])
        segments = c1.multiselect("Default segments", opts, default=[s for s in a.segments if s in opts], key="ls_segs")
        per = c2.number_input("Frequency per N exposure", 1.0, value=float(a.frequency_per), step=1000.0, key="ls_per")
        cap = c3.number_input("Loss ratio cap", 0.5, 50.0, a.lr_cap, 0.5, key="ls_cap")
        new.analytics = a.model_copy(update={"segments": segments, "frequency_per": per, "lr_cap": cap})
        w1, w2 = st.columns(2)
        new.loss_ratio_warn = w1.number_input("Loss ratio warn above", 0.0, 10.0, cfg.loss_ratio_warn, 0.05, key="ls_w")
        new.loss_ratio_fail = w2.number_input("Loss ratio fail above", 0.0, 10.0, cfg.loss_ratio_fail, 0.05, key="ls_f")
        return new

    def render(self, result):
        import plotly.express as px
        import streamlit as st

        from gl_dq.ui import state
        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import line, ordered_categories, series_encoding, style

        a = self.cfg.analytics
        c1, c2, c3 = st.columns([3, 1, 1])
        opts = self.segment_options()
        segs = c1.multiselect("Segment by", opts, default=[s for s in a.segments if s in opts], key="la_segs")
        basis = c2.selectbox("LR basis year", [x for x in ["pol_yr", "loss_yr"] if self.schema.has(x)],
                             index=0 if a.lr_basis == "pol_yr" else 1, key="la_basis")
        cap = c3.number_input("LR cap", 0.5, 50.0, a.lr_cap, 0.5, key="la_cap")
        spec = a.model_copy(update={"segments": segs, "lr_basis": basis, "lr_cap": cap})
        if basis != "pol_yr":
            st.warning("Written premium is booked by policy year. A loss-year loss ratio mixes the two bases, so "
                       "read it as indicative only.")
        with st.spinner("Computing…"):
            t = state.cached_method(self.name, self.cfg, "analytics", a=spec)
        src = self.project.sources
        lr = t["lr_severity"]
        lr = lr.assign(segment=[segment_key({g: r[g] for g in segs}) for _, r in lr.iterrows()])
        enc = series_encoding(lr, "segment", src, segs) if segs else {}
        orders = ordered_categories(lr, enc, basis)
        g1, g2 = st.columns(2)
        fig = line(lr, x=basis, y="loss_ratio", markers=True, hover_data={"premium": ":,.0f", "loss": ":,.0f"},
                   category_orders=orders, **enc)
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        g1.plotly_chart(style(fig, 340, f"Loss ratio by {basis}"), use_container_width=True)
        fig = line(lr, x=basis, y="severity", markers=True, hover_data={"claims": ":,"},
                   category_orders=orders, **enc)
        fig.update_yaxes(rangemode="tozero")
        g2.plotly_chart(style(fig, 340, f"Severity by {basis}"), use_container_width=True)
        with st.expander("Loss ratio / severity table"):
            st.dataframe(lr, hide_index=True, use_container_width=True, column_config={
                "loss_ratio": st.column_config.NumberColumn(format="percent"),
                "premium": st.column_config.NumberColumn(format="localized"),
                "loss": st.column_config.NumberColumn(format="localized"),
                "severity": st.column_config.NumberColumn(format="localized")})

        fq = t["frequency"]
        base = self.m.exposure_base
        fq = fq.assign(segment=[segment_key({g: r[g] for g in segs}) for _, r in fq.iterrows()])
        enc = series_encoding(fq, "segment", src, segs) if segs else dict(color_discrete_sequence=["#2a78d6"])
        # exposure bases are not additive and have different scales: one panel per base
        n_bases = fq[base].nunique()
        fig = px.bar(fq, x="segment" if segs else base, y="frequency", facet_col=base, facet_col_wrap=3,
                     facet_col_spacing=0.07, facet_row_spacing=0.18,
                     hover_data={"claims": ":,", "exposure": ":,.0f"},
                     category_orders=ordered_categories(fq, enc, "segment" if segs else base, base),
                     **{k: v for k, v in enc.items() if not k.startswith("facet")})
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(showticklabels=False, title=None)
        st.plotly_chart(style(fig, 240 * ((n_bases + 2) // 3),
                              f"Claim frequency per {a.frequency_per:,.0f} exposure units (one panel per {base})"),
                        use_container_width=True)

        h = t["policy_lr_hist"]
        special = h[h["bucket"] < 0].groupby(["segment", "label"], as_index=False)["n_policies"].sum()
        dist = h[h["bucket"] >= 0].copy()
        dist["share"] = dist["n_policies"] / h.groupby("segment")["n_policies"].transform("sum")
        enc = series_encoding(dist, "segment", src, segs) if segs else {}
        fig = px.bar(dist, x="lr_from", y="share", barmode="overlay", opacity=0.6,
                     labels={"lr_from": "loss ratio", "share": "share of policies"},
                     hover_data={"n_policies": ":,"}, **enc)
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(bargap=0.02)
        st.plotly_chart(style(fig, 360,
                              f"Policy-level loss ratio distribution (policies with loss; last bin = ≥ {cap:.0%})"),
                        use_container_width=True)
        if not special.empty:
            piv = special.pivot(index="segment", columns="label", values="n_policies").fillna(0)
            piv["total_policies"] = h.groupby("segment")["n_policies"].sum()
            if "zero loss" in piv:
                piv["pct_zero_loss"] = piv["zero loss"] / piv["total_policies"]
            status_table(piv.reset_index(), percent_cols=["pct_zero_loss"])
