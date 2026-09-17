"""Loss summary: allocated loss and claim count by loss year vs source of truth, plus severity,
frequency (by exposure base) and loss ratio analytics with user-chosen segments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from gl_dq.checks._recon import Tolerance, pipeline_agg, reconcile, recon_findings, render_recon, sot_agg, sot_columns
from gl_dq.checks.base import Check, CheckResult
from gl_dq.core.config import CheckConfig
from gl_dq.core.registry import register_check
from gl_dq.core.results import segment_key


class Analytics(BaseModel):
    segments: list[str] = ["src"]
    lr_basis: str = "pol_yr"  # written premium is policy-year based, so LR defaults to policy year
    frequency_per: float = 1000.0  # claims per N exposure units
    lr_cap: float = 5.0
    lr_bins: int = 50


@register_check("loss_recon")
class LossRecon(Check):
    title = "Loss summary"
    icon = "📉"
    description = ("Allocated loss and claim count by loss year vs the pricing study, then severity, frequency and "
                   "loss ratio by segment. Frequency is always split by exposure base, which cannot be added together.")
    default_order = 50

    class Config(CheckConfig):
        segments: list[str] = ["src"]
        time_dim: str = "loss_yr"
        sot_query: str = "sql/sot_loss.sql"
        sot_loss_col: str = "allocation"
        sot_claim_count_col: str = "claim_alloc"
        dim_map: dict[str, str] = {}
        tolerance_loss: Tolerance = Tolerance(abs=5000, pct=0.02)
        tolerance_claims: Tolerance = Tolerance(abs=5, pct=0.02)
        analytics: Analytics = Analytics()

    @property
    def m(self):
        return self.project.measures

    # ---- reconciliation ----------------------------------------------------------
    def recon(self) -> tuple[list[dict], dict[str, pd.DataFrame]]:
        s = self.schema
        dims = list(dict.fromkeys(self.cfg.segments + [self.cfg.time_dim]))
        loss, cnt = s.ref(self.m.loss), s.ref(self.m.claim_count)
        where = f"COALESCE({loss}, 0) <> 0 OR COALESCE({cnt}, 0) <> 0"
        pipe = pipeline_agg(self, dims, {"loss": loss, "claim_count": cnt}, where)
        sot_sql = self.ctx.render_user_sql(self.cfg.sot_query)
        sot = sot_agg(self, sot_sql, dims, self.cfg.dim_map,
                      {"loss": self.cfg.sot_loss_col, "claim_count": self.cfg.sot_claim_count_col})
        rec_loss = reconcile(pipe, sot, dims, "loss", self.cfg.tolerance_loss)
        rec_cnt = reconcile(pipe, sot, dims, "claim_count", self.cfg.tolerance_claims)
        findings = (recon_findings(rec_loss, self.m.loss, "loss", self.cfg.tolerance_loss)
                    + recon_findings(rec_cnt, self.m.claim_count, "claim_count", self.cfg.tolerance_claims))
        return findings, {"recon_loss": rec_loss, "recon_claims": rec_cnt}

    # ---- analytics ---------------------------------------------------------------
    def analytics(self, a: Analytics) -> dict[str, pd.DataFrame]:
        s, m = self.schema, self.m
        segs = s.validate(a.segments)
        out = {}
        lr_dims = list(dict.fromkeys(segs + [a.lr_basis]))
        df = pipeline_agg(self, lr_dims, {"premium": s.ref(m.written_premium), "loss": s.ref(m.loss),
                                          "claims": s.ref(m.claim_count)}, label="loss ratio / severity")
        with np.errstate(divide="ignore", invalid="ignore"):
            df["loss_ratio"] = np.where(df["premium"] > 0, df["loss"] / df["premium"], np.nan)
            df["severity"] = np.where(df["claims"] > 0, df["loss"] / df["claims"], np.nan)
        out["lr_severity"] = df.sort_values(lr_dims)

        f_dims = list(dict.fromkeys(segs + [m.exposure_base]))
        fq = pipeline_agg(self, f_dims, {"claims": s.ref(m.claim_count), "exposure": s.ref(m.exposure)},
                          label="frequency")
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

    def run(self) -> CheckResult:
        findings, tables = self.recon()
        tables.update(self.analytics(self.cfg.analytics))
        return self.result(findings, tables)

    # ---- UI ------------------------------------------------------------------------
    def settings_ui(self, cfg):
        import streamlit as st

        new = cfg.model_copy(deep=True)
        try:
            sot_cols = sot_columns(self, self.ctx.render_user_sql(cfg.sot_query))
        except Exception as e:  # noqa: BLE001
            st.error(f"SOT query failed: {e}")
            sot_cols = []
        available = [d for d in self.segment_options() if cfg.dim_map.get(d, d) in sot_cols and d != cfg.time_dim]
        new.segments = st.multiselect("Reconciliation segments (must exist in SOT)", available,
                                      default=[d for d in cfg.segments if d in available], key="lr_segs")
        c1, c2, c3, c4 = st.columns(4)
        new.tolerance_loss = Tolerance(abs=c1.number_input("Loss abs tol", 0.0, value=cfg.tolerance_loss.abs, key="lr_la"),
                                       pct=c2.number_input("Loss % tol", 0.0, 1.0, cfg.tolerance_loss.pct, 0.005, format="%.3f", key="lr_lp"))
        new.tolerance_claims = Tolerance(abs=c3.number_input("Claims abs tol", 0.0, value=cfg.tolerance_claims.abs, key="lr_ca"),
                                         pct=c4.number_input("Claims % tol", 0.0, 1.0, cfg.tolerance_claims.pct, 0.005, format="%.3f", key="lr_cp"))
        st.caption(f"SOT query: `{cfg.sot_query}` · SOT columns: {', '.join(sot_cols) or 'n/a'}")
        return new

    def render(self, result):
        import plotly.express as px
        import streamlit as st

        from gl_dq.ui import state
        from gl_dq.ui.components import status_table
        from gl_dq.ui.theme import line, series_encoding, style

        tab_r, tab_a = st.tabs(["Reconciliation vs source of truth", "Severity · frequency · loss ratio"])
        dims = list(dict.fromkeys(self.cfg.segments + [self.cfg.time_dim]))
        with tab_r:
            st.markdown(f"**Allocated loss** (`{self.m.loss}`) by {', '.join(dims)}")
            render_recon(st, result.tables["recon_loss"], dims, "Loss", "loss")
            st.divider()
            st.markdown(f"**Claim count** (`{self.m.claim_count}`) by {', '.join(dims)}")
            render_recon(st, result.tables["recon_claims"], dims, "Claims", "claims")
        with tab_a:
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
            g1, g2 = st.columns(2)
            fig = line(lr, x=basis, y="loss_ratio", markers=True, hover_data={"premium": ":,.0f", "loss": ":,.0f"}, **enc)
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            g1.plotly_chart(style(fig, 340, f"Loss ratio by {basis}"), use_container_width=True)
            fig = line(lr, x=basis, y="severity", markers=True, hover_data={"claims": ":,"}, **enc)
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
                         **{k: v for k, v in enc.items() if not k.startswith("facet")})
            fig.update_yaxes(matches=None, showticklabels=True)
            fig.update_xaxes(showticklabels=False, title=None)
            st.plotly_chart(style(fig, 240 * ((n_bases + 2) // 3), f"Claim frequency per {a.frequency_per:,.0f} exposure units (one panel per {base})"),
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
            st.plotly_chart(style(fig, 360, f"Policy-level loss ratio distribution (policies with loss; last bin = ≥ {cap:.0%})"),
                            use_container_width=True)
            if not special.empty:
                piv = special.pivot(index="segment", columns="label", values="n_policies").fillna(0)
                piv["total_policies"] = h.groupby("segment")["n_policies"].sum()
                if "zero loss" in piv:
                    piv["pct_zero_loss"] = piv["zero loss"] / piv["total_policies"]
                status_table(piv.reset_index(), percent_cols=["pct_zero_loss"])
