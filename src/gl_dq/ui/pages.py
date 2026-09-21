"""Overview pages: tracker (workflow progress), recommended preprocessing, knowledge base."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from gl_dq.core.config import dump_yaml
from gl_dq.core.knowledge import ConflictError, export_markdown, preprocessing_spec
from gl_dq.core.results import STATUS_ICON, parse_segment, split_segment_columns
from gl_dq.summary import filter_clause
from gl_dq.tracker import build_tracker, snapshots_for
from gl_dq.ui import state
from gl_dq.ui.components import preprocessing_editor, status_label, step_summary
from gl_dq.ui.theme import CATEGORICAL, SEQ_SCALE, STATUS, entity_colors, limit_series, line, style


def compact(v: float) -> str:
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.1f}{suf}"
    return f"{v:,.0f}"


def summary_page():
    """Front page: what is in the table - records, policy terms and written premium."""
    ctx = state.get_context()
    m = ctx.project.measures
    st.title("📋 Portfolio summary")
    st.caption(f"`{ctx.project.table}` · a policy is one distinct {' + '.join(f'`{c}`' for c in ctx.project.policy_key)} "
               f"· premium is `{m.written_premium}`. Queried live.")

    opts = [o for o in ctx.schema.names() if o in (ctx.project.segment_candidates or []) or o == ctx.project.src_col]
    default = [d for d in [ctx.project.src_col, "covg_type_desc"] if d in opts]
    dims = st.multiselect("Summarize by", opts, default=default, key="sum_dims")

    # filter on the same dimensions: empty = everything
    filters: dict[str, list[str]] = {}
    if dims:
        for col, dim in zip(st.columns(min(len(dims), 4)), dims):
            try:
                values = state.distinct_values(dim)
            except Exception as e:  # noqa: BLE001
                col.caption(f"{dim}: {e}")
                continue
            picked = col.multiselect(f"Filter {dim}", values, default=[], key=f"sum_f_{dim}",
                                     placeholder=f"All ({len(values)})")
            if picked:
                filters[dim] = picked
    where = filter_clause(ctx, filters)

    totals = state.summary([], where)
    t = totals.iloc[0]
    k = st.columns(4)
    k[0].metric("Records", f"{int(t.records):,}")
    k[1].metric("Policies", f"{int(t.policies):,}", help="Distinct policy terms")
    k[2].metric("Written premium", compact(t.premium), help=f"{t.premium:,.2f}")
    k[3].metric("Premium per policy", f"{t.premium_per_policy:,.0f}" if t.policies else "-")
    if where:
        full = state.summary([]).iloc[0]
        shown = "; ".join(f"**{col}**: " + ", ".join(vals) for col, vals in filters.items())
        share = f" ({t.premium / full.premium:.1%} of premium)" if full.premium else ""
        st.caption(f"Filtered to {shown} — {int(t.records):,} of {int(full.records):,} records{share}")
    if not dims:
        st.dataframe(totals, hide_index=True, use_container_width=True)
        return
    df = state.summary(dims, where)
    if df.empty:
        st.warning("No rows match these filters.")
        return

    x, color = dims[0], (dims[1] if len(dims) > 1 else None)
    plot = df.copy()
    if len(dims) > 2:  # keep the charts readable; the tables below still hold every row
        plot["group"] = plot[dims[1:]].astype(str).agg(" · ".join, axis=1)
        color = "group"
    if color:
        plot, dropped = limit_series(plot, color, weight="premium")
        if dropped:
            st.caption(f"Charts show the {plot[color].nunique()} largest {color} values by premium; "
                       f"{len(dropped)} smaller ones are in the tables below.")
    # colour by the column's full domain, so filtering never repaints the values that survive
    if color == ctx.project.src_col:
        known = ctx.project.sources
    elif color in dims:
        try:
            known = state.distinct_values(color)
        except Exception:  # noqa: BLE001
            known = None
    else:
        known = None
    enc = dict(color=color, color_discrete_map=entity_colors(plot[color].astype(str), known)) if color \
        else dict(color_discrete_sequence=[CATEGORICAL[0]])
    cols = st.columns(3)
    for col, (y, title, fmt) in zip(cols, [("premium", "Written premium", ",.0f"),
                                           ("policies", "Policies", ",.0f"),
                                           ("records", "Records", ",.0f")]):
        fig = px.bar(plot, x=x, y=y, barmode="group", hover_data={y: f":{fmt}"}, labels={y: ""}, **enc)
        fig.update_layout(showlegend=bool(color))
        col.plotly_chart(style(fig, 300, title), use_container_width=True)

    if len(dims) > 1:
        st.markdown(f"**By {dims[0]}**")
        _summary_table(state.summary([dims[0]], where), [dims[0]], key="sum_first")
    st.markdown("**By " + ", ".join(dims) + "**")
    _summary_table(df, dims, key="sum_full")
    if len(dims) == 2:
        with st.expander(f"Premium crosstab: {dims[0]} × {dims[1]}"):
            piv = df.pivot_table(index=dims[0], columns=dims[1], values="premium", aggfunc="sum", fill_value=0)
            piv["Total"] = piv.sum(axis=1)
            piv.loc["Total"] = piv.sum()
            st.dataframe(piv.style.format("{:,.0f}"), use_container_width=True)
    with st.expander("SQL"):
        from gl_dq.summary import summary_sql

        st.code(summary_sql(ctx, dims, where), language="sql")


def _summary_table(df, dims, key: str):
    view = df.assign(premium=df["premium"].round(0), premium_per_policy=df["premium_per_policy"].round(0))
    st.dataframe(view, hide_index=True, use_container_width=True, key=key, column_config={
        "records": st.column_config.NumberColumn(format="localized"),
        "policies": st.column_config.NumberColumn(format="localized"),
        "premium": st.column_config.NumberColumn(format="localized"),
        "premium_per_policy": st.column_config.NumberColumn(format="localized"),
        "records_per_policy": st.column_config.NumberColumn(format="%.2f"),
        "premium_share": st.column_config.NumberColumn(format="percent")})
    st.download_button("⬇️ CSV", df.to_csv(index=False), file_name=f"summary_by_{'_'.join(dims) or 'total'}.csv",
                       mime="text/csv", key=f"{key}_dl")


def refresh_controls(ctx, key: str):
    runs = state.runs()
    last = runs.iloc[-1]["run_ts"] if len(runs) else None
    c1, c2 = st.columns([3, 1])
    c1.caption(f"Profile **{ctx.profile}** · table `{ctx.project.table}` · last refresh: **{last or 'never'}**")
    if c2.button("🔄 Refresh now", key=key, use_container_width=True):
        if ctx.project.backend == "databricks" and ctx.project.refresh_job_id:
            from databricks.sdk import WorkspaceClient

            run = WorkspaceClient().jobs.run_now(job_id=int(ctx.project.refresh_job_id))
            st.toast(f"Triggered refresh job run {run.run_id}. Results appear when it finishes.", icon="🚀")
        else:
            from gl_dq.runner import refresh

            with st.spinner("Running all checks…"):
                refresh(ctx, log=lambda *_: None)
            st.toast("Refresh complete", icon="✅")
        state.clear_data_caches()
        st.rerun()


def _var_line(r) -> None:
    flag = STATUS_ICON.get(r.data_status) if isinstance(r.data_status, str) else "·"
    st.markdown(f"{flag} `{r.variable}`" + (" 🔁" if r.reopened else ""))
    meta = [x for x in [r.assignee if isinstance(r.assignee, str) else None,
                        f"{r.days_in_status:g}d in stage" if pd.notna(r.days_in_status) else None] if x]
    if meta:
        st.caption(" · ".join(meta))


def tracker_page():
    ctx = state.get_context()
    wf = ctx.workflow
    st.title("🧭 Cleaning tracker")
    refresh_controls(ctx, "refresh_tracker")
    latest = state.latest_findings()
    if latest.empty:
        st.info("No refresh has been stored yet. Click **Refresh now** (or run `python jobs/refresh.py`).")
        return
    checks = ctx.enabled_checks()
    records = ctx.knowledge.all()
    latest = latest[latest["check"].isin(checks)]
    var_df, long = build_tracker(ctx.schema.names(include_derived=False), latest, records, checks, wf)

    total, done = len(var_df), int(var_df["done"].sum())
    k = st.columns(4 + len(wf.roles))
    k[0].metric("Columns", total)
    k[1].metric("Closed", f"{done} ({done / total:.0%})" if total else "0")
    k[2].metric("Flagged, not started", int((var_df["status"].eq(wf.initial) & var_df["n_flagged"].gt(0)).sum()))
    k[3].metric("🔁 Re-opened", int(var_df["reopened"].sum()), help="Closed columns whose data got worse since they were closed")
    for i, label in enumerate(wf.roles.values()):
        k[4 + i].metric(f"With {label.lower()}", int((var_df["waiting_on"] == label).sum()))
    st.progress(done / total if total else 0.0, text=f"{done} of {total} columns closed · {total - done} to go")

    t_board, t_checks, t_grid, t_edit, t_activity, t_runs = st.tabs(
        ["Workflow board", "Progress by check", "Column × check grid", "Assign & update", "Recent activity", "Run history"])

    with t_board:
        counts = var_df["status"].value_counts()
        flow_keys = {s.key for s in wf.flow()}
        open_stages = [s for s in wf.stages if not s.done and (s.key in flow_keys or counts.get(s.key, 0))]
        st.caption("Open columns by stage, in hand-off order: data status · column · assignee · days in stage. "
                   "Use ➡ in the notes panel of any check page to hand a column to the next stage.")
        for col, s in zip(st.columns(len(open_stages)), open_stages):
            sub = var_df[var_df["status"] == s.key].sort_values(["n_flagged", "days_in_status"], ascending=False)
            with col.container(border=True):
                st.markdown(f"**{s.icon} {s.label}**")
                st.caption(wf.roles[s.role] if s.role else "Unassigned")
                st.markdown(f"### {len(sub)}")
                if s.key == wf.initial:
                    sub = sub[sub["n_flagged"] > 0]
                    st.caption(f"{len(sub)} flagged by the data")
                for r in sub.head(12).itertuples():
                    _var_line(r)
                if len(sub) > 12:
                    st.caption(f"+{len(sub) - 12} more")
        st.markdown("**Closed**")
        done_stages = [s for s in wf.stages if s.done]
        for col, s in zip(st.columns(len(done_stages)), done_stages):
            sub = var_df[var_df["status"] == s.key]
            with col.container(border=True):
                st.markdown(f"**{s.icon} {s.label}**")
                st.markdown(f"### {len(sub)}")
                if s.requires_preprocessing and len(sub):
                    missing = int((sub["n_steps"] == 0).sum())
                    st.caption(f"{int(sub['n_steps'].sum())} preprocessing steps"
                               + (f" · ⚠ {missing} without steps" if missing else ""))
                for r in sub.head(8).itertuples():
                    _var_line(r)

    with t_checks:
        st.markdown("**By check**: columns in scope, closed and flagged")
        for c in checks:
            sub = long[long["check"] == c]
            if sub.empty:
                continue
            n, d = len(sub), int((sub["done"] & ~sub["reopened"]).sum())
            fl = int(sub["data_status"].isin(["warn", "fail"]).sum())
            cls = ctx.checks[c]
            st.progress(d / n, text=f"{cls.icon} {cls.title}: {d}/{n} closed · {fl} flagged")
        st.markdown("**Flagged findings by source and check** (latest run)")
        fl = latest[latest["status"].isin(["warn", "fail"])].copy()
        if fl.empty:
            st.success("Nothing flagged in the latest run.")
        else:
            src_col = ctx.project.src_col
            fl["source"] = [parse_segment(s).get(src_col, "ALL / cross-source") for s in fl["segment"]]
            fl["check"] = pd.Categorical(fl["check"].map(lambda c: ctx.checks[c].title),
                                         [ctx.checks[c].title for c in checks])
            piv = fl.pivot_table(index="check", columns="source", values="status", aggfunc="size", fill_value=0,
                                 observed=True)
            fig = px.imshow(piv, text_auto=True, aspect="auto", color_continuous_scale=SEQ_SCALE, zmin=0)
            style(fig, height=45 * len(piv) + 110)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        prev = state.latest_findings(1)
        if not prev.empty:
            keycols = ["check", "variable", "item", "segment", "metric"]
            cur_f = latest[latest["status"].isin(["warn", "fail"])][keycols].astype(str)
            prev_f = prev[prev["status"].isin(["warn", "fail"]) & prev["check"].isin(checks)][keycols].astype(str)
            new = cur_f.merge(prev_f, how="left", indicator=True).query("_merge == 'left_only'")
            gone = prev_f.merge(cur_f, how="left", indicator=True).query("_merge == 'left_only'")
            st.markdown(f"**Since previous run:** 🆕 {len(new)} newly flagged · ✔️ {len(gone)} no longer flagged")
            if len(new):
                with st.expander("Newly flagged"):
                    st.dataframe(split_segment_columns(new.drop(columns="_merge")), hide_index=True,
                                 use_container_width=True)

    with t_grid:
        c1, c2, c3 = st.columns([2, 2, 2])
        only_flag = c1.toggle("Only columns with flags", True)
        only_open = c2.toggle("Hide closed", False)
        search = c3.text_input("Search column", "")
        view_long = long.copy()
        view_long["cell"] = [(STATUS_ICON.get(d) if isinstance(d, str) else "·") + (" 🔁" if ro else "")
                             for d, ro in zip(view_long["data_status"], view_long["reopened"])]
        grid = view_long.pivot_table(index="variable", columns="check", values="cell", aggfunc="first").fillna("")
        order = [c for c in checks if c in grid.columns]
        grid = grid[order]
        grid.columns = [f"{ctx.checks[c].icon} {ctx.checks[c].title}" for c in order]
        role_cols = {f"{r}_assignee": label for r, label in wf.roles.items()}
        meta = var_df.set_index("variable")[["stage", *role_cols, "n_flagged", "n_steps", "n_notes", "done", "data_status"]]
        grid = meta.rename(columns=role_cols).join(grid, how="inner")
        if only_flag:
            grid = grid[grid["data_status"].isin(["warn", "fail"])]
        if only_open:
            grid = grid[~grid["done"]]
        if search:
            grid = grid[grid.index.str.contains(search, case=False)]
        st.caption("Cells: data status (🟢 pass · 🟠 warn · 🔴 fail · 🔵 info) · 🔁 closed but re-opened by data")
        st.dataframe(grid.drop(columns=["done", "data_status"]), use_container_width=True,
                     height=min(600, 36 * (len(grid) + 1) + 3))

    with t_edit:
        st.caption("Bulk-update stage and assignees. Each saved row writes the column's YAML and a status-history "
                   "entry; closing a column also snapshots its current data status (for 🔁 re-open detection).")
        label_of = {s.key: wf.label(s.key) for s in wf.stages}
        key_of = {v: k for k, v in label_of.items()}
        role_cols = {f"{r}_assignee": label for r, label in wf.roles.items()}
        edit = var_df[["variable", "label", "status", *role_cols, "data_status", "n_flagged", "n_steps"]].copy()
        edit["status"] = edit["status"].map(lambda s: label_of.get(s, s))
        edit["data_status"] = edit["data_status"].map(status_label)
        for c in ["label", *role_cols]:
            edit[c] = edit[c].fillna("")
        edited = st.data_editor(
            edit, hide_index=True, use_container_width=True, height=520, key="tracker_edit",
            disabled=["variable", "data_status", "n_flagged", "n_steps"],
            column_config={"status": st.column_config.SelectboxColumn("stage", options=list(label_of.values()), required=True),
                           **{c: st.column_config.TextColumn(label) for c, label in role_cols.items()}})
        fields = ["status", "label", *role_cols]
        base = edit.set_index("variable")
        changes = [r for r in edited.to_dict("records")
                   if any((r[f] or "") != (base.loc[r["variable"], f] or "") for f in fields)]
        if st.button(f"💾 Save {len(changes)} change(s)", disabled=not changes, type="primary"):
            user, warnings = state.current_user(), []
            for r in changes:
                status = key_of.get(r["status"], r["status"])
                try:
                    rec = ctx.knowledge.update(r["variable"], user, workflow=wf, status=status, label=r["label"] or "",
                                               assignees={role: r[f"{role}_assignee"] for role in wf.roles},
                                               snapshots=snapshots_for(r["variable"], latest))
                    if wf.stage(status).requires_preprocessing and not rec.preprocessing:
                        warnings.append(f"`{r['variable']}`: add recommended preprocessing steps on the 🧰 Preprocessing page")
                except (ConflictError, ValueError) as e:
                    warnings.append(str(e))
            st.session_state.pop("tracker_edit", None)
            st.toast(f"Saved {len(changes)} column(s)", icon="✅")
            for w in warnings:
                st.warning(w)
            if not warnings:
                st.rerun()

    with t_activity:
        rows = [dict(at=ch.at, column=v, by=ch.by, what=f"{wf.label(ch.from_status or wf.initial, False)} → "
                                                        f"{wf.label(ch.to_status, False)}", detail=ch.comment)
                for v, rec in records.items() for ch in rec.status_log]
        rows += [dict(at=n.date, column=v, by=n.author, what="📝 note" + (f" ({n.check})" if n.check else ""), detail=n.text)
                 for v, rec in records.items() for n in rec.notes]
        rows += [dict(at=s.added_at, column=v, by=s.author, what="🧰 preprocessing step", detail=step_summary(wf, s))
                 for v, rec in records.items() for s in rec.preprocessing]
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("at", ascending=False).head(100), hide_index=True,
                         use_container_width=True)
        else:
            st.caption("No activity yet. Move a column to a stage or add a note from any check page.")

    with t_runs:
        runs = state.runs()
        if len(runs):
            fig = line(runs.melt(id_vars=["run_ts"], value_vars=["n_fail", "n_warn"]), x="run_ts", y="value",
                          color="variable", markers=True, labels={"value": "flagged findings", "run_ts": "run"},
                          color_discrete_map={"n_fail": STATUS["fail"], "n_warn": STATUS["warn"]})
            style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(runs.sort_values("run_ts", ascending=False), hide_index=True, use_container_width=True)


def preprocessing_page():
    ctx = state.get_context()
    wf = ctx.workflow
    st.title("🧰 Recommended preprocessing")
    st.caption("Columns left as is in the data and handled in the modeling pipeline. The steps are exported as a "
               "machine-readable spec (`preprocessing_spec.yaml`) that the preprocessing pipeline can be built from.")
    pre_keys = [s.key for s in wf.stages if s.requires_preprocessing]
    records = ctx.knowledge.all()
    spec = preprocessing_spec(records, wf, ctx.project.table)
    in_status = spec["variables"]
    k = st.columns(3)
    k[0].metric("Columns to preprocess", len(in_status))
    k[1].metric("Steps", sum(len(v["steps"]) for v in in_status.values()))
    k[2].metric("⚠ Without steps", sum(1 for v in in_status.values() if not v["steps"]))

    rows = []
    for name, v in in_status.items():
        for s in v["steps"] or [None]:
            if s is None:
                rows.append(dict(column=name, order="", step="⚠ no steps yet", params="", sources="", rationale="", author=""))
                continue
            op = wf.preprocessing_ops.get(s["op"])
            rows.append(dict(column=name, order=str(s["order"]), step=op.label if op else s["op"],
                             params=json.dumps(s["params"]), sources=s["sources"] if isinstance(s["sources"], str)
                             else ", ".join(s["sources"]), rationale=s["rationale"], author=s["author"]))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No columns are in a preprocessing stage yet. Pick a column below and add a step, or choose "
                "“Preprocess in modeling” in the notes panel of any check page.")

    c1, c2, c3 = st.columns(3)
    yaml_text = dump_yaml(spec)
    c1.download_button("⬇️ Spec (YAML)", yaml_text, file_name="preprocessing_spec.yaml", mime="text/yaml",
                       use_container_width=True)
    c2.download_button("⬇️ Spec (JSON)", json.dumps(spec, indent=2), file_name="preprocessing_spec.json",
                       mime="application/json", use_container_width=True)
    if c3.button("📤 Publish to knowledge store", use_container_width=True):
        ctx.knowledge.storage.write_text("exports/preprocessing_spec.yaml", yaml_text)
        st.toast("Published exports/preprocessing_spec.yaml", icon="📤")

    st.divider()
    names = ctx.schema.names(include_derived=False)
    ordered = sorted(names, key=lambda v: (v not in in_status, v))
    var = st.selectbox("Column", ordered, key="pp_page_var",
                       format_func=lambda v: f"{wf.stage(records[v].status).icon if v in records else '⚪'} {v}")
    cur = records[var].status if var in records else wf.initial
    hint = "" if cur in pre_keys or not pre_keys else f" · adding a step moves it to {wf.label(pre_keys[0], False)}"
    st.caption(f"Current stage: {wf.label(cur)}{hint}")
    with st.container(border=True):
        preprocessing_editor(ctx, var, key=f"page_{var}")


def knowledge_page():
    ctx = state.get_context()
    wf = ctx.workflow
    st.title("📚 Knowledge base")
    st.caption("Every column's stage, assignees, notes and recommended preprocessing, stored as YAML (one file per "
               "column) with a full edit history. Export it as a Markdown data dictionary to reuse on other projects.")
    records = ctx.knowledge.all()
    notes = pd.DataFrame([dict(variable=v, date=n.date, author=n.author, check=n.check, src=n.src,
                               tags=n.tags, reusable=n.reusable, note=n.text)
                          for v, rec in records.items() for n in rec.notes])
    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
    q = c1.text_input("Search notes", placeholder="e.g. TRIA, cents, mapping")
    all_tags = sorted({t for ts in notes.get("tags", []) for t in ts}) if not notes.empty else []
    tags = c2.multiselect("Tags", all_tags)
    srcs = c3.multiselect("Source", ctx.project.sources)
    reusable_only = c4.toggle("Reusable", True)
    if notes.empty:
        st.info("No notes yet.")
    else:
        v = notes
        if q:
            v = v[v["note"].str.contains(q, case=False) | v["variable"].str.contains(q, case=False)]
        if tags:
            v = v[v["tags"].map(lambda ts: bool(set(ts) & set(tags)))]
        if srcs:
            v = v[v["src"].isin(srcs)]
        if reusable_only:
            v = v[v["reusable"]]
        st.dataframe(v.assign(tags=v["tags"].map(", ".join)).sort_values("date", ascending=False),
                     hide_index=True, use_container_width=True)

    st.divider()
    c1, c2 = st.columns([1, 1])
    md = export_markdown(records, wf, title=f"{ctx.project.name}: data dictionary & cleaning notes")
    c1.download_button("⬇️ Download data dictionary (.md)", md, file_name="data_dictionary.md", mime="text/markdown")
    if c2.button("📤 Publish to knowledge store (exports/data_dictionary.md)"):
        ctx.knowledge.storage.write_text("exports/data_dictionary.md", md)
        st.toast("Published", icon="📤")

    st.divider()
    if records:
        var = st.selectbox("Column history", sorted(records),
                           format_func=lambda v: f"{wf.stage(records[v].status).icon} {v}")
        for h in ctx.knowledge.history(var)[:25]:
            r = h.get("record", {})
            who = ", ".join(f"{wf.roles.get(k, k)}: {p}" for k, p in (r.get("assignees") or {}).items()) or "unassigned"
            st.caption(f"{h.get('at')} · {h.get('author')} · {h.get('action')} → "
                       f"{wf.label(r.get('status', wf.initial), False)} · {who} · {len(r.get('notes', []))} notes · "
                       f"{len(r.get('preprocessing', []))} steps")
        with st.expander("Raw YAML"):
            st.code(ctx.knowledge.storage.read_text(f"variables/{var}.yaml") or "", language="yaml")
