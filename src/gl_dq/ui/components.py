"""Shared UI building blocks: status tables, the notes/status panel, the check page frame."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gl_dq.core.knowledge import ConflictError, Note, PreprocessingStep
from gl_dq.core.results import STATUS_ICON, STATUS_RANK, worst
from gl_dq.core.workflow import parse_mapping
from gl_dq.tracker import is_reopened, snapshots_for
from gl_dq.ui import state


def status_label(s) -> str:
    return f"{STATUS_ICON.get(s, '⚪')} {s}" if isinstance(s, str) else "⚪"


def status_table(df: pd.DataFrame, percent_cols=(), number_formats: dict | None = None, height="auto", key=None):
    if df is None or df.empty:
        st.caption("No rows.")
        return
    view = df.copy()
    if "status" in view:
        view["status"] = view["status"].map(status_label)
        view = view[["status"] + [c for c in view.columns if c != "status"]]
    cfg = {c: st.column_config.NumberColumn(format="percent") for c in percent_cols if c in view}
    if "segment" in view:  # segment labels are long ("src=BMQ|pol_yr=2019"): give them room
        cfg["segment"] = st.column_config.TextColumn(width="medium")
    for c, fmt in (number_formats or {}).items():
        if c in view:
            cfg[c] = st.column_config.NumberColumn(format=fmt.replace("%,", "%"))
    kwargs = {} if height in (None, "auto") else {"height": height}
    st.dataframe(view, use_container_width=True, hide_index=True, column_config=cfg, key=key, **kwargs)


def status_counts(findings: pd.DataFrame) -> str:
    c = findings["status"].value_counts()
    return "  ".join(f"{STATUS_ICON[s]} {int(c.get(s, 0))} {s}" for s in ["fail", "warn", "pass", "info"] if c.get(s, 0))


# ---- recommended preprocessing editor -------------------------------------------------
def _param_widget(name: str, spec, key: str):
    label = name.replace("_", " ")
    if spec.type == "select":
        opts = spec.options or [""]
        return st.selectbox(label, opts, index=opts.index(spec.default) if spec.default in opts else 0, key=key,
                            help=spec.help or None)
    if spec.type == "number":
        v = st.number_input(label, value=float(spec.default or 0.0), key=key, help=spec.help or None, format="%g")
        return int(v) if float(v).is_integer() and not isinstance(spec.default, float) else v
    if spec.type == "bool":
        return st.checkbox(label, value=bool(spec.default), key=key, help=spec.help or None)
    if spec.type == "list":
        txt = st.text_input(label, value=", ".join(map(str, spec.default or [])), key=key, help=spec.help or None)
        return [x.strip() for x in txt.split(",") if x.strip()]
    if spec.type == "mapping":
        return parse_mapping(st.text_input(label, key=key, help=spec.help or "from=to pairs, comma-separated"))
    return st.text_input(label, value=spec.default or "", key=key, help=spec.help or None) or None


def step_summary(wf, step) -> str:
    op = wf.preprocessing_ops.get(step.op)
    params = ", ".join(f"{k}={v}" for k, v in step.params.items() if v not in (None, "", [], {}))
    scope = f" · sources: {', '.join(step.sources)}" if step.sources else ""
    return f"**{op.label if op else step.op}** `{params}`{scope}" + (f"  \n_{step.rationale}_" if step.rationale else "")


def preprocessing_editor(ctx, variable: str, key: str):
    """List, add and remove recommended preprocessing steps for one column (saved immediately)."""
    wf = ctx.workflow
    rec, _ = ctx.knowledge.get(variable)
    st.markdown(f"**🧰 Recommended preprocessing for `{variable}`**")
    if not rec.preprocessing:
        st.caption("No steps yet. Steps are applied in order when the modeling pipeline is built.")
    for i, step in enumerate(rec.preprocessing):
        a, b = st.columns([6, 1])
        a.markdown(f"{i + 1}. {step_summary(wf, step)}")
        if b.button("🗑", key=f"pp_del_{key}_{i}", help="Remove step"):
            ctx.knowledge.remove_preprocessing_step(variable, i, state.current_user())
            st.rerun()
    ops = wf.preprocessing_ops
    op = st.selectbox("Add a step", list(ops), format_func=lambda k: ops[k].label, key=f"pp_op_{key}")
    if ops[op].description:
        st.caption(ops[op].description)
    params = {name: _param_widget(name, spec, f"pp_{key}_{op}_{name}") for name, spec in ops[op].params.items()}
    sources = st.multiselect("Sources (empty = all)", ctx.project.sources, key=f"pp_src_{key}")
    rationale = st.text_input("Rationale", key=f"pp_why_{key}", placeholder="Why this treatment")
    if st.button("➕ Add step", key=f"pp_add_{key}", type="primary"):
        step = PreprocessingStep(op=op, params={k: v for k, v in params.items() if v not in (None, "", [], {})},
                                 sources=sources, rationale=rationale, author=state.current_user())
        target = next((s.key for s in wf.stages if s.requires_preprocessing), None)
        ctx.knowledge.add_preprocessing_step(variable, step, state.current_user(), set_status=target)
        st.toast(f"Added {ops[op].label} to {variable}", icon="🧰")
        st.rerun()


# ---- notes / status panel ----------------------------------------------------------
def notes_panel(ctx, check_name: str, findings: pd.DataFrame):
    wf = ctx.workflow
    st.markdown("#### 📝 Status & notes")
    if findings.empty:
        st.caption("No variables in this result.")
        return
    f = findings.assign(rank=findings["status"].map(STATUS_RANK).fillna(-1))
    order = f.groupby("variable")["rank"].max().sort_values(ascending=False)
    ws = f.groupby("variable")["status"].agg(worst)
    records = ctx.knowledge.all()
    var = st.selectbox(
        "Variable", list(order.index), key=f"np_var_{check_name}",
        format_func=lambda v: f"{STATUS_ICON.get(ws.get(v))} {v}  {wf.stage(records[v].status).icon if v in records else ''}")

    rec, ver = ctx.knowledge.get(var)
    seen_key = f"np_seen_{check_name}_{var}"
    user = state.current_user()
    sub = findings[findings["variable"] == var]
    cur_status = worst(sub["status"])
    snaps = snapshots_for(var, state.latest_findings())
    snaps.update(snapshots_for(var, findings))
    cur_value = snaps[check_name].judged_value if check_name in snaps else None
    stage = wf.stage(rec.status)
    snap = rec.checks.get(check_name)
    reopened = bool(snap) and is_reopened(rec.status in wf.done_keys(), snap.judged_status, snap.judged_value,
                                          cur_status, cur_value)

    c1, c2 = st.columns(2)
    c1.markdown(f"**Data**<br>{status_label(cur_status)}", unsafe_allow_html=True)
    c2.markdown(f"**Stage**<br>{stage.icon} {stage.label}", unsafe_allow_html=True)
    waiting = wf.waiting_on(rec.status)
    since = f" since {rec.status_since[:10]}" if rec.status_since else ""
    who = rec.assignees.get(stage.role) if stage.role else None
    st.caption((f"Waiting on **{waiting}**" + (f" ({who})" if who else "") if waiting else "Closed" if stage.done
                else "Not started") + since)
    if reopened:
        st.error(f"🔁 Re-opened by data: closed when this check was {snap.judged_status}, now {cur_status}.")

    nxt = wf.next_stage(rec.status)
    if nxt is not None and st.button(f"➡ Move to {nxt.icon} {nxt.label}", key=f"np_next_{check_name}_{var}",
                                     use_container_width=True):
        try:
            ctx.knowledge.update(var, user, workflow=wf, status=nxt.key, snapshots=snaps, expected_version=ver)
            st.toast(f"{var}: {nxt.label}", icon=nxt.icon)
            st.rerun()
        except ConflictError as e:
            st.warning(str(e))

    status = st.selectbox("Status", wf.keys(), index=wf.keys().index(rec.status) if rec.status in wf.keys() else 0,
                          format_func=wf.label, key=f"np_status_{check_name}_{var}")
    if wf.stage(status).description:
        st.caption(wf.stage(status).description)
    if wf.stage(status).requires_preprocessing:
        with st.container(border=True):
            preprocessing_editor(ctx, var, key=f"{check_name}_{var}")
        rec, ver = ctx.knowledge.get(var)  # steps may have been saved

    with st.form(f"np_form_{check_name}_{var}", clear_on_submit=True):
        assignees = {role: st.text_input(f"{label}", rec.assignees.get(role, ""), key=f"np_as_{check_name}_{var}_{role}")
                     for role, label in wf.roles.items()}
        text = st.text_area("Note", placeholder="Findings, business rule, decision… written so it can be reused.")
        a, b = st.columns(2)
        src = a.selectbox("Source", ["(all)"] + ctx.project.sources)
        tags = b.text_input("Tags", placeholder="mapping, business_rule")
        reusable = st.checkbox("Reusable knowledge", value=True)
        submitted = st.form_submit_button("Save", type="primary", use_container_width=True)

    if submitted:
        seen = st.session_state.get(seen_key, ver)
        if seen != ver:
            st.warning("Someone else updated this variable while you were editing. Their changes are shown above; "
                       "please re-apply yours.")
        else:
            note = Note(author=user, text=text.strip(), check=check_name, src=None if src == "(all)" else src,
                        tags=[t.strip() for t in tags.split(",") if t.strip()], reusable=reusable) if text.strip() else None
            changed = (status != rec.status or note is not None
                       or any((v or None) != rec.assignees.get(r) for r, v in assignees.items()))
            if changed:
                try:
                    ctx.knowledge.update(var, user, workflow=wf, status=status, assignees=assignees, note=note,
                                         snapshots=snaps, expected_version=ver)
                    if wf.stage(status).requires_preprocessing and not rec.preprocessing:
                        st.warning("Status saved. Add at least one recommended preprocessing step above.")
                    st.toast(f"Saved {var}", icon="✅")
                    rec, ver = ctx.knowledge.get(var)
                except ConflictError as e:
                    st.warning(str(e))
    st.session_state[seen_key] = ver

    if rec.status_log:
        with st.expander(f"Status history ({len(rec.status_log)})"):
            for ch in reversed(rec.status_log):
                st.caption(f"{ch.at[:16].replace('T', ' ')} · {ch.by}: {wf.label(ch.from_status or wf.initial, False)} → "
                           f"{wf.label(ch.to_status, False)}" + (f" ({ch.comment})" if ch.comment else ""))
    notes = sorted(rec.notes, key=lambda n: n.date, reverse=True)
    st.markdown(f"**Notes ({len(notes)})**")
    if not notes:
        st.caption("No notes yet.")
    for n in notes[:20]:
        scope = " · ".join(x for x in [n.check, n.src] if x)
        tags = " ".join(f"`{t}`" for t in n.tags)
        with st.container(border=True):
            st.markdown(f"{n.text}")
            st.caption(f"{n.date[:10]} · {n.author}" + (f" · {scope}" if scope else "") + (f" · {tags}" if tags else "")
                       + ("" if n.reusable else " · not reusable"))


# ---- check page frame ------------------------------------------------------------------
def check_page(name: str):
    ctx = state.get_context()
    cls = ctx.checks[name]
    cfg = state.session_config(name)
    check = ctx.make_check(name, cfg)
    saved = ctx.check_config(name)

    st.title(f"{cls.icon} {cls.title}")
    st.caption(cls.description)

    with st.expander("⚙️ Settings" + ("  ·  *session changes not saved*" if cfg != saved else "")):
        before = set(st.session_state.keys())
        try:
            new_cfg = check.settings_ui(cfg)
        except Exception as e:  # noqa: BLE001
            st.error(f"Invalid settings: {e}")
            new_cfg = cfg
        created = set(st.session_state.keys()) - before
        st.session_state[f"widgets::{name}"] = list(set(st.session_state.get(f"widgets::{name}", [])) | created)
        b1, b2, b3, _ = st.columns([1, 1.3, 1.3, 3])
        if b1.button("▶ Apply", disabled=new_cfg == cfg, key=f"apply_{name}", type="primary"):
            state.set_session_config(name, new_cfg)
            st.rerun()
        if b2.button("💾 Save to config", disabled=new_cfg == saved, key=f"save_{name}",
                     help="Writes the YAML config used by everyone and by the refresh job"):
            ctx.save_check_config(name, new_cfg)
            state.set_session_config(name, new_cfg)
            st.toast(f"Saved checks/{name}.yaml", icon="💾")
            st.rerun()
        if b3.button("↩ Reset to saved", disabled=cfg == saved and new_cfg == cfg, key=f"reset_{name}"):
            state.reset_session_config(name)
            st.rerun()

    try:
        with st.spinner("Running queries…"):
            result = state.run_check(name, cfg)
    except Exception as e:  # noqa: BLE001
        st.error(f"Check failed: {e}")
        st.exception(e)
        return
    for msg in result.messages:
        st.info(msg)

    main, side = st.columns([3, 1.2], gap="large")
    with main:
        if not result.findings.empty:
            st.markdown(status_counts(result.findings))
        check.render(result)
        with st.expander(f"All findings ({len(result.findings)})"):
            status_table(result.findings.drop(columns=["check"]), key=f"allf_{name}")
        with st.expander("SQL"):
            for label, sql in result.sql.items():
                st.markdown(f"`{label}`")
                st.code(sql, language="sql")
    with side:
        notes_panel(ctx, name, result.findings)
