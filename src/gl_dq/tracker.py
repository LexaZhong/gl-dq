"""Tracker: combine latest findings (data status) with the per-column workflow records."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from gl_dq.core.knowledge import CheckSnapshot, VariableRecord, now_iso
from gl_dq.core.results import STATUS_RANK, worst

REOPEN_REL_CHANGE = 0.10


def data_status(findings: pd.DataFrame) -> pd.DataFrame:
    """Worst status and the worst value at that status per (variable, check)."""
    cols = ["variable", "check", "data_status", "worst_value", "n_flagged"]
    if findings.empty:
        return pd.DataFrame(columns=cols)
    f = findings.copy()
    f["rank"] = f["status"].map(STATUS_RANK).fillna(-1)
    f["absval"] = f["value"].abs()
    g = f.sort_values(["rank", "absval"], ascending=False).groupby(["variable", "check"], as_index=False).first()
    flagged = f[f["status"].isin(["warn", "fail"])].groupby(["variable", "check"]).size().rename("n_flagged")
    g = g.merge(flagged, on=["variable", "check"], how="left").fillna({"n_flagged": 0})
    g = g.rename(columns={"status": "data_status", "absval": "worst_value"})
    return g[cols]


def snapshots_for(variable: str, findings: pd.DataFrame) -> dict[str, CheckSnapshot]:
    """Current data status of every check for a variable, to store when the column is closed."""
    ds = data_status(findings[findings["variable"] == variable]) if not findings.empty else data_status(findings)
    return {r.check: CheckSnapshot(judged_status=r.data_status,
                                   judged_value=None if pd.isna(r.worst_value) else float(r.worst_value),
                                   judged_at=now_iso())
            for r in ds.itertuples()}


def is_reopened(is_done: bool, judged_status: str | None, judged_value: float | None,
                current_status: str | None, current_value: float | None) -> bool:
    """A closed column is re-opened when a check's data got worse after it was closed."""
    if not is_done or current_status not in ("warn", "fail") or judged_status is None:
        return False
    if STATUS_RANK.get(current_status, 0) > STATUS_RANK.get(judged_status, 0):
        return True
    if judged_value is not None and current_value is not None:
        return abs(current_value - judged_value) / max(abs(judged_value), 1e-12) > REOPEN_REL_CHANGE
    return False


def _days_since(ts: str | None) -> float | None:
    if not ts:
        return None
    t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - t).total_seconds() / 86400, 1)


def build_tracker(variables: list[str], findings: pd.DataFrame, records: dict[str, VariableRecord],
                  checks: list[str], workflow) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (one row per column, one row per column x check)."""
    ds = data_status(findings)
    done_keys = workflow.done_keys()
    all_vars = list(dict.fromkeys(variables + sorted(set(ds["variable"])) + sorted(records)))
    all_vars = [v for v in all_vars if v != "_check"]
    idx = {(r.variable, r.check): r for r in ds.itertuples()}
    long_rows = []
    for v in all_vars:
        rec = records.get(v) or VariableRecord(variable=v)
        is_done = rec.status in done_keys
        for c in checks:
            cur = idx.get((v, c))
            snap = rec.checks.get(c)
            if cur is None and snap is None:
                continue
            cur_status = cur.data_status if cur is not None else None
            cur_value = None if cur is None or pd.isna(cur.worst_value) else float(cur.worst_value)
            long_rows.append({
                "variable": v, "check": c, "data_status": cur_status, "worst_value": cur_value,
                "n_flagged": int(cur.n_flagged) if cur is not None else 0, "status": rec.status, "done": is_done,
                "reopened": bool(snap) and is_reopened(is_done, snap.judged_status, snap.judged_value, cur_status, cur_value),
            })
    long = pd.DataFrame(long_rows, columns=["variable", "check", "data_status", "worst_value", "n_flagged",
                                            "status", "done", "reopened"])
    rows = []
    for v in all_vars:
        rec = records.get(v) or VariableRecord(variable=v)
        sub = long[long["variable"] == v]
        reopened = bool(sub["reopened"].any())
        stage = workflow.stage(rec.status)
        rows.append({
            "variable": v, "label": rec.label, "status": rec.status, "stage": workflow.label(rec.status, with_role=False),
            "waiting_on": workflow.waiting_on(rec.status), "assignee": rec.assignees.get(stage.role) if stage.role else None,
            **{f"{role}_assignee": rec.assignees.get(role) for role in workflow.roles},
            "data_status": worst(sub["data_status"].dropna()), "n_flagged": int(sub["n_flagged"].sum()),
            "reopened": reopened, "done": rec.status in done_keys and not reopened,
            "days_in_status": _days_since(rec.status_since), "n_steps": len(rec.preprocessing),
            "n_notes": len(rec.notes), "last_note": rec.notes[-1].text[:120] if rec.notes else None,
            "updated_at": rec.updated_at, "updated_by": rec.updated_by,
        })
    return pd.DataFrame(rows), long
