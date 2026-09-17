"""Run every enabled check and store the findings as one run."""
from __future__ import annotations

import time
import traceback

import pandas as pd

from gl_dq.core.results import FINDING_COLS, new_run_id


def run_all(ctx, checks: list[str] | None = None, log=print) -> tuple[pd.DataFrame, dict]:
    frames, errors = [], {}
    for name in checks or ctx.enabled_checks():
        t0 = time.time()
        try:
            res = ctx.make_check(name).run()
            frames.append(res.findings)
            log(f"  {name:<18} {len(res.findings):>5} findings  {time.time() - t0:5.1f}s")
        except Exception as e:  # keep going; the error is recorded as a failed finding
            errors[name] = traceback.format_exc()
            log(f"  {name:<18} ERROR {e}")
            frames.append(pd.DataFrame([dict(check=name, variable="_check", item="error", segment="ALL",
                                             metric="error", value=None, threshold=None, status="fail",
                                             detail=str(e)[:500])], columns=FINDING_COLS))
    findings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FINDING_COLS)
    return findings, errors


def refresh(ctx, checks: list[str] | None = None, log=print) -> tuple[str, pd.DataFrame, dict]:
    run_id, run_ts = new_run_id()
    log(f"refresh {run_id} · profile={ctx.profile} · table={ctx.project.table}")
    findings, errors = run_all(ctx, checks, log)
    ctx.results.append(findings, run_id, run_ts, ctx.profile)
    counts = findings["status"].value_counts().to_dict()
    log(f"stored {len(findings)} findings: {counts}")
    return run_id, findings, errors
