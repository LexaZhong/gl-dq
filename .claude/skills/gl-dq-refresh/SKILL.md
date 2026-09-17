---
name: gl-dq-refresh
description: Refresh the gl_dq tracker. Validate configs, run all checks (locally or by triggering the Databricks job), deploy the bundle/app when code or config changed, and report what changed since the previous run (newly flagged / cleared findings). Use for "refresh the dashboard", "rerun the checks", "deploy the latest", "what changed since last run".
---

# Refresh & deploy

## Local (synthetic or any duckdb profile)
```bash
.venv/bin/python -m pytest -q                         # configs + SQL + pages still OK
.venv/bin/python jobs/refresh.py --profile ${DQ_PROFILE:-synthetic}
```

## Databricks
1. Confirm the target and warehouse with the user (deploying changes a shared app).
2. `databricks bundle validate -t <target> --var warehouse_id=<id>`
3. If code or bundle config changed: `databricks bundle deploy -t <target> --var warehouse_id=<id>`, then
   `databricks bundle run gl_dq_app -t <target>` to restart the app.
4. New config files → `python jobs/seed_volume.py --catalog <c> --schema <s>` (skips existing files).
5. Trigger the job: `databricks bundle run gl_dq_refresh -t <target>` and watch it to completion.

## Report what changed
```python
import sys; sys.path.insert(0, 'src')
from gl_dq.core.context import load_context
ctx = load_context()
cur, prev = ctx.results.latest(0), ctx.results.latest(1)
key = ["check", "variable", "item", "segment", "metric"]
flag = lambda d: d[d.status.isin(["warn", "fail"])][key].astype(str)
new = flag(cur).merge(flag(prev), how="left", indicator=True).query("_merge=='left_only'")
cleared = flag(prev).merge(flag(cur), how="left", indicator=True).query("_merge=='left_only'")
```
Summarize: run id/time, counts by status, newly flagged (grouped by check), cleared, any
check errors (`variable == '_check'`), and variables re-opened by data (`gl_dq.tracker.build_tracker`).
