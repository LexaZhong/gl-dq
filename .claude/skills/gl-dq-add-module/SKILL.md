---
name: gl-dq-add-module
description: Add a complete new check module (page) to the gl_dq data cleaning dashboard from a plain-English request, e.g. "add a module that checks written premium per policy by state" or "add a page comparing exposure per claim by expn_bs". Creates the Check class, SQL template, YAML config and tests, then verifies them against the data. Use whenever someone wants a new check, page, summary or validation in the tracker.
---

# Add a gl_dq check module

A module = **one Python file + one SQL template + one YAML config + one test**. The registry
auto-discovers it; the app shows it as a page with the shared frame (settings, findings,
SQL, 📝 notes panel). You never edit `app/app.py`.

## 0. Understand the request (ask only if truly ambiguous)
Pin down, in one short list:
- **Question the module answers** and who reads it.
- **Measures** (sums/ratios) and **dimensions/levels** (default `[src]`).
- **What is a problem?** → metric + warn/fail thresholds (a module with no thresholds is fine: emit `info` findings).
- **Variable(s)** the findings should be attached to (drives the tracker and notes).

## 1. Resolve columns — never guess
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from gl_dq.core.context import load_context
ctx = load_context()           # DQ_PROFILE, default synthetic
print(ctx.project.measures); print(ctx.project.derived_columns.keys())
print({c: t for c, t in ctx.schema.columns.items()})
print('existing checks:', list(ctx.checks))"
```
Map the user's words to real columns or `project.measures.*` (prefer measures, so the module
works on every profile, e.g. `measures.claim_count` → `claim_alloc`, so a renamed column is a one-line profile change).
Derived dims (`pol_yr`, `loss_yr`) come from `derived_columns`. If a needed column doesn't exist,
stop and tell the user; propose a derived column in the profile instead of raw SQL in code.
If an existing module already answers it, suggest configuring that one (gl-dq-configure) instead.

## 2. Create the files (copy the templates in this skill's `templates/` dir)
| File | From template | Notes |
|---|---|---|
| `src/gl_dq/checks/<name>.py` | `check_module.py.tmpl` | snake_case `<name>`; set `title`, `icon` (one emoji), `description`, `default_order` (between existing orders) |
| `src/gl_dq/sql/<name>.sql.j2` | `check_sql.sql.j2.tmpl` | aggregate in SQL; never pull raw rows |
| `config/checks/<name>.yaml` | `config.yaml.tmpl` | every field of `Config`, with the defaults |
| `tests/test_<name>.py` | `test_module.py.tmpl` | runs on synthetic data |

### Non-negotiable checklist (the reviewer will check each)
- [ ] Every column goes through `self.schema.ref()` / `sel()` in templates, and `self.schema.validate([...])` runs first. No f-string column names. Literals use `lit()`.
- [ ] Config-authored SQL predicates (like `applies_when`) are allowed **only from YAML**, never from free-text UI inputs.
- [ ] `run()` is pure (no streamlit import at module top); UI imports go inside `settings_ui` / `render`.
- [ ] Findings use the shared schema: `variable, item, segment (segment_key), metric, value, threshold, status (grade()), detail`. Status ∈ pass/info/warn/fail.
- [ ] Levels are user-selectable: `settings_ui` offers `self.segment_options()` in a multiselect and returns a modified **copy** of cfg; widget keys are prefixed with the module name.
- [ ] `render()` uses `gl_dq.ui.components.status_table` for tables and `gl_dq.ui.theme` for charts: `series_encoding()` for series colors (≤ 8 series, else small multiples), `line()` instead of `px.line` (SVG, never WebGL), `SEQ_SCALE` for heatmaps, `DIVERGING` for signed differences, `style(fig, height, title)`. No dual axes; different exposure bases never share an axis.
- [ ] Expensive interactive recomputation goes through `state.cached_method(self.name, self.cfg, "<method>", **kwargs)`.
- [ ] SQL portable across DuckDB and Databricks: use `dialect.percentiles/row/distinct_list` helpers, `CAST(x AS STRING)`, `GROUP BY` ordinals, no `QUALIFY`/`FILTER`/`::` casts.

## 3. Verify (all must pass before you report back)
```bash
.venv/bin/python -m pytest -q tests/test_<name>.py tests/test_databricks_sql.py tests/test_sql_and_config.py
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from gl_dq.core.context import load_context
r = load_context().make_check('<name>').run()
print(r.findings.status.value_counts()); print(list(r.tables)); [print(s) for s in r.sql.values()]"
.venv/bin/python -m pytest -q tests/test_app.py -k <name>
```
Also add `<name>` to `PAGES` in `tests/test_app.py` so the page smoke test covers it, and run
it on the clean data (`DQ_DUCKDB_PATH=data/clean/gl_synth.duckdb`) to make sure it does
not flag clean data unless that is intended.

## 4. Optional deploy
If the user wants it live: `databricks bundle deploy -t <target> --var warehouse_id=<id>` and
upload the YAML to the Volume (`python jobs/seed_volume.py --catalog <c> --schema <s>`; it
skips existing files, so new configs are added without touching edited ones).

## 5. Report
Summarise: page name, what it flags (thresholds), files created, test results, and one
screenshot-worthy finding from the synthetic data. Do not commit unless asked.
