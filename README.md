# gl-dq: GL master data cleaning tracker

A config-driven Streamlit dashboard that summarizes `gl_master` (sources `BOP`, `BMQ`, `CMQ`) and tracks its cleanup for actuaries, business leads, data engineers and data scientists. Every check writes
findings in one common format. Every variable has a review status and actuary notes that are
stored as YAML, so the knowledge carries over to other projects.

| Page | What it answers |
|---|---|
| 📋 Portfolio summary (front page) | What is in the table: record count, policy count (distinct `pol_num` + `pol_eff_dt` + `pol_exp_dt`) and written premium, by `src` × `covg_type_desc` or any other level, with data tables and CSV export |
| 🧭 Cleaning tracker | Workflow board (columns per stage and who they're waiting on), closed vs to go, progress by check, bulk assign, activity, run history |
| 🧰 Preprocessing | Recommended preprocessing steps for columns handled in modeling; exports `preprocessing_spec.yaml` |
| 📚 Knowledge base | Search notes, see edit history, export a Markdown data dictionary |
| 🔑 Key uniqueness | Is the key unique per source and across the whole table? Greedy key suggestion |
| 🔲 Missing rate | Missing share per variable × level, with per-variable thresholds, sentinels and `applies_when` |
| 📏 Business rules | SQL validity rules (date order, event inside policy period, deductible exclusivity…) |
| 📊 Distributions | User-chosen variables and levels, percentile bins (preset or custom), log transforms, PSI, outliers, new categories |
| 💰 Premium reconciliation | `tot_wrtn_prm_amt` vs the pricing-study source of truth by src × coverage (grain is configurable) |
| 📉 Loss summary | `allocation` and claim count by loss year vs source of truth; severity, frequency (per exposure base) and loss ratio by segment |
| 📐 Exposure summary | `expo_amt` by `expn_bs`, premium per exposure, negative or zero exposure, classes on more than one base |

Every check page has ⚙️ **Settings** (Apply for your session, or 💾 **Save to config** for everyone), the
findings table, the SQL it ran, and a 📝 **Status & notes** panel for the selected variable.

## Review workflow (`config/workflow.yaml`)
Each column has one stage, a named assignee per role, and a timestamped status history:

```
⚪ Not started → 🔎 DS investigating → 🧮 Confirming with actuary → 🛠️ With DE for fix
   → 🧪 DS validating fix → [✍️ Actuary sign-off, if require_actuary_signoff] → ✅ Resolved
Other outcomes: ☑️ No issue / accepted as is · ⚙️ Preprocess in modeling · ⛔ Won't fix
```
The notes panel has a **➡ Move to next stage** button. Choosing **⚙️ Preprocess in modeling** opens the
**recommended preprocessing** editor: ordered steps such as impute, map values, cap, unit fix, log, group rare,
bin, exclude rows, derive or custom, each with parameters, sources and a rationale. The steps are exported as
`preprocessing_spec.yaml`. Stages, roles and step types are all configurable in `workflow.yaml`. When a column is
closed, its current data status is saved, so if a later refresh makes the data worse it is flagged 🔁 re-opened.

## Quick start (local, synthetic data)
```bash
python3 -m venv .venv --system-site-packages && .venv/bin/pip install -e ".[local,dev]"
.venv/bin/python synthetic/generate.py --out data/                 # data with injected issues
.venv/bin/python synthetic/generate.py --no-inject --out data/clean/
.venv/bin/python jobs/refresh.py --profile synthetic                # stores a run
DQ_PROFILE=synthetic .venv/bin/python -m streamlit run app/app.py
.venv/bin/python -m pytest -q                                       # 60+ tests, ~10s
```
`synthetic/injected_issues.yaml` lists the 18 data problems that were planted on purpose. The tests check that
each one is flagged and that the clean dataset flags nothing.

## How it fits together
```
config/profiles/<profile>.yaml   table, backend, measures, derived columns (pol_yr, loss_yr), storage locations
config/checks/<check>.yaml       per-check settings (live copy in the UC Volume on Databricks)
config/sql/sot_*.sql             source-of-truth queries  ← fill in the pricing-study SQL here
src/gl_dq/core/                  config, db (DuckDB | Databricks SQL), schema whitelist, storage (local | Volume),
                                 knowledge store, results store, registry
src/gl_dq/checks/<check>.py      one module per page: Config + run() + settings_ui() + render()
src/gl_dq/sql/*.sql.j2           SQL templates (all aggregation happens in the warehouse)
src/gl_dq/summary.py             portfolio summary (records, policy terms, premium) at any level
src/gl_dq/tracker.py             review progress + "re-opened by data" logic
src/gl_dq/ui/                    Streamlit frame, notes panel, chart theme
jobs/refresh.py                  runs all checks and appends findings (parquet locally, Delta on Databricks)
.claude/skills/                  Claude Code skills (below)
```
**Safety:** every column reference is checked against the table schema or configured derived columns.
Values typed in the UI become escaped literals. Raw SQL predicates are accepted only from YAML config, which
is reviewed like code.

**Knowledge store:** `variables/<var>.yaml` (current state) plus `history/<var>/<ts>_<user>.yaml` (one
file per change). Saves use an optimistic version check, so if two people edit the same variable at once,
the second person gets a warning instead of silently overwriting the first person's edit.

## Claude Code skills (`.claude/skills/`)
| Skill | Use it to |
|---|---|
| `gl-dq-add-module` | add a new page from a plain-English request (class + SQL + YAML + tests, verified) |
| `gl-dq-configure` | add or remove variables, thresholds, levels, bins, tolerances, rules, workflow stages and preprocessing step types |
| `gl-dq-investigate` | drill into a flagged finding and save a draft note for an actuary to confirm |
| `gl-dq-knowledge` | search notes, report what's waiting on whom, export the dictionary / preprocessing spec, carry notes to a new project |
| `gl-dq-new-project` | point the framework at another master table |
| `gl-dq-refresh` | validate, deploy, run the refresh job, and report what changed |

They load automatically when Claude Code is opened in this repo. To share them more widely, copy them to
`~/.claude/skills/` or package them as a plugin.

## Deploy to Databricks
1. Fill in `config/sql/sot_premium.sql` and `config/sql/sot_loss.sql`.
2. `databricks bundle deploy -t dev --var warehouse_id=<id> --var catalog=<cat> --var schema=<schema>`
3. `python jobs/seed_volume.py --catalog <cat> --schema <schema>`, then adjust column names in any SQL
   predicates in the Volume YAML if gl_master's columns differ from the synthetic data.
4. Grants for the app's service principal (if the bundle's `uc_securable` resource is not available):
   ```sql
   GRANT USE CATALOG ON CATALOG <cat> TO `<app-sp>`;
   GRANT USE SCHEMA, CREATE TABLE ON SCHEMA <cat>.<schema> TO `<app-sp>`;
   GRANT SELECT ON TABLE <cat>.<schema>.gl_master TO `<app-sp>`;   -- plus the SOT tables
   GRANT READ VOLUME, WRITE VOLUME ON VOLUME <cat>.<schema>.gl_dq TO `<app-sp>`;
   ```
5. `databricks bundle run gl_dq_refresh -t dev`, then `databricks bundle run gl_dq_app -t dev`.

To try it on Databricks with synthetic data first: upload the parquet files and run
`synthetic/load_to_delta.py`, then set `DQ_TABLE`, `DQ_SOT_PREMIUM_TABLE` and `DQ_SOT_LOSS_TABLE` on the app.

The Databricks path is not yet tested against a live workspace. The SQL is rendered with the Databricks
dialect in `tests/test_databricks_sql.py`, but the bundle, grants and app auth still need a first real deploy.
