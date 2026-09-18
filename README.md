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
jobs/check_setup.py              preflight: does the profile match the real table?
jobs/export_extract.py           write a parquet extract of gl_master + the study queries
jobs/validate_sot.py             checks a source-of-truth query and prints a comparison SQL
notebooks/run_in_workspace.py    run the checks from a Databricks notebook (Spark backend)
jobs/seed_volume.py              copies configs + SOT SQL into the UC volume
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

Storage note: findings are parquet files in the volume (`results.type: parquet`), not a Delta table,
so nothing needs CREATE TABLE in the sandbox schema. Switch with `results: {type: delta, table: ...}`.

They load automatically when Claude Code is opened in this repo. To share them more widely, copy them to
`~/.claude/skills/` or package them as a plugin.

## Run it inside your Databricks workspace

Three ways, from least to most permission needed.

**A. Notebook (no warehouse, no app, nothing to deploy)** — runs the checks on the cluster's Spark session:
1. Workspace → Create → **Git folder**, URL `https://github.com/LexaZhong/gl-dq`.
2. Open `notebooks/run_in_workspace.py`, attach a cluster (DBR 14+), fill in the widgets
   (catalog, schema, and the volume path for statuses/notes), Run All.
The `workspace` profile `extends: prod`, so the table, measures, pricing-study queries and check
overrides are inherited — only the backend (Spark instead of a SQL warehouse) and the config/knowledge
locations differ.
It runs preflight, then every check, writes findings to `<catalog>.<schema>.dq_check_results` and displays the
flagged rows and the portfolio summary. Statuses and notes go to a UC volume (`gl_dq`) so they outlive the cluster;
the notebook can create it for you. What you do **not** get here is the dashboard UI: Streamlit cannot render in a
notebook.

**B. Databricks App (the dashboard, shared with the team)** — needs permission to create Apps and a SQL warehouse;
see *Deploy to Databricks* below. The bundle is the easy path, but you can also create the app in the UI
(Compute → Apps → Create app → deploy from a workspace folder) pointing at the Git folder from A, since `app.yaml`
sits at the repo root.

**C. Parquet extract (no warehouse, no Spark, no Delta)** — create it once from a notebook or job:
```bash
python jobs/export_extract.py --profile workspace          # -> <volume>/extract/{gl_master,sot_premium,sot_loss}
python jobs/export_extract.py --profile workspace --sample 200000   # a smaller share
```
then point the dashboard at those files:
```bash
DQ_PROFILE=parquet DQ_PARQUET_TABLE=/Volumes/.../extract/gl_master \
  DQ_PARQUET_SOT_PREMIUM=/Volumes/.../extract/sot_premium \
  DQ_PARQUET_SOT_LOSS=/Volumes/.../extract/sot_loss \
  python jobs/check_setup.py --profile parquet
DQ_PROFILE=parquet DQ_PARQUET_TABLE=... streamlit run app/app.py
```
Each view is a file, a folder or a glob; DuckDB reads them in place (nothing is copied). Column names,
measures, thresholds and the source-of-truth queries are inherited from prod, so an extract is checked
exactly like the table - a test asserts both backends produce identical findings. The study extracts are
optional: without them everything except the two reconciliations still runs. Notebook cell 4 writes the
extract.

**D. From your laptop against the workspace** — the full dashboard, no deployment, only `SELECT` rights:
```bash
export DATABRICKS_HOST=https://<workspace>.azuredatabricks.net DATABRICKS_TOKEN=<pat>
export DATABRICKS_WAREHOUSE_ID=<id> DQ_CATALOG=<cat> DQ_SCHEMA=<schema>
python jobs/check_setup.py --profile prod
DQ_PROFILE=prod DQ_CONFIG_DIR=config DQ_KNOWLEDGE_DIR=data/knowledge_prod streamlit run app/app.py
```

## Deploy to Databricks

**0. Install and authenticate the CLI** (one-off)
```bash
brew install databricks                     # or: pip install databricks-cli
databricks auth login --host https://<your-workspace>.cloud.databricks.com
pip install -e ".[databricks]"              # databricks-sql-connector + sdk for local runs
```

**1. Point it at the real table from your laptop first** — read-only, nothing is deployed, and it
tells you whether the config matches `gl_master` before anything else:
```bash
export DATABRICKS_WAREHOUSE_ID=<sql warehouse id>   # catalog/schema/volume already default to yours
python jobs/check_setup.py --profile prod            # verifies every configured column, source and SOT query
DQ_PROFILE=prod DQ_CONFIG_DIR=config DQ_KNOWLEDGE_DIR=data/knowledge_prod \
  streamlit run app/app.py                           # the whole dashboard, live on gl_master
```
Fix whatever preflight reports in `config/profiles/prod.yaml` (measures, derived columns, policy key,
segment candidates) and in `config/checks/*.yaml` (candidate keys, `applies_when` predicates, business
rules), then fill in `config/sql/sot_premium.sql` and `config/sql/sot_loss.sql` with the pricing-study
queries. Re-run preflight until it is clean.

**1b. Fill in the source-of-truth queries**
`config/sql/sot_premium.sql` and `config/sql/sot_loss.sql` must return the reconciliation dimensions
(named like the pipeline columns, or mapped with `dim_map`) plus the measure columns. Both files carry
the contract and worked examples in their header. Then check them:
```bash
python jobs/validate_sot.py --profile prod --check both              # columns, grain, totals, biggest breaks
python jobs/validate_sot.py --profile prod --check premium_recon --print-sql   # SQL to paste in the SQL editor
```
It names the exact fix when a dimension is missing (add it, drop it from `dims`, or map it), flags nulls
in dimensions and a coarser-grained study, and prints the same numbers the dashboard will show.

**2. Deploy the bundle** (creates the UC volume, the refresh job and the Databricks App)
```bash
databricks bundle validate -t dev --var warehouse_id=<id> --var catalog=<cat> --var schema=<schema>
databricks bundle deploy   -t dev --var warehouse_id=<id> --var catalog=<cat> --var schema=<schema>
python jobs/seed_volume.py --profile prod                      # copies configs + SOT SQL into the volume
```
The volume copy is the live config people edit from the app; `seed_volume.py` never overwrites existing
files unless you pass `--overwrite`.

**Where things are stored** (defaults; move them all with `DQ_VOLUME_DIR`):

| What | Where |
|---|---|
| Table | `na_actuarial_explore.consd_sb_actuarial_sandbox.gl_master` |
| Check configs + SOT SQL | `<volume>/gl_master_cleaning/config` |
| Statuses, notes, preprocessing | `<volume>/gl_master_cleaning/knowledge` |
| Run history (one parquet per run) | `<volume>/gl_master_cleaning/runs` |

`<volume>` is `/Volumes/na_combined_explore_rfnd-risk_cohort/risk-cohort-volume/GL`, i.e. a different
catalog from the table - that is fine, nothing but the volume grant is needed there.

**3. Grant the app's service principal access** (skip if the bundle's `uc_securable` resource worked)
```sql
GRANT USE CATALOG ON CATALOG <cat> TO `<app-sp>`;
GRANT USE SCHEMA, CREATE TABLE ON SCHEMA <cat>.<schema> TO `<app-sp>`;
GRANT SELECT ON TABLE <cat>.<schema>.gl_master TO `<app-sp>`;   -- plus the SOT tables
GRANT READ VOLUME, WRITE VOLUME ON VOLUME <cat>.<schema>.gl_dq TO `<app-sp>`;
```
Find the service principal on the app's page in the workspace (Compute → Apps → gl-dq-tracker).

**4. Run it**
```bash
databricks bundle run gl_dq_refresh -t dev     # first run: writes <cat>.<schema>.dq_check_results
databricks bundle run gl_dq_app -t dev         # starts the app, prints its URL
```
Share the app URL with the team. Anyone who can open it signs in with SSO; their email is recorded as
the author of every note and status change. Unpause the job schedule in `databricks.yml` (it ships
`PAUSED`) once the checks are settled, so the dashboard refreshes on its own.

To try it on Databricks with synthetic data first: upload the parquet files and run
`synthetic/load_to_delta.py`, then set `DQ_TABLE`, `DQ_SOT_PREMIUM_TABLE` and `DQ_SOT_LOSS_TABLE` on the app.

The Databricks path is not yet tested against a live workspace. The SQL is rendered with the Databricks
dialect in `tests/test_databricks_sql.py`, but the bundle, grants and app auth still need a first real deploy.
