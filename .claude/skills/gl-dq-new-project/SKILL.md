---
name: gl-dq-new-project
description: Set up the gl_dq cleaning tracker for another master table (e.g. property_master, auto_master) or another environment. Creates a profile, seeds check configs tailored to the table's columns, and wires the SOT queries and knowledge store. Use for "use this dashboard for property", "new line of business", "point it at the UAT table".
---

# New project / table

1. **Inspect the table**: `DESCRIBE TABLE <catalog>.<schema>.<table>` (or `ctx.db.describe`). Identify:
   source column + values, policy key, date columns for derived years, premium / loss / claim count /
   exposure / exposure-base columns. Ask the user for anything not obvious. Never guess measures.
2. **Create `config/profiles/<project>.yaml`** by copying `prod.yaml` (Databricks) or `synthetic.yaml`
   (local) and changing: `name`, `table`, `src_col`, `sources`, `derived_columns`, `measures`,
   `policy_key`, `segment_candidates`, `sql_vars`, and a **separate** `config_dir` / `knowledge_dir` /
   `results.table`, for example `/Volumes/<c>/<s>/<project>_dq/...`, so the projects never share state.
3. **Seed check configs** into the new config dir: start from `config/checks/*.yaml`, then
   - key_uniqueness: empty `candidate_keys`, run `suggest_key` per source and propose keys to the user;
   - missing_rate: add `applies_when` for structurally null columns (look at null rates by source first);
   - distribution: pick 4–8 key variables (premium, exposure, loss, class, state) with sensible bins/log;
   - business_rules: re-point column names; drop rules that don't apply;
   - SOT queries: write placeholders with TODO comments in `sql/`.
   Validate with `ctx.check_config(name)` for every check.
4. **Carry over knowledge** (optional): use gl-dq-knowledge "Reuse for another project".
5. **Deploy** (Databricks): add a bundle target or a second app resource with `DQ_PROFILE=<project>`;
   run `python jobs/seed_volume.py` with the new volume; run `jobs/refresh.py --profile <project>` once.
6. **Verify**: `DQ_PROFILE=<project> .venv/bin/python jobs/refresh.py` completes with no errors, and
   the app opens with `DQ_PROFILE=<project> streamlit run app/app.py`.
