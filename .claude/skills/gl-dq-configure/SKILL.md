---
name: gl-dq-configure
description: Change what an existing gl_dq check does, without new code. Add or remove variables to profile, change missing-rate thresholds per variable, set levels or segments (src, pol_yr, covg_type_desc…), percentile bins and log scales, reconciliation tolerances and grain, business rules, candidate keys, enabling or disabling a page, the review workflow stages and preprocessing step types. Use for "track X by Y", "loosen/tighten threshold", "add a rule", "add a sign-off stage".
---

# Configure gl_dq checks

Config lives in YAML: `config/checks/<check>.yaml` locally; on Databricks the **live copy is in
the UC Volume** `/Volumes/<catalog>/<schema>/gl_dq/config/checks/`. The dashboard's
"💾 Save to config" writes the same files. The profile (`config/profiles/<profile>.yaml`) holds
table, measures, derived columns and per-profile `check_overrides` (deep-merged).

## Steps
1. **Load the current config and schema** (profile from `$DQ_PROFILE`, default synthetic):
   ```bash
   .venv/bin/python -c "
   import sys; sys.path.insert(0,'src')
   from gl_dq.core.context import load_context
   ctx = load_context(); name='<check>'
   print(ctx.check_config(name).model_dump_json(indent=2))
   print(ctx.checks[name].Config.model_json_schema()['properties'].keys())
   print(ctx.schema.names())"
   ```
   The pydantic `Config` of each check (`src/gl_dq/checks/<check>.py`) is the source of truth for
   field names (`extra="forbid"` rejects typos).
2. **Validate names** against `ctx.schema.names()` (table columns + derived columns). Anything
   used as a level must also be in the profile's `segment_candidates` to appear in the UI.
3. **Edit the YAML** minimally, keeping comments. Cheat sheet:
   | Want | Check | Field |
   |---|---|---|
   | per-variable missing threshold / level / sentinel | missing_rate | `variables.<var>: {warn, fail, group_by, treat_as_missing, applies_when}` |
   | profile a variable, bins, log scale, PSI | distribution | `variables: - {name, type, group_by, percentile_bins: 20 or [..], log_scale: {method: none/log1p/log10/signed_log, y}, hist_bins, psi: {across}, allow_negative, outlier_ratio, top_n}` |
   | key per source | key_uniqueness | `candidate_keys.<SRC or _all>` |
   | validity rule | business_rules | `rules: - {id, description, variables, violation, applies_when, warn, fail}` |
   | value consistency / plausibility | value_checks | `categorical.<col>: {max_values, warn_rows, fail_rows}`; `numeric.<col>: {min, max, allow_negative, allow_zero, discrete, top_value_warn, compare_medians}` |
   | standardize a field / map its values | `transforms.json` | `{apply_to_dashboard, transforms: [{column, standardize: {trim, case, zero_pad, cast}, mapping, unmapped (keep/other/null), description, author, updated}]}`; order is fixed (trim, case, pad, map, cast) and the mapping is written against the STANDARDIZED value |
   | binning schemes for a numeric variable | `binnings.yaml` | `binnings: - {variable, name, method (quantile/equal_width/custom/categorical), bins, cuts, description, author, created}`; cuts are frozen at save time so an experiment stays reproducible - never re-derive them |
   | modelling target, one-ways, interactions | target_analysis | `variables`, `default_target` (frequency/severity/loss_cost/loss_ratio), `max_levels`, `min_claims`, `full_credibility`, `spread_warn`, `interaction_warn`, `where` |
   | rating segments, credibility | segment_mix | `dimensions`, `full_credibility` (1082 claims), `thresholds: {large_share, material_share, z_target, thin_premium_warn, thin_premium_fail}` |
   | exclude rows from EVERY page and the refresh job | `filters.yaml` | `filters: - {key, label, description, enabled, profiles, column + op (in/not_in/is_null/not_null/eq/ne/gt/gte/lt/lte) + values, or expr}` — a rule KEEPS rows its predicate is true for (nulls are excluded unless the rule says otherwise); `{{ raw_table }}` in `expr` is the unfiltered table. Ships all-off; turning one on changes every number, so say what it removes (`state.filter_impact`) when you do |
   | disable a page / change page order | any | `enabled: false`, `order` (lower = higher in the menu) |
   | review workflow: stages, roles, actuary sign-off | `workflow.yaml` | `require_actuary_signoff: true`; `stages: - {key, label, icon, role, done, in_flow, optional, requires_preprocessing, description}` (first stage = not started; never rename keys already used in records) |
   | preprocessing step types | `workflow.yaml` | `preprocessing_ops.<op>: {label, description, params: {<name>: {type: number/text/select/bool/list/mapping, options, default, help}}}` |
   SQL predicates (`applies_when`, `violation`) are trusted config: keep them simple boolean
   expressions over whitelisted columns, and portable across DuckDB and Databricks SQL.
4. **Verify**: `.venv/bin/python -m pytest -q tests/test_sql_and_config.py tests/test_knowledge_tracker.py` (validates every YAML and the workflow), then
   run the check once and show the user what changed (counts of pass/warn/fail before and after).
5. **Publish (Databricks)**: upload changed files to the Volume, e.g.
   `databricks fs cp config/checks/<check>.yaml dbfs:/Volumes/<c>/<s>/gl_dq/config/checks/<check>.yaml --overwrite`.
   Confirm with the user before overwriting a Volume file, because people may have saved changes from the app. Diff first:
   `databricks fs cat dbfs:/Volumes/<c>/<s>/gl_dq/config/checks/<check>.yaml | diff - config/checks/<check>.yaml`.
