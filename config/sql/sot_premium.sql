-- Source of truth: written premium from the pricing study.
-- Must return one row per grain with columns named like the pipeline dimensions
-- (src, covg_type_desc, pol_yr, ...) plus the measure column set in premium_recon.yaml (sot_measure).
-- Jinja variables come from the profile's sql_vars; {{ table }} is the pipeline table.
-- TODO(prod): replace with the pricing study query.
SELECT src, covg_type_desc, pol_yr, wrtn_prm
FROM {{ sot_premium_table }}
