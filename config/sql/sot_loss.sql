-- ============================================================================
-- SOURCE OF TRUTH: allocated loss and claim count from the pricing study.
--
-- Contract (checked by `python jobs/validate_sot.py --check loss_recon`):
--   * one column per dimension in loss_recon.yaml -> segments + time_dim
--     (default: src, loss_yr), named like the pipeline column or mapped in dim_map
--   * two measure columns named in loss_recon.yaml -> sot_loss_col and sot_claim_count_col
--     (default: allocation, claim_alloc)
--   * loss_yr must be the same basis as the pipeline: year(evt_dt), i.e. accident/event year
--   * any grain at least as fine as those dimensions - rows are summed
--
-- Jinja: {{ sot_loss_table }} and other profile `sql_vars`; {{ table }} = the pipeline table.
--
-- TODO(prod): replace the query below with the pricing study.
-- ============================================================================

SELECT src, covg_type_desc, loss_yr, allocation, claim_alloc
FROM {{ sot_loss_table }}

-- ---------------------------------------------------------------------------
-- Worked examples
--
-- 1. Study at accident-year grain with different names -> loss_recon.yaml:
--        dim_map: {src: source_system}
--        sot_loss_col: incurred_alloc
--        sot_claim_count_col: claim_count
--    SELECT source_system, accident_year AS loss_yr, incurred_alloc, claim_count
--    FROM pricing.study.gl_loss_2026
--
-- 2. Study is valued at a specific date (the pipeline is "as of today"):
--    reconcile the same valuation, or expect a systematic difference.
--    SELECT src, YEAR(loss_date) AS loss_yr, SUM(paid + case_reserve) AS allocation,
--           COUNT(DISTINCT claim_nbr) AS claim_alloc
--    FROM pricing.study.gl_claims
--    WHERE valuation_date = DATE '2026-06-30'
--    GROUP BY 1, 2
--
-- 3. Claim counts only, no loss amounts: keep the loss column but return NULL, and raise
--    loss_recon.yaml -> tolerance_loss so the loss comparison is not read as a break.
-- ---------------------------------------------------------------------------
