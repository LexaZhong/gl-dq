-- ============================================================================
-- SOURCE OF TRUTH: allocated loss and claim count, pricing study (BMQ and CMQ only).
-- Used by the prod profile: check_overrides.loss_recon.sot_query
--
-- Grain: src x loss_yr, where loss_yr = YEAR(EVT_DT) - the same event-year basis the
-- pipeline uses (loss_yr = year(evt_dt)). The policy-effective-date window below is a
-- POLICY filter, not a loss filter: it keeps the same policy population as the premium
-- source of truth.
--
-- claim_cnt vs COUNT(DISTINCT OCUR_ID): the pipeline sums an allocated claim count per
-- row, while the study counts distinct occurrences. If one claim touches several coverages
-- or locations these will not agree; compare them once and, if they measure different
-- things, either change sot_claim_count_col or widen tolerance_claims and say so in a note.
--
-- BOP: not covered yet (see the premium file).
--
-- Source: cimm_csm.loss_transx_seg_enriched_2026q2 (profile sql_vars -> sot_loss_table).
-- The policy filter matches policy TERMS (POLHLDR_CONTR_ID + CONTR_EFF_DT against pol_num +
-- pol_eff_dt), so those columns must exist in the loss table too; if they are named differently
-- there, adjust the WHERE clause below.
--
-- {{ table }} is the pipeline table AFTER the global filters in config/filters.yaml, so a filter
-- restricts both sides of the reconciliation; {{ raw_table }} is the unfiltered table.
-- ============================================================================

SELECT
  CASE WHEN BMQ_IND = 'Y' THEN 'BMQ' ELSE 'CMQ' END AS src,
  YEAR(EVT_DT)                                      AS loss_yr,
  SUM(RLA)                                          AS allocation,
  COUNT(DISTINCT OCUR_ID)                           AS claim_cnt
FROM {{ sot_loss_table }}
WHERE LOB = 'GL'
  AND EVT_DT IS NOT NULL
  AND OCUR_ID IS NOT NULL
  AND CONTR_EFF_DT >= DATE '{{ study_from }}'
  AND CONTR_EFF_DT <= DATE '{{ study_to }}'
  AND (POLHLDR_CONTR_ID, CONTR_EFF_DT) IN (
        SELECT DISTINCT pol_num, pol_eff_dt FROM {{ table }} WHERE src IN ('BMQ', 'CMQ')
      )
GROUP BY 1, 2

-- ---------------------------------------------------------------------------
-- Sanity checks to run once in the SQL editor:
--   -- claims with no event date (dropped above, so they would be a silent difference)
--   SELECT COUNT(*) FROM {{ sot_loss_table }}
--   WHERE LOB = 'GL' AND OCUR_ID IS NOT NULL AND EVT_DT IS NULL
--
--   -- is RLA one row per occurrence, or already an allocation across segments?
--   SELECT OCUR_ID, COUNT(*) AS rows, SUM(RLA) AS rla
--   FROM {{ sot_loss_table }} WHERE LOB = 'GL' GROUP BY 1 HAVING COUNT(*) > 1 LIMIT 20
-- ---------------------------------------------------------------------------
