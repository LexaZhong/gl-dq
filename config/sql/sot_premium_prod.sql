-- ============================================================================
-- SOURCE OF TRUTH: written premium, pricing study (BMQ and CMQ only).
-- Used by the prod profile: check_overrides.premium_recon.sot_query
--
-- Reconciled at src x pol_yr - the study has no coverage split, so premium_recon
-- dims are [src, pol_yr] in the prod profile, not [src, covg_type_desc].
--
-- BOP: not covered yet. The prod profile therefore filters the pipeline side to
-- BMQ and CMQ (check_overrides.premium_recon.where). Once a BOP source exists,
-- UNION it in below and drop 'BOP' from that filter.
--
-- Note: the POLHLDR_CONTR_ID filter restricts the study to policies that are already
-- in gl_master, so this reconciles AMOUNTS, not completeness. Policies that exist in
-- the study but are missing from gl_master will not show up here - see the
-- "policies in the study but not in gl_master" query at the bottom.
-- ============================================================================

SELECT
  CASE WHEN BMQ_IND = 'Y' THEN 'BMQ' ELSE 'CMQ' END AS src,
  YEAR(CONTR_EFF_DT)                                AS pol_yr,
  SUM(WP_US_CURY_AMT)                               AS wrtn_prm
FROM {{ sot_premium_table }}
WHERE LOB = 'GL'
  AND CONTR_EFF_DT >= DATE '{{ study_from }}'
  AND CONTR_EFF_DT <= DATE '{{ study_to }}'
  AND POLHLDR_CONTR_ID IN (
        SELECT DISTINCT pol_num FROM {{ table }} WHERE src IN ('BMQ', 'CMQ')
      )
GROUP BY 1, 2

-- ---------------------------------------------------------------------------
-- Completeness check, run separately in the SQL editor:
--   SELECT CASE WHEN BMQ_IND = 'Y' THEN 'BMQ' ELSE 'CMQ' END AS src,
--          YEAR(CONTR_EFF_DT) AS pol_yr,
--          COUNT(DISTINCT POLHLDR_CONTR_ID) AS policies_missing_from_gl_master,
--          SUM(WP_US_CURY_AMT) AS premium_missing
--   FROM {{ sot_premium_table }}
--   WHERE LOB = 'GL'
--     AND CONTR_EFF_DT BETWEEN DATE '{{ study_from }}' AND DATE '{{ study_to }}'
--     AND POLHLDR_CONTR_ID NOT IN (SELECT DISTINCT pol_num FROM {{ table }})
--   GROUP BY 1, 2 ORDER BY premium_missing DESC
-- ---------------------------------------------------------------------------
