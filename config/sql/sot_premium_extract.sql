-- ============================================================================
-- SOURCE OF TRUTH: written premium, read from a local extract of the pricing study.
-- Used by the `parquet` profile: check_overrides.premium_recon.sot_query
--
-- The extract is ALREADY the output of sot_premium_prod.sql (one row per src x pol_yr, with
-- wrtn_prm) - jobs/export_extract.py writes exactly that, and a download from the SQL editor
-- should be the same query. So this is a passthrough: aggregating it again would double-count
-- nothing, but re-running the study's WHERE clauses would fail, because the raw columns
-- (LOB, BMQ_IND, CONTR_EFF_DT, POLHLDR_CONTR_ID) are not in the extract.
--
-- {{ sot_premium_table }} is the `sot_premium` view: DQ_PARQUET_SOT_PREMIUM points it at a
-- .parquet or .csv file, a folder or a glob.
--
-- If your extract is the RAW study table instead, point sot_query back at sql/sot_premium_prod.sql.
-- ============================================================================

SELECT *
FROM {{ sot_premium_table }}
