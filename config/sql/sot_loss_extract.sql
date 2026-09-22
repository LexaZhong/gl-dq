-- ============================================================================
-- SOURCE OF TRUTH: allocated loss and claim count, read from a local extract of the study.
-- Used by the `parquet` profile: check_overrides.loss_recon.sot_query
--
-- Passthrough, for the same reason as sot_premium_extract.sql: the extract is already the output
-- of sot_loss_prod.sql (src x loss_yr, with allocation and claim_cnt), not the raw study table.
--
-- {{ sot_loss_table }} is the `sot_loss` view (DQ_PARQUET_SOT_LOSS: .parquet or .csv).
-- ============================================================================

SELECT *
FROM {{ sot_loss_table }}
