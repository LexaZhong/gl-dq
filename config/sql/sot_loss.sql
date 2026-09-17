-- Source of truth: allocated loss and claim count from the pricing study.
-- Must return columns named like the pipeline dimensions (src, loss_yr, optionally covg_type_desc, ...)
-- plus the measure columns set in loss_recon.yaml (sot_loss_col, sot_claim_count_col).
-- TODO(prod): replace with the pricing study query.
SELECT src, covg_type_desc, loss_yr, allocation, claim_alloc
FROM {{ sot_loss_table }}
