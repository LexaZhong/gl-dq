# Databricks notebook source
# MAGIC %md
# MAGIC ## Load synthetic gl_master into Delta
# MAGIC 1. Locally: `python synthetic/generate.py --out data/`
# MAGIC 2. Upload: `databricks fs cp -r data/ dbfs:/Volumes/<catalog>/<schema>/gl_dq/synthetic/ --overwrite` (parquet files only are needed)
# MAGIC 3. Run this notebook, then point the app at it with env `DQ_TABLE=<catalog>.<schema>.gl_master_synth`,
# MAGIC    `DQ_SOT_PREMIUM_TABLE=...sot_premium_synth`, `DQ_SOT_LOSS_TABLE=...sot_loss_synth`.

# COMMAND ----------
dbutils.widgets.text("catalog", "main")  # noqa: F821
dbutils.widgets.text("schema", "pricing")  # noqa: F821
catalog = dbutils.widgets.get("catalog")  # noqa: F821
schema = dbutils.widgets.get("schema")  # noqa: F821
src = f"/Volumes/{catalog}/{schema}/gl_dq/synthetic"

# COMMAND ----------
for table in ["gl_master_synth", "sot_premium_synth", "sot_loss_synth"]:
    df = spark.read.parquet(f"{src}/{table}.parquet")  # noqa: F821
    for c in [c for c in df.columns if c.endswith("_dt")]:
        df = df.withColumn(c, df[c].cast("date"))
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{table}")
    print(table, spark.table(f"{catalog}.{schema}.{table}").count())  # noqa: F821
