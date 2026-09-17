# Databricks notebook source
# MAGIC %md
# MAGIC # gl-dq in your workspace
# MAGIC Runs the data quality checks on `gl_master` using **this cluster's Spark session** - no SQL
# MAGIC warehouse and no app deployment needed.
# MAGIC
# MAGIC **Setup:** clone `https://github.com/LexaZhong/gl-dq` into your workspace
# MAGIC (Workspace → Create → Git folder), open this notebook from `notebooks/`, attach a cluster
# MAGIC (DBR 14+), fill in the widgets and Run All.
# MAGIC
# MAGIC The interactive dashboard itself is Streamlit and cannot render in a notebook - deploy it as a
# MAGIC Databricks App (see README) or run it locally against this workspace.

# COMMAND ----------
# MAGIC %pip install pydantic>=2.5 jinja2>=3.1 pyyaml>=6.0
# MAGIC %restart_python

# COMMAND ----------
import os
import sys

dbutils.widgets.text("catalog", "na_actuarial_explore")  # noqa: F821
dbutils.widgets.text("schema", "consd_sb_actuarial_sandbox")  # noqa: F821
dbutils.widgets.text("table", "", "Full table name (blank = <catalog>.<schema>.gl_master)")  # noqa: F821
dbutils.widgets.text("sot_premium_table", "", "Premium source of truth (optional)")  # noqa: F821
dbutils.widgets.text("sot_loss_table", "", "Loss source of truth (optional)")  # noqa: F821
dbutils.widgets.text("volume_dir", "/Volumes/na_combined_explore_rfnd-risk_cohort/risk-cohort-volume/GL/gl_master_cleaning", "Volume folder for statuses, notes and run history")  # noqa: F821
dbutils.widgets.dropdown("create_volume", "no", ["no", "yes"], "Create the gl_dq volume if missing")  # noqa: F821

catalog = dbutils.widgets.get("catalog")  # noqa: F821
schema = dbutils.widgets.get("schema")  # noqa: F821
os.environ["DQ_PROFILE"] = "workspace"
os.environ["DQ_CATALOG"], os.environ["DQ_SCHEMA"] = catalog, schema
for widget, env in [("table", "DQ_TABLE"), ("sot_premium_table", "DQ_SOT_PREMIUM_TABLE"),
                    ("sot_loss_table", "DQ_SOT_LOSS_TABLE"), ("volume_dir", "DQ_VOLUME_DIR")]:
    value = dbutils.widgets.get(widget)  # noqa: F821
    if value:
        os.environ[env] = value

# repo root = the folder holding config/ and src/, i.e. the parent of notebooks/
notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
ROOT = os.path.dirname(os.path.dirname("/Workspace" + notebook))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["DQ_CONFIG_DIR"] = os.path.join(ROOT, "config")
# the workspace profile inherits table, measures, source-of-truth queries and check overrides from prod
from gl_dq.core.config import load_project  # noqa: E402

p = load_project("workspace")
print(f"repo:      {ROOT}")
print(f"table:     {p.table}")
print(f"premium SOT: {p.sql_vars['sot_premium_table']}")
print(f"loss SOT:    {p.sql_vars['sot_loss_table']}  (study window {p.sql_vars['study_from']} .. {p.sql_vars['study_to']})")
print(f"knowledge: {p.knowledge_dir}")
print(f"findings:  {p.results.path if p.results.type == 'parquet' else p.results.table}")
print(f"reconciliation covers: {p.check_overrides.get('premium_recon', {}).get('where', 'all sources')}")

# COMMAND ----------
# Knowledge (statuses + notes) must outlive the cluster, so it lives in a UC volume.
if dbutils.widgets.get("create_volume") == "yes":  # noqa: F821
    # only needed if you do not already have a volume; the default path is an existing one
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.gl_dq")  # noqa: F821
    print("volume ready:", f"/Volumes/{catalog}/{schema}/gl_dq")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Preflight
# MAGIC Does the config match the real table, and do the pricing-study queries work?
# MAGIC The source of truth currently covers **BMQ and CMQ only** - BOP is filtered out of both sides
# MAGIC until a BOP study exists.

# COMMAND ----------
sys.path.insert(0, os.path.join(ROOT, "jobs"))
import check_setup  # noqa: E402

try:
    check_setup.main(["--profile", "workspace"])
except SystemExit as e:
    print(f"\nPreflight found problems (exit {e.code}). Fix them in config/ before going further.")

# COMMAND ----------
import validate_sot  # noqa: E402

try:
    validate_sot.main(["--profile", "workspace", "--check", "both"])
except SystemExit as e:
    print(f"\nSource-of-truth problems (exit {e.code}): fix config/sql/sot_*_prod.sql or the dim_map.")

# COMMAND ----------
# MAGIC %md ## 2. Run every check and store the findings

# COMMAND ----------
from gl_dq.core.context import load_context  # noqa: E402
from gl_dq.runner import refresh  # noqa: E402

ctx = load_context("workspace")
run_id, findings, errors = refresh(ctx)
display(findings[findings.status.isin(["warn", "fail"])])  # noqa: F821

# COMMAND ----------
# MAGIC %md ## 3. Portfolio summary

# COMMAND ----------
from gl_dq.summary import summarize  # noqa: E402

display(summarize(ctx, ["src", "covg_type_desc"]))  # noqa: F821

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Set a column's review status from here (optional)
# MAGIC The dashboard is the normal way to do this; a notebook works when you are scripting.

# COMMAND ----------
# from gl_dq.core.knowledge import Note
# from gl_dq.tracker import snapshots_for
# ctx.knowledge.update("expn_bs", "you@company.com", workflow=ctx.workflow, status="investigating",
#                      assignees={"ds": "you@company.com"},
#                      note=Note(author="you@company.com", text="UNK is an unmapped legacy base", src="BMQ"),
#                      snapshots=snapshots_for("expn_bs", ctx.results.latest()))
