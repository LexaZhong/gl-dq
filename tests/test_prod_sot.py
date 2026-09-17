"""The prod source-of-truth queries and the overrides that must match them.

Runs offline against a fake database, so it checks the contract (aliases, dimensions,
filters) without needing the real workspace.
"""
import re
from pathlib import Path

import pytest

from gl_dq.core.config import load_project
from gl_dq.core.context import Context
from gl_dq.core.db import DatabricksDialect
from gl_dq.core.knowledge import KnowledgeStore
from gl_dq.core.registry import discover
from gl_dq.core.results import ParquetResults
from gl_dq.core.schema import TableSchema
from gl_dq.core.storage import LocalStorage
from gl_dq.core.workflow import Workflow

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["src", "pol_num", "pol_eff_dt", "pol_exp_dt", "covg_type_desc", "class1_cd", "expo_amt", "expn_bs",
           "rsk_loc_id", "rsk_itm_id", "loc_st_abbr", "loc_zipcd", "tot_wrtn_prm_amt", "pol_stat", "bi_ded_amt",
           "pd_ded_amt", "csl_ded_amt", "tx_type_nm", "allocation", "claim_alloc", "evt_dt"]


class FakeDB:
    dialect = DatabricksDialect()

    def query(self, sql):
        raise AssertionError("this test must not hit a database")

    def describe(self, table):
        return dict.fromkeys(COLUMNS, "string")


@pytest.fixture
def prod(monkeypatch, tmp_path):
    monkeypatch.setenv("DQ_CATALOG", "cat")
    monkeypatch.setenv("DQ_SCHEMA", "sch")
    monkeypatch.setenv("DQ_CONFIG_DIR", "config")
    p = load_project("prod")
    return Context("prod", p, FakeDB(), TableSchema(p.table, dict.fromkeys(COLUMNS, "string"), p.derived_columns,
                                                    DatabricksDialect()),
                   LocalStorage(ROOT / "config"), KnowledgeStore(LocalStorage(tmp_path)), ParquetResults(tmp_path),
                   discover(), Workflow())


def aliases(sql: str) -> set[str]:
    body = re.sub(r"--[^\n]*", "", sql)
    return set(re.findall(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", body, re.I))


def test_premium_sot_matches_its_config(prod):
    cfg = prod.check_config("premium_recon")
    assert cfg.sot_query == "sql/sot_premium_prod.sql"
    assert cfg.dims == ["src", "pol_yr"], "the study has no coverage split"
    sql = prod.render_user_sql(cfg.sot_query)
    assert aliases(sql) >= set(cfg.dims) | {cfg.sot_measure}
    assert "cimm_csm.premium_transx_seg_enriched_2026q2" in sql and "cat.sch.gl_master" in sql
    assert "GROUP BY" in sql and "BMQ_IND" in sql and "END AS src" in sql


def test_loss_sot_matches_its_config(prod):
    cfg = prod.check_config("loss_recon")
    assert cfg.sot_query == "sql/sot_loss_prod.sql"
    dims = list(dict.fromkeys(cfg.segments + [cfg.time_dim]))
    sql = prod.render_user_sql(cfg.sot_query)
    assert aliases(sql) >= set(dims) | {cfg.sot_loss_col, cfg.sot_claim_count_col}
    assert "YEAR(EVT_DT)" in sql, "loss year must use the same event-date basis as the pipeline"
    assert cfg.tolerance_claims.pct >= 0.05, "distinct occurrences vs allocated counts need a wider tolerance"


@pytest.mark.parametrize("check", ["premium_recon", "loss_recon"])
def test_pipeline_side_is_filtered_to_what_the_study_covers(prod, check):
    """The study is BMQ/CMQ and 2014-2025 only; without the same filter every BOP row is a fake break."""
    cfg = prod.check_config(check)
    assert cfg.where and "BOP" not in cfg.where.replace("'BMQ', 'CMQ'", "")
    assert "src IN ('BMQ', 'CMQ')" in cfg.where and "2014-01-01" in cfg.where and "2025-12-31" in cfg.where
    sql = prod.render_user_sql(cfg.sot_query)
    assert "'2014-01-01'" in sql and "'2025-12-31'" in sql, "study window must match the pipeline filter"


def test_loss_recon_applies_the_filter_to_the_pipeline_query(ctx_injected):
    """A configured `where` must reach the generated SQL (and the analytics), not just the config."""
    cls = ctx_injected.checks["loss_recon"]
    cfg = cls.Config(where="src <> 'BOP'")
    chk = cls(ctx_injected, cfg)
    findings, tables = chk.recon()
    assert "src <> 'BOP'" in chk._sql["pipeline"]
    assert "src=BOP" not in set(tables["recon_loss"]["segment"])
    chk.analytics(cfg.analytics)
    assert all("src <> 'BOP'" in s for k, s in chk._sql.items() if k in ("loss ratio / severity", "frequency"))
