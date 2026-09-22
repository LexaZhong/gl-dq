from pathlib import Path

import pytest
import yaml

from gl_dq.core.config import expand_env
from gl_dq.core.db import DatabricksDialect, DuckDBDialect
from gl_dq.core.schema import UnknownColumnError

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_column_rejected(ctx_injected):
    with pytest.raises(UnknownColumnError):
        ctx_injected.schema.ref("pol_num; DROP TABLE x")
    with pytest.raises(UnknownColumnError):
        ctx_injected.render_sql("agg_by_dims.sql.j2", dims=["nope"], measures={"n": "1"}, where=None)


def test_derived_columns_expand(ctx_injected):
    s = ctx_injected.schema
    assert s.ref("pol_yr") == "(year(pol_eff_dt))"
    assert s.ref("loss_yr") == "(year(evt_dt))"
    sql = ctx_injected.render_sql("agg_by_dims.sql.j2", dims=["src", "pol_yr"], measures={"p": s.ref("tot_wrtn_prm_amt")}, where=None)
    assert '(year(pol_eff_dt)) AS "pol_yr"' in sql
    df = ctx_injected.db.query(sql)
    assert set(df.columns) == {"src", "pol_yr", "p"} and df["pol_yr"].between(2018, 2024).all()


def test_literal_escaping():
    assert DuckDBDialect.lit("O'Brien") == "'O''Brien'"
    assert DuckDBDialect.lit(0.5) == "0.5" and DuckDBDialect.lit(None) == "NULL"
    assert DatabricksDialect().quote("a`b") == "`a``b`"


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("X_SET", "yes")
    monkeypatch.delenv("X_UNSET", raising=False)
    assert expand_env("${X_SET:-no}/${X_UNSET:-dflt}/${X_UNSET}") == "yes/dflt/"
    assert expand_env("${X_UNSET:-${X_SET}.gl_master}") == "yes.gl_master"


def test_prod_profile_parses(monkeypatch):
    from gl_dq.core.config import load_project

    monkeypatch.setenv("DQ_CATALOG", "pricing_cat")
    monkeypatch.setenv("DQ_SCHEMA", "gl")
    monkeypatch.delenv("DQ_TABLE", raising=False)
    p = load_project("prod")
    assert p.table == "pricing_cat.gl.gl_master" and p.backend == "databricks"
    # config/knowledge/run history follow DQ_VOLUME_DIR, not the table's catalog
    assert p.config_dir.endswith("/gl_master_cleaning/config")
    assert p.sql_vars["sot_premium_table"] == "cimm_csm.premium_transx_seg_enriched_2026q2"  # the pricing study
    assert p.sql_vars["study_from"] == "2014-01-01" and p.sql_vars["study_to"] == "2025-12-31"
    assert p.measures.claim_count == "claim_ant"
    monkeypatch.setenv("DQ_SOT_PREMIUM_TABLE", "other.study.table")  # still overridable per environment
    assert load_project("prod").sql_vars["sot_premium_table"] == "other.study.table"


@pytest.mark.parametrize("path", sorted((ROOT / "config" / "checks").glob("*.yaml")), ids=lambda p: p.stem)
def test_check_configs_validate(ctx_injected, path):
    cls = ctx_injected.checks[path.stem]
    cls.Config.model_validate(yaml.safe_load(path.read_text()))  # extra="forbid" catches typos


def test_config_roundtrip(ctx_injected, tmp_path):
    from gl_dq.core.storage import LocalStorage

    store = ctx_injected.config_store
    ctx_injected.config_store = LocalStorage(tmp_path)
    try:
        cfg = ctx_injected.checks["distribution"].Config.model_validate(
            yaml.safe_load((ROOT / "config/checks/distribution.yaml").read_text()))
        cfg.variables[0].percentile_bins = [0.1, 0.9]
        ctx_injected.save_check_config("distribution", cfg)
        assert ctx_injected.check_config("distribution") == cfg
    finally:
        ctx_injected.config_store = store


def test_page_order_and_sections(ctx_injected):
    assert ctx_injected.enabled_checks() == ["key_uniqueness", "missing_rate", "business_rules", "value_checks",
                                             "distribution", "premium_recon", "loss_recon", "exposure",
                                             "segment_mix"]
    sections = ctx_injected.checks_by_category()
    assert list(sections) == ["Checks", "Portfolio analysis"]          # order of first appearance
    assert sections["Portfolio analysis"] == ["segment_mix"]
    assert sections["Checks"][:4] == ["key_uniqueness", "missing_rate", "business_rules", "value_checks"]


def test_workspace_profile_inherits_prod(monkeypatch):
    """The notebook profile must pick up prod's table, source-of-truth queries and overrides."""
    from gl_dq.core.config import load_project

    monkeypatch.setenv("DQ_CATALOG", "cat")
    monkeypatch.setenv("DQ_SCHEMA", "sch")
    monkeypatch.setenv("DQ_CONFIG_DIR", "config")
    for var in ("DQ_TABLE", "DQ_KNOWLEDGE_DIR", "DQ_RESULTS_TABLE"):
        monkeypatch.delenv(var, raising=False)
    w, prod = load_project("workspace"), load_project("prod")
    assert w.backend == "spark" and prod.backend == "databricks"      # only the backend differs
    assert w.table == prod.table and w.measures == prod.measures
    assert w.sql_vars == prod.sql_vars and w.check_overrides == prod.check_overrides
    assert w.results == prod.results and w.results.type == "parquet"
    assert w.config_dir == "config"                                   # from the cloned repo
    assert w.knowledge_dir == f"{VOLUME}/knowledge"                   # survives the cluster


VOLUME = "/Volumes/na_combined_explore_rfnd-risk_cohort/risk-cohort-volume/GL/gl_master_cleaning"


def test_profiles_default_to_the_real_catalog_schema_and_volume(monkeypatch):
    """With no env set both Databricks profiles must resolve fully - no empty path segments."""
    from gl_dq.core.config import load_project

    for var in ("DQ_CATALOG", "DQ_SCHEMA", "DQ_TABLE", "DQ_CONFIG_DIR", "DQ_KNOWLEDGE_DIR", "DQ_RESULTS_PATH",
                "DQ_VOLUME_DIR"):
        monkeypatch.delenv(var, raising=False)
    for name in ("prod", "workspace"):
        p = load_project(name)
        assert p.table == "na_actuarial_explore.consd_sb_actuarial_sandbox.gl_master"
        assert p.knowledge_dir == f"{VOLUME}/knowledge"      # a different catalog from the table: fine
        assert p.results.type == "parquet" and p.results.path == VOLUME
        for path in (p.knowledge_dir, p.config_dir, p.results.path):
            assert "//" not in path.replace("/Volumes", "") and "${" not in path
    assert load_project("prod").config_dir == f"{VOLUME}/config"


def test_volume_dir_moves_everything_together(monkeypatch):
    from gl_dq.core.config import load_project

    for var in ("DQ_CONFIG_DIR", "DQ_KNOWLEDGE_DIR", "DQ_RESULTS_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DQ_VOLUME_DIR", "/Volumes/other/vol/gl")
    p = load_project("prod")
    assert p.config_dir == "/Volumes/other/vol/gl/config"
    assert p.knowledge_dir == "/Volumes/other/vol/gl/knowledge"
    assert p.results.path == "/Volumes/other/vol/gl"


def test_circular_profile_inheritance_is_rejected(tmp_path, monkeypatch):
    import pytest as _pytest
    import yaml as _yaml

    from gl_dq.core.config import load_raw_profile

    (tmp_path / "a.yaml").write_text(_yaml.safe_dump({"extends": str(tmp_path / "b.yaml"), "table": "t"}))
    (tmp_path / "b.yaml").write_text(_yaml.safe_dump({"extends": str(tmp_path / "a.yaml")}))
    with _pytest.raises(ValueError, match="circular"):
        load_raw_profile(str(tmp_path / "a.yaml"))
