"""Project profile + per-check configuration (pydantic models over YAML)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gl_dq import REPO_ROOT

class DerivedColumn(BaseModel):
    expr: str
    type: str = "int"


class Measures(BaseModel):
    written_premium: str = "tot_wrtn_prm_amt"
    loss: str = "allocation"
    claim_count: str = "claim_cnt"
    exposure: str = "expo_amt"
    exposure_base: str = "expn_bs_std"


class ResultsConfig(BaseModel):
    type: Literal["parquet", "delta"] = "parquet"
    path: str | None = "data/results"  # parquet: directory
    table: str | None = None  # delta: catalog.schema.table


class ProjectConfig(BaseModel):
    """A profile: which table, which backend, where config/knowledge/results live."""

    name: str = "GL master"
    backend: Literal["duckdb", "databricks", "spark", "parquet"] = "duckdb"
    # spark = inside a Databricks notebook/job; parquet = read parquet files directly (DuckDB)
    duckdb_path: str | None = None
    warehouse_id: str | None = None
    parquet_views: dict[str, str] = {}  # backend "parquet": table name -> file, folder or glob
    table: str
    src_col: str = "src"
    sources: list[str] = ["BOP", "BMQ", "CMQ"]
    derived_columns: dict[str, DerivedColumn] = {}
    measures: Measures = Measures()
    policy_key: list[str] = ["pol_num", "pol_eff_dt", "pol_exp_dt"]  # identifies one policy term
    segment_candidates: list[str] = []
    sql_vars: dict[str, str] = {}  # template variables for config/filters.yaml rules, e.g. a reference list
    config_dir: str = "config"  # contains checks/*.yaml and sql/*.sql
    knowledge_dir: str = "data/knowledge"
    results: ResultsConfig = ResultsConfig()
    refresh_job_id: str | None = None
    check_overrides: dict[str, dict] = {}
    plugins: list[str] = []  # extra python modules with @register_check classes

    @field_validator("derived_columns", mode="before")
    @classmethod
    def _derived(cls, v):
        return {k: ({"expr": d} if isinstance(d, str) else d) for k, d in (v or {}).items()}


class CheckConfig(BaseModel):
    """Fields shared by every check config. Checks subclass this."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    order: int = 100
    category: str = "Checks"  # sidebar section; "Checks" = is the data right, others group by purpose


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^{}]*))?\}")


def expand_env(text: str) -> str:
    """Expand ${VAR} and ${VAR:-default}; defaults may themselves contain ${...} (innermost first)."""
    prev = None
    while prev != text:
        prev = text
        text = _ENV.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), text)
    return text


def _expand(obj):
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def profile_path(profile: str) -> Path:
    p = Path(profile)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    return REPO_ROOT / "config" / "profiles" / f"{profile}.yaml"


def load_raw_profile(profile: str, _seen: tuple[str, ...] = ()) -> dict:
    """Profile YAML, with `extends: <other profile>` merged underneath it (deep merge, child wins)."""
    if profile in _seen:
        raise ValueError(f"circular profile inheritance: {' -> '.join([*_seen, profile])}")
    raw = yaml.safe_load(profile_path(profile).read_text(encoding="utf-8")) or {}
    base = raw.pop("extends", None)
    return deep_merge(load_raw_profile(base, (*_seen, profile)), raw) if base else raw


def load_project(profile: str | None = None) -> ProjectConfig:
    profile = profile or os.environ.get("DQ_PROFILE", "synthetic")
    return ProjectConfig.model_validate(_expand(load_raw_profile(profile)))


def dump_yaml(data: dict) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)
