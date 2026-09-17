"""Findings: the common long-format output of every check, and their run history store."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FINDING_COLS = ["check", "variable", "item", "segment", "metric", "value", "threshold", "status", "detail"]
STATUS_RANK = {"pass": 0, "info": 1, "warn": 2, "fail": 3}
STATUS_ICON = {"pass": "🟢", "info": "🔵", "warn": "🟠", "fail": "🔴", None: "⚪"}


def segment_key(values: dict) -> str:
    """Canonical segment string: 'src=BMQ|pol_yr=2019' ('ALL' when ungrouped)."""
    if not values:
        return "ALL"
    return "|".join(f"{k}={_fmt(v)}" for k, v in values.items())


def parse_segment(seg: str) -> dict[str, str]:
    if not seg or seg == "ALL":
        return {}
    return dict(part.split("=", 1) for part in seg.split("|"))


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return "<null>"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def grade(value: float | None, warn: float | None, fail: float | None, higher_is_worse: bool = True) -> str:
    """Status for a metric against thresholds (strictly beyond threshold = flagged)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "info"
    v = value if higher_is_worse else -value
    if fail is not None and v > (fail if higher_is_worse else -fail):
        return "fail"
    if warn is not None and v > (warn if higher_is_worse else -warn):
        return "warn"
    return "pass"


def worst(statuses) -> str | None:
    s = [x for x in statuses if x in STATUS_RANK]
    return max(s, key=STATUS_RANK.get) if s else None


def findings_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FINDING_COLS)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    return df


def new_run_id() -> tuple[str, str]:
    ts = datetime.now(timezone.utc)
    return ts.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6], ts.strftime("%Y-%m-%dT%H:%M:%SZ")


class ResultsStore(ABC):
    @abstractmethod
    def append(self, findings: pd.DataFrame, run_id: str, run_ts: str, profile: str) -> None: ...

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """All stored findings with run_id, run_ts columns."""

    def runs(self) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return pd.DataFrame(columns=["run_id", "run_ts", "n_findings", "n_warn", "n_fail"])
        return (df.groupby(["run_id", "run_ts"], as_index=False)
                  .agg(n_findings=("status", "size"),
                       n_warn=("status", lambda s: (s == "warn").sum()),
                       n_fail=("status", lambda s: (s == "fail").sum()))
                  .sort_values("run_ts"))

    def latest(self, offset: int = 0) -> pd.DataFrame:
        """Findings of the latest run (offset=1: the run before)."""
        df = self.load()
        runs = self.runs()
        if len(runs) <= offset:
            return pd.DataFrame(columns=FINDING_COLS + ["run_id", "run_ts"])
        rid = runs.iloc[-1 - offset]["run_id"]
        return df[df["run_id"] == rid].reset_index(drop=True)


class ParquetResults(ResultsStore):
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, findings, run_id, run_ts, profile):
        self.path.mkdir(parents=True, exist_ok=True)
        out = findings.assign(run_id=run_id, run_ts=run_ts, profile=profile)
        out.astype({"detail": "string", "item": "string"}).to_parquet(self.path / f"findings_{run_id}.parquet", index=False)

    def load(self):
        files = sorted(self.path.glob("findings_*.parquet")) if self.path.exists() else []
        if not files:
            return pd.DataFrame(columns=FINDING_COLS + ["run_id", "run_ts", "profile"])
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


class DeltaResults(ResultsStore):
    """Delta table `results.table`; written with Spark when available (Databricks job), else SQL INSERT."""

    def __init__(self, db, table: str):
        self.db, self.table = db, table

    def append(self, findings, run_id, run_ts, profile):
        out = findings.assign(run_id=run_id, run_ts=run_ts, profile=profile)
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
        except ImportError:
            spark = None
        if spark is not None:
            spark.createDataFrame(out.astype({"detail": "string", "item": "string"})).write.mode("append") \
                .option("mergeSchema", "true").saveAsTable(self.table)
            return
        lit = self.db.dialect.lit
        cols = list(out.columns)
        self.db.query(f"CREATE TABLE IF NOT EXISTS {self.table} (check STRING, variable STRING, item STRING, "
                      "segment STRING, metric STRING, value DOUBLE, threshold DOUBLE, status STRING, detail STRING, "
                      "run_id STRING, run_ts STRING, profile STRING)")
        for start in range(0, len(out), 500):
            chunk = out.iloc[start:start + 500]
            values = ",\n".join(
                "(" + ", ".join(lit(None if pd.isna(v) else (float(v) if isinstance(v, (np.floating, np.integer)) else v))
                                for v in row) + ")"
                for row in chunk[cols].itertuples(index=False))
            self.db.query(f"INSERT INTO {self.table} ({', '.join(cols)}) VALUES {values}")

    def load(self):
        try:
            return self.db.query(f"SELECT * FROM {self.table}")
        except Exception:  # table not created yet
            return pd.DataFrame(columns=FINDING_COLS + ["run_id", "run_ts", "profile"])
