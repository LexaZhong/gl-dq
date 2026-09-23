"""Column transforms: standardize a field, then map its values onto an agreed vocabulary.

The two fixes that come out of the values check, made explicit and kept:

  * **standardize** - trim the whitespace, settle the case, left-pad a code to its full width, fix
    the type. Mechanical, and almost always the first thing wrong with a code column.
  * **map values** - 'SALES', 'Sales ' and 'sls' are one exposure base written three ways. A
    mapping says which one is the truth.

Order is fixed and not negotiable: trim -> case -> pad -> map -> cast. Mapping runs *after*
standardizing so one entry catches every spelling ('SALES' covers 'sales ' once trim and case have
run), and the cast runs last so a mapping is always written against text.

The library is stored as JSON (`config/transforms.json`) rather than YAML, because its job is to be
read by the modelling pipeline: `preprocessing_json()` emits every column's whole pipeline -
standardize, map, bin, and the steps recorded against the column in the knowledge store - in the
order they must be applied.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, model_validator

Case = Literal["none", "upper", "lower", "title"]
Cast = Literal["none", "string", "int", "float"]
UNMAPPED = Literal["keep", "null", "other"]

SQL_CAST = {"string": "STRING", "int": "BIGINT", "float": "DOUBLE"}
OTHER_LABEL = "<other>"


class Standardize(BaseModel):
    trim: bool = False
    case: Case = "none"
    zero_pad: int = 0  # left-pad with zeros to this width, e.g. 3 turns "7" into "007"
    cast: Cast = "none"

    @property
    def active(self) -> bool:
        return self.trim or self.case != "none" or self.zero_pad > 0 or self.cast != "none"

    def summary(self) -> str:
        bits = []
        if self.trim:
            bits.append("trim")
        if self.case != "none":
            bits.append(self.case)
        if self.zero_pad:
            bits.append(f"pad to {self.zero_pad}")
        if self.cast != "none":
            bits.append(f"as {self.cast}")
        return ", ".join(bits) or "none"


class ColumnTransform(BaseModel):
    column: str
    standardize: Standardize = Standardize()
    mapping: dict[str, str] = {}  # value seen in the data -> value to use
    unmapped: UNMAPPED = "keep"  # what happens to a value the mapping does not name
    description: str = ""
    author: str = ""
    updated: str = ""

    @model_validator(mode="after")
    def _check(self):
        if self.standardize.zero_pad < 0:
            raise ValueError(f"{self.column}: zero_pad cannot be negative")
        return self

    @property
    def active(self) -> bool:
        return self.standardize.active or bool(self.mapping) or self.unmapped != "keep"

    def summary(self) -> str:
        parts = [self.standardize.summary()] if self.standardize.active else []
        if self.mapping:
            parts.append(f"{len(self.mapping)} value(s) mapped")
        if self.unmapped != "keep":
            parts.append(f"unmapped -> {self.unmapped}")
        return " · ".join(parts) or "no change"

    def stamped(self, author: str) -> ColumnTransform:
        return self.model_copy(update={"author": author,
                                       "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})

    # ---- the pipeline, in the one order it may be applied ---------------------------
    def steps(self) -> list[dict]:
        """The transform as ordered, self-describing steps - what the modelling pipeline reads."""
        out: list[dict] = []
        s = self.standardize
        if s.trim:
            out.append({"op": "trim"})
        if s.case != "none":
            out.append({"op": "case", "to": s.case})
        if s.zero_pad:
            out.append({"op": "zero_pad", "width": s.zero_pad})
        if self.mapping or self.unmapped != "keep":
            out.append({"op": "map_values", "mapping": dict(self.mapping), "unmapped": self.unmapped})
        if s.cast != "none":
            out.append({"op": "cast", "to": s.cast})
        return out


class TransformLibrary(BaseModel):
    # Off by default: the checks are there to report the raw data. Turn it on to see the dashboard
    # as the model will see it, once the mappings are agreed.
    apply_to_dashboard: bool = False
    transforms: list[ColumnTransform] = []

    @model_validator(mode="after")
    def _unique(self):
        cols = [t.column for t in self.transforms]
        if len(cols) != len(set(cols)):
            raise ValueError("duplicate transform (one per column)")
        return self

    def get(self, column: str) -> ColumnTransform | None:
        return next((t for t in self.transforms if t.column == column), None)

    def active(self) -> list[ColumnTransform]:
        return [t for t in self.transforms if t.active]

    def put(self, t: ColumnTransform) -> TransformLibrary:
        return self.model_copy(update={"transforms": [x for x in self.transforms if x.column != t.column] + [t]})

    def drop(self, column: str) -> TransformLibrary:
        return self.model_copy(update={"transforms": [x for x in self.transforms if x.column != column]})

    def fingerprint(self) -> str:
        """Which transforms were in force - stored with a run, and used as a cache key."""
        act = self.active()
        if not act or not self.apply_to_dashboard:
            return ""
        blob = "|".join(f"{t.column}={t.steps()}" for t in sorted(act, key=lambda x: x.column))
        return "+".join(sorted(t.column for t in act)) + "#" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:6]


# ---- SQL -----------------------------------------------------------------------------
def sql_expr(schema, dialect, t: ColumnTransform) -> str:
    """The column, wrapped in its transform. Values become escaped literals; the column is
    whitelisted through schema.ref."""
    x = schema.ref(t.column)
    s = t.standardize
    if s.trim:
        x = f"TRIM({x})"
    if s.case == "upper":
        x = f"UPPER({x})"
    elif s.case == "lower":
        x = f"LOWER({x})"
    elif s.case == "title":
        x = f"INITCAP({x})"
    if s.zero_pad:
        x = f"LPAD(CAST({x} AS STRING), {int(s.zero_pad)}, '0')"
    if t.mapping or t.unmapped != "keep":
        whens = " ".join(f"WHEN {x} = {dialect.lit(k)} THEN {dialect.lit(v)}" for k, v in t.mapping.items())
        fallback = {"keep": x, "null": "NULL", "other": dialect.lit(OTHER_LABEL)}[t.unmapped]
        x = f"CASE {whens} ELSE {fallback} END" if whens else (x if t.unmapped == "keep" else fallback)
    if s.cast != "none":
        x = f"CAST({x} AS {SQL_CAST[s.cast]})"
    return x


# ---- pandas (previews) ----------------------------------------------------------------
def apply_series(values, t: ColumnTransform):
    """The same transform in pandas, for the before/after preview on the values check page."""
    import pandas as pd

    s = pd.Series(values, dtype="object").astype("string")
    st = t.standardize
    if st.trim:
        s = s.str.strip()
    if st.case == "upper":
        s = s.str.upper()
    elif st.case == "lower":
        s = s.str.lower()
    elif st.case == "title":
        s = s.str.title()
    if st.zero_pad:
        s = s.str.pad(st.zero_pad, side="left", fillchar="0")
    if t.mapping or t.unmapped != "keep":
        mapped = s.map(t.mapping)
        if t.unmapped == "keep":
            s = mapped.fillna(s)
        elif t.unmapped == "other":
            s = mapped.fillna(OTHER_LABEL)
        else:
            s = mapped
    if st.cast in ("int", "float"):
        s = pd.to_numeric(s, errors="coerce")
        if st.cast == "int":
            s = s.round().astype("Int64")
    return s


# ---- storage --------------------------------------------------------------------------
def load_transforms(config_store) -> TransformLibrary:
    text = config_store.read_text("transforms.json")
    return TransformLibrary.model_validate(json.loads(text)) if text else TransformLibrary()


def save_transforms(config_store, lib: TransformLibrary) -> None:
    config_store.write_text("transforms.json", json.dumps(lib.model_dump(mode="json"), indent=2) + "\n")


# ---- the modelling handover ------------------------------------------------------------
STEP_ORDER = {"trim": 0, "case": 1, "zero_pad": 2, "map_values": 3, "cast": 4,
              "exclude_rows": 5, "impute": 6, "scale_fix": 7, "cap": 8, "log_transform": 9,
              "group_rare": 10, "bin": 11, "derive": 12, "custom": 13}


def preprocessing_json(ctx) -> dict:
    """Every agreed preprocessing step, per column, in the order it must be applied.

    Three sources, one pipeline: the transforms on this page (standardize, map), the binning
    schemes chosen on the target-analysis page, and whatever the reviewers recorded against the
    column in the knowledge store. Ordering is by STEP_ORDER, so a mapping can never run after the
    binning that depends on it.
    """
    from gl_dq.core.binning import load_binnings

    columns: dict[str, list[dict]] = {}
    notes: dict[str, str] = {}

    for t in ctx.transforms.active():
        columns.setdefault(t.column, []).extend(t.steps())
        if t.description:
            notes[t.column] = t.description

    for b in load_binnings(ctx.config_store).binnings:
        if b.method == "categorical" or not b.cuts:
            continue
        columns.setdefault(b.variable, []).append(
            {"op": "bin", "method": b.method, "scheme": b.name, "cuts": list(b.cuts), "labels": b.labels()})

    wf = ctx.workflow
    keys = {s.key for s in wf.stages if s.requires_preprocessing}
    for name, rec in ctx.knowledge.all().items():
        if rec.status not in keys:
            continue
        for s in rec.preprocessing:
            if s.op == "bin" and any(x["op"] == "bin" for x in columns.get(name, [])):
                continue  # the binning library already carries the resolved cuts
            columns.setdefault(name, []).append(
                {"op": s.op, **s.params, **({"sources": s.sources} if s.sources else {}),
                 "rationale": s.rationale})

    for col, steps in columns.items():
        steps.sort(key=lambda s: STEP_ORDER.get(s["op"], 99))

    return {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "table": ctx.project.table,
            "profile": ctx.profile,
            "filters": [{"key": f.key, "label": f.title, "description": f.description}
                        for f in ctx.filters.active(ctx.profile)],
            "step_order": [k for k, _ in sorted(STEP_ORDER.items(), key=lambda kv: kv[1])],
            "columns": {c: {"steps": columns[c], **({"note": notes[c]} if c in notes else {})}
                        for c in sorted(columns)}}
