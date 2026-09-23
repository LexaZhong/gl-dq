"""Binning schemes: named, reusable ways of cutting a numeric variable into levels.

Fitting a GLM means trying a variable a dozen ways - quintiles, deciles, the ISO bands, a hand-drawn
split around a threshold someone noticed - and keeping the one that separates the target best. That
is a series of experiments, and experiments need three things this module provides:

  * **a name**, so a scheme can be referred to, reused on another target, and adopted by the
    modelling pipeline;
  * **frozen cut points**. A quantile scheme is resolved against the data once, at save time, and
    the resulting cuts are stored. Re-deriving them later would silently make it a different
    scheme, and the comparison against last month's run would be comparing two different things;
  * **provenance** - who made it, when, and why - so the shortlist is readable months later.

The library lives in `config/binnings.yaml`, next to the check configs, so schemes are shared, code
reviewed and versioned like everything else.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

NULL_LABEL = "<null>"
Method = Literal["categorical", "quantile", "equal_width", "custom"]
METHOD_LABELS = {
    "categorical": "Raw levels (no binning)",
    "quantile": "Quantile (equal weight per bin)",
    "equal_width": "Equal width (equal value range)",
    "custom": "Custom cut points",
}


def _num(v: float) -> str:
    """Compact bin-edge label: 2.5M, 250K, 1,000, 0.75."""
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            s = f"{v / div:,.2f}".rstrip("0").rstrip(".")
            return f"{s}{suf}"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.4g}"


class BinSpec(BaseModel):
    """One way of cutting one variable. `cuts` are interior edges: n cuts make n+1 bins."""

    variable: str
    name: str = "default"
    method: Method = "categorical"
    bins: int = 5  # requested bin count for quantile / equal_width, before de-duplication
    cuts: list[float] = []  # resolved interior edges, frozen once saved
    description: str = ""
    author: str = ""
    created: str = ""

    @field_validator("cuts")
    @classmethod
    def _sorted_unique(cls, v):
        # repeated cuts are common on a skewed column (80% zeros gives many identical quantiles)
        return sorted({float(x) for x in v})

    @model_validator(mode="after")
    def _check(self):
        if self.method == "custom" and not self.cuts:
            raise ValueError(f"{self.variable}/{self.name}: a custom binning needs cut points")
        if self.method in ("quantile", "equal_width") and self.bins < 2:
            raise ValueError(f"{self.variable}/{self.name}: {self.method} needs at least 2 bins")
        return self

    @property
    def key(self) -> str:
        return f"{self.variable}::{self.name}"

    @property
    def resolved(self) -> bool:
        return self.method == "categorical" or bool(self.cuts)

    @property
    def n_bins(self) -> int:
        return 1 if self.method == "categorical" else len(self.cuts) + 1

    def summary(self) -> str:
        if self.method == "categorical":
            return "raw levels"
        edges = ", ".join(_num(c) for c in self.cuts[:4]) + ("…" if len(self.cuts) > 4 else "")
        return f"{self.method}, {self.n_bins} bins [{edges}]"

    def labels(self) -> list[str]:
        """Bin labels in order. The numeric prefix is what keeps them sorted on an axis."""
        if self.method == "categorical":
            return []
        edges = self.cuts
        out = [f"01. < {_num(edges[0])}"]
        out += [f"{i + 2:02d}. {_num(edges[i])} to {_num(edges[i + 1])}" for i in range(len(edges) - 1)]
        out.append(f"{len(edges) + 1:02d}. >= {_num(edges[-1])}")
        return out

    def case_sql(self, ref: str, lit) -> str:
        """CASE expression mapping the column to its bin label, built from numeric literals only."""
        if self.method == "categorical":
            return ref
        labels = self.labels()
        parts = [f"WHEN {ref} IS NULL THEN {lit(NULL_LABEL)}"]
        parts += [f"WHEN {ref} < {lit(float(c))} THEN {lit(labels[i])}" for i, c in enumerate(self.cuts)]
        return "CASE " + " ".join(parts) + f" ELSE {lit(labels[-1])} END"

    def stamped(self, author: str) -> BinSpec:
        return self.model_copy(update={"author": author,
                                       "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})


class BinningLibrary(BaseModel):
    binnings: list[BinSpec] = []

    @model_validator(mode="after")
    def _unique(self):
        keys = [b.key for b in self.binnings]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate binning key (variable + name)")
        return self

    def for_variable(self, variable: str) -> list[BinSpec]:
        return [b for b in self.binnings if b.variable == variable]

    def get(self, variable: str, name: str) -> BinSpec | None:
        return next((b for b in self.binnings if b.variable == variable and b.name == name), None)

    def put(self, spec: BinSpec) -> BinningLibrary:
        """Add or replace a scheme, keeping the rest of the library untouched."""
        kept = [b for b in self.binnings if b.key != spec.key]
        return BinningLibrary(binnings=kept + [spec])

    def drop(self, variable: str, name: str) -> BinningLibrary:
        return BinningLibrary(binnings=[b for b in self.binnings
                                        if not (b.variable == variable and b.name == name)])


class BinningSet(BaseModel):
    """The schemes in play for one view - one per variable at most. A model, so it can be a cache key."""

    specs: list[BinSpec] = []

    def get(self, variable: str) -> BinSpec | None:
        return next((s for s in self.specs if s.variable == variable), None)


def resolve(check, spec: BinSpec, where: str | None = None) -> BinSpec:
    """Fill in the cut points from the data. Called once, when a scheme is created or previewed."""
    if spec.method in ("categorical", "custom"):
        return spec
    ref = check.schema.ref(spec.variable)
    filt = f"{ref} IS NOT NULL" + (f" AND ({where})" if where else "")
    if spec.method == "quantile":
        probs = [i / spec.bins for i in range(1, spec.bins)]
        row = check.query(f"{spec.variable} quantiles", check.ctx.render_sql(
            "dist_limits.sql.j2", v=ref, probs=probs, where=filt)).iloc[0]
        raw = [] if row["limits"] is None else [float(x) for x in row["limits"] if x is not None]
    else:
        row = check.query(f"{spec.variable} range", check.ctx.render_sql(
            "dist_limits.sql.j2", v=ref, probs=[0.0], where=filt)).iloc[0]
        lo, hi = float(row["vmin"]), float(row["vmax"])
        step = (hi - lo) / spec.bins
        raw = [lo + step * i for i in range(1, spec.bins)] if hi > lo else []
    # Quantiles of a discrete column repeat - most policies sit on the same limit, so the 20th and
    # 40th percentile are both 1,000,000. Keeping both would make an empty bin that can never be
    # reached. Re-validate rather than model_copy(update=), which skips validators.
    return BinSpec.model_validate({**spec.model_dump(), "cuts": raw})


def load_binnings(config_store) -> BinningLibrary:
    text = config_store.read_text("binnings.yaml")
    return BinningLibrary.model_validate(yaml.safe_load(text) or {}) if text else BinningLibrary()


def save_binnings(config_store, lib: BinningLibrary) -> None:
    from gl_dq.core.config import dump_yaml

    config_store.write_text("binnings.yaml", dump_yaml(lib.model_dump(mode="json")))
