"""Review workflow (stages, roles, hand-offs) and the catalog of preprocessing step types.

Loaded from `<config_dir>/workflow.yaml`; the defaults below mirror config/workflow.yaml.

Default flow for a column:
    not_started -> investigating (DS) -> actuary_review (actuary) -> with_de (DE fixes the column)
    -> ds_validation (DS) -> [actuary_signoff (actuary), if required] -> resolved
Other outcomes: no_issue, preprocess_in_modeling (+ recommended preprocessing steps), wont_fix.
"""
from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator


class Stage(BaseModel):
    key: str
    label: str
    icon: str = "⚪"
    role: str | None = None  # who holds the column in this stage (key of Workflow.roles)
    done: bool = False  # counts as reviewed / closed
    in_flow: bool = True  # part of the main hand-off path (next-step button); outcomes set False
    optional: bool = False  # skipped in the flow unless enabled (e.g. actuary sign-off)
    requires_preprocessing: bool = False  # shows the recommended preprocessing section
    description: str = ""


class ParamSpec(BaseModel):
    type: Literal["number", "text", "select", "bool", "list", "mapping"] = "text"
    options: list[str] = []
    default: Any = None
    help: str = ""


class PreprocOp(BaseModel):
    label: str
    description: str = ""
    params: dict[str, ParamSpec] = {}


DEFAULT_STAGES = [
    Stage(key="not_started", label="Not started", icon="⚪", description="Nobody has looked at this column yet"),
    Stage(key="investigating", label="DS investigating", icon="🔎", role="ds",
          description="A data scientist identified a problem and is investigating"),
    Stage(key="actuary_review", label="Confirming with actuary", icon="🧮", role="actuary",
          description="Findings and the proposed fix are being confirmed with an actuary"),
    Stage(key="with_de", label="With DE for fix", icon="🛠️", role="de",
          description="Settled; a data engineer is transforming / fixing the column"),
    Stage(key="ds_validation", label="DS validating fix", icon="🧪", role="ds",
          description="The fix is in the pipeline; a data scientist validates it"),
    Stage(key="actuary_signoff", label="Actuary sign-off", icon="✍️", role="actuary", optional=True,
          description="Optional final sign-off by an actuary"),
    Stage(key="resolved", label="Resolved", icon="✅", done=True, description="Fixed and validated"),
    Stage(key="no_issue", label="No issue / accepted as is", icon="☑️", done=True, in_flow=False,
          description="Checked; no fix needed"),
    Stage(key="preprocess_in_modeling", label="Preprocess in modeling", icon="⚙️", done=True, in_flow=False,
          requires_preprocessing=True,
          description="Left as is in the data; handled by recommended preprocessing in the modeling pipeline"),
    Stage(key="wont_fix", label="Won't fix", icon="⛔", done=True, in_flow=False,
          description="Known issue that will not be fixed (document why)"),
]

DEFAULT_OPS = {
    "impute": PreprocOp(label="Impute missing", description="Fill missing / sentinel values", params={
        "strategy": ParamSpec(type="select", options=["median", "mean", "mode", "constant", "zero"], default="median"),
        "value": ParamSpec(type="text", help="Only for strategy=constant"),
        "add_indicator": ParamSpec(type="bool", default=True, help="Add a <column>_missing flag"),
        "group_by": ParamSpec(type="list", help="Impute within groups, e.g. src, class1_cd"),
    }),
    "map_values": PreprocOp(label="Map / recode values", description="Recode sentinel or legacy values", params={
        "mapping": ParamSpec(type="mapping", help="from=to pairs, e.g. UNK=null, 000000=null"),
    }),
    "cap": PreprocOp(label="Cap / winsorize", description="Limit extreme values", params={
        "lower_pct": ParamSpec(type="number", default=0.0, help="Lower percentile (0-1)"),
        "upper_pct": ParamSpec(type="number", default=0.99, help="Upper percentile (0-1)"),
        "by": ParamSpec(type="list", help="Compute percentiles within groups, e.g. src"),
    }),
    "scale_fix": PreprocOp(label="Unit / scale correction", description="Fix a unit problem, e.g. cents to dollars",
                           params={"multiply_by": ParamSpec(type="number", default=0.01),
                                   "condition": ParamSpec(type="text", help="Rows to fix, e.g. src = 'BMQ' AND pol_yr = 2019")}),
    "log_transform": PreprocOp(label="Log transform", params={
        "method": ParamSpec(type="select", options=["log1p", "log10", "signed_log"], default="log1p"),
    }),
    "group_rare": PreprocOp(label="Group rare levels", params={
        "min_share": ParamSpec(type="number", default=0.005),
        "other_label": ParamSpec(type="text", default="OTHER"),
    }),
    "bin": PreprocOp(label="Bin", params={
        "method": ParamSpec(type="select", options=["quantile", "fixed_edges"], default="quantile"),
        "n_bins": ParamSpec(type="number", default=10),
        "edges": ParamSpec(type="list", help="For fixed_edges, e.g. 0, 250, 500, 1000"),
    }),
    "exclude_rows": PreprocOp(label="Exclude rows", description="Drop rows from the modeling data", params={
        "condition": ParamSpec(type="text", help="e.g. expo_amt <= 0"),
    }),
    "derive": PreprocOp(label="Derive feature", params={
        "new_column": ParamSpec(type="text"),
        "expression": ParamSpec(type="text", help="e.g. tot_wrtn_prm_amt / NULLIF(expo_amt, 0)"),
    }),
    "custom": PreprocOp(label="Custom", params={"instructions": ParamSpec(type="text")}),
}


class Workflow(BaseModel):
    roles: dict[str, str] = {"ds": "Data scientist", "actuary": "Actuary", "de": "Data engineer"}
    require_actuary_signoff: bool = False
    stages: list[Stage] = DEFAULT_STAGES
    preprocessing_ops: dict[str, PreprocOp] = DEFAULT_OPS

    @model_validator(mode="after")
    def _check(self):
        keys = [s.key for s in self.stages]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate stage keys")
        for s in self.stages:
            if s.role and s.role not in self.roles:
                raise ValueError(f"stage {s.key}: unknown role {s.role!r}")
        if not keys or self.stages[0].done:
            raise ValueError("the first stage must be the not-started stage")
        return self

    # ---- lookups ---------------------------------------------------------------
    @property
    def initial(self) -> str:
        return self.stages[0].key

    def keys(self) -> list[str]:
        return [s.key for s in self.stages]

    def stage(self, key: str) -> Stage:
        for s in self.stages:
            if s.key == key:
                return s
        return Stage(key=key, label=f"{key} (unknown)", icon="❔")

    def done_keys(self) -> set[str]:
        return {s.key for s in self.stages if s.done}

    def flow(self) -> list[Stage]:
        """Main hand-off path, honouring require_actuary_signoff for optional stages."""
        return [s for s in self.stages if s.in_flow and (not s.optional or self.require_actuary_signoff)]

    def next_stage(self, key: str) -> Stage | None:
        path = self.flow()
        idx = next((i for i, s in enumerate(path) if s.key == key), None)
        if idx is None or idx + 1 >= len(path):
            return None
        return path[idx + 1]

    def label(self, key: str, with_role: bool = True) -> str:
        s = self.stage(key)
        role = f" · {self.roles[s.role]}" if with_role and s.role else ""
        return f"{s.icon} {s.label}{role}"

    def waiting_on(self, key: str) -> str | None:
        s = self.stage(key)
        return self.roles.get(s.role) if s.role and not s.done else None


def parse_mapping(text: str | None) -> dict[str, str | None]:
    """'UNK=null, 000000=null, Y=1' -> {'UNK': None, '000000': None, 'Y': '1'}."""
    out: dict[str, str | None] = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = (x.strip() for x in part.split("=", 1))
            out[k] = None if v.lower() in ("null", "none", "") else v
    return out


def load_workflow(config_store) -> Workflow:
    text = config_store.read_text("workflow.yaml")
    return Workflow.model_validate(yaml.safe_load(text) or {}) if text else Workflow()
