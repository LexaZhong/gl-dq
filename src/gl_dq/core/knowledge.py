"""Per-column workflow status, assignees, notes and recommended preprocessing.

One YAML file per variable, with an append-only history:
    variables/<variable>.yaml               current state
    history/<variable>/<utc-ts>_<user>.yaml one file per change (Volumes cannot append)
    exports/data_dictionary.md              generated
    exports/preprocessing_spec.yaml         generated (source for a modeling preprocessing pipeline)
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gl_dq.core.config import dump_yaml
from gl_dq.core.storage import Storage

LEGACY_STATUS = {"in_review": "actuary_review", "accepted_as_is": "no_issue"}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Note(BaseModel):
    date: str = Field(default_factory=now_iso)
    author: str
    text: str
    check: str | None = None
    src: str | None = None
    tags: list[str] = []
    reusable: bool = True


class StatusChange(BaseModel):
    from_status: str | None = None
    to_status: str
    by: str
    at: str = Field(default_factory=now_iso)
    comment: str | None = None


class CheckSnapshot(BaseModel):
    """Data status of one check when the column was closed; used to detect 're-opened by data'."""

    judged_status: str | None = None
    judged_value: float | None = None
    judged_at: str | None = None


class PreprocessingStep(BaseModel):
    op: str  # key of Workflow.preprocessing_ops
    params: dict[str, Any] = {}
    sources: list[str] = []  # empty = all sources
    rationale: str = ""
    author: str | None = None
    added_at: str = Field(default_factory=now_iso)


class VariableRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    variable: str
    label: str | None = None
    description: str | None = None
    status: str = "not_started"
    assignees: dict[str, str] = {}  # role -> person
    status_log: list[StatusChange] = []
    checks: dict[str, CheckSnapshot] = {}
    preprocessing: list[PreprocessingStep] = []
    notes: list[Note] = []
    updated_by: str | None = None
    updated_at: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _legacy(cls, v):
        return LEGACY_STATUS.get(v, v)

    @property
    def status_since(self) -> str | None:
        return self.status_log[-1].at if self.status_log else None


class ConflictError(RuntimeError):
    """The record was changed by someone else since it was loaded."""


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class KnowledgeStore:
    def __init__(self, storage: Storage):
        self.storage = storage

    def _path(self, variable: str) -> str:
        return f"variables/{_safe(variable)}.yaml"

    def get(self, variable: str) -> tuple[VariableRecord, str | None]:
        text = self.storage.read_text(self._path(variable))
        if text is None:
            return VariableRecord(variable=variable), None
        return VariableRecord.model_validate(yaml.safe_load(text)), self.storage.version(self._path(variable))

    def all(self) -> dict[str, VariableRecord]:
        out = {}
        for path in self.storage.list("variables", ".yaml"):
            rec = VariableRecord.model_validate(yaml.safe_load(self.storage.read_text(path)))
            out[rec.variable] = rec
        return out

    def save(self, record: VariableRecord, author: str, expected_version: str | None, action: str) -> str | None:
        """Write the record if nobody changed it since `expected_version` was read."""
        path = self._path(record.variable)
        if self.storage.version(path) != expected_version:
            raise ConflictError(
                f"{record.variable} was modified by someone else since you loaded it. Reload and try again.")
        record.updated_by, record.updated_at = author, now_iso()
        self.storage.write_text(path, dump_yaml(record.model_dump(exclude_none=True)))
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.storage.write_text(
            f"history/{_safe(record.variable)}/{ts}_{_safe(author)}.yaml",
            dump_yaml({"action": action, "author": author, "at": record.updated_at,
                       "record": record.model_dump(exclude_none=True)}))
        return self.storage.version(path)

    def history(self, variable: str) -> list[dict]:
        files = self.storage.list(f"history/{_safe(variable)}", ".yaml")
        return [yaml.safe_load(self.storage.read_text(f)) for f in reversed(files)]

    # ---- mutations (reload -> modify -> save once) ------------------------------------
    def _load_for_update(self, variable: str, expected_version: str | None):
        rec, ver = self.get(variable)
        if expected_version is not None and ver != expected_version:
            raise ConflictError(f"{variable} was modified by someone else since you loaded it. Reload and try again.")
        return rec, ver

    def update(self, variable: str, author: str, *, workflow=None, status: str | None = None,
               assignees: dict[str, str | None] | None = None, label: str | None = None,
               note: Note | None = None, snapshots: dict[str, CheckSnapshot] | None = None,
               comment: str | None = None, expected_version: str | None = None) -> VariableRecord:
        """Change status / assignees / label and/or add a note, as one saved change.

        When the new status is a done status, `snapshots` (check -> data status now) are stored so
        the tracker can flag the column as re-opened if the data gets worse later.
        """
        rec, ver = self._load_for_update(variable, expected_version)
        if status and workflow is not None and status not in workflow.keys():
            raise ValueError(f"unknown status {status!r}")
        if status and status != rec.status:
            rec.status_log.append(StatusChange(from_status=rec.status, to_status=status, by=author, comment=comment))
            rec.status = status
            if workflow is not None and status in workflow.done_keys() and snapshots:
                rec.checks.update(snapshots)
        for role, person in (assignees or {}).items():
            if person:
                rec.assignees[role] = person
            else:
                rec.assignees.pop(role, None)
        if label is not None:
            rec.label = label or None
        if note is not None:
            rec.notes.append(note)
        self.save(rec, author, ver, action="update")
        return rec

    def add_note(self, variable: str, note: Note) -> VariableRecord:
        return self.update(variable, note.author, note=note)

    def add_preprocessing_step(self, variable: str, step: PreprocessingStep, author: str, *, workflow=None,
                               set_status: str | None = None) -> VariableRecord:
        rec, ver = self.get(variable)
        rec.preprocessing.append(step)
        if set_status and rec.status != set_status:
            rec.status_log.append(StatusChange(from_status=rec.status, to_status=set_status, by=author,
                                               comment="recommended preprocessing added"))
            rec.status = set_status
        self.save(rec, author, ver, action="add_preprocessing_step")
        return rec

    def remove_preprocessing_step(self, variable: str, index: int, author: str) -> VariableRecord:
        rec, ver = self.get(variable)
        if 0 <= index < len(rec.preprocessing):
            rec.preprocessing.pop(index)
            self.save(rec, author, ver, action="remove_preprocessing_step")
        return rec


# ---- exports ------------------------------------------------------------------------
def preprocessing_spec(records: dict[str, VariableRecord], workflow, table: str) -> dict:
    """Machine-readable spec of recommended preprocessing for columns in a preprocessing status."""
    keys = {s.key for s in workflow.stages if s.requires_preprocessing}
    variables = {}
    for name in sorted(records):
        r = records[name]
        if r.status in keys:
            variables[name] = {
                "status": r.status,
                "steps": [dict(order=i + 1, op=s.op, params=s.params, sources=s.sources or "all",
                               rationale=s.rationale, author=s.author, added_at=s.added_at)
                          for i, s in enumerate(r.preprocessing)],
            }
    return {"generated_at": now_iso(), "table": table,
            "ops": {k: {"label": o.label, "params": list(o.params)} for k, o in workflow.preprocessing_ops.items()},
            "variables": variables}


def export_markdown(records: dict[str, VariableRecord], workflow, title: str = "Data dictionary & cleaning notes") -> str:
    done = workflow.done_keys()
    lines = [f"# {title}", "", f"_Generated {now_iso()}_", ""]
    n_done = sum(r.status in done for r in records.values())
    lines += [f"**{n_done} / {len(records)}** documented columns closed.", ""]
    for name in sorted(records):
        r = records[name]
        lines += [f"## `{name}`" + (f": {r.label}" if r.label else ""), ""]
        if r.description:
            lines += [r.description, ""]
        lines.append(f"- **Status:** {workflow.label(r.status)}" + (f" (since {r.status_since[:10]})" if r.status_since else ""))
        if r.assignees:
            lines.append("- **Assignees:** " + ", ".join(f"{workflow.roles.get(k, k)}: {v}" for k, v in r.assignees.items()))
        if r.preprocessing:
            lines += ["", "**Recommended preprocessing**", ""]
            for i, s in enumerate(r.preprocessing, 1):
                params = ", ".join(f"{k}={v}" for k, v in s.params.items() if v not in (None, "", [], {}))
                scope = f" [{', '.join(s.sources)}]" if s.sources else ""
                lines.append(f"{i}. `{s.op}`({params}){scope}" + (f": {s.rationale}" if s.rationale else ""))
        reusable = [n for n in r.notes if n.reusable]
        if reusable:
            lines += ["", "**Notes**", ""]
            for n in reusable:
                scope = ", ".join(x for x in [n.check, n.src] if x)
                tags = " ".join(f"`#{t}`" for t in n.tags)
                lines.append(f"- {n.date[:10]} · {n.author}{' · ' + scope if scope else ''} {tags}: {n.text}")
        lines.append("")
    return "\n".join(lines)
