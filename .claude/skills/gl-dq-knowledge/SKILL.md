---
name: gl-dq-knowledge
description: Search, summarize, export or reuse the gl_dq knowledge base (per-column workflow stage, assignees, notes and recommended preprocessing, stored as YAML in the UC Volume or data/knowledge). Use for "what do we know about expn_bs", "how many columns are left", "what is waiting on DE", "export the data dictionary / preprocessing spec", "reuse notes from gl_master for property_master", or progress reports for a business lead.
---

# Knowledge base

Records: `variables/<variable>.yaml`: `status` (a stage key from `config/workflow.yaml`), `assignees`
(role → person), `status_log` (from/to/by/at), `checks` (data snapshot when closed, for re-open
detection), `preprocessing` (ordered steps: op, params, sources, rationale), `notes`.
History: `history/<variable>/*.yaml`. API: `ctx.knowledge` (KnowledgeStore), `ctx.workflow`,
`gl_dq.tracker.build_tracker`, `gl_dq.core.knowledge.preprocessing_spec / export_markdown`.

## Tasks
**Search**
```python
import sys; sys.path.insert(0, 'src')
from gl_dq.core.context import load_context
ctx = load_context()
hits = [(v, n) for v, r in ctx.knowledge.all().items() for n in r.notes
        if 'TERM'.lower() in (n.text + ' '.join(n.tags) + v).lower()]
```
Quote notes verbatim with author and date. Distinguish drafts (`draft` tag / `reusable: false`) from confirmed notes.

**Progress report** (for business leads): build the tracker from the latest run:
```python
from gl_dq.tracker import build_tracker
latest = ctx.results.latest()
var_df, long = build_tracker(ctx.schema.names(include_derived=False), latest, ctx.knowledge.all(),
                             ctx.enabled_checks(), ctx.workflow)
```
Report closed / total, counts per stage in hand-off order, what is waiting on each role
(`var_df.waiting_on`, `assignee`, `days_in_status`: call out anything stuck > 7 days), flagged but
not started, re-opened by data, and columns in `preprocess_in_modeling` without steps. Keep it short, with numbers.

**Export data dictionary**: `export_markdown(ctx.knowledge.all(), ctx.workflow, title=...)` → write to
`exports/data_dictionary.md` in the store (`ctx.knowledge.storage.write_text`) and/or a local file.

**Export preprocessing spec** (input for the modeling preprocessing pipeline):
`preprocessing_spec(ctx.knowledge.all(), ctx.workflow, ctx.project.table)` → YAML to `exports/preprocessing_spec.yaml`.
Each column lists ordered steps `{order, op, params, sources, rationale}`. Op names and parameters are defined in
`workflow.yaml -> preprocessing_ops`.

**Reuse for another project**: load both profiles; for variables with the same name (or a mapping
the user confirms), copy only `reusable: true` notes into the new store as new notes tagged
`carried-over` with `text` prefixed `[from <old table>]`. Never copy stages or assignees: the new
data must be reviewed again. Preprocessing steps may be offered as *suggestions* in a note, not copied as steps.
Show the list before writing.
