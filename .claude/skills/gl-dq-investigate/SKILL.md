---
name: gl-dq-investigate
description: Investigate a flagged gl_dq finding (a failing missing rate, duplicate keys, a PSI shift, premium/loss recon break, exposure anomaly or rule violation). Drills down with SQL, summarizes likely root causes, writes a DRAFT note and moves the column to the 'DS investigating' stage so it can be confirmed with an actuary. Use for "why is BMQ 2019 premium off", "look into the evt_dt nulls", "explain this fail".
---

# Investigate a finding

## 1. Locate the finding
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
import pandas as pd; pd.set_option('display.width',200); pd.set_option('display.max_colwidth',80)
from gl_dq.core.context import load_context
ctx = load_context(); f = ctx.results.latest()
print(f[f.status.isin(['warn','fail'])][['check','variable','item','segment','metric','value','detail']].to_string())"
```
If there is no stored run, run the check live: `ctx.make_check('<check>').run().findings`.
Read what reviewers already know first: `ctx.knowledge.get('<variable>')[0].notes` and search all
reusable notes (`gl-dq-knowledge`), because the issue may be documented already.

## 2. Drill down (read-only SQL via `ctx.db.query`)
Always use `ctx.project.table` and real column names. Aggregate; sample at most ~50 rows.
Typical drills by check:
- **missing_rate**: missing share by src × pol_yr × tx_type_nm / pol_stat; is it concentrated in a load date, state, class?
- **key_uniqueness**: sample duplicate groups (`key_duplicates.sql.j2`); are duplicates exact copies (load issue) or differ in one column (grain issue → which column)?
- **distribution PSI / outliers**: percentiles by year; ratio of medians between the flagged segment and its peers (×100 → cents, ×1000 → thousands); top rows by value.
- **loss ratio out of range** (Portfolio summary): usually a premium or an allocation problem, not both - compare loss and premium separately, and check whether one policy carries the loss (segment deep dive).
- **exposure**: rows with zero/negative exposure by tx_type_nm; class codes with multiple bases and their row counts per base.
- **business_rules**: violating rows by src × pol_yr; dates relative to load or cancellation.

## 3. Write the draft note and set the stage
The workflow (`config/workflow.yaml`, `ctx.workflow`) is: not_started → investigating (DS) →
actuary_review → with_de (DE fixes) → ds_validation (DS) → [actuary_signoff] → resolved; other
outcomes: no_issue, preprocess_in_modeling (+ recommended preprocessing), wont_fix.
Only ever move a column **to `investigating`** (or propose the next stage in your report). Hand-offs
and closing are decisions for the DS, actuary and DE.
```python
from gl_dq.core.knowledge import Note
ctx.knowledge.update("<variable>", "claude-draft (requested by <user>)", workflow=ctx.workflow,
    status="investigating" if ctx.knowledge.get("<variable>")[0].status == ctx.workflow.initial else None,
    assignees={"ds": "<user>"},
    note=Note(author="claude-draft (requested by <user>)", check="<check>", src="<SRC or None>",
              tags=["draft", "<root-cause-tag>"], reusable=False,
              text="<finding in one line>. Evidence: <numbers>. Likely cause: <hypothesis>. "
                   "Proposed fix: <DE transformation> OR proposed preprocessing: <step>. Needs actuary confirmation."))
```
If the likely outcome is "handle in modeling", draft the steps as text in the note (op names from
`ctx.workflow.preprocessing_ops`); don't add steps yourself unless the user asks.
Knowledge writes go to the profile's knowledge store (Volume in prod). **Ask before writing** if
the column is already past `investigating`.

## 4. Report
Give the user: the evidence table (small), the hypothesis ranked by confidence, the SQL used, and
what an actuary needs to confirm. Mention that the note is saved as a draft (`reusable: false`),
and that they should switch `reusable` on once it's confirmed.
