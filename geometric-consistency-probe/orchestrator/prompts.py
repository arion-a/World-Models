"""Builds the prompts sent to Claude Code: the initial task prompt and
the fix prompt sent back after a failed QA pass.

Both prompts are built entirely from repository state -- the canonical
protocol, the (already-synced, see orchestrator/sync.py) task spec, the
orchestrator's own state, and the previous task's result -- so Claude
has everything it needs up front and is never told to ask a human for
routine or implementation decisions. There is no interactive channel
available during a `claude -p ... --permission-prompts none` invocation
anyway; the instructions below make that explicit rather than leaving
Claude to discover it by having a question go unanswered.
"""

from __future__ import annotations

import json
from pathlib import Path

CANONICAL_PROTOCOL_PATH = "research/CANONICAL_RESEARCH_PROTOCOL.md"
RESEARCH_INVARIANTS_PATH = "research/RESEARCH_INVARIANTS.md"


def render_state_summary(state_dict: dict, task: int, attempt: int, max_attempts: int) -> str:
    return json.dumps(
        {
            "current_task": task,
            "attempt": f"{attempt}/{max_attempts}",
            "completed_tasks": state_dict.get("completed_tasks", []),
            "status": state_dict.get("status"),
        },
        indent=2,
    )


def render_previous_result_summary(previous_result: dict | None, previous_task: int) -> str:
    if previous_result is None:
        return f"(no result file found for task {previous_task} -- nothing to summarize)"
    keys = ("task", "implementation_status", "scientific_result", "metrics")
    summary = {k: previous_result[k] for k in keys if k in previous_result}
    return json.dumps(summary, indent=2)


_ORCHESTRATION_HEADER = """\
You are executing Task {task} of an autonomous research pipeline. No
human is available to answer questions during this session -- resolve
ordinary implementation decisions yourself, using the authority order
below; only record a BLOCKED result (see requirement 2) for what
genuinely cannot be resolved this way.

Read and obey, in this authority order (highest first):
1. {canonical_path} -- binding scientific design; never override it.
2. {invariants_path} -- durable rules every task is checked against.
3. {spec_name} -- this task's execution-ready specification (generated
   deterministically from #1 by orchestrator/sync.py).

Current orchestrator state:
{state_summary}

Previous task's result ({prev_result_path}), for context:
{previous_result_summary}

Implement the task completely: code, tests, and the experiment itself
(not a stub). Resolve ordinary implementation decisions (refactoring,
bug fixes, engineering approach, test structure, which specific
scenes/config values to use where the spec gives you latitude) yourself
using the canonical protocol and task specification as your guide --
do not ask the user for routine decisions.

Hard requirements, on top of anything the task specification says:

1. Do not modify or re-implement Tasks 1-5 (see research/RESEARCH_PLAN.md
   for what they are) unless you find a genuine bug in them required to
   complete this task -- if so, make the minimal fix and record exactly
   why in this task's result JSON's "protected_files_justification" field.
2. Do not modify the scientific protocol: do not remove a required
   control or baseline, weaken or skip a required test, change train/test
   methodology, let test-split data influence any fitted parameter,
   replace scene-level splitting, unfreeze the encoder, introduce
   ground-truth leakage into the encoder's input, change the scientific
   hypothesis, remove a required metric, change the interpretation rules,
   or simplify the research question to make the experiment easier. If
   completing this task honestly seems to require any of this, or two
   requirements in the canonical protocol/invariants genuinely conflict,
   or a required credential/model/service is genuinely unavailable, or
   the only way forward would be a destructive/irreversible action
   outside this task's normal workflow: STOP. Do not guess and do not
   silently work around it -- set "implementation_status": "BLOCKED" in
   the result JSON, with a "blocker_category" field (exactly one of
   SCIENTIFIC_CONFLICT, DESTRUCTIVE_ACTION, MISSING_DEPENDENCY, or
   BLOCKER if none of those fit) and a "blocking_issue" field explaining
   precisely what conflicts and why.
3. Write the machine-readable result to state/task_{task:02d}_result.json
   using the schema given in the task specification below.
4. Run the existing test suite (`pytest -m "not slow"` at minimum, from
   the geometric-consistency-probe/ directory) yourself before finishing,
   and report the pass/fail counts honestly in the result JSON's "tests"
   field -- but understand that an independent process will re-run and
   verify this after you finish; your own report is not the basis on
   which this task is accepted.
5. A scientifically negative or weak result is a completely valid,
   acceptable outcome for this task, as long as it was validly measured
   -- do not adjust the experiment, the scene count, the metric, or the
   framing to manufacture a more positive-looking number. Record it
   honestly and move on; do not ask whether a negative result is okay.

The external orchestrator will independently re-verify everything after
you finish -- your own "tests"/"implementation_status" fields are one
input, never the basis of acceptance. Do not ask the user whether to
proceed, whether a result is acceptable, or what to do next.

=== TASK SPECIFICATION ({spec_name}, generated from {canonical_path}) ===

{spec_body}
"""


def render_task_prompt_from_content(
    task: int,
    spec_name: str,
    spec_body: str,
    state_summary: str,
    previous_result_summary: str,
    previous_task: int,
) -> str:
    """Core renderer taking the spec's content directly rather than a
    path -- lets a caller (namely orchestrator/run.py's --dry-run) build
    the exact prompt that would be sent WITHOUT requiring the spec file
    to actually exist on disk yet (a real, non-dry-run sync always
    writes it first; --dry-run must not)."""
    return _ORCHESTRATION_HEADER.format(
        task=task,
        canonical_path=CANONICAL_PROTOCOL_PATH,
        invariants_path=RESEARCH_INVARIANTS_PATH,
        spec_name=spec_name,
        state_summary=state_summary,
        prev_result_path=f"state/task_{previous_task:02d}_result.json",
        previous_result_summary=previous_result_summary,
        spec_body=spec_body,
    )


def render_task_prompt(
    task: int,
    spec_path: str | Path,
    state_summary: str,
    previous_result_summary: str,
    previous_task: int,
) -> str:
    """`spec_path` is the already-synchronized tasks/{{NN}}_*.md path
    (orchestrator.sync.sync_task_spec has already run, for real -- not
    dry_run=True -- by the time this is called, so the file is
    guaranteed to exist; this function does no synchronization itself).
    Prefer render_task_prompt_from_content when the spec content is
    already in hand and the file may not exist on disk (dry-run)."""
    spec_path = Path(spec_path)
    spec_body = spec_path.read_text()
    return render_task_prompt_from_content(task, spec_path.name, spec_body, state_summary, previous_result_summary, previous_task)


_FIX_HEADER = """\
Task {task} did not pass independent QA. This is attempt {attempt} of a
maximum of {max_attempts}. No human is available during this session --
fix the specific problems listed below yourself; do not ask whether or
how to proceed.

=== QA FAILURE REPORT ===

{qa_report}

Fix the identified implementation/methodological failures. Do not
weaken or remove a test to make it pass. Do not remove a required
control or baseline. Do not change the scientific objective, the
train/test methodology, or the interpretation rules to dodge a failing
check. Do not start over from scratch -- fix what's broken.

If, in the course of fixing this, you determine the QA failure actually
reflects a genuine scientific-protocol conflict, a required destructive
action, or a missing external dependency (rather than an implementation
bug) -- STOP, do not guess, and set "implementation_status": "BLOCKED"
with a "blocker_category" field (SCIENTIFIC_CONFLICT, DESTRUCTIVE_ACTION,
MISSING_DEPENDENCY, or BLOCKER) and a "blocking_issue" field explaining
why, in the result JSON.

After fixing, rerun the relevant tests and regenerate the required
artifacts and result JSON.

=== ORIGINAL TASK SPECIFICATION ({spec_name}), for reference ===

{spec_body}
"""


def render_fix_prompt(task: int, spec_path: str | Path, qa_report: str, attempt: int, max_attempts: int) -> str:
    """`qa_report` is a pre-rendered report string (orchestrator.qa.
    render_report() or QAResult.render_report()) -- taking text rather
    than a live QAResult means a fix prompt can be reconstructed purely
    from persisted state/progress.json after an interruption, with no
    need to keep a QAResult object alive across process restarts."""
    spec_path = Path(spec_path)
    spec_body = spec_path.read_text()
    return _FIX_HEADER.format(
        task=task,
        attempt=attempt,
        max_attempts=max_attempts,
        qa_report=qa_report,
        spec_name=spec_path.name,
        spec_body=spec_body,
    )
