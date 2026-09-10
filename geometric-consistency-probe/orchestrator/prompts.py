"""Builds the prompts sent to Claude Code: the initial task prompt (from
tasks/{NN}_*.md, unmodified in substance) and the fix prompt sent back
after a failed QA pass.
"""

from __future__ import annotations

from pathlib import Path


class TaskSpecNotFound(Exception):
    """No tasks/{NN}_*.md file exists yet for this task number.

    Not an error condition for the orchestrator loop: per the user's own
    workflow, task spec files are added one at a time, only after the
    previous task has been accepted. The orchestrator's response to this
    is to stop and wait, not to fail.
    """


def find_task_spec(tasks_dir: str | Path, task: int) -> Path:
    tasks_dir = Path(tasks_dir)
    matches = sorted(tasks_dir.glob(f"{task:02d}_*.md"))
    if not matches:
        raise TaskSpecNotFound(f"no task spec file found for task {task} in {tasks_dir} (expected {task:02d}_*.md)")
    if len(matches) > 1:
        raise TaskSpecNotFound(f"ambiguous: multiple task spec files match task {task}: {matches}")
    return matches[0]


_ORCHESTRATION_HEADER = """\
You are implementing one task in an autonomous, orchestrator-supervised
research pipeline. Read the full task specification below and implement
it completely: code, tests, and the experiment itself (not a stub).

Hard requirements, on top of anything the task specification says:

1. Do not modify or re-implement Tasks 1-5 (see research/RESEARCH_PLAN.md
   for what they are) unless you find a genuine bug in them required to
   complete this task -- if so, make the minimal fix and record exactly
   why in this task's result JSON's "protected_files_justification" field.
2. Follow every rule in research/RESEARCH_INVARIANTS.md. If completing
   this task honestly would require violating one of them, STOP and
   record the conflict in the result JSON rather than silently doing it
   anyway (invariant 15).
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
   framing to manufacture a more positive-looking number.

=== TASK SPECIFICATION (tasks/{spec_name}) ===

{spec_body}
"""


def render_task_prompt(task: int, tasks_dir: str | Path) -> str:
    spec_path = find_task_spec(tasks_dir, task)
    spec_body = spec_path.read_text()
    return _ORCHESTRATION_HEADER.format(task=task, spec_name=spec_path.name, spec_body=spec_body)


_FIX_HEADER = """\
Your previous attempt at Task {task} did not pass independent QA. This
is attempt {attempt} of a maximum of {max_attempts}. Fix the specific
problems listed below -- do not start over from scratch, and do not
change the task's scope or framing to avoid the failing checks.

=== QA FAILURE REPORT ===

{qa_report}

=== ORIGINAL TASK SPECIFICATION (tasks/{spec_name}), for reference ===

{spec_body}
"""


def render_fix_prompt(task: int, tasks_dir: str | Path, qa_report: str, attempt: int, max_attempts: int) -> str:
    """`qa_report` is a pre-rendered report string (orchestrator.qa.
    render_report() or QAResult.render_report()) -- taking text rather
    than a live QAResult means a fix prompt can be reconstructed purely
    from persisted state/progress.json after an interruption, with no
    need to keep a QAResult object alive across process restarts."""
    spec_path = find_task_spec(tasks_dir, task)
    spec_body = spec_path.read_text()
    return _FIX_HEADER.format(
        task=task,
        attempt=attempt,
        max_attempts=max_attempts,
        qa_report=qa_report,
        spec_name=spec_path.name,
        spec_body=spec_body,
    )
