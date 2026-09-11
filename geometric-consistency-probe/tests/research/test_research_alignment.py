"""Research-alignment checks: this file itself is Layer D
(RESEARCH ALIGNMENT) of orchestrator/qa.py's per-task QA -- run directly
via `pytest tests/research`, and also invoked as a subprocess by
orchestrator.qa._layer_d_research_alignment for every task.

These tests check the durable research documents and task specs
themselves, not any experiment's numbers (there are no experiment
results yet -- Task 6 hasn't run). They exist so a future task cannot
quietly delete/water down research/RESEARCH_INVARIANTS.md, so every
task spec carries the sections QA depends on being able to find, and so
state/progress.json always parses and is internally consistent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
TASKS_DIR = REPO_ROOT / "tasks"
STATE_DIR = REPO_ROOT / "state"

REQUIRED_TASK_SECTIONS = (
    "## Objective",
    "## Scientific question",
    "## Implementation requirements",
    "## Required artifacts",
    "## Tests required",
    "## Leakage checks",
    "## Acceptance criteria",
    "## Prohibited shortcuts",
    "## Scientific interpretation limits",
)

# Tasks 6-18: which spec files exist is expected to grow over time (the
# user's own workflow adds tasks/{N+1}_*.md only after task N is
# accepted) -- so these tests check "every spec file that DOES exist is
# well-formed", not "all 13 must exist right now".
TASK_NUMBERS = range(6, 19)


def _find_task_spec(task: int) -> Path | None:
    matches = sorted(TASKS_DIR.glob(f"{task:02d}_*.md"))
    return matches[0] if matches else None


# --- research/RESEARCH_PLAN.md -------------------------------------------


def test_research_plan_exists():
    assert (RESEARCH_DIR / "RESEARCH_PLAN.md").exists(), "research/RESEARCH_PLAN.md must exist"


def test_research_plan_lists_every_task_number():
    text = (RESEARCH_DIR / "RESEARCH_PLAN.md").read_text()
    for n in list(range(1, 6)) + list(TASK_NUMBERS):
        assert re.search(rf"\b{n}\b", text), f"research/RESEARCH_PLAN.md does not mention task {n}"


# --- research/RESEARCH_INVARIANTS.md --------------------------------------


def test_research_invariants_exists():
    assert (RESEARCH_DIR / "RESEARCH_INVARIANTS.md").exists(), "research/RESEARCH_INVARIANTS.md must exist"


def test_research_invariants_has_all_18_numbered_invariants():
    text = (RESEARCH_DIR / "RESEARCH_INVARIANTS.md").read_text()
    missing = [n for n in range(1, 19) if f"{n}. **" not in text]
    assert not missing, f"research/RESEARCH_INVARIANTS.md is missing numbered invariant(s): {missing}"


@pytest.mark.parametrize(
    "keyword",
    [
        "disjoint",  # invariant 4
        "frozen",  # invariant 6
        "linear probe",  # invariant 8
        "negative result",  # invariant 11
        "baseline",  # invariant 14
        "test-set fitting",  # invariant 16
        "self-reported",
        "Claude",  # invariant 18 -- must name what it's guarding against
    ],
)
def test_research_invariants_covers_key_concept(keyword):
    text = (RESEARCH_DIR / "RESEARCH_INVARIANTS.md").read_text()
    assert keyword.lower() in text.lower(), f"research/RESEARCH_INVARIANTS.md does not mention {keyword!r}"


# --- tasks/{NN}_*.md -------------------------------------------------------


@pytest.mark.parametrize("task", TASK_NUMBERS)
def test_task_spec_well_formed_if_present(task):
    spec_path = _find_task_spec(task)
    if spec_path is None:
        pytest.skip(f"tasks/{task:02d}_*.md not added yet (expected -- added sequentially after prior task acceptance)")

    text = spec_path.read_text()
    assert text.strip(), f"{spec_path} is empty"

    missing_sections = [s for s in REQUIRED_TASK_SECTIONS if s not in text]
    assert not missing_sections, f"{spec_path} is missing required section(s): {missing_sections}"

    assert f"Task {task}" in text or f"# Task {task}" in text, f"{spec_path} does not clearly identify itself as Task {task}"


def test_at_least_task_6_spec_exists():
    """Task 6 is next per state/progress.json -- its spec must exist for
    the orchestrator to have anything to do."""
    assert _find_task_spec(6) is not None, "tasks/06_*.md must exist so the orchestrator can start Task 6"


def test_no_task_spec_files_out_of_declared_range():
    """NN_name.md (the main Tasks 6-18 pipeline) or NNX_name.md, a single
    uppercase-letter-suffixed sub-task explicitly scoped as a narrower
    follow-up to task NN (e.g. tasks/07B_latent_transformation_discovery.md,
    Task 7B -- has its own separate orchestrator state machine in
    experiments/task7b_latent_transformation_discovery/state.py, is never
    advanced by orchestrator/run.py, and orchestrator/sync.py's own
    tasks_dir.glob(f"{task:02d}_*.md") for task=7 does not match an
    "07B_..." filename, so this does not collide with or get mistaken for
    the main pipeline's own Task 7 spec). Still guards against a stray or
    out-of-range file: the leading digits must name a real task number."""
    for path in TASKS_DIR.glob("*.md"):
        match = re.match(r"^(\d+)([A-Z]?)_", path.name)
        assert match, f"{path.name} does not follow the NN_name.md or NNX_name.md convention"
        n = int(match.group(1))
        assert n in TASK_NUMBERS, f"{path.name} has a task number ({n}) outside the declared 6-18 pipeline"


# --- state/progress.json ---------------------------------------------------


def test_progress_json_parses_and_is_internally_consistent():
    """A lightweight, dependency-free re-check of the same invariant
    orchestrator.state.OrchestratorState.validate() enforces -- kept
    independent (no import of orchestrator.state) so this test still
    catches a corrupted progress.json even if orchestrator/state.py
    itself has a bug in its own validate() method."""
    progress_path = STATE_DIR / "progress.json"
    if not progress_path.exists():
        pytest.skip("state/progress.json not created yet (orchestrator bootstraps it on first run)")

    data = json.loads(progress_path.read_text())
    assert isinstance(data["current_task"], int)
    assert 1 <= data["current_task"] <= 19
    completed = data.get("completed_tasks", [])
    assert data["current_task"] not in completed, "current_task must not already be in completed_tasks"
    assert sorted(set(completed)) == list(range(1, data["current_task"])), "completed_tasks must be exactly [1..current_task-1], no gaps"


def test_progress_json_reflects_tasks_1_through_5_complete_at_minimum():
    progress_path = STATE_DIR / "progress.json"
    if not progress_path.exists():
        pytest.skip("state/progress.json not created yet")
    data = json.loads(progress_path.read_text())
    completed = set(data.get("completed_tasks", []))
    already_done_or_current = completed | {data["current_task"]}
    assert {1, 2, 3, 4, 5} <= already_done_or_current, "Tasks 1-5 (already complete in this repository's real history) must not be un-recorded"
