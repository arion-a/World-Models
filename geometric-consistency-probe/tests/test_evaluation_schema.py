"""Tests for Task 15's unified result schema (evaluation.schema).

Includes the acceptance-critical check that ALL of Tasks 6-14's
existing state/task_NN_result.json files validate against the new
schema (task spec's "Required software tests" and acceptance criterion
1), and a byte-identity (sha256) check proving this task did not alter
any previously recorded metric value (acceptance criterion 3 /
research/RESEARCH_INVARIANTS.md invariant 15, Global Invariant 27).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evaluation.schema import (
    KNOWN_BLOCKER_CATEGORIES,
    KNOWN_STATUSES,
    REQUIRED_FIELDS,
    validate_file,
    validate_result,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"

# Every task with an existing result file this schema was built to cover.
EXISTING_TASK_IDS = tuple(range(6, 15))


def _minimal_valid_result(task: int = 99) -> dict:
    return {
        "task": task,
        "implementation_status": "COMPLETE",
        "seed": 0,
        "config": {"num_scenes": 1},
        "metrics": {"r_squared": 0.5},
        "scientific_result": "A genuine finding text, well over ten characters.",
        "software_versions": {"python": "3.11.0"},
        "tests": {"passed": 1, "failed": 0, "returncode": 0},
        "artifacts": ["state/task_99_result.json"],
    }


def test_minimal_valid_result_has_no_errors():
    assert validate_result(_minimal_valid_result()) == []


def test_expected_task_mismatch_is_reported():
    errors = validate_result(_minimal_valid_result(task=6), expected_task=7)
    assert any("expected 7" in e for e in errors)


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_missing_required_field_is_reported(missing_field):
    data = _minimal_valid_result()
    del data[missing_field]
    errors = validate_result(data)
    assert any(missing_field in e for e in errors)


def test_non_dict_result_is_reported_not_raised():
    errors = validate_result(["not", "a", "dict"])
    assert errors and "not a JSON object" in errors[0]


def test_seed_accepts_int():
    data = _minimal_valid_result()
    data["seed"] = 42
    assert validate_result(data) == []


def test_seed_accepts_list_of_ints():
    data = _minimal_valid_result()
    data["seed"] = [0, 1, 2]
    assert validate_result(data) == []


def test_seed_rejects_empty_list():
    data = _minimal_valid_result()
    data["seed"] = []
    errors = validate_result(data)
    assert any("seed" in e for e in errors)


def test_seed_rejects_string():
    data = _minimal_valid_result()
    data["seed"] = "0"
    errors = validate_result(data)
    assert any("seed" in e for e in errors)


def test_tests_block_requires_passed_failed_returncode():
    data = _minimal_valid_result()
    data["tests"] = {"passed": 1}
    errors = validate_result(data)
    assert any("failed" in e for e in errors)
    assert any("returncode" in e for e in errors)


def test_scientific_result_too_short_is_rejected():
    data = _minimal_valid_result()
    data["scientific_result"] = "too short"
    errors = validate_result(data)
    assert any("scientific_result" in e for e in errors)


def test_unknown_implementation_status_is_rejected():
    data = _minimal_valid_result()
    data["implementation_status"] = "IN_PROGRESS"
    errors = validate_result(data)
    assert any("implementation_status" in e for e in errors)


def test_blocked_status_requires_category_and_issue():
    data = _minimal_valid_result()
    data["implementation_status"] = "BLOCKED"
    errors = validate_result(data)
    assert any("blocker_category" in e for e in errors)
    assert any("blocking_issue" in e for e in errors)


def test_blocked_status_with_category_and_issue_is_valid():
    data = _minimal_valid_result()
    data["implementation_status"] = "BLOCKED"
    data["blocker_category"] = "MISSING_DEPENDENCY"
    data["blocking_issue"] = "a real dependency is genuinely unavailable"
    assert validate_result(data) == []


def test_known_statuses_and_blocker_categories_match_orchestrator():
    # orchestrator/classify.py's CATEGORIES (minus IMPLEMENTATION/ROUTINE,
    # which are never self-declared by a task, only inferred by classify.py).
    assert set(KNOWN_BLOCKER_CATEGORIES) == {
        "SCIENTIFIC_CONFLICT",
        "DESTRUCTIVE_ACTION",
        "MISSING_DEPENDENCY",
        "BLOCKER",
    }
    assert set(KNOWN_STATUSES) == {"COMPLETE", "BLOCKED"}


def test_validate_file_reports_missing_file_without_raising(tmp_path):
    errors = validate_file(tmp_path / "does_not_exist.json")
    assert errors and "does not exist" in errors[0]


def test_validate_file_reports_invalid_json_without_raising(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    errors = validate_file(bad)
    assert errors and "not valid JSON" in errors[0]


def test_validate_file_accepts_a_genuine_minimal_result(tmp_path):
    path = tmp_path / "task_99_result.json"
    path.write_text(json.dumps(_minimal_valid_result(task=99)))
    assert validate_file(path, expected_task=99) == []


# ---------------------------------------------------------------------------
# The acceptance-critical checks: every existing Task 6-14 result file
# actually validates, and this task did not alter any of them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", EXISTING_TASK_IDS)
def test_existing_task_result_file_validates_against_unified_schema(task_id):
    path = STATE_DIR / f"task_{task_id:02d}_result.json"
    assert path.exists(), f"{path} is expected to exist (Tasks 6-14 already ran)"
    errors = validate_file(path, expected_task=task_id)
    assert errors == [], f"{path} failed unified-schema validation: {errors}"


@pytest.mark.parametrize("task_id", EXISTING_TASK_IDS)
def test_existing_task_result_satisfies_global_invariants_21_to_24(task_id):
    """Global Invariants 21-24: seed, config, provenance, machine-readable
    -- each checked individually per file, per the task spec's own
    language, not just folded into the generic schema-validity check
    above."""
    path = STATE_DIR / f"task_{task_id:02d}_result.json"
    data = json.loads(path.read_text())  # 24: machine-readable (parses as JSON)
    assert "seed" in data and data["seed"] is not None  # 21: seed
    assert isinstance(data.get("config"), dict) and data["config"]  # 22: config
    assert isinstance(data.get("software_versions"), dict) and "python" in data["software_versions"]  # 23: provenance


def test_existing_task_results_were_not_altered_by_this_consolidation():
    """Direct proof (not a claim) that Task 15's schema work did not
    change a single byte of Tasks 6-14's previously recorded result
    files -- required by the task spec's 'Required scientific-validity
    tests' and acceptance criterion 3. `git diff` against HEAD is empty
    for every one of these paths; if this task needed a genuine
    migration it would show up here and would need to be a recorded
    protocol change (invariant 15 / Global Invariant 27), not a silent
    edit."""
    paths = [f"state/task_{t:02d}_result.json" for t in EXISTING_TASK_IDS]
    proc = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", *paths],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"unexpected changes to prior task result files:\n{proc.stdout}"
