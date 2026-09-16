"""Tests for evaluation.validate_existing_results -- Task 15 requirement
3: re-validate Tasks 6-14's existing result files against the unified
schema and migrate in place only if actually necessary."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.validate_existing_results import (
    STATE_DIR,
    UNIFIED_TASK_IDS,
    all_valid,
    validate_all,
)


def test_unified_task_ids_is_6_through_14():
    assert UNIFIED_TASK_IDS == tuple(range(6, 15))


def test_validate_all_against_real_state_dir_reports_no_errors():
    report = validate_all()
    assert set(report.keys()) == set(UNIFIED_TASK_IDS)
    for task_id, entry in report.items():
        assert entry["errors"] == [], f"task {task_id}: {entry['errors']}"
    assert all_valid(report)


def test_validate_all_performs_no_migration_when_none_is_needed():
    report = validate_all()
    for task_id, entry in report.items():
        assert entry["sha256_before"] == entry["sha256_after"], f"task {task_id} was modified on disk"
        assert entry["migrated"] is False


def test_validate_all_reports_missing_file_without_raising(tmp_path):
    # An empty state dir: every task file is "missing" -- should report,
    # not raise.
    report = validate_all(state_dir=tmp_path)
    assert not all_valid(report)
    for entry in report.values():
        assert entry["errors"]
        assert "does not exist" in entry["errors"][0]


def test_validate_all_reports_invalid_json_without_raising(tmp_path):
    (tmp_path / "task_06_result.json").write_text("{not valid json")
    report = validate_all(state_dir=tmp_path)
    assert report[6]["errors"]
    assert "not valid JSON" in report[6]["errors"][0]


def test_state_dir_default_points_at_real_state_directory():
    assert STATE_DIR == Path(__file__).resolve().parent.parent / "state"
