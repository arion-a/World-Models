"""Unit tests for orchestrator/sync.py: deterministic reconciliation of
tasks/{NN}_*.md with research/CANONICAL_RESEARCH_PROTOCOL.md.

These exercise the module directly against throwaway temp directories
(no fake git repo needed -- sync.py only touches the filesystem, never
git), complementing the indirect coverage sync.py already gets via
tests/orchestrator/conftest.py's make_fake_repo (which calls
sync_task_spec for real to build its fixtures).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator import sync

CANONICAL_ONE_TASK = """\
# TASK 6 — FAKE TASK SIX

### Purpose

Fake purpose for task 6.

### Scientific question

Fake scientific question for task 6.

### Why this task exists

Fake why this task exists for task 6.

### Relationship to previous tasks

Fake relationship to previous tasks for task 6.

### Hypothesis

Fake hypothesis for task 6.

### Mathematical formulation

Fake mathematical formulation for task 6.

### Inputs

Fake inputs for task 6.

### Outputs

Fake outputs for task 6.

### Experimental protocol

Fake experimental protocol for task 6.

### Dataset requirements

Fake dataset requirements for task 6.

### Train/test protocol

Fake train/test protocol for task 6.

### Controls

Fake controls for task 6.

### Baselines

Fake baselines for task 6.

### Metrics

Fake metrics for task 6.

### Required artifacts

Fake required artifacts for task 6.

### Required software tests

Fake required software tests for task 6.

### Required scientific-validity tests

Fake required scientific-validity tests for task 6.

### Required research-alignment checks

Fake required research-alignment checks for task 6.

### Required leakage checks

Fake required leakage checks for task 6.

### Reproducibility requirements

Fake reproducibility requirements for task 6.

### Failure conditions

Fake failure conditions for task 6.

### Acceptance criteria

Fake acceptance criteria for task 6.

### Interpretation rules

Fake interpretation rules for task 6.

### Explicit prohibited shortcuts

Fake explicit prohibited shortcuts for task 6.

### What a positive result means

Fake what a positive result means for task 6.

### What a negative result means

Fake what a negative result means for task 6.

### What this task does NOT establish

Fake what this task does not establish for task 6.

## Validation performed on this document

Not a task -- must not be parsed as one.
"""


def _write_canonical(tmp_path: Path, text: str = CANONICAL_ONE_TASK) -> Path:
    research = tmp_path / "research"
    research.mkdir(parents=True, exist_ok=True)
    path = research / "CANONICAL_RESEARCH_PROTOCOL.md"
    path.write_text(text)
    return path


# --- parse_canonical_task_section ---------------------------------------------


def test_parse_finds_the_requested_task():
    parsed = sync.parse_canonical_task_section(CANONICAL_ONE_TASK, 6)
    assert parsed is not None
    title, fields = parsed
    assert title == "FAKE TASK SIX"
    assert set(fields) == set(sync.CANONICAL_FIELD_ORDER)
    assert fields["Purpose"] == "Fake purpose for task 6."


def test_parse_returns_none_for_absent_task():
    assert sync.parse_canonical_task_section(CANONICAL_ONE_TASK, 99) is None


def test_parse_stops_at_validation_section_not_field_bodies():
    parsed = sync.parse_canonical_task_section(CANONICAL_ONE_TASK, 6)
    _, fields = parsed
    assert "Not a task" not in fields["What this task does NOT establish"]


def test_parse_stops_at_next_task_header():
    two_tasks = CANONICAL_ONE_TASK.replace(
        "## Validation performed on this document\n\nNot a task -- must not be parsed as one.\n",
        "",
    ) + "# TASK 7 — FAKE TASK SEVEN\n\n### Purpose\n\nFake purpose for task 7.\n"
    parsed6 = sync.parse_canonical_task_section(two_tasks, 6)
    assert "task 7" not in parsed6[1]["Purpose"].lower()
    parsed7 = sync.parse_canonical_task_section(two_tasks, 7)
    assert parsed7[1]["Purpose"] == "Fake purpose for task 7."


# --- render_task_spec_from_canonical: purity/determinism ----------------------


def test_render_is_deterministic():
    parsed = sync.parse_canonical_task_section(CANONICAL_ONE_TASK, 6)
    title, fields = parsed
    rendered_a = sync.render_task_spec_from_canonical(6, title, fields)
    rendered_b = sync.render_task_spec_from_canonical(6, title, fields)
    assert rendered_a == rendered_b


def test_render_contains_all_nine_required_headers():
    parsed = sync.parse_canonical_task_section(CANONICAL_ONE_TASK, 6)
    title, fields = parsed
    rendered = sync.render_task_spec_from_canonical(6, title, fields)
    for header_name, _ in sync.SECTION_MAP:
        assert f"## {header_name}" in rendered


def test_render_missing_field_renders_placeholder_not_a_crash():
    rendered = sync.render_task_spec_from_canonical(6, "Fake title", {})
    assert "not specified in the canonical protocol" in rendered


# --- sync_task_spec: the full read-canonical/write-derived cycle --------------


def test_sync_creates_the_task_file(tmp_path):
    _write_canonical(tmp_path)
    result = sync.sync_task_spec(6, tmp_path)
    assert result.changed is True
    assert result.path.exists()
    assert result.path.name == "06_fake_task_six.md"
    assert "Fake purpose for task 6." in result.path.read_text()


def test_sync_is_idempotent(tmp_path):
    _write_canonical(tmp_path)
    first = sync.sync_task_spec(6, tmp_path)
    second = sync.sync_task_spec(6, tmp_path)
    assert first.changed is True
    assert second.changed is False
    assert first.path == second.path


def test_sync_overwrites_hand_edits_to_match_canonical(tmp_path):
    _write_canonical(tmp_path)
    result = sync.sync_task_spec(6, tmp_path)
    result.path.write_text("hand-edited nonsense that does not match the canonical protocol\n")

    resynced = sync.sync_task_spec(6, tmp_path)
    assert resynced.changed is True
    assert "hand-edited nonsense" not in resynced.path.read_text()
    assert "Fake purpose for task 6." in resynced.path.read_text()


def test_sync_reuses_existing_filename_under_a_different_slug(tmp_path):
    _write_canonical(tmp_path)
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "06_some_other_hand_chosen_name.md").write_text("stale content\n")

    result = sync.sync_task_spec(6, tmp_path)
    assert result.path.name == "06_some_other_hand_chosen_name.md"
    assert not (tasks_dir / "06_fake_task_six.md").exists()
    assert "Fake purpose for task 6." in result.path.read_text()


def test_sync_dry_run_does_not_write(tmp_path):
    _write_canonical(tmp_path)
    result = sync.sync_task_spec(6, tmp_path, dry_run=True)
    assert result.changed is True
    assert not result.path.exists(), "dry_run=True must never write to disk"
    assert "Fake purpose for task 6." in result.content


def test_sync_dry_run_reports_changed_false_when_already_in_sync(tmp_path):
    _write_canonical(tmp_path)
    sync.sync_task_spec(6, tmp_path)  # real sync first
    result = sync.sync_task_spec(6, tmp_path, dry_run=True)
    assert result.changed is False
    assert result.content == result.path.read_text()


def test_sync_dry_run_never_overwrites_a_stale_file(tmp_path):
    _write_canonical(tmp_path)
    sync.sync_task_spec(6, tmp_path)
    tasks_dir = tmp_path / "tasks"
    spec_path = next(tasks_dir.glob("06_*.md"))
    spec_path.write_text("stale hand-edit\n")

    result = sync.sync_task_spec(6, tmp_path, dry_run=True)
    assert result.changed is True
    assert spec_path.read_text() == "stale hand-edit\n"
    assert "Fake purpose for task 6." in result.content


def test_sync_raises_when_canonical_has_no_section_for_task(tmp_path):
    _write_canonical(tmp_path)
    with pytest.raises(sync.CanonicalSectionNotFound):
        sync.sync_task_spec(99, tmp_path)


def test_sync_raises_when_canonical_file_itself_is_missing(tmp_path):
    with pytest.raises(sync.CanonicalSectionNotFound):
        sync.sync_task_spec(6, tmp_path)


def test_sync_reflects_a_canonical_edit_on_next_call(tmp_path):
    canonical_path = _write_canonical(tmp_path)
    first = sync.sync_task_spec(6, tmp_path)
    assert "Fake purpose for task 6." in first.path.read_text()

    updated_text = CANONICAL_ONE_TASK.replace("Fake purpose for task 6.", "Updated purpose for task 6.")
    canonical_path.write_text(updated_text)

    second = sync.sync_task_spec(6, tmp_path)
    assert second.changed is True
    assert "Updated purpose for task 6." in second.path.read_text()
    assert "Fake purpose for task 6." not in second.path.read_text()
