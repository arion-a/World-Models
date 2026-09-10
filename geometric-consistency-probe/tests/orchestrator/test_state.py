"""Unit tests for orchestrator/state.py: loading, persistence, next-task
determination, malformed-state handling, transition legality, and the
structural guarantee that current_task can only advance via
advance_task() after status == PASSED.
"""

from __future__ import annotations

import json

import pytest

from orchestrator import state as st


# --- bootstrap / loading ----------------------------------------------------


def test_bootstrap_reflects_tasks_1_through_5_complete():
    s = st.OrchestratorState.bootstrap()
    assert s.current_task == 6
    assert s.completed_tasks == [1, 2, 3, 4, 5]
    assert s.status == st.READY


def test_load_missing_file_bootstraps_without_writing(tmp_path):
    path = tmp_path / "progress.json"
    s = st.load(path)
    assert s.current_task == 6
    assert not path.exists(), "load() must not write anything itself"


def test_load_existing_file_round_trips(tmp_path):
    path = tmp_path / "progress.json"
    s = st.OrchestratorState.bootstrap()
    st.save(s, path)
    reloaded = st.load(path)
    assert reloaded.to_dict() == s.to_dict()


def test_save_is_atomic_no_leftover_tmp_file(tmp_path):
    path = tmp_path / "progress.json"
    st.save(st.OrchestratorState.bootstrap(), path)
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == []


# --- malformed state handling ------------------------------------------------


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{not json")
    with pytest.raises(st.StateError):
        st.load(path)


def test_load_rejects_current_task_also_in_completed(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"current_task": 6, "completed_tasks": [1, 2, 3, 4, 5, 6], "status": "READY"}))
    with pytest.raises(st.StateError):
        st.load(path)


def test_load_rejects_gap_in_completed_tasks(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"current_task": 6, "completed_tasks": [1, 2, 4, 5], "status": "READY"}))
    with pytest.raises(st.StateError):
        st.load(path)


def test_load_rejects_unknown_status(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"current_task": 6, "completed_tasks": [1, 2, 3, 4, 5], "status": "TOTALLY_MADE_UP"}))
    with pytest.raises(st.StateError):
        st.load(path)


def test_load_rejects_missing_current_task_field(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"completed_tasks": [1, 2, 3, 4, 5], "status": "READY"}))
    with pytest.raises(st.StateError):
        st.load(path)


def test_load_rejects_out_of_range_task(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"current_task": 99, "completed_tasks": list(range(1, 99)), "status": "READY"}))
    with pytest.raises(st.StateError):
        st.load(path)


def test_load_rejects_duplicate_completed_tasks(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"current_task": 6, "completed_tasks": [1, 1, 2, 3, 4, 5], "status": "READY"}))
    with pytest.raises(st.StateError):
        st.load(path)


# --- next_task / is_complete -------------------------------------------------


def test_next_task_returns_current_task_when_incomplete():
    s = st.OrchestratorState.bootstrap()
    assert s.next_task() == 6
    assert not s.is_complete()


def test_next_task_returns_none_when_all_18_complete():
    s = st.OrchestratorState(current_task=19, completed_tasks=list(range(1, 19)), status=st.READY)
    s.validate()
    assert s.next_task() is None
    assert s.is_complete()


# --- transition legality -----------------------------------------------------


@pytest.mark.parametrize(
    "start,target",
    [
        (st.READY, st.QA),
        (st.READY, st.PASSED),
        (st.READY, st.FIXING),
        (st.RUNNING, st.PASSED),
        (st.RUNNING, st.FIXING),
        (st.PASSED, st.RUNNING),
        (st.FAILED, st.RUNNING),
        (st.BLOCKED, st.RUNNING),
        (st.QA, st.RUNNING),
    ],
)
def test_illegal_transitions_are_rejected(start, target):
    s = st.OrchestratorState.bootstrap()
    s.status = start
    with pytest.raises(st.StateError):
        s.transition(target)
    assert s.status == start, "a rejected transition must not partially apply"


def test_passed_to_blocked_is_legal():
    """QA passing does not guarantee the checkpoint commit itself can be
    made (orchestrator/run.py's _finish_passed_task falls back to
    BLOCKED when both create_checkpoint and its get_status recovery
    fallback fail) -- this must be a legal transition, not something
    that crashes the orchestrator with an illegal-transition StateError
    on top of the underlying git failure."""
    s = st.OrchestratorState.bootstrap()
    s.status = st.PASSED
    s.transition(st.BLOCKED)
    assert s.status == st.BLOCKED


def test_self_transition_is_a_safe_no_op():
    """This is what makes resuming after an interruption safe: whichever
    status the process died in, re-entering the loop and re-asserting
    that same status must not raise."""
    for status in st.STATUSES:
        s = st.OrchestratorState.bootstrap()
        s.status = status
        s.transition(status)
        assert s.status == status


def test_full_happy_path_transition_sequence():
    s = st.OrchestratorState.bootstrap()
    s.transition(st.RUNNING, task=6)
    s.transition(st.QA, task=6)
    s.transition(st.PASSED, task=6)
    assert s.status == st.PASSED


def test_qa_can_go_to_fixing_or_blocked():
    s = st.OrchestratorState.bootstrap()
    s.transition(st.RUNNING)
    s.transition(st.QA)
    s.transition(st.FIXING)
    assert s.status == st.FIXING
    s.transition(st.RUNNING)
    s.transition(st.QA)
    s.transition(st.BLOCKED)
    assert s.status == st.BLOCKED


# --- advance_task: the single point of "task ordering" enforcement ----------


def test_advance_task_requires_passed_status():
    s = st.OrchestratorState.bootstrap()
    for bad_status in (st.READY, st.RUNNING, st.QA, st.FIXING, st.FAILED, st.BLOCKED):
        s.status = bad_status
        with pytest.raises(st.StateError):
            s.advance_task("deadbeef")
        assert s.current_task == 6, "current_task must never move without status == PASSED"


def test_advance_task_moves_current_task_and_records_commit():
    s = st.OrchestratorState.bootstrap()
    s.transition(st.RUNNING)
    s.transition(st.QA)
    s.transition(st.PASSED)
    new_task = s.advance_task("abc123")
    assert new_task == 7
    assert s.current_task == 7
    assert s.completed_tasks == [1, 2, 3, 4, 5, 6]
    assert s.commits["6"] == "abc123"
    assert s.status == st.READY


def test_advance_task_is_idempotent_safe_against_double_call():
    """After advancing once, calling advance_task again must fail (status
    is READY, not PASSED) rather than silently advancing twice."""
    s = st.OrchestratorState.bootstrap()
    s.transition(st.RUNNING)
    s.transition(st.QA)
    s.transition(st.PASSED)
    s.advance_task("abc123")
    with pytest.raises(st.StateError):
        s.advance_task("def456")
    assert s.current_task == 7


# --- reset_task (explicit operator override only) ---------------------------


def test_reset_task_only_valid_from_blocked_or_failed():
    s = st.OrchestratorState.bootstrap()
    with pytest.raises(st.StateError):
        s.reset_task(6)  # status is READY, not BLOCKED/FAILED


def test_reset_task_returns_to_ready_and_clears_attempts():
    s = st.OrchestratorState.bootstrap()
    s.transition(st.RUNNING)
    s.record_attempt(6)
    s.record_attempt(6)
    s.transition(st.QA)
    s.transition(st.BLOCKED)
    s.reset_task(6)
    assert s.status == st.READY
    assert s.attempts["6"] == 0
    assert s.current_task == 6  # reset does not skip or advance the task


def test_reset_task_rejects_wrong_task_number():
    s = st.OrchestratorState.bootstrap()
    s.transition(st.RUNNING)
    s.transition(st.QA)
    s.transition(st.BLOCKED)
    with pytest.raises(st.StateError):
        s.reset_task(7)  # current_task is 6, not 7


# --- record_attempt / record_qa ----------------------------------------------


def test_record_attempt_increments_per_task():
    s = st.OrchestratorState.bootstrap()
    assert s.record_attempt(6) == 1
    assert s.record_attempt(6) == 2
    assert s.attempts["6"] == 2


def test_record_qa_stores_summary_by_task():
    s = st.OrchestratorState.bootstrap()
    s.record_qa(6, {"task": 6, "passed": False, "layers": []})
    assert s.last_qa["6"]["passed"] is False


def test_history_is_capped():
    s = st.OrchestratorState.bootstrap()
    s.transition(st.RUNNING)
    for _ in range(600):
        s.transition(st.RUNNING)  # self-transition no-op, still appends a history entry
    assert len(s.history) <= st._MAX_HISTORY
