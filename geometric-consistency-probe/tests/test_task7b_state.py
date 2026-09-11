"""Task 7B's own state machine: terminal statuses block further
transitions, and save/load round-trips exactly -- writes only
state/task_07b_result.json-shaped files (never touches Task 7's own
state/task_07_result.json, which this module doesn't even import)."""
from __future__ import annotations

import pytest

from experiments.task7b_latent_transformation_discovery import state as st


def test_fresh_state_is_not_started():
    s = st.Task7BState()
    assert s.status == st.NOT_STARTED


def test_transition_updates_status_and_history():
    s = st.Task7BState()
    s.transition(st.PROTOCOL_FROZEN, protocol_hash="abc123")
    assert s.status == st.PROTOCOL_FROZEN
    assert s.history[-1]["to"] == st.PROTOCOL_FROZEN


@pytest.mark.parametrize("terminal", [st.COMPLETE, st.COMPLETE_SCALE_LIMITED, st.FAILED_SOFTWARE, st.INVALID_CONTAMINATED])
def test_terminal_statuses_block_further_transitions(terminal):
    s = st.Task7BState(status=terminal)
    with pytest.raises(st.StateError):
        s.transition(st.PROTOCOL_FROZEN)


def test_save_and_load_round_trip(tmp_path):
    s = st.Task7BState()
    s.transition(st.PROTOCOL_FROZEN, protocol_hash="deadbeef")
    s.transition(st.DATA_VALIDATED)
    path = tmp_path / "task_07b_result.json"
    st.save(s, path)
    reloaded = st.load(path)
    assert reloaded.status == st.DATA_VALIDATED
    assert len(reloaded.history) == 2


def test_load_missing_file_returns_fresh_state(tmp_path):
    s = st.load(tmp_path / "does_not_exist.json")
    assert s.status == st.NOT_STARTED
