"""Integration tests for orchestrator/run.py's main loop: the state
machine, the repair loop, git checkpointing, dry-run, and
interruption/resume -- all against an isolated throwaway git repo, with
`claude_client.invoke_claude` monkeypatched so no real `claude` CLI is
ever invoked.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from orchestrator import claude_client, git_ops, qa, run, state
from tests.orchestrator.conftest import make_fake_repo, write_valid_result


@pytest.fixture
def repo(tmp_path):
    return make_fake_repo(tmp_path / "repo", task=6)


def _args(repo, **overrides):
    ns = run.build_arg_parser().parse_args([])
    ns.repo_root = str(repo)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _fake_claude_factory(behaviors):
    """behaviors: list of callables, one per invocation, each given
    (prompt, repo_root) and returning a ClaudeResult. Raises if called
    more times than behaviors provided (catches accidental extra
    invocations)."""
    calls = {"n": 0}

    def fake_invoke(prompt, repo_root, config=None, log_path=None):
        idx = calls["n"]
        calls["n"] += 1
        if idx >= len(behaviors):
            raise AssertionError(f"invoke_claude called more times ({idx + 1}) than expected ({len(behaviors)})")
        result = behaviors[idx](prompt, repo_root)
        if log_path is not None:
            # The real invoke_claude always writes a log; mirror that here
            # so tests can verify run.py wires log_path through correctly.
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(f"=== fake invocation {idx + 1} ===\nprompt: {len(prompt)} chars\nreturncode: {result.returncode}\n\n")
        return result

    fake_invoke.calls = calls
    return fake_invoke


def _writes_valid_result(task=6):
    def _behavior(prompt, repo_root):
        write_valid_result(repo_root, task=task)
        return claude_client.ClaudeResult(returncode=0, stdout="{}", stderr="", parsed_json={}, duration_seconds=0.1, command=["claude"])

    return _behavior


def _writes_nothing():
    def _behavior(prompt, repo_root):
        return claude_client.ClaudeResult(returncode=0, stdout="{}", stderr="", parsed_json={}, duration_seconds=0.1, command=["claude"])

    return _behavior


# --- successful advancement --------------------------------------------------


def test_successful_task_advances_state_and_creates_checkpoint(repo, monkeypatch):
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_valid_result()]))
    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 0

    s = state.load(repo / "state" / "progress.json")
    assert s.current_task == 7
    assert 6 in s.completed_tasks
    assert s.status == state.READY
    assert s.commits["6"] is not None

    log = subprocess.run(["git", "log", "--format=%s"], cwd=repo, capture_output=True, text=True, check=True)
    assert any(line.startswith("task-06-pass") for line in log.stdout.splitlines())


def test_qa_log_and_task_log_are_written(repo, monkeypatch):
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_valid_result()]))
    run.main(["--repo-root", str(repo)])
    assert (repo / "logs" / "task_06.log").exists()
    assert (repo / "logs" / "task_06_qa.log").exists()
    assert "PASS" in (repo / "logs" / "task_06_qa.log").read_text()


# --- repair loop --------------------------------------------------------------


def test_repair_loop_succeeds_on_second_attempt(repo, monkeypatch):
    monkeypatch.setattr(
        run.claude_client,
        "invoke_claude",
        _fake_claude_factory([_writes_nothing(), _writes_valid_result()]),
    )
    exit_code = run.main(["--repo-root", str(repo), "--max-fix-attempts", "3"])
    assert exit_code == 0
    s = state.load(repo / "state" / "progress.json")
    assert s.current_task == 7
    assert s.attempts["6"] == 2


def test_repair_loop_uses_fix_prompt_referencing_prior_failure(repo, monkeypatch):
    prompts_seen = []

    def behavior_1(prompt, repo_root):
        prompts_seen.append(prompt)
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    def behavior_2(prompt, repo_root):
        prompts_seen.append(prompt)
        write_valid_result(repo_root, task=6)
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([behavior_1, behavior_2]))
    run.main(["--repo-root", str(repo), "--max-fix-attempts", "3"])

    assert len(prompts_seen) == 2
    assert "attempt 2" in prompts_seen[1]
    assert "QA FAILURE REPORT" in prompts_seen[1]
    assert "does not exist" in prompts_seen[1] or "FAIL" in prompts_seen[1]


def test_max_fix_attempts_exhausted_marks_blocked_and_stops(repo, monkeypatch):
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_nothing(), _writes_nothing(), _writes_nothing()]))
    exit_code = run.main(["--repo-root", str(repo), "--max-fix-attempts", "3"])
    assert exit_code == 1

    s = state.load(repo / "state" / "progress.json")
    assert s.status == state.BLOCKED
    assert s.current_task == 6, "a blocked task must never advance"
    assert 6 not in s.completed_tasks


def test_blocked_task_refuses_to_run_again_without_reset(repo, monkeypatch):
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_nothing()] * 3))
    run.main(["--repo-root", str(repo), "--max-fix-attempts", "3"])

    # A second invocation must refuse outright -- no Claude call at all.
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([]))
    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 1
    s = state.load(repo / "state" / "progress.json")
    assert s.status == state.BLOCKED
    assert s.current_task == 6


def test_reset_task_allows_retry_after_blocked(repo, monkeypatch):
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_nothing()] * 3))
    run.main(["--repo-root", str(repo), "--max-fix-attempts", "3"])
    assert state.load(repo / "state" / "progress.json").status == state.BLOCKED

    exit_code = run.main(["--repo-root", str(repo), "--reset-task", "6"])
    assert exit_code == 0
    assert state.load(repo / "state" / "progress.json").status == state.READY

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_valid_result()]))
    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 0
    assert state.load(repo / "state" / "progress.json").current_task == 7


# --- Claude invocation failure ------------------------------------------------


def test_claude_invocation_error_marks_failed_not_blocked(repo, monkeypatch):
    def raise_error(prompt, repo_root, config=None, log_path=None):
        raise claude_client.ClaudeInvocationError("claude executable not found")

    monkeypatch.setattr(run.claude_client, "invoke_claude", raise_error)
    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 2
    s = state.load(repo / "state" / "progress.json")
    assert s.status == state.FAILED
    assert s.current_task == 6


# --- dry-run: absolutely no side effects -------------------------------------


def test_dry_run_makes_no_changes(repo, monkeypatch, capsys):
    def fail_if_called(*a, **k):
        raise AssertionError("invoke_claude must never be called during --dry-run")

    monkeypatch.setattr(run.claude_client, "invoke_claude", fail_if_called)

    status_before = git_ops.get_status(repo)
    exit_code = run.main(["--repo-root", str(repo), "--dry-run"])
    assert exit_code == 0

    assert not (repo / "state" / "progress.json").exists(), "dry-run must not write progress.json"
    status_after = git_ops.get_status(repo)
    assert status_before.commit == status_after.commit
    assert status_after.clean

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "next task: 6" in out


def test_dry_run_does_not_write_a_stale_task_spec(tmp_path, capsys):
    """Regression: --dry-run must never call the real (writing)
    sync_task_spec -- a genuinely stale tasks/{NN}_*.md is exactly the
    kind of fact --dry-run exists to report, not silently fix on disk.
    (Caught by running --dry-run against the real repo, which has a
    genuinely stale tasks/07_geometric_consistency.md -- the other
    dry-run tests all use fixtures whose spec already matches the
    canonical protocol, which hid this bug.)"""
    repo = make_fake_repo(tmp_path / "repo", canonical_tasks={6: "Fake task six"})
    spec_path = next((repo / "tasks").glob("06_*.md"))
    original_content = spec_path.read_text()
    spec_path.write_text("this file is stale and disagrees with the canonical protocol\n")

    exit_code = run.main(["--repo-root", str(repo), "--dry-run"])
    assert exit_code == 0

    assert spec_path.read_text() == "this file is stale and disagrees with the canonical protocol\n", (
        "dry-run must not have overwritten the stale spec"
    )
    out = capsys.readouterr().out
    assert "task spec sync: would update" in out
    # The prompt preview must still reflect the CORRECT (synced) content
    # the real run would actually send, not the stale on-disk content.
    assert "Fake purpose for task 6." in original_content  # sanity: canonical content differs from the stale file


def test_dry_run_reports_task_spec_missing_without_erroring(tmp_path, capsys):
    # No '# TASK 6 -- ...' section in the canonical protocol at all --
    # sync.sync_task_spec(6, ...) must raise CanonicalSectionNotFound
    # rather than the orchestrator fabricating a task.
    repo = make_fake_repo(tmp_path / "repo2", canonical_tasks={})
    exit_code = run.main(["--repo-root", str(repo), "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Canonical protocol has no section for task 6 yet" in out


def test_missing_task_spec_stops_cleanly_in_real_run(tmp_path):
    """Per the user's own workflow: task N+1's canonical section is only
    written after task N is accepted. If it's not there yet, the
    orchestrator must stop cleanly -- not error, not fabricate work."""
    repo = make_fake_repo(tmp_path / "repo3", canonical_tasks={})
    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 0
    assert not (repo / "state" / "progress.json").exists()


# --- protection: dirty tree refuses to start ---------------------------------


def test_dirty_tree_refuses_to_start_without_allow_dirty(repo, monkeypatch):
    (repo / "unrelated_uncommitted.txt").write_text("someone's in-progress work\n")
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([]))
    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 2
    assert (repo / "unrelated_uncommitted.txt").exists(), "must never discard unrelated uncommitted work"
    assert not (repo / "state" / "progress.json").exists()


# --- interruption / resume behavior ------------------------------------------


def test_resume_from_passed_status_finishes_checkpoint_without_reinvoking_claude(repo, monkeypatch):
    """Simulates a crash between QA passing and the checkpoint commit
    completing: status is PASSED, current_task is still 6, but the
    result file (and hence git working tree) already reflects a
    completed task. Resuming must finish the checkpoint, not re-run
    Claude or QA."""
    write_valid_result(repo, task=6)
    s = state.OrchestratorState.bootstrap()
    s.transition(state.RUNNING, task=6)
    s.record_attempt(6)
    s.transition(state.QA, task=6)
    s.record_qa(6, {"task": 6, "passed": True, "layers": []})
    s.transition(state.PASSED, task=6)
    state.save(s, repo / "state" / "progress.json")

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([]))  # must not be called
    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 0

    resumed = state.load(repo / "state" / "progress.json")
    assert resumed.current_task == 7
    assert 6 in resumed.completed_tasks
    log = subprocess.run(["git", "log", "--format=%s"], cwd=repo, capture_output=True, text=True, check=True)
    assert any(line.startswith("task-06-pass") for line in log.stdout.splitlines())


def test_resume_from_fixing_status_sends_a_fix_prompt_not_initial_prompt(repo, monkeypatch):
    """Simulates a crash right after QA failed once and status was set
    to FIXING. Resuming must send attempt 2 as a FIX prompt (using the
    persisted last_qa), not restart at attempt 1 with the initial
    prompt."""
    s = state.OrchestratorState.bootstrap()
    s.transition(state.RUNNING, task=6)
    s.record_attempt(6)  # attempt 1 already used
    s.transition(state.QA, task=6)
    s.record_qa(6, {"task": 6, "passed": False, "layers": [{"name": "A. SOFTWARE CORRECTNESS", "passed": False, "details": ["result file missing"]}]})
    s.transition(state.FIXING, task=6)
    state.save(s, repo / "state" / "progress.json")

    prompts_seen = []

    def behavior(prompt, repo_root):
        prompts_seen.append(prompt)
        write_valid_result(repo_root, task=6)
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([behavior]))
    exit_code = run.main(["--repo-root", str(repo), "--max-fix-attempts", "3"])
    assert exit_code == 0
    assert len(prompts_seen) == 1
    assert "attempt 2" in prompts_seen[0]
    assert "result file missing" in prompts_seen[0]

    resumed = state.load(repo / "state" / "progress.json")
    assert resumed.current_task == 7
    assert resumed.attempts["6"] == 2


def test_resume_from_qa_status_reruns_qa_without_reinvoking_claude(repo, monkeypatch):
    """Simulates a crash DURING qa.run_qa itself (Claude's attempt-1 work
    already finished and status moved to QA, but the QA subprocess was
    killed before recording a result): resuming must re-run QA on the
    attempt already sitting in the working tree, never re-invoke Claude
    for a new attempt -- both to avoid wasting an attempt with no QA
    report to build a fix prompt from, and because this exact resume
    path used to crash (see the next test)."""
    write_valid_result(repo, task=6)
    s = state.OrchestratorState.bootstrap()
    s.transition(state.RUNNING, task=6)
    s.record_attempt(6)  # attempt 1 already used
    s.transition(state.QA, task=6)
    # No record_qa call -- the crash happened before QA finished.
    state.save(s, repo / "state" / "progress.json")

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([]))  # must not be called
    exit_code = run.main(["--repo-root", str(repo), "--allow-dirty"])
    assert exit_code == 0

    resumed = state.load(repo / "state" / "progress.json")
    assert resumed.current_task == 7
    assert 6 in resumed.completed_tasks
    assert resumed.attempts["6"] == 1  # never incremented for a phantom new attempt


def test_resume_from_qa_status_then_later_claude_error_does_not_crash(repo, monkeypatch):
    """Regression test for the exact crash this fixes: resuming mid-QA,
    that re-run QA correctly fails (an ordinary QA failure, not a
    declared blocker), the loop proceeds to a genuinely NEW attempt, and
    THAT invocation raises ClaudeInvocationError. Before the fix, the
    resumed attempt's status was still nominally QA when the new
    invoke_claude call was made, so the except-block's
    transition(FAILED) raised StateError (QA -> FAILED is not an
    allowed transition) instead of failing cleanly."""
    s = state.OrchestratorState.bootstrap()
    s.transition(state.RUNNING, task=6)
    s.record_attempt(6)  # attempt 1 already used, no valid result written
    s.transition(state.QA, task=6)
    state.save(s, repo / "state" / "progress.json")

    def raise_error(prompt, repo_root, config=None, log_path=None):
        raise claude_client.ClaudeInvocationError("timed out after 3600s")

    monkeypatch.setattr(run.claude_client, "invoke_claude", raise_error)
    exit_code = run.main(["--repo-root", str(repo), "--allow-dirty", "--max-fix-attempts", "3"])
    assert exit_code == 2

    resumed = state.load(repo / "state" / "progress.json")
    assert resumed.status == state.FAILED
    assert resumed.attempts["6"] == 2  # attempt 1 (skipped-invoke) + the new attempt 2


def test_resume_completed_task_is_a_pure_skip(tmp_path, monkeypatch):
    """If progress.json already shows a later task as current (task 6
    completed by a previous, separate run), starting the orchestrator
    must work on the NEW current task, never redo task 6."""
    # Canonical protocol has no '# TASK 7 -- ...' section yet -> the
    # orchestrator must stop cleanly on task 7, and crucially must NEVER
    # re-invoke Claude for the already-completed task 6.
    repo = make_fake_repo(tmp_path / "repo", canonical_tasks={6: "Fake task six"})
    s = state.OrchestratorState(current_task=7, completed_tasks=[1, 2, 3, 4, 5, 6], status=state.READY)
    s.validate()
    state.save(s, repo / "state" / "progress.json")
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([]))
    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 0
    resumed = state.load(repo / "state" / "progress.json")
    assert resumed.current_task == 7
    assert resumed.completed_tasks == [1, 2, 3, 4, 5, 6]


# --- --loop behavior ----------------------------------------------------------


def test_loop_stops_when_next_spec_file_is_missing(tmp_path, monkeypatch):
    """--loop must not fabricate task 7 just because task 6 passed -- it
    stops and waits once the canonical protocol has no '# TASK 7 -- ...'
    section yet, per the user's own 'insert the next task only after
    this one is accepted' workflow."""
    repo = make_fake_repo(tmp_path / "repo", canonical_tasks={6: "Fake task six"})
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_valid_result()]))
    exit_code = run.main(["--repo-root", str(repo), "--loop"])
    assert exit_code == 0
    s = state.load(repo / "state" / "progress.json")
    assert s.current_task == 7  # task 6 passed...
    # ...but no infinite/incorrect advancement past 7 happened, and no
    # crash occurred just because the canonical protocol has no task 7
    # section yet (a single invoke_claude call was expected and used).


def test_loop_continues_across_two_available_tasks(tmp_path, monkeypatch):
    # Canonical protocol covers tasks 6 and 7 but not 8 -- orchestrator.
    # sync generates each tasks/{NN}_*.md for real as --loop reaches it
    # (nothing hand-copied), and --loop must stop cleanly once it reaches
    # 8, which has no canonical section yet.
    repo = make_fake_repo(tmp_path / "repo", canonical_tasks={6: "Fake task six", 7: "Fake task seven"})
    calls = {"n": 0}

    def behavior(prompt, repo_root):
        calls["n"] += 1
        task_num = 6 if calls["n"] == 1 else 7
        write_valid_result(repo_root, task=task_num, artifact_name=f"report_{task_num}.md")
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([behavior, behavior]))
    exit_code = run.main(["--repo-root", str(repo), "--loop"])
    assert exit_code == 0
    s = state.load(repo / "state" / "progress.json")
    assert s.current_task == 8
    assert s.completed_tasks == [1, 2, 3, 4, 5, 6, 7]


# --- command construction / research invariant enforcement are covered in
# test_claude_client.py and test_qa.py respectively; sanity-check here that
# run.py actually wires QA's research-alignment layer into the pass/fail
# decision end-to-end.


def test_watered_down_research_invariants_blocks_the_whole_pipeline(repo, monkeypatch):
    text = (repo / "research" / "RESEARCH_INVARIANTS.md").read_text()
    lines = [l for l in text.splitlines() if not l.startswith("18. **")]
    (repo / "research" / "RESEARCH_INVARIANTS.md").write_text("\n".join(lines))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "tamper with invariants"], cwd=repo, check=True, capture_output=True)

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_valid_result(), _writes_valid_result(), _writes_valid_result()]))
    exit_code = run.main(["--repo-root", str(repo), "--max-fix-attempts", "3"])
    assert exit_code == 1
    s = state.load(repo / "state" / "progress.json")
    assert s.status == state.BLOCKED
    assert s.current_task == 6
