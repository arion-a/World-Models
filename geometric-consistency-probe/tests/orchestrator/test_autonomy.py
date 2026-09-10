"""End-to-end demonstrations that the orchestrator drives a task from
READY through to the next task's READY state -- or to a correctly
justified BLOCKED stop -- with ZERO human/conversational interaction at
any point, per research/CANONICAL_RESEARCH_PROTOCOL.md's autonomous
task state machine (READY -> ... -> PASS -> CHECKPOINT -> ADVANCE, or
-> BLOCKED for a genuine external blocker).

Every test here uses a fully mocked `claude_client.invoke_claude` (never
the real CLI) and asserts on real, on-disk orchestrator state -- these
are not "inspect the code and assert it looks autonomous" tests; each
one runs orchestrator.run.main() for real against an isolated fake repo
and checks the resulting state/progress.json, git log, and blocker
records.
"""

from __future__ import annotations

import subprocess

import pytest

from orchestrator import claude_client, classify, git_ops, state, sync
from tests.orchestrator.conftest import make_fake_repo, write_blocked_result, write_valid_result
from tests.orchestrator.test_run import _fake_claude_factory, _writes_valid_result, repo  # noqa: F401 -- reused fixture

from orchestrator import run


# --- 1 & 2: a stale tasks/{NN}_*.md is automatically synchronized, then
# the (now up to date) task automatically proceeds -- no "should I sync
# it?" question anywhere in this path. --------------------------------------


def test_stale_task_spec_is_synced_then_task_proceeds_automatically(tmp_path, monkeypatch):
    repo = make_fake_repo(tmp_path / "repo", canonical_tasks={6: "Fake task six"})
    spec_path = next((repo / "tasks").glob("06_*.md"))
    spec_path.write_text("this file is stale and disagrees with the canonical protocol\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "simulate a stale hand-edited task spec"], cwd=repo, check=True, capture_output=True)

    prompts_seen = []

    def behavior(prompt, repo_root):
        prompts_seen.append(prompt)
        write_valid_result(repo_root, task=6)
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([behavior]))
    exit_code = run.main(["--repo-root", str(repo)])

    assert exit_code == 0, "sync + proceed must complete with no human input"
    assert "Fake purpose for task 6." in spec_path.read_text(), "canonical protocol content must have overwritten the stale hand-edit"
    assert "this file is stale" not in spec_path.read_text()

    s = state.load(repo / "state" / "progress.json")
    assert s.current_task == 7
    assert 6 in s.completed_tasks
    assert s.sync["6"]["changed"] is True, "the sync event must be recorded in state, not just silently done"

    # The prompt Claude actually received must reflect the synced (not
    # stale) content.
    assert "Fake purpose for task 6." in prompts_seen[0]
    assert "this file is stale" not in prompts_seen[0]


# --- 6: a scientifically negative result is a valid outcome and must
# advance exactly like a positive one -- QA never inspects a metric's
# VALUE, only its presence/validity, so this is not a special case in
# the code, but it is a scenario worth demonstrating explicitly. -----------


def test_negative_scientific_result_advances_automatically(repo, monkeypatch):
    def behavior(prompt, repo_root):
        write_valid_result(
            repo_root,
            task=6,
            extra_fields={
                "scientific_result": "The learned map performed no better than the persistence baseline -- a clear negative result.",
                "metrics": {
                    "learned_W_T": {"r2": -0.3, "mean_cosine_similarity": 0.05, "mean_relative_l2_error": 1.4},
                    "persistence_baseline": {"r2": 0.02, "mean_cosine_similarity": 0.1, "mean_relative_l2_error": 0.9},
                    "random_pair_control": {"r2": -0.31, "mean_cosine_similarity": 0.01, "mean_relative_l2_error": 1.5},
                },
            },
        )
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([behavior]))
    exit_code = run.main(["--repo-root", str(repo)])

    assert exit_code == 0, "a validly-measured negative result must still advance -- it is not a failure"
    s = state.load(repo / "state" / "progress.json")
    assert s.current_task == 7
    assert 6 in s.completed_tasks
    assert s.last_qa["6"]["passed"] is True


# --- 9: canonical protocol overrides a stale task file, even when the
# stale file was committed as if it were authoritative. --------------------


def test_canonical_protocol_wins_over_committed_stale_spec(tmp_path):
    repo = make_fake_repo(tmp_path / "repo", canonical_tasks={6: "Fake task six"})
    spec_path = next((repo / "tasks").glob("06_*.md"))
    original = spec_path.read_text()
    spec_path.write_text(original.replace("Fake purpose for task 6.", "A HUMAN HAND-WROTE THIS AND IT DISAGREES"))
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "commit a stale spec as if authoritative"], cwd=repo, check=True, capture_output=True)

    result = sync.sync_task_spec(6, repo)
    assert result.changed is True
    assert "A HUMAN HAND-WROTE THIS" not in result.path.read_text()
    assert "Fake purpose for task 6." in result.path.read_text()


# --- 11/12/13: genuine external blockers stop the pipeline (BLOCKED),
# they do not become an "ask the human" moment or an arbitrary retry. -----


@pytest.mark.parametrize(
    "category,issue",
    [
        (classify.SCIENTIFIC_CONFLICT, "the task spec requires unfreezing the encoder, which invariant 1 forbids"),
        (classify.DESTRUCTIVE_ACTION, "completing this task as specified would require a force-push to main"),
        (classify.MISSING_DEPENDENCY, "the facebook/vjepa2-vitl-fpc64-256 checkpoint is not reachable from this environment"),
    ],
)
def test_genuine_blocker_categories_stop_the_pipeline_without_wasting_repair_attempts(repo, monkeypatch, category, issue):
    def behavior(prompt, repo_root):
        write_blocked_result(repo_root, task=6, blocker_category=category, blocking_issue=issue)
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    # Only ONE behavior is provided even though --max-fix-attempts is 3:
    # a genuine blocker must go straight to BLOCKED on the first QA
    # failure, never consume a second or third repair attempt.
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([behavior]))
    exit_code = run.main(["--repo-root", str(repo), "--max-fix-attempts", "3"])

    assert exit_code == 1
    s = state.load(repo / "state" / "progress.json")
    assert s.status == state.BLOCKED
    assert s.current_task == 6, "a blocked task must never advance"
    assert s.attempts["6"] == 1, "no repair attempt should have been spent on an unretryable blocker"
    assert s.blockers["6"]["category"] == category
    assert s.blockers["6"]["reason"] == issue


# --- 18: a checkpoint failure must prevent advancement -- QA passing is
# not enough on its own; the state must not show the task complete until
# the commit genuinely exists. ----------------------------------------------


def test_checkpoint_failure_prevents_advancement(repo, monkeypatch):
    """A total checkpoint failure (create_checkpoint fails AND the
    get_status fallback used to detect an already-committed checkpoint
    also fails) must land in BLOCKED, never silently advance the task as
    if a commit existed. Only get_status calls made from that fallback
    (i.e. after create_checkpoint has already been tried) are made to
    fail here -- the legitimate earlier get_status calls (the initial
    dirty-tree check, capturing pre_task_commit) must keep working
    normally, or this test would be exercising the wrong failure."""
    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([_writes_valid_result()]))

    real_get_status = git_ops.get_status
    checkpoint_attempted = {"v": False}

    def failing_create_checkpoint(*a, **k):
        checkpoint_attempted["v"] = True
        raise git_ops.GitError("simulated checkpoint failure: disk full")

    def flaky_get_status(repo_root):
        if checkpoint_attempted["v"]:
            raise git_ops.GitError("simulated checkpoint failure: disk full")
        return real_get_status(repo_root)

    monkeypatch.setattr(run.git_ops, "create_checkpoint", failing_create_checkpoint)
    monkeypatch.setattr(run.git_ops, "get_status", flaky_get_status)

    exit_code = run.main(["--repo-root", str(repo)])
    assert exit_code == 2

    s = state.load(repo / "state" / "progress.json")
    assert s.current_task == 6, "must not advance without a real checkpoint commit"
    assert 6 not in s.completed_tasks
    assert s.status == state.BLOCKED


# --- 20 / FINAL VALIDATION: explicit Task 7 chains, RUNNING -> QA ->
# PASS -> CHECKPOINT -> Task 8, and the repair-path variant, each with
# zero human interaction. ----------------------------------------------------


@pytest.fixture
def repo_at_task_7(tmp_path):
    repo = make_fake_repo(tmp_path / "repo", canonical_tasks={7: "Fake task seven", 8: "Fake task eight"})
    s = state.OrchestratorState(current_task=7, completed_tasks=[1, 2, 3, 4, 5, 6], status=state.READY)
    s.validate()
    state.save(s, repo / "state" / "progress.json")
    return repo


def test_task_7_clean_pass_chain_reaches_task_8_with_zero_interaction(repo_at_task_7, monkeypatch):
    def behavior(prompt, repo_root):
        write_valid_result(repo_root, task=7)
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([behavior]))
    exit_code = run.main(["--repo-root", str(repo_at_task_7)])

    assert exit_code == 0
    s = state.load(repo_at_task_7 / "state" / "progress.json")
    assert s.status == state.READY
    assert s.current_task == 8
    assert 7 in s.completed_tasks
    assert s.commits["7"] is not None

    log = subprocess.run(["git", "log", "--format=%s"], cwd=repo_at_task_7, capture_output=True, text=True, check=True)
    assert any(line.startswith("task-07-pass") for line in log.stdout.splitlines())


def test_task_7_repair_path_reaches_task_8_with_zero_interaction(repo_at_task_7, monkeypatch):
    def fails_first(prompt, repo_root):
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])  # no result file -> QA fails

    def fixes_on_retry(prompt, repo_root):
        write_valid_result(repo_root, task=7)
        return claude_client.ClaudeResult(0, "{}", "", {}, 0.1, ["claude"])

    monkeypatch.setattr(run.claude_client, "invoke_claude", _fake_claude_factory([fails_first, fixes_on_retry]))
    exit_code = run.main(["--repo-root", str(repo_at_task_7), "--max-fix-attempts", "3"])

    assert exit_code == 0
    s = state.load(repo_at_task_7 / "state" / "progress.json")
    assert s.current_task == 8
    assert 7 in s.completed_tasks
    assert s.attempts["7"] == 2, "must have gone through exactly one repair round"


# --- 14: no interactive approval surface is ever offered to Claude ---------


def test_claude_is_always_invoked_with_no_interactive_prompt_surface():
    argv = claude_client.build_command("a prompt", claude_client.ClaudeConfig())
    assert "--permission-prompts" in argv
    assert argv[argv.index("--permission-prompts") + 1] == "none"
