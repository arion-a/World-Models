"""Unit tests for orchestrator/git_ops.py, against real throwaway git
repositories under tmp_path (git itself is fast/deterministic/local, so
these are not mocked -- only the `claude` CLI is mocked elsewhere).
"""

from __future__ import annotations

import subprocess

import pytest

from orchestrator import git_ops


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=path, check=True, capture_output=True)
    (path / "a.txt").write_text("initial\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)
    return path


@pytest.fixture
def repo(tmp_path):
    return _init_repo(tmp_path / "repo")


def test_get_status_on_clean_repo(repo):
    status = git_ops.get_status(repo)
    assert status.clean
    assert len(status.commit) == 40


def test_get_status_on_dirty_repo(repo):
    (repo / "a.txt").write_text("changed\n")
    status = git_ops.get_status(repo)
    assert not status.clean


def test_require_clean_tree_passes_when_clean(repo):
    status = git_ops.require_clean_tree(repo)
    assert status.clean


def test_require_clean_tree_raises_when_dirty_and_does_not_touch_anything(repo):
    (repo / "a.txt").write_text("changed\n")
    (repo / "untracked.txt").write_text("new\n")
    with pytest.raises(git_ops.GitError):
        git_ops.require_clean_tree(repo)
    # the dirty state must be untouched -- no silent stash/discard
    assert (repo / "untracked.txt").exists()
    assert (repo / "a.txt").read_text() == "changed\n"


def test_create_checkpoint_commits_with_task_prefix(repo):
    (repo / "new_file.py").write_text("x = 1\n")
    commit_hash = git_ops.create_checkpoint(repo, task=6, message_body="details here")
    assert len(commit_hash) == 40
    log = subprocess.run(["git", "log", "-1", "--format=%s%n%b"], cwd=repo, capture_output=True, text=True, check=True)
    assert log.stdout.startswith("task-06-pass")
    assert "details here" in log.stdout


def test_create_checkpoint_refuses_empty_commit(repo):
    """If Claude claimed success but changed literally nothing, that must
    surface as an error, not a silent empty commit."""
    with pytest.raises(git_ops.GitError):
        git_ops.create_checkpoint(repo, task=6)


def test_create_checkpoint_never_amends_previous_commit(repo):
    before = git_ops.get_status(repo).commit
    (repo / "f1.py").write_text("1\n")
    c1 = git_ops.create_checkpoint(repo, task=6)
    (repo / "f2.py").write_text("2\n")
    c2 = git_ops.create_checkpoint(repo, task=7)
    assert len({before, c1, c2}) == 3, "each checkpoint must be a brand-new commit"
    log = subprocess.run(["git", "log", "--format=%H"], cwd=repo, capture_output=True, text=True, check=True)
    assert len(log.stdout.strip().splitlines()) == 3


def test_diff_name_only_working_tree_vs_commit(repo):
    before = git_ops.get_status(repo).commit
    (repo / "b.txt").write_text("new\n")
    (repo / "a.txt").write_text("changed\n")
    changed = git_ops.diff_name_only(repo, before, to_commit=None)
    assert set(changed) == {"a.txt", "b.txt"}


def test_diff_name_only_between_two_commits(repo):
    before = git_ops.get_status(repo).commit
    (repo / "b.txt").write_text("new\n")
    after = git_ops.create_checkpoint(repo, task=6)
    changed = git_ops.diff_name_only(repo, before, after)
    assert "b.txt" in changed


def test_record_state_commit_adds_a_followup_commit(repo):
    (repo / "state").mkdir()
    (repo / "state" / "progress.json").write_text('{"a": 1}\n')
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add progress.json"], cwd=repo, check=True, capture_output=True)

    (repo / "state" / "progress.json").write_text('{"a": 2}\n')
    new_commit = git_ops.record_state_commit(repo, task=6, commit_hash="deadbeef")
    assert new_commit is not None
    log = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=repo, capture_output=True, text=True, check=True)
    assert "task-06-pass" in log.stdout


def test_record_state_commit_returns_none_when_nothing_changed(repo):
    (repo / "state").mkdir()
    (repo / "state" / "progress.json").write_text('{"a": 1}\n')
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add progress.json"], cwd=repo, check=True, capture_output=True)

    result = git_ops.record_state_commit(repo, task=6, commit_hash="deadbeef")
    assert result is None


def test_git_ops_never_uses_shell_true():
    """Static guard: this module must invoke git via argument arrays,
    never shell=True (which would make it injectable via crafted paths
    or messages)."""
    import inspect

    source = inspect.getsource(git_ops)
    assert "shell=True" not in source


# --- repo_root as a SUBDIRECTORY of the actual git repository --------------
#
# This is the real deployment shape (geometric-consistency-probe/ is a
# subdirectory of the World-Models git repo, not its own repo), and it
# broke both require_clean_tree()'s state/-exclusion and qa.py's
# PROTECTED_PATHS matching the first time the orchestrator ran for
# real: `git status --porcelain` / `git diff --name-only` always report
# paths relative to the git TOP-LEVEL, never relative to cwd, so every
# reported path came back prefixed with the subdirectory name (e.g.
# "geometric-consistency-probe/state/progress.json"), which no
# repo_root-relative prefix check ("state/", "generation/", ...) could
# ever match. None of the tests above caught this because `repo` there
# IS the git top-level. These tests pin the nested case specifically.


@pytest.fixture
def nested_repo(tmp_path):
    """git init'd at tmp_path/outer; the orchestrator's repo_root is the
    subdirectory tmp_path/outer/inner -- mirrors geometric-consistency-
    probe/ living inside the World-Models repo."""
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init"], cwd=outer, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=outer, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=outer, check=True, capture_output=True)
    inner = outer / "inner"
    (inner / "generation").mkdir(parents=True)
    (inner / "generation" / "core.py").write_text("VALUE = 1\n")
    (inner / "a.txt").write_text("initial\n")
    subprocess.run(["git", "add", "-A"], cwd=outer, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=outer, check=True, capture_output=True)
    return inner


def test_repo_relative_prefix_reflects_subdirectory_nesting(nested_repo):
    assert git_ops.repo_relative_prefix(nested_repo) == "inner/"


def test_repo_relative_prefix_is_empty_at_git_toplevel(repo):
    assert git_ops.repo_relative_prefix(repo) == ""


def test_require_clean_tree_ignores_state_dir_when_repo_root_is_nested(nested_repo):
    """The exact bug: state/progress.json modified inside a nested
    repo_root must still be excluded from the dirty check, even though
    git reports it as 'inner/state/progress.json'."""
    (nested_repo / "state").mkdir()
    (nested_repo / "state" / "progress.json").write_text('{"a": 1}\n')
    subprocess.run(["git", "add", "-A"], cwd=nested_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add progress.json"], cwd=nested_repo, check=True, capture_output=True)

    (nested_repo / "state" / "progress.json").write_text('{"a": 2}\n')
    status = git_ops.require_clean_tree(nested_repo)  # must NOT raise
    assert not status.clean  # state/ change is real, just excluded from the check


def test_require_clean_tree_still_catches_unrelated_dirty_files_when_nested(nested_repo):
    (nested_repo / "unrelated.txt").write_text("someone's work\n")
    with pytest.raises(git_ops.GitError):
        git_ops.require_clean_tree(nested_repo)


def test_diff_name_only_strips_nesting_prefix(nested_repo):
    before = git_ops.get_status(nested_repo).commit
    (nested_repo / "generation" / "core.py").write_text("VALUE = 2\n")
    changed = git_ops.diff_name_only(nested_repo, before, to_commit=None)
    assert changed == ["generation/core.py"], "paths must be repo_root-relative, not prefixed with the outer repo's subdirectory name"
