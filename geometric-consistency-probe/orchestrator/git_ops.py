"""Safe git helpers for the orchestrator's checkpoint commits.

Every call here uses argument-array subprocess invocation
(`subprocess.run(["git", ...])`), never invoking git through a shell and
never building a concatenated shell string, per the orchestrator's
safety requirements. Nothing here force-pushes, resets, rewrites
history, or deletes anything -- the only git operations exposed are:
read status/log, and create ordinary new commits.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitError(Exception):
    """A git command failed or the repository is in an unexpected state."""


@dataclass
class GitStatus:
    clean: bool
    porcelain: str
    commit: str
    branch: str


def _run_git(args: list[str], cwd: str | Path, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def get_status(repo_root: str | Path) -> GitStatus:
    porcelain = _run_git(["status", "--porcelain"], repo_root)
    if porcelain.returncode != 0:
        raise GitError(f"git status failed: {porcelain.stderr.strip()}")

    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    if commit.returncode != 0:
        raise GitError(f"git rev-parse HEAD failed: {commit.stderr.strip()}")

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if branch.returncode != 0:
        raise GitError(f"git rev-parse --abbrev-ref HEAD failed: {branch.stderr.strip()}")

    text = porcelain.stdout
    return GitStatus(clean=(text.strip() == ""), porcelain=text, commit=commit.stdout.strip(), branch=branch.stdout.strip())


# state/ holds the orchestrator's own bookkeeping (progress.json,
# task_NN_result.json): the orchestrator itself writes there continuously
# (every status transition saves progress.json) outside of any checkpoint
# commit, so it can legitimately be "dirty" before a task even starts --
# e.g. right after a BLOCKED task is reset to READY for a retry, before
# any new checkpoint has happened. That is expected, not a bystander's
# unrelated work, so it is excluded from the "is this tree safe to start
# a fresh task on" check below. It is NOT excluded from create_checkpoint
# ()'s `git add -A` -- that intentionally commits it.
DEFAULT_IGNORED_FOR_DIRTY_CHECK = ("state/", "logs/")


def _porcelain_path(line: str) -> str:
    # Porcelain format: "XY PATH" or "XY PATH1 -> PATH2" for renames.
    path = line[3:].strip()
    return path.split(" -> ")[-1]


def require_clean_tree(repo_root: str | Path, ignore_paths: tuple[str, ...] = DEFAULT_IGNORED_FOR_DIRTY_CHECK) -> GitStatus:
    """Refuse to proceed if there are pre-existing uncommitted changes
    unrelated to the orchestrator's own work. This is what makes the
    checkpoint commit in create_checkpoint() safe to `git add -A`: if the
    tree was clean (modulo `ignore_paths`) before a task started,
    everything staged afterward is that task's own change, never a
    bystander's unrelated edit."""
    status = get_status(repo_root)
    relevant_lines = [
        line for line in status.porcelain.splitlines() if line.strip() and not any(_porcelain_path(line).startswith(p) for p in ignore_paths)
    ]
    if relevant_lines:
        raise GitError(
            "repository has uncommitted changes -- refusing to start a task "
            "(would risk folding unrelated changes into a task checkpoint "
            "commit). Commit or stash them first, or pass --allow-dirty if "
            "you have verified they are safe to include.\n" + status.porcelain
        )
    return status


def create_checkpoint(repo_root: str | Path, task: int, message_body: str = "") -> str:
    """Stage everything and commit as `task-{NN}-pass`. Only ever called
    after QA has independently passed a task (orchestrator/run.py), and
    only ever adds a new commit -- never amends, never force-pushes."""
    add = _run_git(["add", "-A"], repo_root)
    if add.returncode != 0:
        raise GitError(f"git add -A failed: {add.stderr.strip()}")

    status = _run_git(["status", "--porcelain"], repo_root)
    if status.stdout.strip() == "":
        raise GitError(f"task {task} passed QA but produced no file changes to commit -- refusing to create an empty checkpoint")

    message = f"task-{task:02d}-pass" + (f"\n\n{message_body}" if message_body else "")
    commit = _run_git(["commit", "-m", message], repo_root)
    if commit.returncode != 0:
        raise GitError(f"git commit failed: {commit.stderr.strip()}")

    rev = _run_git(["rev-parse", "HEAD"], repo_root)
    if rev.returncode != 0:
        raise GitError(f"git rev-parse HEAD failed after commit: {rev.stderr.strip()}")
    return rev.stdout.strip()


def record_state_commit(repo_root: str | Path, task: int, commit_hash: str) -> str | None:
    """A small follow-up commit that records the checkpoint's own commit
    hash inside state/progress.json (written after create_checkpoint(),
    since the hash isn't known until that commit exists). Returns the new
    HEAD, or None if progress.json had nothing new to commit (e.g. the
    hash was already recorded by the caller before create_checkpoint)."""
    status = _run_git(["status", "--porcelain", "--", "state/progress.json"], repo_root)
    if status.stdout.strip() == "":
        return None
    add = _run_git(["add", "state/progress.json"], repo_root)
    if add.returncode != 0:
        raise GitError(f"git add state/progress.json failed: {add.stderr.strip()}")
    commit = _run_git(["commit", "-m", f"task-{task:02d}-pass: record checkpoint commit hash"], repo_root)
    if commit.returncode != 0:
        raise GitError(f"git commit (progress.json hash record) failed: {commit.stderr.strip()}")
    rev = _run_git(["rev-parse", "HEAD"], repo_root)
    return rev.stdout.strip()


def diff_name_only(repo_root: str | Path, from_commit: str, to_commit: str | None = None) -> list[str]:
    """Files changed between `from_commit` and `to_commit` (for QA's
    regression / protected-files layer). `to_commit=None` compares
    against the current *working tree* (uncommitted changes included) --
    this is what QA needs, since it runs before any checkpoint commit
    exists for the task under review.

    IMPORTANT: `git diff` alone never reports brand-new *untracked*
    files (only modifications to already-tracked ones), and a task's own
    new files are untracked at QA time (nothing has been `git add`-ed
    yet). So when comparing against the working tree, this also unions
    in untracked files from `git status --porcelain` -- otherwise a task
    that added a new file under a protected path (e.g. a new file in
    generation/) would silently evade the protected-files check.
    """
    args = ["diff", "--name-only", from_commit] if to_commit is None else ["diff", "--name-only", from_commit, to_commit]
    result = _run_git(args, repo_root)
    if result.returncode != 0:
        raise GitError(f"git diff failed: {result.stderr.strip()}")
    changed = {line for line in result.stdout.splitlines() if line.strip()}

    if to_commit is None:
        status = _run_git(["status", "--porcelain", "--untracked-files=all"], repo_root)
        if status.returncode != 0:
            raise GitError(f"git status failed: {status.stderr.strip()}")
        for line in status.stdout.splitlines():
            if line.startswith("??"):
                changed.add(line[3:].strip())

    return sorted(changed)
