"""Fixtures for orchestrator unit tests.

Every test in this package operates on a fully isolated, throwaway git
repository under `tmp_path` -- NEVER on the real geometric-consistency-
probe checkout. This matters: orchestrator.git_ops.create_checkpoint
makes real commits, and these tests must not touch this project's own
git history.

No test in this package makes a real call to the `claude` CLI --
`orchestrator.claude_client.invoke_claude` is always monkeypatched. Real
subprocesses ARE used for `git` and for the fake repo's own (trivial,
fast) pytest suite, since those are cheap, deterministic, and not what
"mock Claude" is about.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from orchestrator import sync

FAKE_INVARIANTS_MD = "\n".join(f"{n}. **Fake invariant {n}.** Placeholder text for testing." for n in range(1, 19))

# Fake tasks/{NN}_*.md files are generated FOR REAL by orchestrator.sync
# from this fake canonical protocol document, exercising the actual
# sync/parse/render code path rather than a hand-maintained parallel
# format that could silently drift from it. Covers tasks 6-8 so
# multi-task --loop tests have somewhere to advance to.
FAKE_TASK_TITLES = {6: "Fake task six", 7: "Fake task seven", 8: "Fake task eight"}


def _fake_canonical_task_section(task: int, title: str) -> str:
    fields = "\n\n".join(f"### {name}\n\nFake {name.lower()} for task {task}." for name in sync.CANONICAL_FIELD_ORDER)
    return f"# TASK {task} — {title.upper()}\n\n{fields}\n\n"


def _render_fake_canonical_protocol(titles: dict[int, str]) -> str:
    return "".join(_fake_canonical_task_section(t, title) for t, title in titles.items()) + (
        "## Validation performed on this document\n\nFake validation section -- not a task.\n"
    )


FAKE_CANONICAL_PROTOCOL = _render_fake_canonical_protocol(FAKE_TASK_TITLES)


def _run_git(args, cwd):
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


def make_fake_repo(root: Path, task: int = 6, nested: bool = False, canonical_tasks: dict[int, str] | None = None) -> Path:
    """`nested=True` git-inits at `root` but puts all the fake-repo
    content under `root/inner`, returning that subdirectory as
    repo_root -- mirrors this project's actual deployment shape
    (geometric-consistency-probe/ is a subdirectory of the World-Models
    git repo, not its own repo root). `git status`/`git diff` report
    paths relative to the git TOP-LEVEL, never relative to cwd, so this
    nested shape is the only way to catch a prefix-matching bug like
    orchestrator/git_ops.py's DEFAULT_IGNORED_FOR_DIRTY_CHECK or
    orchestrator/qa.py's PROTECTED_PATHS silently never matching -- a
    real bug this project's first live run hit that `nested=False`
    fixtures could not have caught.

    `task` only decides which fake result/checkpoint helpers a test
    defaults to reasoning about -- it is NOT what makes a task's
    canonical section exist or not (that's real orchestrator.sync
    behavior, driven only by the canonical protocol text). Pass
    `canonical_tasks` (default FAKE_TASK_TITLES) to control exactly
    which '# TASK N -- ...' sections the fake canonical protocol
    contains -- e.g. `canonical_tasks={}` for a repo where no task has
    a canonical section yet, or `canonical_tasks={6: "Fake task six"}`
    for one where only task 6 does."""
    content_root = (root / "inner") if nested else root
    git_root = root
    titles = FAKE_TASK_TITLES if canonical_tasks is None else canonical_tasks

    (content_root / "tasks").mkdir(parents=True, exist_ok=True)
    (content_root / "state").mkdir(parents=True, exist_ok=True)
    (content_root / "research").mkdir(parents=True, exist_ok=True)
    (content_root / "tests" / "research").mkdir(parents=True, exist_ok=True)
    (content_root / "generation").mkdir(parents=True, exist_ok=True)

    (content_root / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\nlogs/\n")
    (content_root / "research" / "RESEARCH_INVARIANTS.md").write_text(FAKE_INVARIANTS_MD)
    (content_root / "research" / "CANONICAL_RESEARCH_PROTOCOL.md").write_text(_render_fake_canonical_protocol(titles))
    (content_root / "tests" / "research" / "test_fake_alignment.py").write_text("def test_ok():\n    assert True\n")
    (content_root / "tests" / "test_dummy.py").write_text("def test_ok():\n    assert True\n")
    (content_root / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow test\n")
    (content_root / "generation" / "core.py").write_text("# protected placeholder module\nVALUE = 1\n")

    # Generate tasks/{NN}_*.md FOR REAL via the actual sync module, for
    # every task the fake canonical protocol covers -- not hand-written,
    # so these fixtures can never drift from what sync.py actually does.
    for t in titles:
        sync.sync_task_spec(t, content_root)

    _run_git(["init"], git_root)
    _run_git(["config", "user.email", "test@example.com"], git_root)
    _run_git(["config", "user.name", "Test"], git_root)
    _run_git(["add", "-A"], git_root)
    _run_git(["commit", "-m", "initial fake repo"], git_root)
    return content_root


def write_valid_result(root: Path, task: int, artifact_name: str = "report.md", extra_fields: dict | None = None) -> None:
    """Simulates what a genuinely-completed Claude implementation would
    leave behind: a schema-complete state/task_NN_result.json plus the
    artifact file(s) it declares."""
    (root / artifact_name).write_text("# Fake report\nSubstantive content, not a stub.\n")
    result = {
        "task": task,
        "implementation_status": "COMPLETE",
        "scientific_result": "The learned map achieved higher held-out R2 than every baseline, a modest positive effect.",
        "transform": "camera_rotation",
        "encoder": {"name": "VJEPAEncoder", "checkpoint": "facebook/vjepa2-vitl-fpc64-256", "pretrained": True, "frozen": True},
        "dataset": {"train_scene_ids": ["scene_0000", "scene_0001"], "test_scene_ids": ["scene_0002", "scene_0003"]},
        "metrics": {
            "learned_W_T": {"r2": 0.42, "mean_cosine_similarity": 0.7, "mean_relative_l2_error": 0.3},
            "persistence_baseline": {"r2": 0.1, "mean_cosine_similarity": 0.4, "mean_relative_l2_error": 0.6},
            "random_pair_control": {"r2": 0.02, "mean_cosine_similarity": 0.1, "mean_relative_l2_error": 0.9},
        },
        "tests": {"passed": 3, "failed": 0},
        "artifacts": [artifact_name],
        "config": {"num_scenes": 4},
        "seed": 0,
    }
    if extra_fields:
        result.update(extra_fields)
    (root / "state" / f"task_{task:02d}_result.json").write_text(json.dumps(result, indent=2))


def write_blocked_result(root: Path, task: int, blocker_category: str, blocking_issue: str) -> None:
    """Simulates a task that correctly identified it could not proceed
    and reported so honestly (per research/CANONICAL_RESEARCH_PROTOCOL.md's
    'STOP and record the conflict' rule) -- e.g. a genuine scientific
    conflict, a missing credential, or a required destructive action."""
    result = {
        "task": task,
        "implementation_status": "BLOCKED",
        "blocker_category": blocker_category,
        "blocking_issue": blocking_issue,
        "scientific_result": "No experiment was executed -- see blocking_issue.",
    }
    (root / "state" / f"task_{task:02d}_result.json").write_text(json.dumps(result, indent=2))


@pytest.fixture
def fake_repo(tmp_path) -> Path:
    return make_fake_repo(tmp_path / "repo", task=6)
