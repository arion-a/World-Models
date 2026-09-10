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

FAKE_INVARIANTS_MD = "\n".join(f"{n}. **Fake invariant {n}.** Placeholder text for testing." for n in range(1, 19))

FAKE_TASK_SPEC_TEMPLATE = """\
# Task {task} — Fake task for orchestrator unit tests

## Objective

Fake objective.

## Scientific question

Fake question.

## Implementation requirements

Fake requirements.

## Required artifacts

Fake artifacts.

## Tests required

Fake tests.

## Leakage checks

Fake leakage checks.

## Acceptance criteria

Fake acceptance criteria.

## Prohibited shortcuts

Fake prohibited shortcuts.

## Scientific interpretation limits

Fake limits.
"""


def _run_git(args, cwd):
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result


def make_fake_repo(root: Path, task: int = 6) -> Path:
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "research").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "research").mkdir(parents=True, exist_ok=True)
    (root / "generation").mkdir(parents=True, exist_ok=True)

    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\nlogs/\n")
    (root / "research" / "RESEARCH_INVARIANTS.md").write_text(FAKE_INVARIANTS_MD)
    (root / "tests" / "research" / "test_fake_alignment.py").write_text("def test_ok():\n    assert True\n")
    (root / "tests" / "test_dummy.py").write_text("def test_ok():\n    assert True\n")
    (root / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow test\n")
    (root / "tasks" / f"{task:02d}_fake.md").write_text(FAKE_TASK_SPEC_TEMPLATE.format(task=task))
    (root / "generation" / "core.py").write_text("# protected placeholder module\nVALUE = 1\n")

    _run_git(["init"], root)
    _run_git(["config", "user.email", "test@example.com"], root)
    _run_git(["config", "user.name", "Test"], root)
    _run_git(["add", "-A"], root)
    _run_git(["commit", "-m", "initial fake repo"], root)
    return root


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


@pytest.fixture
def fake_repo(tmp_path) -> Path:
    return make_fake_repo(tmp_path / "repo", task=6)
