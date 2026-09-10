"""Independent QA: the orchestrator's own verification of a task, run
AFTER Claude reports it is finished.

research/RESEARCH_INVARIANTS.md invariant 18: "The orchestrator must not
mark a task PASS solely because Claude reports success." Concretely,
that means every check in this module reads from disk, runs a real
subprocess, or parses an artifact -- none of it takes Claude's own
"implementation_status": "COMPLETE" at face value as anything more than
one field to cross-check against reality.

Seven layers, matching the orchestrator's brief:

  A. SOFTWARE CORRECTNESS -- tests actually pass, result file exists and
     parses, no NaN/Inf, referenced artifacts actually exist on disk and
     are non-empty, no stub/placeholder markers left behind.
  B. TASK ACCEPTANCE     -- the task's own declared required fields are
     present and structurally sane.
  C. SCIENTIFIC VALIDITY -- the experiment actually measured something
     (a metrics block with real baseline comparisons), not just asserted
     a conclusion in prose.
  D. RESEARCH ALIGNMENT  -- research/RESEARCH_INVARIANTS.md is intact,
     and tests/research/test_research_alignment.py passes.
  E. DATA LEAKAGE        -- declared train/test scene ID sets are
     disjoint, everywhere they appear in the result.
  F. REPRODUCIBILITY     -- seed and config are recorded.
  G. REGRESSION          -- the full existing test suite still passes,
     and any change to a protected (Tasks 1-5) file carries an explicit
     justification.

IMPORTANT, honestly stated limitation: layers B/C's checks are
structural and heuristic (are the right fields present, are they
non-trivial, do baseline comparisons exist) -- this script cannot
adjudicate whether an experiment's *scientific design* is sound in the
way a human or Task 18's adversarial audit can. What it CAN and DOES
guarantee is that a task cannot pass by lying about whether it ran,
whether tests pass, whether files exist, or whether splits leak.

Distinguishing SOFTWARE FAILURE from SCIENTIFIC NEGATIVE RESULT: no
layer here ever inspects the *value* of a reported metric (e.g. "is R2
above some threshold") -- only whether it is present, finite, and
backed by an actual baseline comparison. A weak or negative scientific
result therefore passes QA exactly as readily as a strong one; only a
missing/invalid/inconsistent/leaking measurement fails it.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from orchestrator import git_ops

# Core Tasks 1-5 modules. A task 6+ implementation touching these
# without an explicit justification recorded in its own result JSON
# fails the REGRESSION layer -- see research/RESEARCH_INVARIANTS.md and
# each task spec's "Depends on" header.
PROTECTED_PATHS = (
    "generation/",
    "transforms/",
    "encoders/",
    "DESIGN.md",
    "IMPLEMENTATION_NOTES.md",
    "configs/config.py",
)

# Tasks whose result JSON is expected to declare a train/test scene
# split (DATA LEAKAGE layer applies). Infrastructure-only tasks (10, 15,
# 17) and the report/release/audit tasks are exempt by default -- update
# this as tasks 7-18 actually get specced/implemented in more detail.
SPLIT_BEARING_TASKS = {6, 7, 8, 9, 11, 12, 13, 14, 18}

# Fields (dotted paths into the result JSON) required for EVERY task,
# regardless of task-specific schema. Extend TASK_SPECIFIC_FIELDS below
# for a given task's extra requirements once its own schema solidifies.
GENERIC_REQUIRED_FIELDS = ("task", "implementation_status", "scientific_result")

TASK_SPECIFIC_FIELDS: dict[int, tuple[str, ...]] = {
    6: (
        "transform",
        "encoder",
        "dataset",
        "dataset.train_scene_ids",
        "dataset.test_scene_ids",
        "metrics",
        "seed",
        "config",
    ),
}

STUB_MARKERS = ("TODO: implement", "NotImplementedError", "raise NotImplementedError", "# stub", "pass  # placeholder")


class QAConfig:
    def __init__(
        self,
        test_command: tuple[str, ...] = ("pytest", "-m", "not slow", "-q"),
        research_test_command: tuple[str, ...] = ("pytest", "tests/research", "-q"),
        test_timeout_seconds: int = 900,
    ):
        self.test_command = test_command
        self.research_test_command = research_test_command
        self.test_timeout_seconds = test_timeout_seconds


@dataclass
class LayerResult:
    name: str
    passed: bool
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "details": self.details}


@dataclass
class QAResult:
    task: int
    layers: list[LayerResult]

    @property
    def passed(self) -> bool:
        return all(layer.passed for layer in self.layers)

    @property
    def failed_layers(self) -> list[LayerResult]:
        return [layer for layer in self.layers if not layer.passed]

    def to_dict(self) -> dict:
        return {"task": self.task, "passed": self.passed, "layers": [layer.to_dict() for layer in self.layers]}

    def render_report(self) -> str:
        return render_report(self.to_dict())


def render_report(qa_dict: dict) -> str:
    """Render a QA report from its dict form (QAResult.to_dict()) --
    factored out so a report can be reconstructed from persisted state
    (state/progress.json's last_qa[task]) without needing a live
    QAResult instance, e.g. when building a fix prompt after resuming
    from an interruption."""
    lines = [f"QA report for task {qa_dict.get('task')}: {'PASS' if qa_dict.get('passed') else 'FAIL'}"]
    for layer in qa_dict.get("layers", []):
        lines.append(f"[{'PASS' if layer.get('passed') else 'FAIL'}] {layer.get('name')}")
        for detail in layer.get("details", []):
            lines.append(f"    - {detail}")
    return "\n".join(lines)


def run_qa(task: int, repo_root: str | Path, result_path: str | Path, pre_task_commit: str, config: QAConfig | None = None) -> QAResult:
    """Run all seven QA layers for `task` and return the aggregate result.

    `pre_task_commit` is the commit hash recorded before Claude was
    invoked for this attempt -- used by the REGRESSION layer to diff
    against the (still uncommitted) working tree.
    """
    config = config or QAConfig()
    repo_root = Path(repo_root)
    result_path = Path(result_path)

    result_json, result_json_error = _load_result_json(result_path)

    layer_a, test_output = _layer_a_software_correctness(repo_root, result_path, result_json, result_json_error, config)
    layer_b = _layer_b_task_acceptance(task, result_json)
    layer_c = _layer_c_scientific_validity(task, result_json)
    layer_d = _layer_d_research_alignment(repo_root, config)
    layer_e = _layer_e_data_leakage(task, result_json)
    layer_f = _layer_f_reproducibility(result_json)
    layer_g = _layer_g_regression(repo_root, task, result_json, pre_task_commit, test_output, config)

    return QAResult(task=task, layers=[layer_a, layer_b, layer_c, layer_d, layer_e, layer_f, layer_g])


# ---------------------------------------------------------------------------
# Helpers shared across layers
# ---------------------------------------------------------------------------


def _load_result_json(result_path: Path) -> tuple[dict | None, str | None]:
    if not result_path.exists():
        return None, f"{result_path} does not exist"
    try:
        return json.loads(result_path.read_text()), None
    except json.JSONDecodeError as exc:
        return None, f"{result_path} is not valid JSON: {exc}"


def _get_path(data: dict, dotted: str):
    """dotted='dataset.train_scene_ids' -> data['dataset']['train_scene_ids'].
    Returns a sentinel _MISSING if any segment is absent."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


_MISSING = object()


def _find_all_paths(data, key_name: str, prefix: str = "") -> list[tuple[str, object]]:
    """Recursively find every occurrence of a key anywhere in a nested
    dict/list structure -- used to locate train_scene_ids/test_scene_ids
    regardless of how deeply a task nests its per-transform results."""
    found = []
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else k
            if k == key_name:
                found.append((path, v))
            found.extend(_find_all_paths(v, key_name, path))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            found.extend(_find_all_paths(v, key_name, f"{prefix}[{i}]"))
    return found


def _find_non_finite(data, prefix: str = "") -> list[str]:
    """Recursively find NaN/Inf anywhere in a JSON-loaded structure."""
    bad = []
    if isinstance(data, dict):
        for k, v in data.items():
            bad.extend(_find_non_finite(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            bad.extend(_find_non_finite(v, f"{prefix}[{i}]"))
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            bad.append(f"{prefix} = {data}")
    return bad


def _run(cmd: tuple[str, ...], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False)


# ---------------------------------------------------------------------------
# Layer A: software correctness
# ---------------------------------------------------------------------------


def _layer_a_software_correctness(repo_root, result_path, result_json, result_json_error, config):
    details = []
    passed = True
    test_output = ""

    try:
        test_proc = _run(config.test_command, repo_root, config.test_timeout_seconds)
        test_output = test_proc.stdout + test_proc.stderr
        if test_proc.returncode != 0:
            passed = False
            details.append(f"test command {list(config.test_command)} exited {test_proc.returncode}")
            details.append(_tail(test_output, 20))
        else:
            details.append(f"test command {list(config.test_command)} exited 0")
    except subprocess.TimeoutExpired:
        passed = False
        test_output = ""
        details.append(f"test command {list(config.test_command)} timed out after {config.test_timeout_seconds}s")

    if result_json_error:
        passed = False
        details.append(f"result file invalid: {result_json_error}")
    else:
        non_finite = _find_non_finite(result_json)
        if non_finite:
            passed = False
            details.append(f"NaN/Inf found in result JSON: {non_finite}")

        artifacts = result_json.get("artifacts", []) if isinstance(result_json, dict) else []
        for artifact in artifacts:
            # Artifacts are declared relative to the repo root (result_path
            # lives at <repo_root>/state/task_NN_result.json, so its
            # grandparent is <repo_root>), not relative to state/.
            candidate = Path(artifact) if Path(artifact).is_absolute() else Path(result_path).parents[1] / artifact
            if not candidate.exists():
                passed = False
                details.append(f"declared artifact does not exist: {artifact}")
            elif candidate.is_file() and candidate.stat().st_size == 0:
                passed = False
                details.append(f"declared artifact is an empty file: {artifact}")

        stub_hits = _scan_for_stub_markers(result_json)
        if stub_hits:
            passed = False
            details.append(f"stub/placeholder markers found in result JSON text fields: {stub_hits}")

        all_zero = _all_metrics_are_zero(result_json)
        if all_zero:
            passed = False
            details.append("every numeric value under 'metrics' is exactly 0.0 -- looks like an unfilled template, not a genuine run")

    return LayerResult("A. SOFTWARE CORRECTNESS", passed, details), test_output


def _tail(text: str, n_lines: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def _scan_for_stub_markers(result_json: dict) -> list[str]:
    hits = []

    def _walk(node):
        if isinstance(node, str):
            for marker in STUB_MARKERS:
                if marker in node:
                    hits.append(marker)
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(result_json)
    return hits


def _all_metrics_are_zero(result_json: dict) -> bool:
    metrics = result_json.get("metrics") if isinstance(result_json, dict) else None
    if not metrics:
        return False
    numbers = []

    def _walk(node):
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            numbers.append(float(node))
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(metrics)
    return bool(numbers) and all(n == 0.0 for n in numbers)


# ---------------------------------------------------------------------------
# Layer B: task acceptance
# ---------------------------------------------------------------------------


def _layer_b_task_acceptance(task: int, result_json: dict | None) -> LayerResult:
    details = []
    passed = True

    if result_json is None:
        return LayerResult("B. TASK ACCEPTANCE", False, ["no result JSON to check (see layer A)"])

    required = list(GENERIC_REQUIRED_FIELDS) + list(TASK_SPECIFIC_FIELDS.get(task, ()))
    for field_path in required:
        value = _get_path(result_json, field_path)
        if value is _MISSING:
            passed = False
            details.append(f"missing required field: {field_path}")
        elif value in ("", None, [], {}):
            passed = False
            details.append(f"required field is empty: {field_path}")

    declared_task = result_json.get("task")
    if declared_task != task:
        passed = False
        details.append(f"result JSON declares task={declared_task!r}, expected {task}")

    status = result_json.get("implementation_status")
    if status != "COMPLETE":
        passed = False
        details.append(f"implementation_status is {status!r}, expected 'COMPLETE'")

    return LayerResult("B. TASK ACCEPTANCE", passed, details)


# ---------------------------------------------------------------------------
# Layer C: scientific validity
# ---------------------------------------------------------------------------


def _layer_c_scientific_validity(task: int, result_json: dict | None) -> LayerResult:
    details = []
    passed = True

    if result_json is None:
        return LayerResult("C. SCIENTIFIC VALIDITY", False, ["no result JSON to check (see layer A)"])

    scientific_result = result_json.get("scientific_result")
    if not isinstance(scientific_result, str) or len(scientific_result.strip()) < 10:
        passed = False
        details.append("'scientific_result' is missing or too short to be a genuine finding")

    if task in SPLIT_BEARING_TASKS:
        metrics = result_json.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            passed = False
            details.append("no 'metrics' block found for an experiment-type task")
        else:
            metrics_text = json.dumps(metrics).lower()
            has_baseline = any(word in metrics_text for word in ("baseline", "control", "persistence", "random"))
            if not has_baseline:
                passed = False
                details.append("'metrics' has no identifiable baseline/control comparison (invariant 14)")

    return LayerResult("C. SCIENTIFIC VALIDITY", passed, details)


# ---------------------------------------------------------------------------
# Layer D: research alignment
# ---------------------------------------------------------------------------


def _layer_d_research_alignment(repo_root: Path, config: QAConfig) -> LayerResult:
    details = []
    passed = True

    invariants_path = repo_root / "research" / "RESEARCH_INVARIANTS.md"
    if not invariants_path.exists():
        return LayerResult("D. RESEARCH ALIGNMENT", False, ["research/RESEARCH_INVARIANTS.md is missing"])

    text = invariants_path.read_text()
    missing_invariants = [n for n in range(1, 19) if f"{n}. **" not in text]
    if missing_invariants:
        passed = False
        details.append(f"research/RESEARCH_INVARIANTS.md is missing numbered invariant(s): {missing_invariants}")

    research_tests_dir = repo_root / "tests" / "research"
    if not research_tests_dir.exists():
        passed = False
        details.append("tests/research/ does not exist")
    else:
        try:
            proc = _run(config.research_test_command, repo_root, config.test_timeout_seconds)
            if proc.returncode != 0:
                passed = False
                details.append(f"{list(config.research_test_command)} exited {proc.returncode}")
                details.append(_tail(proc.stdout + proc.stderr, 20))
        except subprocess.TimeoutExpired:
            passed = False
            details.append("research alignment tests timed out")

    return LayerResult("D. RESEARCH ALIGNMENT", passed, details)


# ---------------------------------------------------------------------------
# Layer E: data leakage
# ---------------------------------------------------------------------------


def _layer_e_data_leakage(task: int, result_json: dict | None) -> LayerResult:
    if task not in SPLIT_BEARING_TASKS:
        return LayerResult("E. DATA LEAKAGE", True, ["not applicable to this task (no train/test split declared by spec)"])

    if result_json is None:
        return LayerResult("E. DATA LEAKAGE", False, ["no result JSON to check (see layer A)"])

    train_occurrences = _find_all_paths(result_json, "train_scene_ids")
    test_occurrences = _find_all_paths(result_json, "test_scene_ids")

    if not train_occurrences or not test_occurrences:
        return LayerResult(
            "E. DATA LEAKAGE",
            False,
            [f"expected train_scene_ids/test_scene_ids somewhere in the result JSON for task {task}, found train={len(train_occurrences)} test={len(test_occurrences)}"],
        )

    details = []
    passed = True
    for train_path, train_ids in train_occurrences:
        if not isinstance(train_ids, list) or not train_ids:
            passed = False
            details.append(f"{train_path} is empty or not a list")

    for test_path, test_ids in test_occurrences:
        if not isinstance(test_ids, list) or not test_ids:
            passed = False
            details.append(f"{test_path} is empty or not a list")

    # Pair up train/test occurrences that live at the same nesting level
    # (same parent path) and check disjointness there; also check every
    # train set against every test set globally, since a scene leaking
    # between ANY declared train set and ANY declared test set anywhere
    # in the document is a leakage bug regardless of nesting.
    for train_path, train_ids in train_occurrences:
        if not isinstance(train_ids, list):
            continue
        for test_path, test_ids in test_occurrences:
            if not isinstance(test_ids, list):
                continue
            overlap = set(train_ids) & set(test_ids)
            if overlap:
                passed = False
                details.append(f"scene ID leakage between {train_path} and {test_path}: {sorted(overlap)}")

    return LayerResult("E. DATA LEAKAGE", passed, details)


# ---------------------------------------------------------------------------
# Layer F: reproducibility
# ---------------------------------------------------------------------------


def _layer_f_reproducibility(result_json: dict | None) -> LayerResult:
    if result_json is None:
        return LayerResult("F. REPRODUCIBILITY", False, ["no result JSON to check (see layer A)"])

    details = []
    passed = True

    seed = result_json.get("seed")
    if seed is None:
        passed = False
        details.append("'seed' is missing")

    cfg = result_json.get("config")
    if cfg in (None, "", {}):
        passed = False
        details.append("'config' is missing")

    return LayerResult("F. REPRODUCIBILITY", passed, details)


# ---------------------------------------------------------------------------
# Layer G: regression
# ---------------------------------------------------------------------------


def _layer_g_regression(repo_root: Path, task: int, result_json: dict | None, pre_task_commit: str, test_output: str, config: QAConfig) -> LayerResult:
    details = []
    passed = True

    try:
        changed_files = git_ops.diff_name_only(repo_root, pre_task_commit, to_commit=None)
    except git_ops.GitError as exc:
        return LayerResult("G. REGRESSION", False, [f"could not compute git diff: {exc}"])

    protected_changed = [f for f in changed_files if any(f.startswith(p) for p in PROTECTED_PATHS)]
    if protected_changed:
        justification = (result_json or {}).get("protected_files_justification")
        if not justification:
            passed = False
            details.append(
                f"protected Task 1-5 file(s) changed with no 'protected_files_justification' recorded: {protected_changed}"
            )
        else:
            details.append(f"protected file(s) changed WITH recorded justification: {protected_changed} -- {justification!r}")

    # The fast suite's pass/fail was already established in Layer A; a
    # separate exit code isn't re-run here to avoid doubling the (real)
    # render/model-loading cost of the test suite. Regression's own
    # unique contribution is the protected-files check above.
    if not test_output:
        details.append("no test output available to inspect for regression (see layer A)")

    return LayerResult("G. REGRESSION", passed, details)
