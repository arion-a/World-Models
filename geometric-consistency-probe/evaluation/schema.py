"""The unified result schema for Tasks 6-14 (Task 15).

`research/CANONICAL_RESEARCH_PROTOCOL.md`'s "TASK 15" section requires
"one shared result schema ... that is a strict superset of every field
Tasks 6-14 actually produced (fields extended, never dropped; optional
fields for task-type-specific data)". Concretely, the union of every
top-level key that appears anywhere in `state/task_0{6..9}_result.json`
/ `state/task_1{0..4}_result.json` (verified empirically against those
nine files, not assumed) splits into:

- `REQUIRED_FIELDS`: the 9 keys present in *every* one of those nine
  files (`task`, `implementation_status`, `seed`, `config`, `metrics`,
  `scientific_result`, `software_versions`, `tests`, `artifacts`).
  These are the fields this schema actually enforces.
- Everything else (`transform`, `motion_config`, `scale_ladder`,
  `occlusion_scenario`, ... -- see `KNOWN_OPTIONAL_FIELDS` for the full
  list) is task-specific and stays optional: this schema never rejects
  a result for carrying one of these, or for carrying a genuinely new
  field no prior task used (that is what "strict superset" and "fields
  extended, never dropped" mean in practice -- an allow-list of known
  optional fields for documentation purposes, not a deny-list of
  anything else).

This module intentionally does not depend on the `jsonschema` package
(not in `requirements.txt`, and not otherwise used anywhere in this
repository) -- the checks below are the plain-Python equivalent of a
JSON Schema with `"required"` set to `REQUIRED_FIELDS`,
`"additionalProperties": true`, and per-field type constraints, which
is all this task's own requirements call for.

`validate_result` is deliberately silent about anything it is not
asked to check -- see research/RESEARCH_INVARIANTS.md invariant 15 /
Global Invariant 27: a schema that silently invented a default for a
missing field would hide exactly the kind of gap this task exists to
surface (this project's own "no hidden defaults" requirement for Task
15's tests).
"""

from __future__ import annotations

import json
from pathlib import Path

# Present in every one of Tasks 6-14's existing result files.
REQUIRED_FIELDS: tuple[str, ...] = (
    "task",
    "implementation_status",
    "seed",
    "config",
    "metrics",
    "scientific_result",
    "software_versions",
    "tests",
    "artifacts",
)

# Sub-fields required inside 'tests' (present in all nine files;
# 'deselected'/'command' appear in some but are not required).
TESTS_REQUIRED_SUBFIELDS: tuple[str, ...] = ("passed", "failed", "returncode")

# Sub-fields required inside 'software_versions' (present in all nine).
SOFTWARE_VERSIONS_REQUIRED_SUBFIELDS: tuple[str, ...] = ("python",)

KNOWN_STATUSES: tuple[str, ...] = ("COMPLETE", "BLOCKED")

# orchestrator/classify.py's declared blocker categories.
KNOWN_BLOCKER_CATEGORIES: tuple[str, ...] = (
    "SCIENTIFIC_CONFLICT",
    "DESTRUCTIVE_ACTION",
    "MISSING_DEPENDENCY",
    "BLOCKER",
)

# Task-specific optional fields actually seen across state/task_06..14
# result files, documented here (not enforced) purely so a reader can
# see at a glance what "extended, never dropped" produced. New tasks
# are free to introduce further fields not listed here; that is not a
# schema violation.
KNOWN_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "appearance_physical_equality_checks",
        "baseline_r2_comparison",
        "comparison_table_markdown",
        "confound_analysis",
        "confound_conditions_for_leakage_check",
        "counterfactual_conditions",
        "dataset",
        "encoder",
        "encoders",
        "fitting",
        "geometric_vs_appearance_confound_investigation",
        "invariance_computation",
        "magnitude_matching_method",
        "margins",
        "motion_config",
        "null_transform_sanity_check",
        "num_failed_runs",
        "numerical_parity_with_task6",
        "occlusion_scenario",
        "occlusion_verification",
        "pixel_diff_stats",
        "pose_convention",
        "protected_files_justification",
        "protocol_notes",
        "render_params",
        "research_alignment_note",
        "results",
        "reused_from_task6",
        "reused_from_task7",
        "scale_budget_justification",
        "scale_ladder",
        "scale_points",
        "scene_render_alignment_check",
        "scientific_qa",
        "scrambled_identity_control",
        "scrambled_label_control",
        "seeds",
        "shuffled_label_seed",
        "software",
        "static_clip_control",
        "task6_reproduction_check",
        "train_fraction",
        "transform",
        "transform_params",
        "transforms_evaluated",
        "trend_analysis",
        "trivial_cue_investigation",
        "variables_probed",
        "wall_clock_seconds_total",
        "windowing_scheme",
        # Task 15 itself (this task's own result -- see below).
        "migration_report",
        "cli_dispatch_check",
    }
)


def validate_result(data: object, *, expected_task: int | None = None) -> list[str]:
    """Validate a parsed result dict against the unified schema.

    Returns a list of human-readable error strings; an empty list means
    the result validates. Never raises on a malformed `data` -- the
    caller (a schema-validation test, or `evaluation.run_task`) decides
    what to do with a non-empty error list, matching this repository's
    established pattern of QA layers reporting failures as data
    (`orchestrator/qa.py`'s `LayerResult`) rather than raising.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return [f"result is not a JSON object (got {type(data).__name__})"]

    for field_name in REQUIRED_FIELDS:
        if field_name not in data:
            errors.append(f"missing required field: {field_name!r}")

    if errors:
        # Every remaining check below indexes into fields that were
        # just confirmed present -- bail out now rather than raising
        # KeyError further down.
        return errors

    task_value = data["task"]
    if not isinstance(task_value, int) or isinstance(task_value, bool):
        errors.append(f"'task' must be an int, got {type(task_value).__name__}")
    elif expected_task is not None and task_value != expected_task:
        errors.append(f"'task' is {task_value!r}, expected {expected_task!r}")

    status = data["implementation_status"]
    if not isinstance(status, str) or not status:
        errors.append("'implementation_status' must be a non-empty string")
    elif status not in KNOWN_STATUSES:
        errors.append(f"'implementation_status' is {status!r}, expected one of {KNOWN_STATUSES}")
    elif status == "BLOCKED":
        category = data.get("blocker_category")
        if category not in KNOWN_BLOCKER_CATEGORIES:
            errors.append(f"status is BLOCKED but 'blocker_category' is {category!r}, expected one of {KNOWN_BLOCKER_CATEGORIES}")
        if not isinstance(data.get("blocking_issue"), str) or not data.get("blocking_issue", "").strip():
            errors.append("status is BLOCKED but 'blocking_issue' is missing or empty")

    seed = data["seed"]
    seed_ok = isinstance(seed, int) and not isinstance(seed, bool)
    seed_ok = seed_ok or (isinstance(seed, list) and seed and all(isinstance(s, int) and not isinstance(s, bool) for s in seed))
    if not seed_ok:
        errors.append("'seed' must be an int or a non-empty list of ints")

    config = data["config"]
    if not isinstance(config, dict):
        errors.append(f"'config' must be an object, got {type(config).__name__}")

    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        errors.append(f"'metrics' must be an object, got {type(metrics).__name__}")

    scientific_result = data["scientific_result"]
    if not isinstance(scientific_result, str) or len(scientific_result.strip()) < 10:
        errors.append("'scientific_result' must be a non-trivial string (>= 10 characters)")

    software_versions = data["software_versions"]
    if not isinstance(software_versions, dict):
        errors.append(f"'software_versions' must be an object, got {type(software_versions).__name__}")
    else:
        for sub in SOFTWARE_VERSIONS_REQUIRED_SUBFIELDS:
            if sub not in software_versions:
                errors.append(f"'software_versions' is missing sub-field: {sub!r}")

    tests = data["tests"]
    if not isinstance(tests, dict):
        errors.append(f"'tests' must be an object, got {type(tests).__name__}")
    else:
        for sub in TESTS_REQUIRED_SUBFIELDS:
            if sub not in tests:
                errors.append(f"'tests' is missing sub-field: {sub!r}")
            elif not isinstance(tests[sub], int) or isinstance(tests[sub], bool):
                errors.append(f"'tests.{sub}' must be an int, got {type(tests[sub]).__name__}")

    artifacts = data["artifacts"]
    if not isinstance(artifacts, list) or not all(isinstance(a, str) for a in artifacts):
        errors.append("'artifacts' must be a list of strings")

    return errors


def validate_file(path: str | Path, *, expected_task: int | None = None) -> list[str]:
    """Load and validate a result JSON file. A missing file or invalid
    JSON is itself reported as a validation error (never raised), so
    callers can treat `validate_file` uniformly with `validate_result`.
    """
    path = Path(path)
    if not path.exists():
        return [f"{path} does not exist"]
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]
    return validate_result(data, expected_task=expected_task)
