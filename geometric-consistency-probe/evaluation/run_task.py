"""Unified CLI entry point over Tasks 6-14's existing experiment code
(Task 15, requirement 2): `python -m evaluation.run_task --task 6..14`.

This module does not reimplement any task's experiment logic -- it
imports each task's existing `experiments.taskNN_*` module and calls
its existing `main()`, exactly as `python -m experiments.taskNN_*`
already would. `--task` accepts a single id (`6`), a comma-separated
list (`6,8,10`), or a range using either `-` or the task spec's own
`..` notation (`6-9`, `6..14`); any extra CLI arguments after `--task`
are forwarded verbatim to each dispatched task's own argument parser
(e.g. `--config`, `--skip_tests`), so nothing about a task's own
configuration surface changes.

Unknown task ids, an empty `--task` selector, or a malformed range
raise `ValueError` immediately -- this module never silently skips or
guesses at a task selector (the task's own "invalid configuration fails
loudly, not silently" requirement).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType

# Task id -> the module implementing that task's existing experiment
# entry point. Verified against experiments/*.py: every one of these
# defines main() driven by argparse with a --config flag, matching
# python -m experiments.taskNN_* usage already documented for each task.
TASK_MODULES: dict[int, str] = {
    6: "experiments.task6_camera_rotation",
    7: "experiments.task7_geometric_consistency",
    8: "experiments.task8_physical_state",
    9: "experiments.task9_appearance_invariance",
    10: "experiments.task10_baselines",
    11: "experiments.task11_scale",
    12: "experiments.task12_temporal_consistency",
    13: "experiments.task13_occlusion",
    14: "experiments.task14_counterfactuals",
}


def parse_task_selector(selector: str) -> list[int]:
    """Parse a --task value into an ordered, de-duplicated list of task
    ids. Accepts 'N', 'N,M,...', 'N-M', and 'N..M' (mixed/comma-joined
    forms allowed, e.g. '6,8-10'). Raises ValueError on anything
    malformed or on an id outside TASK_MODULES -- never guesses."""
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("--task must be a non-empty string")

    ids: list[int] = []
    for raw_part in selector.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid --task selector: empty segment in {selector!r}")

        sep = ".." if ".." in part else ("-" if "-" in part else None)
        if sep is None:
            try:
                ids.append(int(part))
            except ValueError as exc:
                raise ValueError(f"invalid --task value {part!r}: not an integer") from exc
            continue

        lo_str, _, hi_str = part.partition(sep)
        try:
            lo, hi = int(lo_str), int(hi_str)
        except ValueError as exc:
            raise ValueError(f"invalid --task range {part!r}: endpoints must be integers") from exc
        if lo > hi:
            raise ValueError(f"invalid --task range {part!r}: start {lo} is greater than end {hi}")
        ids.extend(range(lo, hi + 1))

    unknown = sorted({t for t in ids if t not in TASK_MODULES})
    if unknown:
        raise ValueError(f"unknown task id(s) {unknown}; known tasks are {sorted(TASK_MODULES)}")

    seen: set[int] = set()
    ordered: list[int] = []
    for task_id in ids:
        if task_id not in seen:
            seen.add(task_id)
            ordered.append(task_id)
    return ordered


def resolve_task_module(task_id: int) -> ModuleType:
    if task_id not in TASK_MODULES:
        raise ValueError(f"unknown task id {task_id}; known tasks are {sorted(TASK_MODULES)}")
    module = importlib.import_module(TASK_MODULES[task_id])
    if not hasattr(module, "main"):
        raise AttributeError(f"{TASK_MODULES[task_id]} has no main() entry point")
    return module


def run_task(task_id: int, extra_argv: list[str] | None = None) -> None:
    """Dispatch to task `task_id`'s existing main(), with `extra_argv`
    forwarded as that task's own sys.argv (so its own argparse parses
    exactly what it would parsing `python -m experiments.taskNN_* ...`
    directly)."""
    module = resolve_task_module(task_id)
    extra_argv = list(extra_argv or [])
    prog = f"python -m {TASK_MODULES[task_id]}"
    previous_argv = sys.argv
    try:
        sys.argv = [prog, *extra_argv]
        module.main()
    finally:
        sys.argv = previous_argv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Unified dispatcher for Tasks 6-14's existing experiment entry points (Task 15).",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task id, comma-separated list, or range: '6', '6,8', '6-9', or '6..14'.",
    )
    args, extra_argv = parser.parse_known_args(argv)

    task_ids = parse_task_selector(args.task)
    for task_id in task_ids:
        run_task(task_id, extra_argv)


if __name__ == "__main__":
    main()
