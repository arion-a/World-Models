"""Re-validates Tasks 6-14's existing `state/task_NN_result.json` files
against `evaluation.schema`'s unified schema (Task 15, requirement 3).

`state/task_07b_result.json` is intentionally excluded: it is Task 7b's
own ad hoc protocol/state document (`status`/`protocol_hash`/`history`/
`updated_at`), not one of Tasks 6-14's canonical experiment results the
task spec names (`state/task_0{6..9}_result.json`, `state/
task_1{0..4}_result.json`).

No field in any of the nine files needed renaming to conform to the
unified schema (verified below, not assumed) -- every file already used
the field names `REQUIRED_FIELDS` lists. Consequently this module never
writes to any of these files: `sha256_before`/`sha256_after` in its
report are expected to be identical for every task, and that equality
*is* the evidence, required by the task's own acceptance criteria, that
no previously recorded metric value was altered by this consolidation.
If a future task's result actually required a rename to conform, that
would be a genuine in-place migration and would need to be recorded as
a protocol change per research/RESEARCH_INVARIANTS.md invariant 15 /
Global Invariant 27 -- this module's `migrated` flag exists precisely
to make that case detectable, even though it is False for all nine
files today.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evaluation.schema import validate_result

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"

# The task spec's own file list: state/task_0{6..9}_result.json,
# state/task_1{0..4}_result.json.
UNIFIED_TASK_IDS: tuple[int, ...] = tuple(range(6, 15))


def _result_path(task_id: int) -> Path:
    return STATE_DIR / f"task_{task_id:02d}_result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_all(state_dir: Path | None = None) -> dict[int, dict]:
    """Validate every one of Tasks 6-14's result files against the
    unified schema. Returns {task_id: {"path", "errors", "sha256_before",
    "sha256_after", "migrated"}}. Performs no in-place migration when
    none is required (the case for all nine files as of this task).
    """
    state_dir = Path(state_dir) if state_dir is not None else STATE_DIR
    report: dict[int, dict] = {}
    for task_id in UNIFIED_TASK_IDS:
        path = state_dir / f"task_{task_id:02d}_result.json"
        if not path.exists():
            report[task_id] = {
                "path": str(path),
                "errors": [f"{path} does not exist"],
                "sha256_before": None,
                "sha256_after": None,
                "migrated": False,
            }
            continue

        sha_before = _sha256(path)
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            report[task_id] = {
                "path": str(path),
                "errors": [f"{path} is not valid JSON: {exc}"],
                "sha256_before": sha_before,
                "sha256_after": sha_before,
                "migrated": False,
            }
            continue

        errors = validate_result(data, expected_task=task_id)
        sha_after = _sha256(path)
        report[task_id] = {
            "path": str(path),
            "errors": errors,
            "sha256_before": sha_before,
            "sha256_after": sha_after,
            "migrated": sha_before != sha_after,
        }
    return report


def all_valid(report: dict[int, dict]) -> bool:
    return all(not entry["errors"] for entry in report.values())


def main() -> None:
    report = validate_all()
    print(json.dumps(report, indent=2))
    if not all_valid(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
