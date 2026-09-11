"""Task 7B's own orchestrator state machine (contract Sec. 14) --
completely separate from orchestrator/state.py's READY/RUNNING/QA/PASSED
machine, which governs Tasks 6-18 as a whole. Writes ONLY
state/task_07b_result.json; never touches state/task_07_result.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

NOT_STARTED = "NOT_STARTED"
PROTOCOL_FROZEN = "PROTOCOL_FROZEN"
DATA_VALIDATED = "DATA_VALIDATED"
LEARNING_CURVE_COMPLETE = "LEARNING_CURVE_COMPLETE"
MODEL_SELECTED = "MODEL_SELECTED"
TEST_EVALUATED = "TEST_EVALUATED"
STRUCTURE_TESTED = "STRUCTURE_TESTED"
COMPLETE_SCALE_LIMITED = "COMPLETE_SCALE_LIMITED"
COMPLETE = "COMPLETE"
FAILED_SOFTWARE = "FAILED_SOFTWARE"
INVALID_CONTAMINATED = "INVALID_CONTAMINATED"

ORDER = [
    NOT_STARTED, PROTOCOL_FROZEN, DATA_VALIDATED, LEARNING_CURVE_COMPLETE,
    MODEL_SELECTED, TEST_EVALUATED, STRUCTURE_TESTED,
]
TERMINAL = {COMPLETE, COMPLETE_SCALE_LIMITED, FAILED_SOFTWARE, INVALID_CONTAMINATED}

ADVANCEMENT_REQUIREMENTS = {
    PROTOCOL_FROZEN: "hash, splits, schedule, models, controls, and rules exist; no test evaluation exists",
    DATA_VALIDATED: "scene/transform/render/encoder/split/leakage QA passes",
    LEARNING_CURVE_COMPLETE: "every feasible preregistered N and seed is represented, including failures",
    MODEL_SELECTED: "selection record proves validation-only choice and complexity tie-break",
    TEST_EVALUATED: "one sealed test artifact with all controls and contamination check",
    STRUCTURE_TESTED: "every mathematically valid predeclared check has a result or NOT_APPLICABLE justification",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateError(Exception):
    pass


@dataclass
class Task7BState:
    status: str = NOT_STARTED
    protocol_hash: str | None = None
    history: list[dict] = field(default_factory=list)
    updated_at: str = ""

    def transition(self, new_status: str, **fields) -> None:
        if self.status in TERMINAL:
            raise StateError(f"cannot transition out of terminal status {self.status!r} except via a versioned remediation plan")
        old = self.status
        self.status = new_status
        self.updated_at = _now_iso()
        self.history.append({"from": old, "to": new_status, "timestamp": self.updated_at, **fields})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task7BState":
        return cls(status=d.get("status", NOT_STARTED), protocol_hash=d.get("protocol_hash"), history=list(d.get("history", [])), updated_at=d.get("updated_at", ""))


def load(path: Path) -> Task7BState:
    path = Path(path)
    if not path.exists():
        return Task7BState()
    return Task7BState.from_dict(json.loads(path.read_text()))


def save(state: Task7BState, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2))
    tmp.replace(path)
