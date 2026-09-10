"""Persistent orchestrator state (state/progress.json).

This is the single source of truth for "which task are we on." It must
survive interruption: restarting the orchestrator must never re-run a
task already recorded as PASSED, and must never advance past a task
whose status is not PASSED (research/RESEARCH_INVARIANTS.md invariant
18, applied to the orchestrator's own operation). Both properties are
enforced structurally here, not just by convention in run.py:

  * `advance_task()` is the ONLY way `current_task` moves forward, and
    it refuses to run unless `status == PASSED`.
  * `transition()` is the ONLY way `status` changes, and it checks
    `ALLOWED_TRANSITIONS` -- there is no code path that sets `PASSED`
    directly from `READY`/`RUNNING`/`FIXING`, only from `QA`.
  * `save()` always calls `validate()` first and writes atomically
    (temp file + os.replace), so a crash mid-write cannot corrupt
    progress.json into a state that silently looks fine on next load.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

READY = "READY"
RUNNING = "RUNNING"
QA = "QA"
FIXING = "FIXING"
PASSED = "PASSED"
FAILED = "FAILED"
BLOCKED = "BLOCKED"

STATUSES = (READY, RUNNING, QA, FIXING, PASSED, FAILED, BLOCKED)

# current_status -> set of statuses it may legally move to.
# FAILED/BLOCKED are terminal: resuming them requires an explicit
# operator reset (OrchestratorState.reset_task), never an automatic
# transition. PASSED -> READY happens only inside advance_task().
# PASSED -> BLOCKED is the one other legal exit from PASSED: QA already
# passed, but the checkpoint commit itself could not be created OR
# recovered (orchestrator/run.py's _finish_passed_task) -- a genuine
# MISSING_DEPENDENCY-shaped infra failure (e.g. git/disk failure), never
# an automatic re-attempt; the task must not be silently reported
# advanced without a real commit backing it.
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    READY: {RUNNING},
    RUNNING: {QA, FAILED},
    QA: {PASSED, FIXING, BLOCKED},
    FIXING: {RUNNING},
    PASSED: {READY, BLOCKED},
    FAILED: set(),
    BLOCKED: set(),
}

# Tasks 1-5 are already complete in this repository's real history --
# see research/RESEARCH_PLAN.md's task table. A fresh progress.json
# bootstraps to exactly that fact, not to "nothing done yet".
FIRST_PENDING_TASK = 6
LAST_TASK = 18
DEFAULT_COMPLETED_TASKS = tuple(range(1, FIRST_PENDING_TASK))

_MAX_HISTORY = 500


class StateError(Exception):
    """Malformed or invalid orchestrator state. Never silently repaired."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class OrchestratorState:
    current_task: int
    completed_tasks: list[int] = field(default_factory=lambda: list(DEFAULT_COMPLETED_TASKS))
    status: str = READY
    attempts: dict[str, int] = field(default_factory=dict)
    commits: dict[str, str | None] = field(default_factory=dict)
    last_qa: dict[str, dict | None] = field(default_factory=dict)
    # Per-task record that orchestrator.sync ran and whether it changed
    # tasks/{NN}_*.md to match research/CANONICAL_RESEARCH_PROTOCOL.md --
    # this is what makes "the task file was reconciled with the canonical
    # protocol" an auditable, non-interactive fact instead of something
    # anyone had to be asked about.
    sync: dict[str, dict] = field(default_factory=dict)
    # Per-task record of WHY a task is BLOCKED, when it's for a reason
    # more specific than "repair attempts exhausted" -- see
    # orchestrator/classify.py. None/absent means either not blocked, or
    # blocked only because MAX_FIX_ATTEMPTS was exhausted with no more
    # specific category identified.
    blockers: dict[str, dict] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    updated_at: str = ""

    # -- construction -----------------------------------------------------

    @classmethod
    def bootstrap(cls) -> "OrchestratorState":
        """The state a fresh checkout starts in: Tasks 1-5 complete, Task
        6 READY. Used when state/progress.json does not exist yet -- never
        used to overwrite an existing file."""
        state = cls(current_task=FIRST_PENDING_TASK, completed_tasks=list(DEFAULT_COMPLETED_TASKS), status=READY)
        state.updated_at = _now_iso()
        return state

    @classmethod
    def from_dict(cls, data: dict) -> "OrchestratorState":
        if not isinstance(data, dict):
            raise StateError(f"progress.json must contain a JSON object, got {type(data).__name__}")
        try:
            state = cls(
                current_task=data["current_task"],
                completed_tasks=list(data.get("completed_tasks", [])),
                status=data.get("status", READY),
                attempts=dict(data.get("attempts", {})),
                commits=dict(data.get("commits", {})),
                last_qa=dict(data.get("last_qa", {})),
                sync=dict(data.get("sync", {})),
                blockers=dict(data.get("blockers", {})),
                history=list(data.get("history", [])),
                updated_at=data.get("updated_at", ""),
            )
        except KeyError as exc:
            raise StateError(f"progress.json missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise StateError(f"progress.json has a malformed field: {exc}") from exc
        state.validate()
        return state

    def to_dict(self) -> dict:
        return asdict(self)

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        if not isinstance(self.current_task, int):
            raise StateError(f"current_task must be an int, got {type(self.current_task).__name__}")
        if not (1 <= self.current_task <= LAST_TASK + 1):
            raise StateError(f"current_task {self.current_task} out of range [1, {LAST_TASK + 1}]")

        if not isinstance(self.completed_tasks, list) or not all(isinstance(t, int) for t in self.completed_tasks):
            raise StateError("completed_tasks must be a list of ints")
        if len(set(self.completed_tasks)) != len(self.completed_tasks):
            raise StateError(f"completed_tasks contains duplicates: {self.completed_tasks}")
        for t in self.completed_tasks:
            if not (1 <= t <= LAST_TASK):
                raise StateError(f"completed_tasks contains out-of-range task {t}")

        if self.current_task in self.completed_tasks:
            raise StateError(
                f"current_task {self.current_task} is also listed in completed_tasks "
                f"-- a task cannot be both 'the one we're on' and 'done'"
            )
        # current_task must be the smallest not-yet-completed task: no gaps
        # left behind (e.g. completed=[1,2,4], current=5 silently skipping 3).
        expected_completed = set(range(1, self.current_task))
        if set(self.completed_tasks) != expected_completed:
            raise StateError(
                f"completed_tasks {sorted(self.completed_tasks)} is inconsistent with "
                f"current_task {self.current_task} (expected exactly {sorted(expected_completed)})"
            )

        if self.status not in STATUSES:
            raise StateError(f"status {self.status!r} is not one of {STATUSES}")

        for label, mapping, value_check in (
            ("attempts", self.attempts, lambda v: isinstance(v, int) and v >= 0),
            ("commits", self.commits, lambda v: v is None or isinstance(v, str)),
        ):
            if not isinstance(mapping, dict):
                raise StateError(f"{label} must be a dict")
            for k, v in mapping.items():
                if not _is_task_key(k):
                    raise StateError(f"{label} has a non-task-number key: {k!r}")
                if not value_check(v):
                    raise StateError(f"{label}[{k!r}] has an invalid value: {v!r}")

        if not isinstance(self.last_qa, dict):
            raise StateError("last_qa must be a dict")
        for k, v in self.last_qa.items():
            if not _is_task_key(k):
                raise StateError(f"last_qa has a non-task-number key: {k!r}")
            if v is not None and not isinstance(v, dict):
                raise StateError(f"last_qa[{k!r}] must be a dict or null")

        for label, mapping in (("sync", self.sync), ("blockers", self.blockers)):
            if not isinstance(mapping, dict):
                raise StateError(f"{label} must be a dict")
            for k, v in mapping.items():
                if not _is_task_key(k):
                    raise StateError(f"{label} has a non-task-number key: {k!r}")
                if not isinstance(v, dict):
                    raise StateError(f"{label}[{k!r}] must be a dict")

        if not isinstance(self.history, list):
            raise StateError("history must be a list")

    # -- transitions ------------------------------------------------------

    def transition(self, new_status: str, **event_fields) -> None:
        if new_status not in STATUSES:
            raise StateError(f"cannot transition to unknown status {new_status!r}")
        if new_status == self.status:
            # Reaffirming the current status is a no-op, not a state
            # change -- this is what makes resuming after an interruption
            # safe: whichever status the process died in, re-entering the
            # orchestrator loop and re-asserting that same status must not
            # raise just because it isn't a "new" transition.
            self._append_history("transition", from_status=self.status, to_status=new_status, note="resumed (no-op)", **event_fields)
            return
        allowed = ALLOWED_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise StateError(f"illegal transition {self.status!r} -> {new_status!r} (allowed: {sorted(allowed)})")
        old_status = self.status
        self.status = new_status
        self._append_history("transition", from_status=old_status, to_status=new_status, **event_fields)

    def record_attempt(self, task: int) -> int:
        key = str(task)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        self._append_history("attempt", task=task, attempt=self.attempts[key])
        return self.attempts[key]

    def record_qa(self, task: int, qa_summary: dict) -> None:
        self.last_qa[str(task)] = qa_summary
        self._append_history("qa", task=task, passed=qa_summary.get("passed"))

    def record_sync(self, task: int, changed: bool, path: str, title: str) -> None:
        """Records that orchestrator.sync reconciled tasks/{NN}_*.md with
        research/CANONICAL_RESEARCH_PROTOCOL.md for this task -- a
        routine, deterministic, non-interactive fact, logged for audit
        rather than asked about."""
        self.sync[str(task)] = {"changed": changed, "path": path, "title": title, "timestamp": _now_iso()}
        self._append_history("sync", task=task, changed=changed)

    def record_blocker(self, task: int, category: str, reason: str) -> None:
        """Records WHY a task is BLOCKED when it's for a more specific
        reason than 'repair attempts exhausted' -- see
        orchestrator/classify.py's category names."""
        self.blockers[str(task)] = {"category": category, "reason": reason, "timestamp": _now_iso()}
        self._append_history("blocker", task=task, category=category)

    def advance_task(self, commit_hash: str | None) -> int:
        """The ONLY way current_task moves forward. Requires status ==
        PASSED (i.e. QA actually passed this task) -- this is the
        structural enforcement of 'no advancing after failed QA'."""
        if self.status != PASSED:
            raise StateError(f"cannot advance_task: status is {self.status!r}, not PASSED")
        finished_task = self.current_task
        self.completed_tasks = sorted(set(self.completed_tasks) | {finished_task})
        self.commits[str(finished_task)] = commit_hash
        self.current_task = finished_task + 1
        self.transition(READY, task=finished_task, commit=commit_hash)
        self.validate()
        return self.current_task

    def reset_task(self, task: int) -> None:
        """Explicit operator override to retry a BLOCKED/FAILED task from
        scratch. Never called automatically by the orchestrator loop."""
        if task != self.current_task:
            raise StateError(f"reset_task({task}) does not match current_task {self.current_task}")
        if self.status not in (BLOCKED, FAILED):
            raise StateError(f"reset_task only valid from BLOCKED/FAILED, current status is {self.status!r}")
        self.attempts[str(task)] = 0
        self.blockers.pop(str(task), None)
        old_status = self.status
        self.status = READY
        self._append_history("reset", task=task, from_status=old_status)

    def next_task(self) -> int | None:
        """The next task to work on, or None if Task 18 (LAST_TASK) is
        already complete."""
        if self.current_task > LAST_TASK:
            return None
        return self.current_task

    def is_complete(self) -> bool:
        return self.next_task() is None

    def _append_history(self, event: str, **fields) -> None:
        entry = {"event": event, "timestamp": _now_iso(), **fields}
        self.history.append(entry)
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_MAX_HISTORY:]


def _is_task_key(key) -> bool:
    if not isinstance(key, str):
        return False
    try:
        int(key)
    except ValueError:
        return False
    return True


def load(path: str | Path) -> OrchestratorState:
    """Load state/progress.json, bootstrapping a fresh one if it doesn't
    exist yet. Raises StateError (never silently repairs) if the file
    exists but is malformed -- an operator must look at that by hand."""
    path = Path(path)
    if not path.exists():
        return OrchestratorState.bootstrap()
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise StateError(f"{path} is not valid JSON: {exc}") from exc
    return OrchestratorState.from_dict(raw)


def save(state: OrchestratorState, path: str | Path) -> None:
    """Validate, then write atomically (temp file + os.replace) so an
    interruption mid-write cannot leave progress.json half-written."""
    state.validate()
    state.updated_at = _now_iso()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=False) + "\n")
    os.replace(tmp_path, path)
