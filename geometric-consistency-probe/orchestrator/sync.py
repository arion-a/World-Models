"""Deterministic reconciliation of tasks/{NN}_*.md with
research/CANONICAL_RESEARCH_PROTOCOL.md.

Per the canonical authority hierarchy, the task-specific execution file
is a DERIVED artifact of the canonical protocol, never an independent
source of truth. Whether tasks/{NN}_*.md matches the canonical
document's "# TASK N -- ..." section is therefore a ROUTINE, DERIVABLE
fact -- never something to ask a human about. This module makes it a
deterministic function of two files on disk: read the canonical
section, render it into the (fixed, tested) 9-section execution format,
and write it if the existing file doesn't already match exactly. There
is no LLM call and no judgment involved, by design -- see
research/CANONICAL_RESEARCH_PROTOCOL.md's "IMPORTANT DISTINCTION"
section (this is category A, ROUTINE DERIVABLE DECISION).

A consequence worth stating plainly: hand-edits to tasks/{NN}_*.md are
NOT preserved across a sync. If the canonical document and the task
file disagree, the canonical document wins and the task file is
overwritten to match it (research/CANONICAL_RESEARCH_PROTOCOL.md's own
stated rule). Edit the canonical document, not the generated task file.

Whether the canonical document even HAS a section for a given task
number is the one condition that still means "stop and wait" (raises
CanonicalSectionNotFound) -- there is nothing to derive a task from
until its scientific design exists, and inventing one would be exactly
the kind of unrequested scope decision Claude must never make silently
(research/RESEARCH_INVARIANTS.md invariant 15).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# The 27 field names research/CANONICAL_RESEARCH_PROTOCOL.md uses for
# every task (its "REQUIRED TASK FORMAT" section), in the order they
# appear under each "# TASK N -- ..." heading.
CANONICAL_FIELD_ORDER = (
    "Purpose",
    "Scientific question",
    "Why this task exists",
    "Relationship to previous tasks",
    "Hypothesis",
    "Mathematical formulation",
    "Inputs",
    "Outputs",
    "Experimental protocol",
    "Dataset requirements",
    "Train/test protocol",
    "Controls",
    "Baselines",
    "Metrics",
    "Required artifacts",
    "Required software tests",
    "Required scientific-validity tests",
    "Required research-alignment checks",
    "Required leakage checks",
    "Reproducibility requirements",
    "Failure conditions",
    "Acceptance criteria",
    "Interpretation rules",
    "Explicit prohibited shortcuts",
    "What a positive result means",
    "What a negative result means",
    "What this task does NOT establish",
)

# How the 27 canonical fields map onto the 9 headers
# tests/research/test_research_alignment.py requires every tasks/{NN}_*.md
# to have. Multi-field sections prefix each contributing field with a
# bold inline label so provenance stays traceable; single-field sections
# render the body directly.
SECTION_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Objective", ("Purpose", "Why this task exists", "Relationship to previous tasks")),
    ("Scientific question", ("Scientific question", "Hypothesis", "Mathematical formulation")),
    (
        "Implementation requirements",
        ("Inputs", "Outputs", "Experimental protocol", "Dataset requirements", "Train/test protocol", "Controls", "Baselines", "Metrics"),
    ),
    ("Required artifacts", ("Required artifacts",)),
    ("Tests required", ("Required software tests", "Required scientific-validity tests", "Required research-alignment checks")),
    ("Leakage checks", ("Required leakage checks",)),
    ("Acceptance criteria", ("Failure conditions", "Acceptance criteria")),
    ("Prohibited shortcuts", ("Explicit prohibited shortcuts",)),
    ("Scientific interpretation limits", ("Interpretation rules", "What a positive result means", "What a negative result means", "What this task does NOT establish")),
)

_TASK_HEADER_RE = re.compile(r"^# TASK (\d+) — (.+)$", re.MULTILINE)
_SECTION_STOP_RE = re.compile(r"^(?:# TASK \d+ —|## Validation performed)", re.MULTILINE)
_FIELD_HEADER_RE = re.compile(r"^### (.+)$", re.MULTILINE)


class CanonicalSectionNotFound(Exception):
    """No '# TASK N -- ...' section exists yet in the canonical protocol
    for this task number. This is the ROUTINE 'wait for the research
    contract to be extended' condition, not an error -- the orchestrator
    stops cleanly and waits rather than fabricating a task."""


@dataclass
class SyncResult:
    task: int
    path: Path
    changed: bool
    title: str
    content: str


def parse_canonical_task_section(canonical_text: str, task: int) -> tuple[str, dict[str, str]] | None:
    """Returns (title, {field_name: body}) for '# TASK {task} -- TITLE',
    or None if no such section exists."""
    match = None
    for m in _TASK_HEADER_RE.finditer(canonical_text):
        if int(m.group(1)) == task:
            match = m
            break
    if match is None:
        return None

    title = match.group(2).strip()
    start = match.end()
    stop_match = _SECTION_STOP_RE.search(canonical_text, start)
    end = stop_match.start() if stop_match else len(canonical_text)
    body = canonical_text[start:end]

    field_matches = list(_FIELD_HEADER_RE.finditer(body))
    fields: dict[str, str] = {}
    for i, fm in enumerate(field_matches):
        name = fm.group(1).strip()
        field_start = fm.end()
        field_end = field_matches[i + 1].start() if i + 1 < len(field_matches) else len(body)
        fields[name] = body[field_start:field_end].strip()
    return title, fields


def render_task_spec_from_canonical(task: int, title: str, fields: dict[str, str]) -> str:
    """Pure, deterministic: same (task, title, fields) always renders the
    same markdown, byte for byte -- this is what makes repeated syncs
    idempotent (a no-op once already synchronized)."""
    lines = [
        f"# Task {task} — {title}",
        "",
        f"**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "
        f'"TASK {task} — {title.upper()}" section.** This file is generated '
        "deterministically from that section by `orchestrator/sync.py` -- "
        "hand edits here are not preserved across a sync; edit the "
        "canonical document instead. If the two ever disagree, the "
        "canonical document wins.",
        "",
    ]
    for section_name, field_names in SECTION_MAP:
        lines.append(f"## {section_name}")
        lines.append("")
        parts = []
        for fname in field_names:
            body = fields.get(fname, "").strip()
            if not body:
                continue
            parts.append(body if len(field_names) == 1 else f"**{fname}.** {body}")
        lines.append("\n\n".join(parts) if parts else "(not specified in the canonical protocol for this task.)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "task"


def sync_task_spec(
    task: int,
    repo_root: str | Path,
    canonical_path: Path | None = None,
    tasks_dir: Path | None = None,
    dry_run: bool = False,
) -> SyncResult:
    """Reconciles tasks/{NN}_*.md with the canonical protocol's section
    for `task`. Idempotent: calling this repeatedly with no canonical
    change produces `changed=False` every time after the first sync.

    `dry_run=True` computes and returns exactly what a real sync would
    do (including `changed`) WITHOUT writing anything -- required so
    `orchestrator/run.py --dry-run` can genuinely make no changes, since
    a stale spec is exactly the kind of thing a dry-run is meant to
    report on, not silently fix.

    Raises CanonicalSectionNotFound if the canonical document has no
    section for this task yet -- the orchestrator's caller should treat
    that exactly like the old tasks/{NN}_*.md-missing case: stop and
    wait, do not fail.
    """
    repo_root = Path(repo_root)
    canonical_path = canonical_path or (repo_root / "research" / "CANONICAL_RESEARCH_PROTOCOL.md")
    tasks_dir = tasks_dir or (repo_root / "tasks")

    if not canonical_path.exists():
        raise CanonicalSectionNotFound(f"{canonical_path} does not exist")

    canonical_text = canonical_path.read_text()
    parsed = parse_canonical_task_section(canonical_text, task)
    if parsed is None:
        raise CanonicalSectionNotFound(f"no '# TASK {task} —' section in {canonical_path}")
    title, fields = parsed

    rendered = render_task_spec_from_canonical(task, title, fields)

    existing_matches = sorted(tasks_dir.glob(f"{task:02d}_*.md"))
    if existing_matches:
        path = existing_matches[0]
        changed = (not path.exists()) or path.read_text() != rendered
    else:
        path = tasks_dir / f"{task:02d}_{_slugify(title)}.md"
        changed = True

    if changed and not dry_run:
        tasks_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)

    return SyncResult(task=task, path=path, changed=changed, title=title, content=rendered)
