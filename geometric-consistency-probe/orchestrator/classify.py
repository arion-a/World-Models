"""Classifies WHY a task reported implementation_status == "BLOCKED" so
the orchestrator can decide, without asking a human, whether to keep
repairing or to stop immediately.

Authority hierarchy (research/CANONICAL_RESEARCH_PROTOCOL.md, "CORE
REQUIREMENT" / "IMPORTANT DISTINCTION"):

  A. ROUTINE / B. IMPLEMENTATION -- resolvable from the canonical
     protocol, the task specification, or ordinary engineering judgment.
     Not a real blocker at all: treated as an ordinary QA failure,
     eligible for the normal automatic repair loop up to
     MAX_FIX_ATTEMPTS.
  C. SCIENTIFIC_CONFLICT -- two authoritative scientific requirements
     genuinely conflict (e.g. the task spec asks for something
     research/RESEARCH_INVARIANTS.md forbids).
  D. DESTRUCTIVE_ACTION -- completing the task as specified would
     require a destructive/irreversible action outside the permitted
     workflow (force-push, rm -rf, resetting shared history, etc.).
  E. MISSING_DEPENDENCY -- a required credential, model, dataset, or
     service is genuinely unavailable in this environment.
  BLOCKER -- a real blocker was reported but doesn't fit any of the
     above categories confidently.

C, D, E, and unclassified BLOCKER all mean: retrying will not help, so
the repair loop is skipped and the task goes straight to BLOCKED for a
human to look at (orchestrator/run.py's STOP_REPAIR_CATEGORIES). A, B
mean: this is exactly what the automatic repair loop exists for.

This is necessarily heuristic for whatever a task's own result JSON
doesn't self-declare via `blocker_category` -- it cannot verify that a
claimed external blocker really is one. What it DOES guarantee is that
the orchestrator never burns repair attempts on a category of problem
no amount of re-prompting can fix, and never silently powers through a
category that requires a human decision.
"""

from __future__ import annotations

ROUTINE = "ROUTINE"
IMPLEMENTATION = "IMPLEMENTATION"
SCIENTIFIC_CONFLICT = "SCIENTIFIC_CONFLICT"
DESTRUCTIVE_ACTION = "DESTRUCTIVE_ACTION"
MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
BLOCKER = "BLOCKER"

CATEGORIES = (ROUTINE, IMPLEMENTATION, SCIENTIFIC_CONFLICT, DESTRUCTIVE_ACTION, MISSING_DEPENDENCY, BLOCKER)

# Categories where retrying cannot help -- skip the remaining automatic
# repair attempts and go straight to BLOCKED.
STOP_REPAIR_CATEGORIES = (SCIENTIFIC_CONFLICT, DESTRUCTIVE_ACTION, MISSING_DEPENDENCY, BLOCKER)

_MISSING_DEPENDENCY_KEYWORDS = (
    "credential",
    "api key",
    "api_key",
    "no network",
    "network access",
    "not reachable",
    "unreachable",
    "no access to",
    "quota",
    "rate limit",
    "unavailable",
    "not installed",
    "license",
    "no gpu",
    "out of memory",
    "no approval surface",
)
_DESTRUCTIVE_KEYWORDS = (
    "force push",
    "force-push",
    "rm -rf",
    "reset --hard",
    "drop table",
    "delete the repository",
    "rewrite history",
    "truncate",
)
_SCIENTIFIC_CONFLICT_KEYWORDS = (
    "conflicts with",
    "contradicts",
    "invariant violat",
    "cannot satisfy both",
    "mutually exclusive",
    "incompatible with the research design",
    "requires changing the research design",
    "protocol conflict",
)


def classify_blocker(result_json: dict | None) -> str:
    """ROUTINE means "there is no declared blocker at all" -- i.e. this
    isn't a BLOCKED-status result, so the ordinary QA-fail/repair path
    applies unchanged. Every other return value means implementation_
    status == "BLOCKED" was actually reported.
    """
    if not isinstance(result_json, dict):
        return ROUTINE
    if result_json.get("implementation_status") != "BLOCKED":
        return ROUTINE

    declared = result_json.get("blocker_category")
    if declared in CATEGORIES and declared != ROUTINE:
        return declared

    text = " ".join(str(result_json.get(k, "")) for k in ("blocking_issue", "scientific_result", "next_steps")).lower()

    if any(kw in text for kw in _DESTRUCTIVE_KEYWORDS):
        return DESTRUCTIVE_ACTION
    if any(kw in text for kw in _SCIENTIFIC_CONFLICT_KEYWORDS):
        return SCIENTIFIC_CONFLICT
    if any(kw in text for kw in _MISSING_DEPENDENCY_KEYWORDS):
        return MISSING_DEPENDENCY
    return BLOCKER
