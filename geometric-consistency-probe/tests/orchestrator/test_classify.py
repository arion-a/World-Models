"""Unit tests for orchestrator/classify.py's classify_blocker(): the
mechanism that decides, without asking a human, whether a QA-failing
task's declared blocker is retryable (ordinary implementation/
methodological failure -- keep repairing) or a genuine external blocker
(SCIENTIFIC_CONFLICT / DESTRUCTIVE_ACTION / MISSING_DEPENDENCY / BLOCKER
-- stop immediately, no repair attempt wasted).
"""

from __future__ import annotations

from orchestrator import classify


# --- not a blocker at all: ordinary QA-fail path applies ---------------------


def test_non_dict_result_is_routine():
    assert classify.classify_blocker(None) == classify.ROUTINE


def test_missing_implementation_status_is_routine():
    assert classify.classify_blocker({"task": 6}) == classify.ROUTINE


def test_complete_status_is_routine():
    assert classify.classify_blocker({"task": 6, "implementation_status": "COMPLETE"}) == classify.ROUTINE


# --- self-declared blocker_category is trusted verbatim ----------------------


def test_declared_scientific_conflict_is_trusted():
    result = {
        "implementation_status": "BLOCKED",
        "blocker_category": "SCIENTIFIC_CONFLICT",
        "blocking_issue": "task spec asks for something totally unrelated to any keyword list",
    }
    assert classify.classify_blocker(result) == classify.SCIENTIFIC_CONFLICT


def test_declared_destructive_action_is_trusted():
    result = {"implementation_status": "BLOCKED", "blocker_category": "DESTRUCTIVE_ACTION", "blocking_issue": "n/a"}
    assert classify.classify_blocker(result) == classify.DESTRUCTIVE_ACTION


def test_declared_missing_dependency_is_trusted():
    result = {"implementation_status": "BLOCKED", "blocker_category": "MISSING_DEPENDENCY", "blocking_issue": "n/a"}
    assert classify.classify_blocker(result) == classify.MISSING_DEPENDENCY


def test_declared_bare_blocker_is_trusted():
    result = {"implementation_status": "BLOCKED", "blocker_category": "BLOCKER", "blocking_issue": "n/a"}
    assert classify.classify_blocker(result) == classify.BLOCKER


def test_declared_category_not_in_stop_list_falls_through_to_heuristics():
    # A nonsense/unknown declared category must not be trusted blindly --
    # only a real CATEGORIES member is honored; otherwise fall back to
    # keyword heuristics on the text fields.
    result = {
        "implementation_status": "BLOCKED",
        "blocker_category": "NOT_A_REAL_CATEGORY",
        "blocking_issue": "the required credential is unavailable in this environment",
    }
    assert classify.classify_blocker(result) == classify.MISSING_DEPENDENCY


def test_declared_routine_is_rejected_and_falls_back_to_heuristics():
    # A BLOCKED result cannot declare itself ROUTINE -- that would defeat
    # the whole point of the escape hatch. Falls back to the keyword
    # heuristics (or BLOCKER) instead of ever returning ROUTINE here.
    result = {"implementation_status": "BLOCKED", "blocker_category": "ROUTINE", "blocking_issue": "something vague"}
    assert classify.classify_blocker(result) != classify.ROUTINE


# --- keyword heuristics when blocker_category is absent ----------------------


def test_missing_dependency_keyword_heuristic():
    result = {"implementation_status": "BLOCKED", "blocking_issue": "no approval surface was available for this Bash command"}
    assert classify.classify_blocker(result) == classify.MISSING_DEPENDENCY


def test_missing_dependency_credential_keyword():
    result = {"implementation_status": "BLOCKED", "blocking_issue": "the required API key is not configured"}
    assert classify.classify_blocker(result) == classify.MISSING_DEPENDENCY


def test_destructive_keyword_heuristic():
    result = {"implementation_status": "BLOCKED", "blocking_issue": "completing this would require a force-push to main"}
    assert classify.classify_blocker(result) == classify.DESTRUCTIVE_ACTION


def test_scientific_conflict_keyword_heuristic():
    result = {"implementation_status": "BLOCKED", "blocking_issue": "the task spec conflicts with research invariant 3"}
    assert classify.classify_blocker(result) == classify.SCIENTIFIC_CONFLICT


def test_scientific_conflict_keyword_checked_in_scientific_result_field_too():
    result = {"implementation_status": "BLOCKED", "scientific_result": "this is mutually exclusive with the frozen-encoder invariant"}
    assert classify.classify_blocker(result) == classify.SCIENTIFIC_CONFLICT


def test_no_matching_keyword_falls_back_to_bare_blocker():
    result = {"implementation_status": "BLOCKED", "blocking_issue": "something went wrong that doesn't match any known pattern"}
    assert classify.classify_blocker(result) == classify.BLOCKER


def test_destructive_keyword_takes_priority_over_scientific_conflict_keyword():
    # Both keyword sets match -- destructive is checked first because an
    # irreversible action is the more urgent category to flag correctly.
    result = {
        "implementation_status": "BLOCKED",
        "blocking_issue": "this conflicts with invariant 5 and would also require rm -rf on the dataset directory",
    }
    assert classify.classify_blocker(result) == classify.DESTRUCTIVE_ACTION


# --- STOP_REPAIR_CATEGORIES contract ------------------------------------------


def test_stop_repair_categories_excludes_routine_and_implementation():
    assert classify.ROUTINE not in classify.STOP_REPAIR_CATEGORIES
    assert classify.IMPLEMENTATION not in classify.STOP_REPAIR_CATEGORIES


def test_stop_repair_categories_includes_all_genuine_blockers():
    for cat in (classify.SCIENTIFIC_CONFLICT, classify.DESTRUCTIVE_ACTION, classify.MISSING_DEPENDENCY, classify.BLOCKER):
        assert cat in classify.STOP_REPAIR_CATEGORIES
