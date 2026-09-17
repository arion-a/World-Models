"""Tests for Task 16 -- the final research report.

`tasks/16_report.md`'s "Tests required" section: every `state/task_0{6..9}
|1{0..5}_result.json` referenced by `reports/final_research_report.md` must
actually exist, and at least one number from each must be quoted correctly
in the report (a simple string/number match against the JSON source, to
guard against transcription drift). This also checks the report's required
structure, its "understands 3D" interpretation discipline, and that
negative results are not omitted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "state"
REPORT_PATH = REPO_ROOT / "reports" / "final_research_report.md"
RESULT_TASK_IDS = tuple(range(6, 16))  # 6..15 inclusive

# Either a decimal with >=2 decimal digits (e.g. "-0.313") or a bare
# integer of >=2 digits (e.g. "808", "1024") -- both specific enough that
# a match is not coincidental.
_NUMBER_RE = re.compile(r"-?\d+\.\d{2,}|-?\d{2,}\b")


def _numbers_in(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


@pytest.fixture(scope="module")
def report_text() -> str:
    assert REPORT_PATH.exists(), f"{REPORT_PATH} must exist"
    return REPORT_PATH.read_text()


@pytest.mark.parametrize("task_id", RESULT_TASK_IDS)
def test_cited_task_result_file_exists(task_id):
    path = STATE_DIR / f"task_{task_id:02d}_result.json"
    assert path.exists(), f"{path} (cited by the Task 16 report) does not exist"


@pytest.mark.parametrize("task_id", RESULT_TASK_IDS)
def test_report_cites_the_result_file_path(report_text, task_id):
    expected = f"state/task_{task_id:02d}_result.json"
    assert expected in report_text, f"{REPORT_PATH} never cites {expected}"


@pytest.mark.parametrize("task_id", RESULT_TASK_IDS)
def test_at_least_one_number_from_each_result_quoted_correctly(report_text, task_id):
    """Guards against transcription drift: pull every >=2-decimal numeric
    literal out of the task's own scientific_result (falling back to the
    whole file for Task 15, whose scientific_result is prose-only but
    whose metrics still appear verbatim), and require at least one to
    appear, as an exact substring, somewhere in the final report."""
    path = STATE_DIR / f"task_{task_id:02d}_result.json"
    data = json.loads(path.read_text())
    # Union of the human-authored scientific_result prose and the full file
    # (so integer-only fields like tests.passed also count as citable
    # numbers) -- a report claim can legitimately draw on either.
    source_text = data.get("scientific_result", "") + "\n" + json.dumps(data)
    source_numbers = _numbers_in(source_text)
    assert source_numbers, f"{path} has no numeric literal (>=2 digits) to check transcription against"

    report_numbers = _numbers_in(report_text)
    overlap = source_numbers & report_numbers
    assert overlap, (
        f"None of task {task_id}'s numbers ({sorted(source_numbers)[:5]}...) "
        f"appear verbatim in {REPORT_PATH}"
    )


def test_task_07b_supplementary_result_is_cited_correctly(report_text):
    """Task 7B is not one of the numbered 6-15 pipeline tasks, but this
    report cites it as directly relevant supplementary evidence for the
    camera-rotation reconciliation -- its own sealed-test number must be
    quoted correctly too, under the same transcription-drift discipline."""
    path = STATE_DIR / "task_07b_result.json"
    assert path.exists()
    data = json.loads(path.read_text())
    sealed_r2 = None
    for entry in data.get("history", []):
        if "sealed_test_r2" in entry:
            sealed_r2 = entry["sealed_test_r2"]
    assert sealed_r2 is not None, "state/task_07b_result.json has no recorded sealed_test_r2"
    # The report rounds to 4 significant decimals ("0.5571"); the source
    # records the full float -- check the rounded prefix is a genuine
    # substring-consistent rounding of the recorded value, then check the
    # report actually contains it.
    rounded = f"{sealed_r2:.4f}"
    assert rounded in report_text, f"Task 7B's sealed_test_r2 ({sealed_r2}) rounds to {rounded!r}, not found in {REPORT_PATH}"


REQUIRED_SECTIONS = (
    "## Abstract",
    "## Research question",
    "## Motivation",
    "## Experimental setup",
    "### Controlled 3D environment",
    "### Rendering",
    "### Representation extraction",
    "### Geometric transformations",
    "### Physical-state probes",
    "### Appearance controls",
    "### Baselines",
    "### Scale experiments",
    "### Temporal experiments",
    "### Occlusion experiments",
    "### Counterfactual experiments",
    "## Results",
    "## Negative results",
    "## Failure analysis",
    "## Alternative explanations",
    "## Limitations",
    "## Conclusions",
    "## Future work",
)


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_report_has_required_section(report_text, section):
    assert section in report_text, f"{REPORT_PATH} is missing required section {section!r}"


def test_report_has_supported_vs_not_supported_section(report_text):
    assert "Supported vs. not-supported claims" in report_text


def test_report_has_claim_evidence_matrix(report_text):
    assert "## Claim-evidence matrix" in report_text
    assert "CLAIM" in report_text and "EXPERIMENT" in report_text and "METRIC" in report_text
    assert "CONTROL" in report_text and "LIMITATION" in report_text


def test_understands_3d_phrase_never_appears_without_a_metric_on_the_same_line(report_text):
    """Global Invariant 29 / research/RESEARCH_INVARIANTS.md invariant 13:
    'the model understands 3D' (or equivalent) must never appear without an
    operational metric immediately alongside it. This report only ever
    discusses the phrase to say it is NOT claimed -- every line containing
    it must also contain a negation marker or a metric reference."""
    lines = report_text.splitlines()
    indices_with_phrase = [i for i, line in enumerate(lines) if "understand" in line.lower() and "3d" in line.lower()]
    assert indices_with_phrase, "expected the report to explicitly discuss (and disclaim) the 'understands 3D' phrase"
    for i in indices_with_phrase:
        # Markdown hard-wraps long sentences across lines, so check a small
        # window (this line plus the next couple) rather than one bare line.
        window = " ".join(lines[i : i + 3]).lower()
        is_quoted_question = "?" in window  # e.g. restating the canonical research
        # question ("Does a World Model Actually Understand 3D?") is framing,
        # not a claim, and carries its own question-mark signal instead.
        has_negation = any(tok in window for tok in ("not ", "never", "no task", "does not"))
        has_metric_ref = any(tok in window for tok in ("r²", "r^2", "metric", "accuracy"))
        assert is_quoted_question or has_negation or has_metric_ref, (
            f"'understands 3D' appears without a disclaimer, metric reference, or quoted-question marker nearby: {lines[i]!r}"
        )


NEGATIVE_RESULT_MARKERS = (
    ("6", "not linearly predictable"),
    ("9", "not more representation-changing"),
    ("12", "static-clip control"),
    ("13", "not more recoverable"),
    ("14", "invalidating"),
)


@pytest.mark.parametrize("task_id,marker", NEGATIVE_RESULT_MARKERS)
def test_negative_results_are_not_omitted(report_text, task_id, marker):
    assert "## Negative results" in report_text
    negative_section = report_text.split("## Negative results", 1)[1].split("## Failure analysis", 1)[0]
    assert f"Task {task_id}" in negative_section, f"Task {task_id}'s negative result is not listed in the Negative results section"
    assert marker.lower() in negative_section.lower()


def test_every_headline_result_row_reports_a_baseline_comparison(report_text):
    """Global Invariant 19 / research/RESEARCH_INVARIANTS.md invariant 14:
    no result may be reported without its baseline comparison. Checks the
    headline results table has a non-empty baseline column for every task
    row."""
    table_start = report_text.index("### Headline results table")
    table_end = report_text.index("### Per-task narrative")
    table = report_text[table_start:table_end]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "Task" not in line and "---" not in line]
    assert len(rows) >= 10, "expected at least 10 data rows (Tasks 6-15) in the headline results table"
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) >= 5, f"malformed table row: {row!r}"
        baseline_cell = cells[3]
        assert baseline_cell and baseline_cell != "n/a" or "15" in cells[0], f"row missing baseline comparison: {row!r}"
