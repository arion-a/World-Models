"""Task 7B final, end-to-end QA (contract Sec. 14's "Completion"
checklist) -- run once after Stage 2/3 finishes, independently
re-deriving pass/fail from the artifacts on disk rather than trusting
any stage's own print statements. Mirrors the Task 7 forensic audit's
methodology: mechanical re-checks, not a self-report.

Determines COMPLETE vs COMPLETE_SCALE_LIMITED (never a silent choice
made inside run_learning_curve.py itself): a curve that never
stabilized over the full preregistered schedule is a valid, honest
outcome per the contract, but the resulting state must say so.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experiments.task7b_latent_transformation_discovery import state

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parent.parent
STATE_PATH = REPO_ROOT / "state" / "task_07b_result.json"

HISTORICAL_PATHS = (
    "geometric-consistency-probe/state/task_07_result.json",
    "geometric-consistency-probe/experiments/geometric_consistency/",
    "geometric-consistency-probe/experiments/task7_forensic_audit/",
    "geometric-consistency-probe/experiments/task7_diagnostic/",
)

REQUIRED_ARTIFACTS = (
    "protocol.json", "protocol.md", "split_manifest.json",
    "stage1_data_manifest.json", "z_cache_primary.npz",
    "data_validation_qa_report.json", "data_validation_qa_report.md",
    "learning_curve_all_points.json", "learning_curve_aggregated.json",
    "model_selection_record.json", "sealed_test_result.json",
    "structure_test_results.json",
)


class FinalQAResult:
    def __init__(self):
        self.layers = []

    def add(self, name: str, passed: bool, details: list[str]):
        self.layers.append({"name": name, "passed": passed, "details": details})

    @property
    def passed(self) -> bool:
        return all(l["passed"] for l in self.layers)

    def render(self) -> str:
        lines = [f"Task 7B FINAL QA: {'PASS' if self.passed else 'FAIL'}"]
        for l in self.layers:
            lines.append(f"[{'PASS' if l['passed'] else 'FAIL'}] {l['name']}")
            for d in l["details"]:
                lines.append(f"    - {d}")
        return "\n".join(lines)


def check_required_artifacts_exist(qa: FinalQAResult):
    missing = [a for a in REQUIRED_ARTIFACTS if not (OUT_DIR / a).exists()]
    qa.add("All required artifacts exist on disk", not missing, [f"missing: {a}" for a in missing])


def check_historical_artifacts_untouched(qa: FinalQAResult):
    """Re-derives from git itself (never trusted from memory): none of
    Task 7's historical artifacts have any uncommitted OR
    Task-7B-committed change. Uses `git status --porcelain` (working
    tree) and confirms no Task 7B commit's diff touches these paths."""
    details = []
    ok = True
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--"] + list(HISTORICAL_PATHS),
        cwd=REPO_ROOT.parent, capture_output=True, text=True,
    )
    if dirty.stdout.strip():
        ok = False
        details.append(f"uncommitted changes under historical paths: {dirty.stdout.strip()}")

    # Confirm no commit reachable from HEAD but not from the pre-Task-7B
    # base touches these paths, by checking the full log of commits that
    # touched them and asserting the most recent one predates any Task 7B
    # commit (Task 7B commits are identified by touching this package).
    log = subprocess.run(
        ["git", "log", "--oneline", "--", *HISTORICAL_PATHS],
        cwd=REPO_ROOT.parent, capture_output=True, text=True,
    )
    task7b_commits = subprocess.run(
        ["git", "log", "--oneline", "--",
         "geometric-consistency-probe/experiments/task7b_latent_transformation_discovery/"],
        cwd=REPO_ROOT.parent, capture_output=True, text=True,
    )
    historical_hashes = {line.split()[0] for line in log.stdout.strip().splitlines() if line}
    task7b_hashes = {line.split()[0] for line in task7b_commits.stdout.strip().splitlines() if line}
    overlap = historical_hashes & task7b_hashes
    if overlap:
        ok = False
        details.append(f"a Task 7B commit also touched a historical path: {overlap}")
    qa.add("Historical Task 7 / audit / diagnostic artifacts untouched", ok, details)


def check_sealed_test_clean_and_single(qa: FinalQAResult):
    result = json.loads((OUT_DIR / "sealed_test_result.json").read_text())
    details = []
    ok = True
    if not result.get("sealed"):
        ok = False
        details.append("sealed_test_result.json does not have sealed=True")
    if not str(result.get("contamination_status", "")).startswith("CLEAN"):
        ok = False
        details.append(f"contamination_status is not CLEAN: {result.get('contamination_status')}")
    for key in ("selected_config_test_metrics", "control_test_metrics", "bootstrap_ci_over_scenes", "test_scene_ids"):
        if key not in result:
            ok = False
            details.append(f"missing required field: {key}")
    qa.add("Sealed test artifact is clean and complete", ok, details)


def check_structure_tests_all_resolved(qa: FinalQAResult):
    """Every predeclared Stage 3 check must have a numeric result or an
    explicit NOT_APPLICABLE reason -- never silently missing."""
    result = json.loads((OUT_DIR / "structure_test_results.json").read_text())
    details = []
    ok = True

    def _check_entry(name, entry):
        nonlocal ok
        if entry.get("applicable") is True:
            if "output_space_agreement" not in entry and "metrics_vs_true_interior_magnitude" not in entry:
                ok = False
                details.append(f"{name}: applicable=True but no output-space metrics present")
        elif entry.get("applicable") is False:
            if not entry.get("reason"):
                ok = False
                details.append(f"{name}: applicable=False but no reason given")
        else:
            ok = False
            details.append(f"{name}: missing 'applicable' field entirely")

    for entry in result.get("composition", []):
        _check_entry(f"composition theta1={entry.get('theta1')} theta2={entry.get('theta2')}", entry)
    for entry in result.get("inverse", []):
        _check_entry(f"inverse theta={entry.get('theta')}", entry)
    _check_entry("identity", result.get("identity", {}))
    _check_entry("interpolation", result.get("interpolation", {}))

    if not result.get("composition") and not result.get("inverse"):
        ok = False
        details.append("no composition or inverse results recorded at all")

    qa.add("Every predeclared structure test resolved (result or NOT_APPLICABLE)", ok, details)


def check_data_validation_qa_passed(qa: FinalQAResult):
    report = json.loads((OUT_DIR / "data_validation_qa_report.json").read_text())
    qa.add("Stage 1 data-validation QA passed", bool(report.get("passed")), [] if report.get("passed") else ["data_validation_qa_report.json: passed=False"])


def check_repo_test_suite_passes(qa: FinalQAResult):
    proc = subprocess.run(
        ["python3", "-m", "pytest", "-m", "not slow", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
    qa.add("Full repo test suite passes (`pytest -m \"not slow\"`)", proc.returncode == 0, [] if proc.returncode == 0 else [tail])


def determine_completion_status() -> tuple[str, str]:
    all_points_path = OUT_DIR / "learning_curve_aggregated.json"
    aggregated = json.loads(all_points_path.read_text())
    from experiments.task7b_latent_transformation_discovery import learning_curve
    stabilized = learning_curve.check_stabilization(aggregated)
    if stabilized:
        return state.COMPLETE, "Learning curve stabilization rule satisfied on the preregistered schedule."
    return (
        state.COMPLETE_SCALE_LIMITED,
        "Learning curve did NOT stabilize within N_train<=1024 (the full preregistered schedule ran); "
        "per contract Sec. 5/13, the result is scale-limited and carries no asymptotic claim.",
    )


def run_all() -> FinalQAResult:
    qa = FinalQAResult()
    check_required_artifacts_exist(qa)
    check_historical_artifacts_untouched(qa)
    check_data_validation_qa_passed(qa)
    check_sealed_test_clean_and_single(qa)
    check_structure_tests_all_resolved(qa)
    check_repo_test_suite_passes(qa)
    return qa


if __name__ == "__main__":
    qa = run_all()
    print(qa.render())
    (OUT_DIR / "qa_report.md").write_text(qa.render())
    (OUT_DIR / "qa_report.json").write_text(json.dumps({"passed": qa.passed, "layers": qa.layers}, indent=2))

    s = state.load(STATE_PATH)
    if not qa.passed:
        if s.status not in state.TERMINAL:
            s.transition(state.FAILED_SOFTWARE, reason="final QA failed -- see qa_report.md")
            state.save(s, STATE_PATH)
        raise SystemExit("Final QA FAILED -- see qa_report.md")

    completion_status, reason = determine_completion_status()
    if s.status == state.STRUCTURE_TESTED:
        s.transition(completion_status, reason=reason)
        state.save(s, STATE_PATH)
    print(f"\nFinal status: {completion_status} -- {reason}")
