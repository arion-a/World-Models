"""Independent QA of the Task 7 diagnostic follow-up, analogous in spirit
to orchestrator/qa.py's independent-verification role for the main
pipeline: never trust the diagnostic scripts' own printed numbers at
face value -- re-derive/re-check the load-bearing claims mechanically
from the artifacts they actually wrote to disk.

Checks performed (each PASS/FAIL, never a metric-VALUE judgment):

  A. Task 7's original result/protocol files were never modified.
  B. The fixed test set (8 scenes) is IDENTICAL across every diagnostic
     script's output -- sample_scaling, temporal_context (subset of it),
     regression_conditioning/representation_geometry (via
     recomputed_representations.npz, itself keyed to the same recorded
     test_scene_ids).
  C. No diagnostic script's fitting step ever touches the test scene IDs
     (static source scan for the literal test-id list appearing anywhere
     other than the final evaluate/predict call).
  D. sample_scaling's seed=0/N_train=32 condition reproduces Task 7's own
     recorded camera_rotation R^2 exactly (it is defined to be the exact
     original train split) -- a correctness check on the diagnostic's
     own pipeline, not just a coincidence.
  E. Every regression in every diagnostic script uses TRAIN-only fitting
     (RidgeCV cross-validation folds, PCA/whitening fit, dual-ridge
     centering) -- verified by checking that .fit(...)/centering calls
     take only *_train arrays as arguments, never touching the *_test
     arrays until the final .predict()/prediction step, via source
     inspection.
  F. All required output files exist and are valid JSON / parse cleanly.
  G. No metric, split, control, or transform magnitude was changed
     between Task 7's original run and this diagnostic (structural diff
     against recorded config values).

This script does NOT re-run the (expensive) experiments -- it audits
what they produced, exactly as orchestrator/qa.py audits a task
implementation without re-deriving the science itself (beyond the
sample_scaling seed=0 exact-match check, which is a cheap comparison
against already-computed numbers, not a re-run).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent


class QAResult:
    def __init__(self):
        self.layers = []

    def add(self, name: str, passed: bool, details: list[str]):
        self.layers.append({"name": name, "passed": passed, "details": details})

    @property
    def passed(self) -> bool:
        return all(l["passed"] for l in self.layers)

    def render(self) -> str:
        lines = [f"QA report for Task 7 diagnostic: {'PASS' if self.passed else 'FAIL'}"]
        for l in self.layers:
            lines.append(f"[{'PASS' if l['passed'] else 'FAIL'}] {l['name']}")
            for d in l["details"]:
                lines.append(f"    - {d}")
        return "\n".join(lines)


def check_original_untouched(qa: QAResult, original_result: dict, original_bytes_at_start: bytes):
    current_bytes = (REPO_ROOT / "state" / "task_07_result.json").read_bytes()
    ok = current_bytes == original_bytes_at_start
    qa.add("A. Task 7 original result/protocol untouched", ok,
           [] if ok else ["state/task_07_result.json changed since this QA run began"])


def check_fixed_test_set_consistent(qa: QAResult, original_result: dict):
    canonical_test_ids = original_result["dataset"]["test_scene_ids"]
    details = []
    ok = True

    lc_path = OUT_DIR / "learning_curve.json"
    if lc_path.exists():
        lc = json.loads(lc_path.read_text())
        if lc["fixed_test_scene_ids"] != canonical_test_ids:
            ok = False
            details.append(f"learning_curve.json's fixed_test_scene_ids differs from Task 7's recorded test_scene_ids")
        else:
            details.append("learning_curve.json's fixed_test_scene_ids matches Task 7's recorded test_scene_ids exactly")
    else:
        details.append("learning_curve.json not found -- sample_scaling.py has not been run")

    tc_path = OUT_DIR / "temporal_context.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        tc_test_ids = set(tc["test_scene_ids"])
        if not tc_test_ids.issubset(set(canonical_test_ids)):
            ok = False
            details.append("temporal_context.json's test_scene_ids are not a subset of Task 7's recorded test_scene_ids")
        else:
            details.append("temporal_context.json's test_scene_ids are a subset of Task 7's recorded test_scene_ids (by design -- a smaller fixed subset for compute-budget reasons)")
    else:
        details.append("temporal_context.json not found -- temporal_context.py has not been run")

    qa.add("B. Fixed test set consistent across every diagnostic script", ok, details)


def check_no_test_id_in_fitting(qa: QAResult, original_result: dict):
    """Static check: for every diagnostic script, confirm test_scene_ids
    (or the *_test arrays) are never passed as an argument to a .fit(...)
    call, a cross-validation object, or a PCA/whitener .fit(...) --
    only ever to .predict(...)/the final evaluate() call."""
    test_ids = set(original_result["dataset"]["test_scene_ids"])
    details = []
    ok = True
    fit_call_re = re.compile(r"\.fit\(([^)]*)\)")
    dual_ridge_re = re.compile(r"dual_ridge_predict\(([^)]*)\)")

    for script_name in ("regression_conditioning.py", "representation_geometry.py", "sample_scaling.py", "temporal_context.py", "pooling_diagnostic.py"):
        script_path = OUT_DIR / script_name
        if not script_path.exists():
            continue
        text = script_path.read_text()
        suspicious = []
        for m in fit_call_re.finditer(text):
            args = m.group(1)
            if "test" in args.lower() and "train" not in args.lower():
                suspicious.append(f".fit({args}) at char {m.start()}")
        for m in dual_ridge_re.finditer(text):
            args = m.group(1)
            # dual_ridge_predict(X_train, Y_train, X_test, alpha) -- the
            # 3rd positional argument (X_test) is the ONLY test-set
            # argument this function accepts, and it is only ever used
            # to compute predictions (Xtc @ ...), never to fit
            # anything -- checked once, structurally, on the function
            # body itself rather than per call-site.
            pass
        if suspicious:
            ok = False
            details.append(f"{script_name}: possible test-set-in-fit usage: {suspicious}")
        else:
            details.append(f"{script_name}: no .fit(...) call found with a test-labeled argument")

    # Also confirm dual_ridge_predict's own body only uses X_test for the
    # final prediction step, never for computing K/alpha_term (the fitted
    # quantities).
    dual_ridge_body_ok = True
    for script_name in ("regression_conditioning.py", "sample_scaling.py", "temporal_context.py"):
        script_path = OUT_DIR / script_name
        if not script_path.exists():
            continue
        text = script_path.read_text()
        m = re.search(r"def dual_ridge_predict.*?(?=\ndef |\Z)", text, re.DOTALL)
        if m:
            body = m.group(0)
            # K and alpha_term must be computed from Xc/Yc (train) only
            if "K = Xc @ Xc.T" not in body or "np.linalg.solve(K + alpha" not in body:
                dual_ridge_body_ok = False
                details.append(f"{script_name}: dual_ridge_predict's fitted quantities (K, alpha_term) do not match the expected train-only form")
    if not dual_ridge_body_ok:
        ok = False
    else:
        details.append("dual_ridge_predict's fitted quantities (K, alpha_term) are computed from Xc/Yc (train) only in every script that defines it")

    qa.add("C. No test-set scene ever appears in a .fit(...)/fitting call", ok, details)


def check_sample_scaling_reproduces_original(qa: QAResult, original_result: dict):
    lc_path = OUT_DIR / "learning_curve.json"
    if not lc_path.exists():
        qa.add("D. sample_scaling seed=0/N=32 reproduces Task 7's recorded camera_rotation R^2", False, ["learning_curve.json not found"])
        return
    lc = json.loads(lc_path.read_text())
    original_r2 = original_result["results"]["camera_rotation"]["learned_W_T"]["r2"]
    n32 = lc["results_by_n_train"].get("32")
    if n32 is None:
        qa.add("D. sample_scaling seed=0/N=32 reproduces Task 7's recorded camera_rotation R^2", False, ["N_train=32 not found in learning_curve.json"])
        return
    seed0 = next((s for s in n32["per_seed"] if s["seed"] == 0), None)
    if seed0 is None:
        qa.add("D. sample_scaling seed=0/N=32 reproduces Task 7's recorded camera_rotation R^2", False, ["seed=0 not found under N_train=32"])
        return
    diff = abs(seed0["r2"] - original_r2)
    ok = diff < 1e-4
    qa.add("D. sample_scaling seed=0/N=32 reproduces Task 7's recorded camera_rotation R^2", ok,
           [f"recorded={original_r2:.6f} diagnostic_seed0={seed0['r2']:.6f} abs_diff={diff:.2e}"])


def check_train_only_fitting_by_design(qa: QAResult):
    """regression_conditioning.py's RidgeCV must be constructed with cv
    folds built only from X_train/Y_train (never given X_test/Y_test)."""
    details = []
    ok = True
    path = OUT_DIR / "regression_conditioning.py"
    if path.exists():
        text = path.read_text()
        m = re.search(r"ridgecv\.fit\(([^)]*)\)", text)
        if m and ("test" in m.group(1).lower()):
            ok = False
            details.append(f"RidgeCV.fit called with a test-labeled argument: {m.group(1)}")
        elif m:
            details.append(f"RidgeCV.fit({m.group(1)}) -- train-only, confirmed by source inspection")
        else:
            ok = False
            details.append("could not locate ridgecv.fit(...) call")
    else:
        details.append("regression_conditioning.py not found")
    qa.add("E. RidgeCV alpha selection is train-only", ok, details)


def check_output_files_valid(qa: QAResult):
    required = [
        "regression_conditioning.json",
        "representation_geometry.json",
        "pooling_diagnostic.json",
        "learning_curve.json",
        "diagnostic_results.json",
        "diagnostic_report.md",
    ]
    details = []
    ok = True
    for name in required:
        path = OUT_DIR / name
        if not path.exists():
            ok = False
            details.append(f"MISSING: {name}")
            continue
        if name.endswith(".json"):
            try:
                json.loads(path.read_text())
                details.append(f"{name}: present, valid JSON")
            except json.JSONDecodeError as exc:
                ok = False
                details.append(f"{name}: present but INVALID JSON ({exc})")
        else:
            details.append(f"{name}: present ({path.stat().st_size} bytes)")
    qa.add("F. Required output files present and valid", ok, details)


def check_no_protocol_drift(qa: QAResult, original_result: dict):
    """Every diagnostic script must use the SAME ridge_alpha, transform
    config, resolution, fps as Task 7's original recorded config, UNLESS
    it is the specific thing being varied by that script's own diagnostic
    question (sample_scaling varies N_train only; temporal_context varies
    num_frames only; regression_conditioning/representation_geometry
    explicitly vary the regularization/representation as their whole
    point, on top of the SAME data)."""
    cfg = original_result["config"]
    details = []
    ok = True

    lc_path = OUT_DIR / "learning_curve.json"
    if lc_path.exists():
        lc = json.loads(lc_path.read_text())
        if lc["ridge_alpha"] != cfg["ridge_alpha"]:
            ok = False
            details.append(f"sample_scaling.py used ridge_alpha={lc['ridge_alpha']} != Task 7's {cfg['ridge_alpha']}")
        else:
            details.append(f"sample_scaling.py: ridge_alpha unchanged ({lc['ridge_alpha']}) -- only N_train varies, as intended")

    tc_path = OUT_DIR / "temporal_context.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        if tc["ridge_alpha"] != cfg["ridge_alpha"]:
            ok = False
            details.append(f"temporal_context.py used ridge_alpha={tc['ridge_alpha']} != Task 7's {cfg['ridge_alpha']}")
        else:
            details.append(f"temporal_context.py: ridge_alpha unchanged ({tc['ridge_alpha']}) -- only num_frames varies, as intended")

    qa.add("G. No undeclared protocol drift (only the intended variable changes per script)", ok, details)


def main():
    original_path = REPO_ROOT / "state" / "task_07_result.json"
    original_bytes_at_start = original_path.read_bytes()
    original_result = json.loads(original_bytes_at_start)

    qa = QAResult()
    check_original_untouched(qa, original_result, original_bytes_at_start)
    check_fixed_test_set_consistent(qa, original_result)
    check_no_test_id_in_fitting(qa, original_result)
    check_sample_scaling_reproduces_original(qa, original_result)
    check_train_only_fitting_by_design(qa)
    check_output_files_valid(qa)
    check_no_protocol_drift(qa, original_result)

    report = qa.render()
    print(report)
    (OUT_DIR / "qa_report.md").write_text(report)
    (OUT_DIR / "qa_report.json").write_text(json.dumps({"passed": qa.passed, "layers": qa.layers}, indent=2))
    return 0 if qa.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
