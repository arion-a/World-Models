"""Diagnostic B: regression conditioning.

Compares the original fixed-alpha=10.0 ridge against:
  (1) OLS (alpha=0) -- expected to be pathological at D=1024 > N_train=32.
  (2) Ridge with alpha selected by TRAIN-ONLY k-fold cross-validation
      (sklearn RidgeCV, cv folds built only from the 32 train scenes --
      the held-out test set is never touched during alpha selection).

Uses the REAL representations independently re-derived by the Task 7
forensic audit (experiments/task7_forensic_audit/recomputed_representations.npz)
-- no new rendering/encoding needed, and no contact with
state/task_07_result.json or experiments/geometric_consistency/.

For every one of the six transforms (not just camera_rotation), so this
diagnostic's conclusion generalizes across the whole original comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = REPO_ROOT / "experiments" / "task7_forensic_audit"


def r_squared(pred, target):
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def cosine_sim(pred, target):
    num = np.sum(pred * target, axis=-1)
    denom = np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1)
    denom = np.where(denom == 0, 1e-12, denom)
    return float(np.mean(num / denom))


def main():
    recorded = json.loads((REPO_ROOT / "state" / "task_07_result.json").read_text())
    transforms = list(recorded["transforms_evaluated"])
    train_ids = recorded["dataset"]["train_scene_ids"]
    test_ids = recorded["dataset"]["test_scene_ids"]

    npz = np.load(AUDIT_DIR / "recomputed_representations.npz")
    Z_train = np.stack([npz[f"Z__{sid}"] for sid in train_ids])
    Z_test = np.stack([npz[f"Z__{sid}"] for sid in test_ids])

    alpha_grid = np.array([0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])

    results = {}
    for t in transforms:
        Zp_train = np.stack([npz[f"Zp__{t}__{sid}"] for sid in train_ids])
        Zp_test = np.stack([npz[f"Zp__{t}__{sid}"] for sid in test_ids])

        entry = {}

        # (0) original fixed alpha=10.0 -- sanity re-check, should match recorded exactly
        from sklearn.linear_model import Ridge
        ridge10 = Ridge(alpha=10.0, fit_intercept=True).fit(Z_train, Zp_train)
        pred10 = ridge10.predict(Z_test)
        entry["original_fixed_alpha_10"] = {"r2": r_squared(pred10, Zp_test), "cosine": cosine_sim(pred10, Zp_test)}

        # (1) OLS
        ols = LinearRegression(fit_intercept=True).fit(Z_train, Zp_train)
        pred_ols = ols.predict(Z_test)
        entry["ols_alpha_0"] = {"r2": r_squared(pred_ols, Zp_test), "cosine": cosine_sim(pred_ols, Zp_test)}

        # (2) RidgeCV -- alpha selected via train-only 4-fold CV (N_train=32 -> 8 per fold),
        # never touches Z_test/Zp_test during selection.
        n_splits = 4
        ridgecv = RidgeCV(alphas=alpha_grid, fit_intercept=True, cv=n_splits)
        # RidgeCV's default scorer is per-output R^2 averaged -- fine here since it never
        # sees the test set; store the selected alpha explicitly for the record.
        ridgecv.fit(Z_train, Zp_train)
        selected_alpha = float(np.atleast_1d(ridgecv.alpha_).mean()) if hasattr(ridgecv, "alpha_") else None
        pred_cv = ridgecv.predict(Z_test)
        entry["ridge_cv_train_only"] = {
            "selected_alpha": selected_alpha,
            "cv_folds": n_splits,
            "alpha_grid": alpha_grid.tolist(),
            "r2": r_squared(pred_cv, Zp_test),
            "cosine": cosine_sim(pred_cv, Zp_test),
        }

        results[t] = entry
        print(f"{t}: fixed_a10 r2={entry['original_fixed_alpha_10']['r2']:.4f}  "
              f"OLS r2={entry['ols_alpha_0']['r2']:.4f}  "
              f"CV(a={selected_alpha:g}) r2={entry['ridge_cv_train_only']['r2']:.4f}")

    (OUT_DIR / "regression_conditioning.json").write_text(json.dumps(results, indent=2))

    # cross-check original_fixed_alpha_10 against the recorded/reproduced numbers
    max_diff = max(
        abs(results[t]["original_fixed_alpha_10"]["r2"] - recorded["results"][t]["learned_W_T"]["r2"])
        for t in transforms
    )
    print(f"\nsanity cross-check vs state/task_07_result.json: max abs diff = {max_diff:.2e}")


if __name__ == "__main__":
    main()
