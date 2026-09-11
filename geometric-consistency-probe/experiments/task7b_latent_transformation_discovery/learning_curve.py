"""Task 7B Stage 1 learning curve (contract Sec. 5-7): for every
N_train in the frozen schedule and every seed, fit the ENTIRE M0-M6
hierarchy (plus controls) on the nested training subset, evaluate on the
FIXED validation split (never test), and record every candidate's
metrics -- including failures. Selection (highest mean validation R^2,
tie-break by parameter count then hierarchy order) happens once, after
every scheduled point has run, in `select_configuration` below.

Nothing in this module ever reads Z_test/Zp_test -- only Z_val/Zp_val
and the nested Z_train/Zp_train subsets. This is asserted structurally
in tests/test_task7b_learning_curve.py, not just true by convention.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.task7b_latent_transformation_discovery import data, models
from experiments.task7b_latent_transformation_discovery.metrics import UNDEFINED, bootstrap_ci, evaluate

OUT_DIR = Path(__file__).resolve().parent


def _load_representations(z_cache_path: Path) -> dict[str, np.ndarray]:
    npz = np.load(z_cache_path)
    Z, Zp = {}, {}
    for key in npz.files:
        if key.startswith("Z__"):
            Z[key[len("Z__"):]] = npz[key]
        elif key.startswith("Zp__"):
            Zp[key[len("Zp__"):]] = npz[key]
    return Z, Zp


def _stack(ids: list[str] | tuple[str, ...], Z: dict, Zp: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([Z[i] for i in ids]), np.stack([Zp[i] for i in ids])


def _candidate_grid():
    """Yields (model_key, fit_kwargs) for every hyperparameter combination
    in the frozen protocol's model_grid, in the fixed M0-M6 order."""
    yield ("M0_persistence", {})
    yield ("M1_scalar", {})
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        yield ("M2_diagonal", {"alpha": alpha})
    for r in models.LOW_RANK_RANKS:
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            yield (f"M3_low_rank_r{r}", {"r": r, "alpha": alpha})
    for alpha in (1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0):
        yield ("M4_full_linear", {"alpha": alpha})
    for alpha in (1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0):
        yield ("M5_affine", {"alpha": alpha})
    for h in models.MLP_HIDDEN_SIZES:
        yield (f"M6_residual_mlp_h{h}", {"h": h})


def _fit_candidate(model_key: str, kwargs: dict, Z_train, Zp_train, Z_val, Zp_val, seed: int) -> models.FittedModel:
    if model_key == "M0_persistence":
        return models.fit_m0_persistence(Z_train, Zp_train)
    if model_key == "M1_scalar":
        return models.fit_m1_scalar(Z_train, Zp_train)
    if model_key == "M2_diagonal":
        return models.fit_m2_diagonal(Z_train, Zp_train, alpha=kwargs["alpha"])
    if model_key.startswith("M3_low_rank"):
        return models.fit_m3_low_rank(Z_train, Zp_train, r=kwargs["r"], alpha=kwargs["alpha"])
    if model_key == "M4_full_linear":
        return models.fit_m4_full_linear(Z_train, Zp_train, alpha=kwargs["alpha"])
    if model_key == "M5_affine":
        return models.fit_m5_affine(Z_train, Zp_train, alpha=kwargs["alpha"])
    if model_key.startswith("M6_residual_mlp"):
        return models.fit_m6_residual_mlp(Z_train, Zp_train, Z_val, Zp_val, h=kwargs["h"], seed=seed)
    raise ValueError(f"unknown model_key {model_key}")


def run_point(n_train: int, seed: int, train_ids, val_ids, Z: dict, Zp: dict) -> dict:
    """Fits every candidate for one (N_train, seed) point, evaluated on
    validation only. Returns a JSON-safe record."""
    Z_train, Zp_train = _stack(train_ids, Z, Zp)
    Z_val, Zp_val = _stack(val_ids, Z, Zp)

    results = {}
    for model_key, kwargs in _candidate_grid():
        t0 = time.time()
        try:
            fitted = _fit_candidate(model_key, kwargs, Z_train, Zp_train, Z_val, Zp_val, seed)
            metric_bundle = evaluate(fitted.predict(Z_val), Zp_val).to_dict()
            status = "ok"
        except Exception as exc:  # noqa: BLE001 -- a failed candidate is a recorded result, never a crash
            metric_bundle = {"r2": UNDEFINED, "mse": UNDEFINED, "relative_l2_error": UNDEFINED, "cosine_similarity": UNDEFINED}
            status = f"FAILED: {exc!r}"
            fitted = None
        results[f"{model_key}::{json.dumps(kwargs, sort_keys=True)}"] = {
            "model_key": model_key,
            "hyperparams": kwargs,
            "n_params": fitted.n_params if fitted is not None else None,
            "status": status,
            "val_metrics": metric_bundle,
            "fit_seconds": time.time() - t0,
        }

    # controls -- same train/val split, fixed alpha (a representative
    # mid-grid value; controls are diagnostic, not part of the selection
    # competition)
    control_alpha = 10.0
    train_mean = models.fit_train_mean_control(Z_train, Zp_train)
    persistence = models.fit_m0_persistence(Z_train, Zp_train)
    shuffled = models.fit_shuffled_pair_control(Z_train, Zp_train, alpha=control_alpha, seed=seed)
    controls = {
        "control_train_mean": evaluate(train_mean.predict(Z_val), Zp_val).to_dict(),
        "control_persistence": evaluate(persistence.predict(Z_val), Zp_val).to_dict(),
        "control_shuffled_pair": evaluate(shuffled.predict(Z_val), Zp_val).to_dict(),
    }

    return {
        "n_train": n_train,
        "seed": seed,
        "n_train_scene_ids": list(train_ids),
        "n_val": len(val_ids),
        "candidates": results,
        "controls": controls,
    }


def select_best_candidate(point_record: dict) -> tuple[str, dict]:
    """Highest val R^2; ties within 0.01 -> lower n_params, then earlier
    hierarchy position (dict insertion order from _candidate_grid, which
    IS the hierarchy order)."""
    numeric = [
        (key, entry) for key, entry in point_record["candidates"].items()
        if entry["status"] == "ok" and isinstance(entry["val_metrics"]["r2"], (int, float))
    ]
    if not numeric:
        raise RuntimeError(f"No candidate produced a defined R^2 at N_train={point_record['n_train']}, seed={point_record['seed']}")
    best_r2 = max(entry["val_metrics"]["r2"] for _, entry in numeric)
    tied = [(key, entry) for key, entry in numeric if best_r2 - entry["val_metrics"]["r2"] <= 0.01]
    tied.sort(key=lambda ke: (ke[1]["n_params"], list(point_record["candidates"].keys()).index(ke[0])))
    return tied[0]


def aggregate_over_seeds(points: list[dict]) -> dict:
    """For a fixed N_train, aggregate the SELECTED-best-per-seed R^2
    values into a mean + bootstrap CI -- used by the stopping rule."""
    n_train = points[0]["n_train"]
    assert all(p["n_train"] == n_train for p in points)
    best_r2_per_seed = []
    for p in points:
        _, entry = select_best_candidate(p)
        best_r2_per_seed.append(entry["val_metrics"]["r2"])
    ci = bootstrap_ci(best_r2_per_seed, seed=0)
    return {"n_train": n_train, "per_seed_best_r2": best_r2_per_seed, **ci}


def check_stabilization(aggregated_by_n: list[dict]) -> bool:
    """contract Sec. 5's stopping rule: final two scheduled points differ
    by <0.02 absolute mean AND their 95% CIs overlap."""
    if len(aggregated_by_n) < 2:
        return False
    a, b = aggregated_by_n[-2], aggregated_by_n[-1]
    if a["mean"] == UNDEFINED or b["mean"] == UNDEFINED:
        return False
    diff_ok = abs(a["mean"] - b["mean"]) < 0.02
    overlap_ok = not (a["high"] < b["low"] or b["high"] < a["low"])
    return diff_ok and overlap_ok
