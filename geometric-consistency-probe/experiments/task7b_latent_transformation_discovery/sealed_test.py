"""Task 7B sealed test (contract Sec. 7): evaluates ONLY the single
already-selected (model, hyperparameter) configuration plus the
pre-specified controls, exactly once, on the fixed test split.

Contamination guard: refuses to run if a sealed_test_result.json already
exists (a rerun is permitted only for a documented software failure
discovered BEFORE result inspection -- that is an explicit,
human-reviewed override, never this function's default path).

Validation set for the final refit: the selected (model, hyperparameter)
configuration is refit on `train_ids` before being evaluated on
`test_ids`. If that configuration is M6 (residual MLP), refitting needs
a genuine held-out slice for early stopping -- this MUST be the
protocol's own fixed `val_ids` (already used during Stage 1 model
selection, so reusing it here adds no new information the model/
hyperparameter choice wasn't already exposed to), and MUST NOT be
`test_ids`. Passing Z_test/Zp_test as the val arguments here would mean
the sealed test's own early stopping is tuned against the test set --
exactly the leakage this whole protocol exists to prevent -- so
`val_ids` is a required argument, not optional.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.task7b_latent_transformation_discovery import learning_curve, models
from experiments.task7b_latent_transformation_discovery.metrics import bootstrap_ci, evaluate

OUT_DIR = Path(__file__).resolve().parent
SEALED_TEST_PATH = OUT_DIR / "sealed_test_result.json"


def run_sealed_test(
    selected_model_key: str,
    selected_hyperparams: dict,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    Z: dict,
    Zp: dict,
    seed: int,
    force: bool = False,
) -> dict:
    if SEALED_TEST_PATH.exists() and not force:
        raise RuntimeError(
            f"{SEALED_TEST_PATH} already exists -- the sealed test may run only once. "
            "A rerun requires `force=True`, used only for a documented software failure "
            "discovered BEFORE the existing result was inspected (contract Sec. 7)."
        )

    Z_train, Zp_train = learning_curve._stack(train_ids, Z, Zp)
    Z_val, Zp_val = learning_curve._stack(val_ids, Z, Zp)
    Z_test, Zp_test = learning_curve._stack(test_ids, Z, Zp)

    t0 = time.time()
    # val_ids (never test_ids) supplies early stopping for M6 -- M0-M5
    # ignore the val arguments entirely, per learning_curve._fit_candidate.
    selected = learning_curve._fit_candidate(selected_model_key, selected_hyperparams, Z_train, Zp_train, Z_val, Zp_val, seed)
    selected_metrics = evaluate(selected.predict(Z_test), Zp_test).to_dict()

    control_alpha = selected_hyperparams.get("alpha", 10.0)
    train_mean = models.fit_train_mean_control(Z_train, Zp_train)
    persistence = models.fit_m0_persistence(Z_train, Zp_train)
    shuffled = models.fit_shuffled_pair_control(Z_train, Zp_train, alpha=control_alpha, seed=seed)

    control_metrics = {
        "control_train_mean": evaluate(train_mean.predict(Z_test), Zp_test).to_dict(),
        "control_persistence": evaluate(persistence.predict(Z_test), Zp_test).to_dict(),
        "control_shuffled_pair": evaluate(shuffled.predict(Z_test), Zp_test).to_dict(),
    }

    # per-scene metrics + bootstrap CI over scenes (contract's explicit
    # "report per-scene metrics" + "95% bootstrap CIs over scenes")
    per_scene_r2 = []
    pred_test = selected.predict(Z_test)
    for i, scene_id in enumerate(test_ids):
        r2_i = evaluate(pred_test[i : i + 1], Zp_test[i : i + 1]).to_dict()["r2"]
        per_scene_r2.append(r2_i)
    scene_ci = bootstrap_ci(per_scene_r2, seed=0)

    result = {
        "task": "7B",
        "sealed": True,
        "selected_model_key": selected_model_key,
        "selected_hyperparams": selected_hyperparams,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "seed": seed,
        "selected_config_test_metrics": selected_metrics,
        "control_test_metrics": control_metrics,
        "difference_vs_persistence_r2": (
            selected_metrics["r2"] - control_metrics["control_persistence"]["r2"]
            if isinstance(selected_metrics["r2"], (int, float)) and isinstance(control_metrics["control_persistence"]["r2"], (int, float))
            else "undefined"
        ),
        "difference_vs_best_control_r2": (
            selected_metrics["r2"] - max(
                v["r2"] for v in control_metrics.values() if isinstance(v["r2"], (int, float))
            )
            if isinstance(selected_metrics["r2"], (int, float))
            else "undefined"
        ),
        "per_scene_r2": per_scene_r2,
        "bootstrap_ci_over_scenes": scene_ci,
        "test_scene_ids": list(test_ids),
        "fit_seconds": time.time() - t0,
        "contamination_status": "CLEAN -- test scenes were not accessed by any fitting, normalization, early-stopping, or selection step before this call (early stopping, where applicable, used val_ids only)",
    }

    SEALED_TEST_PATH.write_text(json.dumps(result, indent=2))
    return result
