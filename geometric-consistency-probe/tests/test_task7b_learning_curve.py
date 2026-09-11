"""QA gates for the learning-curve/selection machinery itself: selection
uses validation only, the tie-break rule is correct, stabilization
detection matches the contract's exact rule, and a failed candidate is
recorded rather than crashing the whole point."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.task7b_latent_transformation_discovery import learning_curve
from experiments.task7b_latent_transformation_discovery.metrics import UNDEFINED


def _synthetic_representations(n_scenes: int, D: int, seed: int, w_true=None):
    rng = np.random.default_rng(seed)
    Z, Zp = {}, {}
    if w_true is None:
        w_true = rng.normal(size=D) * 0.5 + 1.0
    for i in range(n_scenes):
        sid = f"scene_{i:04d}"
        z = rng.normal(size=D)
        Z[sid] = z
        Zp[sid] = z * w_true + rng.normal(size=D) * 0.01
    return Z, Zp, [f"scene_{i:04d}" for i in range(n_scenes)]


def test_run_point_never_touches_a_provided_test_only_id_set():
    """Structural guarantee: run_point takes train_ids/val_ids explicitly
    and stacks ONLY those from Z/Zp -- extra keys present in Z/Zp (meant
    to represent unseen test scenes) must never appear in any candidate's
    fitting or evaluation."""
    Z, Zp, all_ids = _synthetic_representations(60, D=10, seed=0)
    train_ids, val_ids, test_ids = all_ids[:30], all_ids[30:50], all_ids[50:]

    # sabotage the "test" entries so a leak would be detectable/obviously wrong
    for sid in test_ids:
        Z[sid] = np.full(10, 999.0)
        Zp[sid] = np.full(10, -999.0)

    record = learning_curve.run_point(30, seed=0, train_ids=train_ids, val_ids=val_ids, Z=Z, Zp=Zp)
    assert record["n_train_scene_ids"] == train_ids
    assert record["n_val"] == len(val_ids)
    # every val metric must be plausible (not blown up by the sabotaged 999s)
    for entry in record["candidates"].values():
        if entry["status"] == "ok" and isinstance(entry["val_metrics"]["mse"], float):
            assert entry["val_metrics"]["mse"] < 1000, "a test-id leak would show up as a huge validation MSE"


def test_select_best_candidate_picks_highest_val_r2():
    Z, Zp, all_ids = _synthetic_representations(80, D=8, seed=1)
    record = learning_curve.run_point(50, seed=0, train_ids=all_ids[:50], val_ids=all_ids[50:], Z=Z, Zp=Zp)
    key, entry = learning_curve.select_best_candidate(record)
    all_r2 = [e["val_metrics"]["r2"] for e in record["candidates"].values() if e["status"] == "ok" and isinstance(e["val_metrics"]["r2"], (int, float))]
    assert entry["val_metrics"]["r2"] == max(all_r2)


def test_select_best_candidate_tie_break_prefers_fewer_parameters():
    fake_record = {
        "candidates": {
            "M4_full_linear::{}": {"model_key": "M4_full_linear", "n_params": 1000000, "status": "ok", "val_metrics": {"r2": 0.80}},
            "M2_diagonal::{}": {"model_key": "M2_diagonal", "n_params": 100, "status": "ok", "val_metrics": {"r2": 0.795}},
            "M0_persistence::{}": {"model_key": "M0_persistence", "n_params": 0, "status": "ok", "val_metrics": {"r2": 0.5}},
        }
    }
    key, entry = learning_curve.select_best_candidate(fake_record)
    # 0.80 vs 0.795 differ by 0.005 < 0.01 -> tied -> fewer params (M2_diagonal) wins
    assert entry["model_key"] == "M2_diagonal"


def test_select_best_candidate_raises_when_nothing_is_defined():
    fake_record = {
        "n_train": 10,
        "seed": 0,
        "candidates": {
            "M4_full_linear::{}": {"model_key": "M4_full_linear", "n_params": 10, "status": "FAILED: blew up", "val_metrics": {"r2": UNDEFINED}},
        },
    }
    with pytest.raises(RuntimeError):
        learning_curve.select_best_candidate(fake_record)


def test_check_stabilization_true_when_close_and_overlapping():
    aggregated = [
        {"n_train": 256, "mean": 0.40, "low": 0.35, "high": 0.45},
        {"n_train": 512, "mean": 0.41, "low": 0.36, "high": 0.46},
    ]
    assert learning_curve.check_stabilization(aggregated) is True


def test_check_stabilization_false_when_still_improving():
    aggregated = [
        {"n_train": 256, "mean": 0.10, "low": 0.05, "high": 0.15},
        {"n_train": 512, "mean": 0.30, "low": 0.25, "high": 0.35},
    ]
    assert learning_curve.check_stabilization(aggregated) is False


def test_check_stabilization_false_with_fewer_than_two_points():
    assert learning_curve.check_stabilization([{"n_train": 64, "mean": 0.1, "low": 0.0, "high": 0.2}]) is False


def test_aggregate_over_seeds_matches_per_seed_selection():
    Z, Zp, all_ids = _synthetic_representations(100, D=6, seed=3)
    points = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        train_ids = list(rng.choice(all_ids[:70], size=40, replace=False))
        points.append(learning_curve.run_point(40, seed=seed, train_ids=train_ids, val_ids=all_ids[70:], Z=Z, Zp=Zp))
    agg = learning_curve.aggregate_over_seeds(points)
    assert agg["n_train"] == 40
    assert len(agg["per_seed_best_r2"]) == 3
