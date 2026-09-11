"""QA gate (contract Sec. 12): verify metric calculations against
hand-computable cases, and confirm the constant-target case returns the
literal string "undefined", never 0.0 or a silent nan."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.task7b_latent_transformation_discovery.metrics import (
    UNDEFINED,
    bootstrap_ci,
    cosine_similarity,
    evaluate,
    mse,
    multi_output_r2,
    relative_l2_error,
)
from metrics.common import r_squared as legacy_r_squared


def test_r2_perfect_prediction_is_exactly_one():
    target = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    assert multi_output_r2(target, target) == pytest.approx(1.0)


def test_r2_mean_prediction_is_exactly_zero():
    target = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    mean_pred = np.broadcast_to(target.mean(axis=0), target.shape)
    assert multi_output_r2(mean_pred, target) == pytest.approx(0.0, abs=1e-12)


def test_r2_hand_computable_case():
    # target = [[0],[2]], pred = [[1],[1]] -> mean=1, ss_tot=(0-1)^2+(2-1)^2=2
    # ss_res=(1-0)^2+(1-2)^2=2 -> R2 = 1 - 2/2 = 0
    target = np.array([[0.0], [2.0]])
    pred = np.array([[1.0], [1.0]])
    assert multi_output_r2(pred, target) == pytest.approx(0.0)


def test_r2_constant_target_is_the_string_undefined():
    target = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    pred = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = multi_output_r2(pred, target)
    assert result == UNDEFINED
    assert result != 0.0


def test_r2_matches_legacy_metrics_common_implementation():
    """Task 7B's R^2 must be the SAME formula as metrics/common.py's
    r_squared (SS_res/SS_tot against the eval set's own mean) -- cross
    checked directly rather than trusted to coincidentally agree."""
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(20, 16))
    target = rng.normal(size=(20, 16))
    new = multi_output_r2(pred, target)
    legacy = legacy_r_squared(pred, target)
    assert new == pytest.approx(legacy)


def test_mse_hand_case():
    pred = np.array([[1.0, 2.0]])
    target = np.array([[2.0, 4.0]])
    assert mse(pred, target) == pytest.approx((1.0 + 4.0) / 2)


def test_relative_l2_error_hand_case():
    pred = np.array([[3.0, 4.0]])  # norm 5
    target = np.array([[0.0, 0.0]])
    # target norm is 0 -> denominator clamped to 1e-12, huge but finite
    val = relative_l2_error(pred, target)
    assert val > 1e6


def test_cosine_similarity_hand_case():
    pred = np.array([[1.0, 0.0]])
    target = np.array([[0.0, 1.0]])
    assert cosine_similarity(pred, target) == pytest.approx(0.0, abs=1e-9)
    assert cosine_similarity(pred, pred) == pytest.approx(1.0)


def test_evaluate_bundle_reports_undefined_r2_without_crashing():
    target = np.ones((4, 3)) * 7.0
    pred = np.random.default_rng(1).normal(size=(4, 3))
    bundle = evaluate(pred, target).to_dict()
    assert bundle["r2"] == UNDEFINED
    assert isinstance(bundle["mse"], float)


def test_bootstrap_ci_contains_the_mean():
    values = [0.1, 0.15, 0.12, 0.09, 0.11, 0.13]
    result = bootstrap_ci(values, n_resamples=500, seed=0)
    assert result["low"] <= result["mean"] <= result["high"]
    assert result["n"] == len(values)


def test_bootstrap_ci_handles_all_undefined_gracefully():
    result = bootstrap_ci([UNDEFINED, UNDEFINED], n_resamples=100, seed=0)
    assert result["mean"] == UNDEFINED
    assert result["n"] == 0
