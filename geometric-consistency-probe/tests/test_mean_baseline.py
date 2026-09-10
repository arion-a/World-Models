"""Unit tests for baselines/mean_baseline.py (Task 6's new required control)."""

from __future__ import annotations

import inspect

import numpy as np

from baselines.mean_baseline import evaluate_mean_baseline
from metrics.common import cosine_similarity, r_squared, relative_l2_error


def test_mean_baseline_matches_manual_r2_computation():
    Z_prime_train = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)  # mean = [2, 0]
    Z_prime_test = np.array([[2.0, 1.0], [2.0, -1.0], [4.0, 0.0]], dtype=np.float32)
    result = evaluate_mean_baseline("camera_rotation", Z_prime_train, Z_prime_test)

    mean_vec = Z_prime_train.mean(axis=0)
    pred = np.broadcast_to(mean_vec, Z_prime_test.shape)
    ss_res = np.sum((Z_prime_test - pred) ** 2)
    ss_tot = np.sum((Z_prime_test - Z_prime_test.mean(axis=0)) ** 2)
    expected_r2 = 1.0 - ss_res / ss_tot
    assert np.isclose(result.r2, expected_r2)
    assert np.isclose(result.mean_cosine_similarity, float(np.mean(cosine_similarity(pred, Z_prime_test))))
    assert np.isclose(result.mean_relative_l2_error, float(np.mean(relative_l2_error(pred, Z_prime_test))))


def test_mean_baseline_result_carries_the_transform_name():
    Z_prime_train = np.ones((5, 4), dtype=np.float32)
    Z_prime_test = np.ones((3, 4), dtype=np.float32)
    result = evaluate_mean_baseline("camera_rotation", Z_prime_train, Z_prime_test)
    assert result.transform_name == "camera_rotation"


def test_mean_baseline_prediction_ignores_the_untransformed_representation():
    """The whole point of this control is that the prediction depends
    only on Z'_train, never on Z itself -- checked structurally by the
    function's signature never accepting a Z/Z_train argument."""
    params = list(inspect.signature(evaluate_mean_baseline).parameters)
    assert params == ["transform_name", "Z_prime_train", "Z_prime_test"]


def test_mean_baseline_is_finite_on_random_inputs():
    rng = np.random.default_rng(0)
    Z_prime_train = rng.normal(size=(20, 8)).astype(np.float32)
    Z_prime_test = rng.normal(loc=5.0, size=(10, 8)).astype(np.float32)
    result = evaluate_mean_baseline("camera_rotation", Z_prime_train, Z_prime_test)
    assert np.isfinite(result.r2)
    assert np.isfinite(result.mean_cosine_similarity)
    assert np.isfinite(result.mean_relative_l2_error)


def test_mean_baseline_every_test_row_gets_the_identical_prediction():
    Z_prime_train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean_vec = Z_prime_train.mean(axis=0)
    Z_prime_test = np.array([[0.0, 0.0], [10.0, 10.0], [-5.0, 5.0]], dtype=np.float32)

    result = evaluate_mean_baseline("camera_rotation", Z_prime_train, Z_prime_test)
    # Cross-check against a hand-built constant prediction matrix.
    pred = np.tile(mean_vec, (Z_prime_test.shape[0], 1))
    expected = r_squared(pred, Z_prime_test)
    assert np.isclose(result.r2, expected)
