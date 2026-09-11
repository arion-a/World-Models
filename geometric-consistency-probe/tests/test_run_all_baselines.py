"""Unit tests for Task 10's consolidated baseline framework
(baselines/run_all_baselines.py).

Synthetic Z arrays throughout -- no bpy, no torch, no network (Required
software tests: "Every baseline executes without error on real data" is
covered separately by the real Task 10 experiment
(experiments/task10_baselines.py) and its state/task_10_result.json, not
here; this file's job is the fast, always-run regression/shape/leakage
checks).
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from baselines.identity_baseline import IdentityBaselineResult, evaluate_identity_baseline
from baselines.mean_baseline import MeanBaselineResult, evaluate_mean_baseline
from baselines.run_all_baselines import (
    CORE_BASELINE_NAMES,
    FULL_BASELINE_NAMES,
    AlternateEncoderBaselineResult,
    run_all_baselines,
    run_baselines_core_three,
    run_probe_baselines,
)
from baselines.shuffled_pairing_baseline import ShuffledPairingBaselineResult, evaluate_shuffled_pairing_baseline


def _synthetic_arrays(seed: int = 0, n_train: int = 24, n_test: int = 8, d: int = 6):
    rng = np.random.default_rng(seed)
    Z_train = rng.normal(size=(n_train, d)).astype(np.float32)
    W_true = rng.normal(size=(d, d)).astype(np.float32)
    b_true = rng.normal(size=(d,)).astype(np.float32)
    noise = 0.05
    Zp_train = Z_train @ W_true + b_true + noise * rng.normal(size=(n_train, d)).astype(np.float32)
    Z_test = rng.normal(size=(n_test, d)).astype(np.float32)
    Zp_test = Z_test @ W_true + b_true + noise * rng.normal(size=(n_test, d)).astype(np.float32)
    return Z_train, Zp_train, Z_test, Zp_test


def _pixel_and_random_arrays(seed_offset: int):
    """A separate synthetic 'alternate encoder' representation set --
    deliberately a different feature dimensionality/distribution than
    the primary encoder's, since a real pixel-statistics/random-encoder
    representation is never the same array as the primary Z (Task 10's
    'no baseline may receive privileged ground-truth information' /
    fair-comparison requirement -- checked here by construction: these
    arrays come from an entirely separate synthetic source).
    """
    return _synthetic_arrays(seed=100 + seed_offset, n_train=24, n_test=8, d=3)


# --- run_baselines_core_three: parity with calling each function directly --


def test_core_three_matches_calling_each_baseline_function_directly():
    Z_train, Zp_train, Z_test, Zp_test = _synthetic_arrays()
    alpha, seed = 10.0, 0

    core = run_baselines_core_three("camera_rotation", Z_train, Zp_train, Z_test, Zp_test, alpha=alpha, seed=seed)

    expected_persistence = evaluate_identity_baseline("camera_rotation", Z_test, Zp_test)
    expected_mean = evaluate_mean_baseline("camera_rotation", Zp_train, Zp_test)
    expected_shuffled = evaluate_shuffled_pairing_baseline(
        "camera_rotation", Z_train, Zp_train, Z_test, Zp_test, alpha=alpha, seed=seed
    )

    assert core["persistence"] == expected_persistence
    assert core["mean"] == expected_mean
    assert core["shuffled_pairing"] == expected_shuffled


def test_core_three_returns_the_original_dataclass_types():
    Z_train, Zp_train, Z_test, Zp_test = _synthetic_arrays()
    core = run_baselines_core_three("camera_rotation", Z_train, Zp_train, Z_test, Zp_test)
    assert isinstance(core["persistence"], IdentityBaselineResult)
    assert isinstance(core["mean"], MeanBaselineResult)
    assert isinstance(core["shuffled_pairing"], ShuffledPairingBaselineResult)


# --- run_all_baselines: full five, uniform shape, no silent skip -----------


def test_run_all_baselines_returns_exactly_the_five_required_baselines():
    Z_train, Zp_train, Z_test, Zp_test = _synthetic_arrays()
    pZ_train, pZp_train, pZ_test, pZp_test = _pixel_and_random_arrays(0)
    rZ_train, rZp_train, rZ_test, rZp_test = _pixel_and_random_arrays(1)

    result = run_all_baselines(
        Z_train, Zp_train, Z_test, Zp_test, "camera_rotation", alpha=10.0, seed=0,
        pixel_Z_train=pZ_train, pixel_Zp_train=pZp_train, pixel_Z_test=pZ_test, pixel_Zp_test=pZp_test,
        random_Z_train=rZ_train, random_Zp_train=rZp_train, random_Z_test=rZ_test, random_Zp_test=rZp_test,
    )

    assert set(result.keys()) == set(FULL_BASELINE_NAMES)
    assert set(CORE_BASELINE_NAMES) <= set(FULL_BASELINE_NAMES)
    for name, res in result.items():
        fields = {f.name for f in dataclasses.fields(res)}
        assert fields == {"transform_name", "r2", "mean_cosine_similarity", "mean_relative_l2_error"}, name
        assert res.transform_name == "camera_rotation"
        assert np.isfinite(res.r2)
        assert np.isfinite(res.mean_cosine_similarity)
        assert np.isfinite(res.mean_relative_l2_error)


def test_run_all_baselines_core_three_are_bit_identical_to_calling_them_directly():
    """Consolidating must not change any baseline's math (Task 10's
    'Failure conditions': 'a baseline's math silently changed while
    consolidating it')."""
    Z_train, Zp_train, Z_test, Zp_test = _synthetic_arrays()
    pZ_train, pZp_train, pZ_test, pZp_test = _pixel_and_random_arrays(0)
    rZ_train, rZp_train, rZ_test, rZp_test = _pixel_and_random_arrays(1)

    result = run_all_baselines(
        Z_train, Zp_train, Z_test, Zp_test, "camera_rotation", alpha=10.0, seed=0,
        pixel_Z_train=pZ_train, pixel_Zp_train=pZp_train, pixel_Z_test=pZ_test, pixel_Zp_test=pZp_test,
        random_Z_train=rZ_train, random_Zp_train=rZp_train, random_Z_test=rZ_test, random_Zp_test=rZp_test,
    )

    assert result["persistence"] == evaluate_identity_baseline("camera_rotation", Z_test, Zp_test)
    assert result["mean"] == evaluate_mean_baseline("camera_rotation", Zp_train, Zp_test)
    assert result["shuffled_pairing"] == evaluate_shuffled_pairing_baseline(
        "camera_rotation", Z_train, Zp_train, Z_test, Zp_test, alpha=10.0, seed=0
    )


def test_run_all_baselines_pixel_and_random_use_the_identical_fit_evaluate_protocol():
    """DESIGN.md Sec 12 items 3-4: the pixel-statistics/random-encoder
    baselines must re-run the SAME fit-rho(T)-and-evaluate procedure the
    primary 'learned W_T' uses (metrics.equivariance.evaluate_equivariance),
    not some other scoring rule -- checked here by reproducing the
    pixel_statistics result from evaluate_equivariance directly on the
    same alternate-encoder arrays.
    """
    from metrics.equivariance import evaluate_equivariance

    Z_train, Zp_train, Z_test, Zp_test = _synthetic_arrays()
    pZ_train, pZp_train, pZ_test, pZp_test = _pixel_and_random_arrays(0)
    rZ_train, rZp_train, rZ_test, rZp_test = _pixel_and_random_arrays(1)

    result = run_all_baselines(
        Z_train, Zp_train, Z_test, Zp_test, "camera_rotation", alpha=10.0, seed=0,
        pixel_Z_train=pZ_train, pixel_Zp_train=pZp_train, pixel_Z_test=pZ_test, pixel_Zp_test=pZp_test,
        random_Z_train=rZ_train, random_Zp_train=rZp_train, random_Z_test=rZ_test, random_Zp_test=rZp_test,
    )

    expected_pixel, _ = evaluate_equivariance("camera_rotation", pZ_train, pZp_train, pZ_test, pZp_test, alpha=10.0)
    expected_random, _ = evaluate_equivariance("camera_rotation", rZ_train, rZp_train, rZ_test, rZp_test, alpha=10.0)
    assert np.isclose(result["pixel_statistics"].r2, expected_pixel.r2)
    assert np.isclose(result["random_encoder"].r2, expected_random.r2)
    assert isinstance(result["pixel_statistics"], AlternateEncoderBaselineResult)
    assert isinstance(result["random_encoder"], AlternateEncoderBaselineResult)


def test_run_all_baselines_requires_all_eight_pixel_and_random_arrays():
    """No silent skip: omitting any of the pixel_*/random_* arrays is a
    loud TypeError, not a quietly-incomplete result (Task 10's 'Failure
    conditions')."""
    Z_train, Zp_train, Z_test, Zp_test = _synthetic_arrays()
    pZ_train, pZp_train, pZ_test, pZp_test = _pixel_and_random_arrays(0)

    with pytest.raises(TypeError):
        run_all_baselines(  # missing every random_* argument
            Z_train, Zp_train, Z_test, Zp_test, "camera_rotation",
            pixel_Z_train=pZ_train, pixel_Zp_train=pZp_train, pixel_Z_test=pZ_test, pixel_Zp_test=pZp_test,
        )


def test_run_all_baselines_is_reproducible_given_the_same_seed():
    Z_train, Zp_train, Z_test, Zp_test = _synthetic_arrays()
    pZ_train, pZp_train, pZ_test, pZp_test = _pixel_and_random_arrays(0)
    rZ_train, rZp_train, rZ_test, rZp_test = _pixel_and_random_arrays(1)
    kwargs = dict(
        pixel_Z_train=pZ_train, pixel_Zp_train=pZp_train, pixel_Z_test=pZ_test, pixel_Zp_test=pZp_test,
        random_Z_train=rZ_train, random_Zp_train=rZp_train, random_Z_test=rZ_test, random_Zp_test=rZp_test,
    )
    r1 = run_all_baselines(Z_train, Zp_train, Z_test, Zp_test, "camera_rotation", alpha=10.0, seed=3, **kwargs)
    r2 = run_all_baselines(Z_train, Zp_train, Z_test, Zp_test, "camera_rotation", alpha=10.0, seed=3, **kwargs)
    for name in FULL_BASELINE_NAMES:
        assert r1[name] == r2[name]


# --- run_probe_baselines (Task 8-style Z -> y probes) -----------------------


def test_run_probe_baselines_mean_prediction_ignores_z():
    rng = np.random.default_rng(0)
    Z_train = rng.normal(size=(10, 4)).astype(np.float32)
    y_train = rng.normal(size=(10, 2)).astype(np.float32)
    Z_test = rng.normal(size=(5, 4)).astype(np.float32)
    y_test = rng.normal(size=(5, 2)).astype(np.float32)

    result = run_probe_baselines(
        Z_train, y_train, Z_test, y_test,
        predict_fn=lambda zt, yt, zte: np.broadcast_to(yt.mean(axis=0), (zte.shape[0], yt.shape[1])),
        permutation=np.arange(10),
    )
    expected_mean = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
    assert np.allclose(result["mean"], expected_mean)


def test_run_probe_baselines_shuffled_label_uses_the_given_permutation():
    rng = np.random.default_rng(0)
    Z_train = rng.normal(size=(10, 4)).astype(np.float32)
    y_train = np.arange(10, dtype=np.float32).reshape(10, 1)
    Z_test = rng.normal(size=(5, 4)).astype(np.float32)
    y_test = rng.normal(size=(5, 1)).astype(np.float32)
    perm = np.roll(np.arange(10), 1)

    seen = {}

    def predict_fn(zt, yt, zte):
        seen["yt"] = yt
        return np.zeros((zte.shape[0], yt.shape[1]), dtype=np.float32)

    run_probe_baselines(Z_train, y_train, Z_test, y_test, predict_fn=predict_fn, permutation=perm)
    assert np.array_equal(seen["yt"], y_train[perm])


def test_run_probe_baselines_signature_has_no_defaults_for_the_permutation():
    """The permutation must be supplied by the caller (kept independently
    testable/inspectable, e.g. task8's own shuffled_label_permutation
    guard against a degenerate identity permutation) rather than
    generated silently inside this function."""
    params = inspect.signature(run_probe_baselines).parameters
    assert params["permutation"].default is inspect.Parameter.empty
