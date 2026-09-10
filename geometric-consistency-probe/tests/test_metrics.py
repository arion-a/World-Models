import numpy as np

from metrics.common import cosine_similarity, r_squared, relative_l2_error


def test_cosine_similarity_identical_vectors():
    a = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    sims = cosine_similarity(a, a)
    assert np.allclose(sims, 1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert np.isclose(cosine_similarity(a, b)[0], 0.0)


def test_relative_l2_error_zero_for_identical():
    a = np.random.default_rng(0).normal(size=(5, 4))
    err = relative_l2_error(a, a)
    assert np.allclose(err, 0.0)


def test_relative_l2_error_guards_zero_target():
    pred = np.array([[1.0, 0.0]])
    target = np.array([[0.0, 0.0]])
    # ||pred-target||=1, ||target||=0 -> guarded denominator, error should be large but finite.
    err = relative_l2_error(pred, target)
    assert np.isfinite(err[0])
    assert err[0] > 1e6


def test_r_squared_perfect_prediction():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(10, 3))
    assert np.isclose(r_squared(target, target), 1.0)


def test_r_squared_mean_prediction_is_zero():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(20, 2))
    mean_pred = np.tile(target.mean(axis=0, keepdims=True), (20, 1))
    assert np.isclose(r_squared(mean_pred, target), 0.0, atol=1e-8)


def test_r_squared_worse_than_mean_is_negative():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(20, 2))
    bad_pred = -10 * target
    assert r_squared(bad_pred, target) < 0


def test_r_squared_handles_1d_vs_column_vector_shape_mismatch():
    # Regression test: sklearn's Ridge.predict returns a flat (N,) array
    # for a single-output fit even when the target it was fit on was a
    # (N, 1) column vector. Naively using np.atleast_2d on a (N,) array
    # gives (1, N), not (N, 1), which silently broadcasts (N, 1) - (1, N)
    # into an (N, N) matrix and produces a nonsensical score. Both input
    # shapes for the same underlying data must give the same answer.
    target_col = np.array([[1.0], [2.0], [3.0], [4.0]])
    pred_flat = np.array([1.1, 1.9, 3.2, 3.8])
    pred_col = pred_flat.reshape(-1, 1)
    assert np.isclose(r_squared(pred_flat, target_col), r_squared(pred_col, target_col))
    assert r_squared(pred_flat, target_col) > 0.9
