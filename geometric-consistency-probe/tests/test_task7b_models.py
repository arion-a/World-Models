"""QA gate (contract Sec. 12): unit-test every model's dimensions,
parameter count, deterministic-seed behavior, no-NaN/no-Inf, and
synthetic known-mapping recovery at a well-determined toy shape.

Parameter-count discrepancy note (see models.py's module docstring):
the contract's own worked example for M6 ("2Dh+2D+h" formula vs. its
listed example counts 16,408/32,808/65,616 at D=1024) is internally
inconsistent -- neither the stated formula nor a straightforward
alternative reading of the M6 equation (Zhat'=Z+B tanh(AZ+a)+b, whose
correct parameter count is 2Dh+D+h: A is h*D, a is h, B is D*h, b is D)
reproduces those exact three numbers together. test_m6_parameter_count_*
below asserts against 2Dh+D+h -- the count for the equation actually
implemented -- and documents the discrepancy rather than silently
matching an unverified example.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.task7b_latent_transformation_discovery.metrics import multi_output_r2
from experiments.task7b_latent_transformation_discovery.models import (
    LOW_RANK_RANKS,
    fit_m0_persistence,
    fit_m1_scalar,
    fit_m2_diagonal,
    fit_m3_low_rank,
    fit_m4_full_linear,
    fit_m5_affine,
    fit_m6_residual_mlp,
    fit_shuffled_pair_control,
    fit_train_mean_control,
)

WELL_DETERMINED_N = 400  # >> D so every model, including full linear (D^2 params), is well-posed
WELL_DETERMINED_D = 12


def _toy_data(seed=0, n=WELL_DETERMINED_N, d=WELL_DETERMINED_D, noise_std=0.0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    return Z, rng


# --- M0 persistence -----------------------------------------------------------


def test_m0_persistence_zero_params_and_identity_prediction():
    Z, rng = _toy_data()
    model = fit_m0_persistence(Z, Z + rng.normal(size=Z.shape))
    assert model.n_params == 0
    assert np.array_equal(model.predict(Z), Z)


# --- M1 scalar ------------------------------------------------------------------


def test_m1_scalar_recovers_known_scale():
    Z, rng = _toy_data(seed=1)
    a_true = 2.5
    Zp = a_true * Z
    model = fit_m1_scalar(Z, Zp)
    assert model.n_params == 1
    assert model.hyperparams["a"] == pytest.approx(a_true, abs=1e-6)
    assert multi_output_r2(model.predict(Z), Zp) == pytest.approx(1.0, abs=1e-8)


# --- M2 diagonal ----------------------------------------------------------------


def test_m2_diagonal_recovers_known_weights():
    Z, rng = _toy_data(seed=2)
    w_true = rng.uniform(0.5, 3.0, size=WELL_DETERMINED_D)
    Zp = Z * w_true
    model = fit_m2_diagonal(Z, Zp, alpha=0.0)
    assert model.n_params == WELL_DETERMINED_D
    assert multi_output_r2(model.predict(Z), Zp) == pytest.approx(1.0, abs=1e-8)


# --- M3 low-rank residual -------------------------------------------------------


@pytest.mark.parametrize("r", LOW_RANK_RANKS[:3])  # 2,4,8 -- keep the toy test fast
def test_m3_low_rank_recovers_known_low_rank_operator(r):
    Z, rng = _toy_data(seed=3, d=32, n=800)
    D = Z.shape[1]
    U_true = rng.normal(size=(D, r)) * 0.1
    V_true = rng.normal(size=(r, D)) * 0.1
    L_true = U_true @ V_true
    Zp = Z + Z @ L_true.T
    model = fit_m3_low_rank(Z, Zp, r=r, alpha=1e-6)
    assert model.n_params == 2 * D * r
    r2 = multi_output_r2(model.predict(Z), Zp)
    assert r2 == pytest.approx(1.0, abs=1e-4)


def test_m3_rejects_rank_not_less_than_d():
    Z, rng = _toy_data(seed=3, d=8)
    with pytest.raises(ValueError):
        fit_m3_low_rank(Z, Z, r=8)


# --- M4 full linear ---------------------------------------------------------------


def test_m4_full_linear_recovers_known_matrix():
    Z, rng = _toy_data(seed=4)
    D = Z.shape[1]
    W_true = rng.normal(size=(D, D)) * 0.2
    Zp = Z @ W_true.T
    model = fit_m4_full_linear(Z, Zp, alpha=1e-6)
    assert model.n_params == D * D
    assert multi_output_r2(model.predict(Z), Zp) == pytest.approx(1.0, abs=1e-4)


# --- M5 affine ---------------------------------------------------------------


def test_m5_affine_recovers_known_matrix_and_bias():
    Z, rng = _toy_data(seed=5)
    D = Z.shape[1]
    W_true = rng.normal(size=(D, D)) * 0.2
    b_true = rng.normal(size=D) * 0.5
    Zp = Z @ W_true.T + b_true
    model = fit_m5_affine(Z, Zp, alpha=1e-6)
    assert model.n_params == D * D + D
    assert multi_output_r2(model.predict(Z), Zp) == pytest.approx(1.0, abs=1e-4)


# --- M6 residual MLP --------------------------------------------------------------


def test_m6_parameter_count_matches_the_implemented_equation():
    D, h = WELL_DETERMINED_D, 8
    Z, rng = _toy_data(seed=6, n=200)
    Zp = Z + rng.normal(size=Z.shape) * 0.01
    Z_val, Zp_val = Z[:50], Zp[:50]
    model = fit_m6_residual_mlp(Z, Zp, Z_val, Zp_val, h=h, seed=0, max_epochs=50, patience=10)
    assert model.n_params == 2 * D * h + D + h


def test_m6_recovers_a_small_known_nonlinear_residual():
    rng = np.random.default_rng(7)
    n, d, h_true = 600, 10, 4
    Z = rng.normal(size=(n, d)) * 0.5
    A_true = rng.normal(size=(h_true, d)) * 0.3
    B_true = rng.normal(size=(d, h_true)) * 0.3
    Zp = Z + np.tanh(Z @ A_true.T) @ B_true.T

    split = 500
    Z_train, Zp_train = Z[:split], Zp[:split]
    Z_val, Zp_val = Z[split:], Zp[split:]

    model = fit_m6_residual_mlp(Z_train, Zp_train, Z_val, Zp_val, h=8, seed=0, max_epochs=4000, patience=200, lr=5e-3)
    r2 = multi_output_r2(model.predict(Z_val), Zp_val)
    assert r2 > 0.85, f"expected the MLP to recover a known small nonlinear residual, got r2={r2}"


def test_m6_is_deterministic_given_the_same_seed():
    Z, rng = _toy_data(seed=8, n=200)
    Zp = Z + rng.normal(size=Z.shape) * 0.05
    Z_val, Zp_val = Z[:40], Zp[:40]
    m_a = fit_m6_residual_mlp(Z, Zp, Z_val, Zp_val, h=8, seed=42, max_epochs=100, patience=20)
    m_b = fit_m6_residual_mlp(Z, Zp, Z_val, Zp_val, h=8, seed=42, max_epochs=100, patience=20)
    assert np.allclose(m_a.predict(Z_val), m_b.predict(Z_val))


# --- controls ---------------------------------------------------------------------


def test_train_mean_control_uses_only_train_data():
    Z_train, rng = _toy_data(seed=9)
    Zp_train = Z_train + 3.0
    model = fit_train_mean_control(Z_train, Zp_train)
    pred = model.predict(rng.normal(size=(5, WELL_DETERMINED_D)))
    assert np.allclose(pred, Zp_train.mean(axis=0))


def test_shuffled_pair_control_destroys_correspondence_but_uses_same_fitting_path():
    Z, rng = _toy_data(seed=10)
    D = Z.shape[1]
    W_true = rng.normal(size=(D, D)) * 0.2
    Zp = Z @ W_true.T  # a real, learnable correspondence

    real_model = fit_m5_affine(Z, Zp, alpha=1.0)
    shuffled_model = fit_shuffled_pair_control(Z, Zp, alpha=1.0, seed=0)

    real_r2 = multi_output_r2(real_model.predict(Z), Zp)
    shuffled_r2 = multi_output_r2(shuffled_model.predict(Z), Zp)
    assert real_r2 > 0.9
    assert shuffled_r2 < real_r2 - 0.5, "shuffling correspondence must measurably destroy fit quality"
    assert shuffled_model.n_params == real_model.n_params


# --- no-NaN/no-Inf sweep across the whole hierarchy at the toy shape ------------


def test_no_nan_or_inf_anywhere_in_the_hierarchy():
    Z, rng = _toy_data(seed=11)
    Zp = Z @ (rng.normal(size=(WELL_DETERMINED_D, WELL_DETERMINED_D)) * 0.1) + rng.normal(size=WELL_DETERMINED_D) * 0.1
    Z_val, Zp_val = Z[:80], Zp[:80]

    models = [
        fit_m0_persistence(Z, Zp),
        fit_m1_scalar(Z, Zp),
        fit_m2_diagonal(Z, Zp, alpha=1.0),
        fit_m3_low_rank(Z, Zp, r=2, alpha=1.0),
        fit_m4_full_linear(Z, Zp, alpha=1.0),
        fit_m5_affine(Z, Zp, alpha=1.0),
        fit_m6_residual_mlp(Z, Zp, Z_val, Zp_val, h=8, seed=0, max_epochs=200, patience=30),
        fit_train_mean_control(Z, Zp),
        fit_shuffled_pair_control(Z, Zp, alpha=1.0, seed=0),
    ]
    for model in models:
        pred = model.predict(Z_val)
        assert np.all(np.isfinite(pred)), f"{model.name} produced non-finite predictions"
