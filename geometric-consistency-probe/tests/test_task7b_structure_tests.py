"""Verifies the Stage 3 structure-test math (composition/inverse/identity/
interpolation) against KNOWN synthetic linear operators, before it is
ever applied to real fitted models -- contract Sec. 12's "verify metric
calculations against hand-computable cases" extended to the operator-
level Frobenius discrepancies this module introduces."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.task7b_latent_transformation_discovery import structure_tests as st


def _linear_predict(W, b=None):
    b = np.zeros(W.shape[0]) if b is None else b
    return lambda Z: Z @ W.T + b


def test_operator_matrix_recovers_a_known_linear_map():
    rng = np.random.default_rng(0)
    D = 6
    W_true = rng.normal(size=(D, D))
    predict = _linear_predict(W_true)
    W_recovered = st.operator_matrix(type("M", (), {"predict": staticmethod(predict)})(), D)
    assert W_recovered == pytest.approx(W_true, abs=1e-9)


def test_operator_matrix_returns_none_for_nonzero_bias():
    D = 4
    W = np.eye(D)
    b = np.ones(D)
    predict = _linear_predict(W, b)
    result = st.operator_matrix(type("M", (), {"predict": staticmethod(predict)})(), D)
    assert result is None


def test_composition_holds_exactly_for_commuting_known_matrices():
    """Two rotation-like matrices about the same axis compose additively
    -- constructed here directly (not via the renderer) as a pure math
    check of the composition_test function itself."""
    rng = np.random.default_rng(1)
    D = 8
    # simulate a 1-parameter family: A(theta) = exp(theta * G) for a fixed
    # generator G (guarantees A(a)A(b) = A(a+b) exactly)
    G = rng.normal(size=(D, D)) * 0.05
    G = G - G.T  # skew-symmetric -> generates a genuine rotation group

    def A(theta):
        # small-angle approx of matrix exponential, exact enough for this test's tolerance
        n = 40
        M = np.eye(D)
        term = np.eye(D)
        for k in range(1, n):
            term = term @ (theta * G) / k
            M = M + term
        return M

    theta1, theta2 = 0.3, 0.5
    W1, W2, W_sum = A(theta1), A(theta2), A(theta1 + theta2)
    Z_test = rng.normal(size=(20, D))

    result = st.composition_test(_linear_predict(W1), _linear_predict(W2), _linear_predict(W_sum), Z_test, W1=W1, W2=W2, W_sum=W_sum)
    assert result["normalized_frobenius_operator_discrepancy"] < 1e-6
    assert result["output_space_agreement"]["r2"] == pytest.approx(1.0, abs=1e-4)


def test_composition_detects_a_genuine_violation():
    rng = np.random.default_rng(2)
    D = 5
    W1 = rng.normal(size=(D, D))
    W2 = rng.normal(size=(D, D))
    W_sum_wrong = rng.normal(size=(D, D))  # NOT W2 @ W1 -- a deliberate violation
    Z_test = rng.normal(size=(20, D))
    result = st.composition_test(_linear_predict(W1), _linear_predict(W2), _linear_predict(W_sum_wrong), Z_test, W1=W1, W2=W2, W_sum=W_sum_wrong)
    assert result["normalized_frobenius_operator_discrepancy"] > 0.5


def test_inverse_holds_for_a_true_inverse_pair():
    rng = np.random.default_rng(3)
    D = 6
    W = rng.normal(size=(D, D)) * 0.3 + np.eye(D)
    W_inv = np.linalg.inv(W)
    Z_test = rng.normal(size=(15, D))
    result = st.inverse_test(_linear_predict(W), _linear_predict(W_inv), Z_test, W=W, W_neg=W_inv)
    assert result["normalized_frobenius_operator_discrepancy"] < 1e-8
    assert result["output_space_agreement"]["r2"] == pytest.approx(1.0, abs=1e-6)


def test_identity_holds_for_the_true_identity():
    D = 5
    Z_test = np.random.default_rng(4).normal(size=(10, D))
    result = st.identity_test(_linear_predict(np.eye(D)), Z_test, W_zero=np.eye(D), b_zero=np.zeros(D))
    assert result["normalized_frobenius_w_discrepancy"] == pytest.approx(0.0, abs=1e-12)
    assert result["bias_norm_relative_to_target_scale"] == pytest.approx(0.0, abs=1e-12)


def test_identity_detects_a_genuine_violation():
    D = 5
    rng = np.random.default_rng(5)
    W_wrong = rng.normal(size=(D, D))
    Z_test = rng.normal(size=(10, D))
    result = st.identity_test(_linear_predict(W_wrong), Z_test, W_zero=W_wrong)
    assert result["normalized_frobenius_w_discrepancy"] > 0.5


def test_interpolation_recovers_the_true_interior_value_for_a_linear_family():
    rng = np.random.default_rng(6)
    D = 6

    def A(theta):
        return np.eye(D) * (1 + 0.01 * theta)  # exactly linear in theta

    theta_a, theta_b, theta_interior = 10.0, 30.0, 20.0
    lam = (theta_interior - theta_a) / (theta_b - theta_a)
    Z_test = rng.normal(size=(12, D))
    Zp_true_interior = Z_test @ A(theta_interior).T

    result = st.interpolation_test(_linear_predict(A(theta_a)), _linear_predict(A(theta_b)), lam, Z_test, Zp_true_interior)
    assert result["metrics_vs_true_interior_magnitude"]["r2"] == pytest.approx(1.0, abs=1e-6)
    assert result["exploratory_continuous_generator_analysis"] is False


def test_not_applicable_helper_records_a_reason():
    result = st.not_applicable("transform family lacks a validated composition law over this range")
    assert result["applicable"] is False
    assert "composition law" in result["reason"]
