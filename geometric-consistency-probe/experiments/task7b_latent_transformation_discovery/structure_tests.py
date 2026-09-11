"""Task 7B Stage 3 structure tests (contract Sec. 9): composition,
inverse, identity, and interpolation, run on held-out TEST scenes using
operators selected entirely on train/validation data -- no refitting
here. Each test is only run for a transform family verified to satisfy a
valid one-parameter composition law (contract Sec. 8/9's physical
pre-check); tests/test_task7b_composition_law.py already verifies this
for camera_rotation at fixed elevation=0.

For LINEAR selected forms (M1/M2/M3/M4), operator-level discrepancies use
the exact Frobenius-norm formulas the contract specifies. For non-linear
forms (M6) or M5 (affine, which has a bias term the contract handles
separately), only output-space comparisons are reported for
composition/inverse (no single "W" matrix exists to compare directly for
M6; M5's affine composition is handled via its own homogeneous-coordinate
note).
"""
from __future__ import annotations

import numpy as np

from experiments.task7b_latent_transformation_discovery.metrics import evaluate

LINEAR_MODEL_PREFIXES = ("M1_scalar", "M2_diagonal", "M3_low_rank", "M4_full_linear")


def is_linear_model(model_key: str) -> bool:
    return any(model_key.startswith(p) for p in LINEAR_MODEL_PREFIXES)


def operator_matrix(fitted_model, D: int) -> np.ndarray | None:
    """Recovers the effective D x D linear operator W such that
    predict(Z) ~= Z @ W.T, by probing with the identity basis -- works
    for any of the linear model forms (M1-M4) regardless of their
    internal parameterization (scalar/diagonal/low-rank/full), without
    each needing to expose its own W attribute. Returns None for models
    with a bias term (would need probing with a zero input too) or
    non-linear models -- callers should use output-space comparisons for
    those instead (contract Sec. 9's affine/nonlinear provisions)."""
    basis = np.eye(D)
    zero = np.zeros((1, D))
    pred_basis = fitted_model.predict(basis)
    pred_zero = fitted_model.predict(zero)
    if not np.allclose(pred_zero, 0.0, atol=1e-6):
        return None  # has a nonzero bias -- not a pure linear map, use output-space comparisons
    return pred_basis.T  # W such that predict(Z) = Z @ W.T


def composition_test(
    F_theta1_predict, F_theta2_predict, F_sum_predict, Z_test: np.ndarray, W1: np.ndarray | None = None, W2: np.ndarray | None = None, W_sum: np.ndarray | None = None
) -> dict:
    """F_theta2(F_theta1(Z)) vs F_(theta1+theta2)(Z), output-space error,
    plus the operator-level Frobenius discrepancy when all three
    operators are linear (W1, W2, W_sum given)."""
    composed_output = F_theta2_predict(F_theta1_predict(Z_test))
    direct_output = F_sum_predict(Z_test)
    output_metrics = evaluate(composed_output, direct_output).to_dict()

    result = {"output_space_agreement": output_metrics, "applicable": True}
    if W1 is not None and W2 is not None and W_sum is not None:
        composed_W = W2 @ W1
        frob_num = np.linalg.norm(composed_W - W_sum, ord="fro")
        frob_den = np.linalg.norm(W_sum, ord="fro")
        result["normalized_frobenius_operator_discrepancy"] = float(frob_num / frob_den) if frob_den > 1e-12 else "undefined"
    return result


def inverse_test(F_theta_predict, F_neg_theta_predict, Z_test: np.ndarray, W: np.ndarray | None = None, W_neg: np.ndarray | None = None) -> dict:
    """F_-theta(F_theta(Z)) vs Z."""
    roundtrip = F_neg_theta_predict(F_theta_predict(Z_test))
    output_metrics = evaluate(roundtrip, Z_test).to_dict()

    result = {"output_space_agreement": output_metrics, "applicable": True}
    if W is not None and W_neg is not None:
        D = W.shape[0]
        frob_num = np.linalg.norm(W_neg @ W - np.eye(D), ord="fro")
        frob_den = np.linalg.norm(np.eye(D), ord="fro")
        result["normalized_frobenius_operator_discrepancy"] = float(frob_num / frob_den)
    return result


def identity_test(F_zero_predict, Z_test: np.ndarray, W_zero: np.ndarray | None = None, b_zero: np.ndarray | None = None) -> dict:
    """F_0(Z) vs Z, with no fitted exception."""
    pred = F_zero_predict(Z_test)
    output_metrics = evaluate(pred, Z_test).to_dict()
    result = {"output_space_agreement": output_metrics, "applicable": True}
    if W_zero is not None:
        D = W_zero.shape[0]
        frob_num = np.linalg.norm(W_zero - np.eye(D), ord="fro")
        frob_den = np.linalg.norm(np.eye(D), ord="fro")
        result["normalized_frobenius_w_discrepancy"] = float(frob_num / frob_den)
    if b_zero is not None:
        result["bias_norm_relative_to_target_scale"] = float(np.linalg.norm(b_zero))
    return result


def interpolation_test(F_theta_a_predict, F_theta_b_predict, lam: float, Z_test: np.ndarray, Zp_true_interior: np.ndarray) -> dict:
    """Predeclared output-space linear interpolation rule:
    (1-lambda) F_theta_a(Z) + lambda F_theta_b(Z), compared against the
    TRUE (never-fit) interior magnitude's target. No operator-group /
    matrix-log claim is made -- see contract Sec. 9's explicit warning."""
    pred_interior = (1 - lam) * F_theta_a_predict(Z_test) + lam * F_theta_b_predict(Z_test)
    metrics = evaluate(pred_interior, Zp_true_interior).to_dict()
    return {
        "rule": "output_space_linear_interpolation",
        "lambda": lam,
        "exploratory_continuous_generator_analysis": False,
        "metrics_vs_true_interior_magnitude": metrics,
        "applicable": True,
    }


NOT_APPLICABLE = {"applicable": False, "reason": None}


def not_applicable(reason: str) -> dict:
    return {"applicable": False, "reason": reason}
