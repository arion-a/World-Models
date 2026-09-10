import numpy as np

from baselines.identity_baseline import evaluate_identity_baseline
from baselines.shuffled_pairing_baseline import evaluate_shuffled_pairing_baseline
from metrics.equivariance import evaluate_equivariance
from probes.linear_rep_transform import LinearRepTransform
from probes.linear_state_probe import LinearStateProbe


def _synthetic_linear_pairs(n, d, seed, noise=0.0):
    rng = np.random.default_rng(seed)
    W_true = rng.normal(size=(d, d)) * 0.3 + np.eye(d)
    b_true = rng.normal(size=d) * 0.1
    Z = rng.normal(size=(n, d))
    Z_prime = Z @ W_true.T + b_true + noise * rng.normal(size=(n, d))
    return Z, Z_prime, W_true, b_true


def test_linear_rep_transform_recovers_near_exact_linear_map():
    Z, Z_prime, W_true, b_true = _synthetic_linear_pairs(n=200, d=6, seed=0, noise=0.0)
    rho = LinearRepTransform.fit(Z, Z_prime, alpha=1e-6)
    pred = rho.predict(Z)
    assert np.allclose(pred, Z_prime, atol=1e-2)


def test_equivariance_high_r2_on_clean_linear_relation():
    Z_train, Zp_train, W_true, b_true = _synthetic_linear_pairs(n=150, d=8, seed=1, noise=0.0)
    # Test pairs must follow the SAME linear map as train, so re-use W_true/b_true.
    rng2 = np.random.default_rng(99)
    Z_test = rng2.normal(size=(50, 8))
    Zp_test = Z_test @ W_true.T + b_true

    result, rho = evaluate_equivariance("dummy", Z_train, Zp_train, Z_test, Zp_test, alpha=1e-3)
    assert result.r2 > 0.95
    assert result.mean_cosine_similarity > 0.95


def test_equivariance_low_r2_when_relation_is_random_noise():
    rng = np.random.default_rng(0)
    d = 10
    Z_train = rng.normal(size=(100, d))
    Zp_train = rng.normal(size=(100, d))  # unrelated to Z_train
    Z_test = rng.normal(size=(40, d))
    Zp_test = rng.normal(size=(40, d))
    result, _ = evaluate_equivariance("dummy", Z_train, Zp_train, Z_test, Zp_test, alpha=10.0)
    assert result.r2 < 0.3


def test_identity_baseline_matches_manual_computation():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(20, 5))
    Zp = Z + 0.01 * rng.normal(size=(20, 5))  # nearly unchanged
    result = evaluate_identity_baseline("dummy", Z, Zp)
    assert result.r2 > 0.9  # identity is a great predictor when nothing changes


def test_shuffled_pairing_baseline_is_worse_than_true_pairing():
    Z_train, Zp_train, W_true, b_true = _synthetic_linear_pairs(n=150, d=8, seed=3, noise=0.0)
    rng2 = np.random.default_rng(123)
    Z_test = rng2.normal(size=(50, 8))
    Zp_test = Z_test @ W_true.T + b_true

    true_result, _ = evaluate_equivariance("dummy", Z_train, Zp_train, Z_test, Zp_test, alpha=1e-3)
    shuffled_result = evaluate_shuffled_pairing_baseline("dummy", Z_train, Zp_train, Z_test, Zp_test, alpha=1e-3)
    assert true_result.r2 > shuffled_result.r2


def test_linear_state_probe_recovers_linear_target():
    rng = np.random.default_rng(0)
    d = 6
    Z_train = rng.normal(size=(100, d))
    w = rng.normal(size=d)
    y_train = (Z_train @ w).reshape(-1, 1)
    Z_test = rng.normal(size=(30, d))
    y_test = (Z_test @ w).reshape(-1, 1)

    probe = LinearStateProbe.fit(Z_train, y_train, alpha=1e-6)
    assert probe.score(Z_test, y_test) > 0.95
