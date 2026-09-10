"""Baseline: fit rho(T) on deliberately mismatched (Z_i, Z'_j) pairs, i != j.

This is a negative control for the fitting procedure itself: if a ridge
map fit on *scrambled* correspondences scores nearly as well as the map
fit on true correspondences, the apparent "equivariance" is an artifact
of the regression setup (e.g. dataset-level structure, regularization
shrinking everything toward the mean) rather than genuine scene-by-scene
predictability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.common import cosine_similarity, r_squared, relative_l2_error
from probes.linear_rep_transform import LinearRepTransform


@dataclass
class ShuffledPairingBaselineResult:
    transform_name: str
    r2: float
    mean_cosine_similarity: float
    mean_relative_l2_error: float


def evaluate_shuffled_pairing_baseline(
    transform_name: str,
    Z_train: np.ndarray,
    Z_prime_train: np.ndarray,
    Z_test: np.ndarray,
    Z_prime_test: np.ndarray,
    alpha: float = 10.0,
    seed: int = 0,
) -> ShuffledPairingBaselineResult:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Z_prime_train))
    # Guard against an identity permutation (rare, but possible for tiny N).
    if len(perm) > 1 and np.all(perm == np.arange(len(perm))):
        perm = np.roll(perm, 1)
    shuffled_targets = Z_prime_train[perm]

    rho = LinearRepTransform.fit(Z_train, shuffled_targets, alpha=alpha)
    pred = rho.predict(Z_test)
    return ShuffledPairingBaselineResult(
        transform_name=transform_name,
        r2=r_squared(pred, Z_prime_test),
        mean_cosine_similarity=float(np.mean(cosine_similarity(pred, Z_prime_test))),
        mean_relative_l2_error=float(np.mean(relative_l2_error(pred, Z_prime_test))),
    )
