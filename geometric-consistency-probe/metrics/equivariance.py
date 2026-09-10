"""Component A: geometric equivariance.

Given a fitted rho(T) = W_T (probes/linear_rep_transform.py) and a
held-out set of (Z, Z') pairs from scenes NOT used to fit W_T, we report:

  * held-out R^2 of the affine fit (how much of Z' variance the learned
    map explains beyond just predicting the mean of Z')
  * mean cosine similarity between predicted and actual Z'
  * mean relative L2 error

All three are reported together because they are not redundant: a
representation could get high cosine similarity (right direction) but
poor R^2 (wrong scale), or vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.common import cosine_similarity, r_squared, relative_l2_error
from probes.linear_rep_transform import LinearRepTransform


@dataclass
class EquivarianceResult:
    transform_name: str
    r2: float
    mean_cosine_similarity: float
    mean_relative_l2_error: float
    n_train: int
    n_test: int


def evaluate_equivariance(
    transform_name: str,
    Z_train: np.ndarray,
    Z_prime_train: np.ndarray,
    Z_test: np.ndarray,
    Z_prime_test: np.ndarray,
    alpha: float = 10.0,
) -> tuple[EquivarianceResult, LinearRepTransform]:
    """Fit rho(T) on the train split, evaluate geometric consistency on the test split.

    Train and test scenes must be disjoint -- see representations/dataset.py
    for the split logic, and tests/test_scene_split.py for the check that
    enforces it.
    """
    rho = LinearRepTransform.fit(Z_train, Z_prime_train, alpha=alpha)
    Z_prime_pred = rho.predict(Z_test)
    result = EquivarianceResult(
        transform_name=transform_name,
        r2=r_squared(Z_prime_pred, Z_prime_test),
        mean_cosine_similarity=float(np.mean(cosine_similarity(Z_prime_pred, Z_prime_test))),
        mean_relative_l2_error=float(np.mean(relative_l2_error(Z_prime_pred, Z_prime_test))),
        n_train=len(Z_train),
        n_test=len(Z_test),
    )
    return result, rho
