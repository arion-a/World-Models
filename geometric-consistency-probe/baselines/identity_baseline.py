"""Baseline: rho(T) = Identity, i.e. predict Z' = Z.

If this trivial "nothing changes" baseline already scores well on the
equivariance metrics, a high score for a *learned* W_T is not evidence of
anything beyond "the representation doesn't move much under this
transform" (which is an invariance finding, not an equivariance one).
Always report this next to the learned rho(T) result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.common import cosine_similarity, r_squared, relative_l2_error


@dataclass
class IdentityBaselineResult:
    transform_name: str
    r2: float
    mean_cosine_similarity: float
    mean_relative_l2_error: float


def evaluate_identity_baseline(transform_name: str, Z_test: np.ndarray, Z_prime_test: np.ndarray) -> IdentityBaselineResult:
    pred = Z_test
    return IdentityBaselineResult(
        transform_name=transform_name,
        r2=r_squared(pred, Z_prime_test),
        mean_cosine_similarity=float(np.mean(cosine_similarity(pred, Z_prime_test))),
        mean_relative_l2_error=float(np.mean(relative_l2_error(pred, Z_prime_test))),
    )
