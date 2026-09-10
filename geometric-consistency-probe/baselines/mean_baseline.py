"""Baseline: rho(T) = constant, predicting Z' as the training-set mean of Z'.

    Z_hat' = mean(Z'_train)

ignoring the input Z entirely. If a learned W_T does not clearly beat
this, apparent "equivariance" evidence would really just be "transformed
representations cluster near a fixed point regardless of the
untransformed scene," not a genuine per-scene predictable transformation.
Always report this next to the learned rho(T) result, matching
identity_baseline.py's role as a control on the other trivial hypothesis
(Z_hat' = Z, no change at all).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.common import cosine_similarity, r_squared, relative_l2_error


@dataclass
class MeanBaselineResult:
    transform_name: str
    r2: float
    mean_cosine_similarity: float
    mean_relative_l2_error: float


def evaluate_mean_baseline(transform_name: str, Z_prime_train: np.ndarray, Z_prime_test: np.ndarray) -> MeanBaselineResult:
    """Z_hat' = mean(Z'_train), broadcast to every test example.

    `Z_prime_train` must come only from TRAIN scenes -- the mean is a
    fitted parameter and must never see a test-split representation
    (research/RESEARCH_INVARIANTS.md invariant 16). Deliberately takes
    no `Z`/`Z_train` argument at all: this baseline's prediction does not
    depend on the untransformed representation, by construction.
    """
    mean_vec = Z_prime_train.mean(axis=0)
    pred = np.broadcast_to(mean_vec, Z_prime_test.shape)
    return MeanBaselineResult(
        transform_name=transform_name,
        r2=r_squared(pred, Z_prime_test),
        mean_cosine_similarity=float(np.mean(cosine_similarity(pred, Z_prime_test))),
        mean_relative_l2_error=float(np.mean(relative_l2_error(pred, Z_prime_test))),
    )
