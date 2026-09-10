"""Component B: geometric invariance, measured on the raw (Z, Z') pairs
without fitting anything -- no learned map, just "how similar are the
representations before and after the transform".

This is the natural counterpart to equivariance: for the control
transforms (lighting_change, texture_change), which should NOT move the
physical state, we want raw cosine similarity to be HIGH (the encoder
ignores appearance) -- unlike equivariance, which is about whether change
is *predictable* by a map, invariance is about whether there is much
change to predict in the first place.

Reporting both together is what lets us tell the two apart: a transform
with high raw similarity (high invariance) that also gets a trivially
high equivariance R^2 with rho(T) ~= Identity would indicate "nothing
happened", not "a rich, correctly-predicted transformation".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.common import cosine_similarity, relative_l2_error


@dataclass
class InvarianceResult:
    transform_name: str
    mean_cosine_similarity: float
    mean_relative_l2_error: float
    n: int


def evaluate_invariance(transform_name: str, Z: np.ndarray, Z_prime: np.ndarray) -> InvarianceResult:
    return InvarianceResult(
        transform_name=transform_name,
        mean_cosine_similarity=float(np.mean(cosine_similarity(Z, Z_prime))),
        mean_relative_l2_error=float(np.mean(relative_l2_error(Z, Z_prime))),
        n=len(Z),
    )
