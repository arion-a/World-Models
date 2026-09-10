"""Learn rho(T): a linear representation-space transformation, Z' ~ W_T Z + b.

This is the central object of the V0 experiment:

    Z' ≈ rho(T) Z

We use ridge regression (closed form / sklearn `Ridge`) because the
project principle is "prefer simple linear probes ... before using neural
probes" -- a linear map is the simplest non-trivial hypothesis for rho(T),
and it is exactly solvable, so there is no optimizer, no learning rate,
and no training-instability confound to worry about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class LinearRepTransform:
    """Fitted affine map W_T Z + b approximating Z -> Z' for one transform type."""

    alpha: float
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    _model: Ridge | None = None

    @classmethod
    def fit(cls, Z: np.ndarray, Z_prime: np.ndarray, alpha: float = 10.0) -> "LinearRepTransform":
        """Z, Z_prime: (N_train, D) representation pairs from N_train distinct scenes."""
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Z, Z_prime)
        return cls(alpha=alpha, W=model.coef_, b=model.intercept_, _model=model)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.W.T + self.b
