"""Component C: accessibility of physical state from representations.

A linear ridge probe Z -> s, where `s` is some scalar or vector summary
of the physical state S (e.g. camera azimuth, an object's rotation
angle). A high held-out R^2 means that quantity is *linearly decodable*
from Z; it does NOT by itself mean the representation "understands" that
quantity causally or would support the same decoding under a different
scene distribution -- see DESIGN.md "Interpretation guardrails".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from metrics.common import r_squared


@dataclass
class LinearStateProbe:
    alpha: float
    _model: Ridge | None = None

    @classmethod
    def fit(cls, Z: np.ndarray, targets: np.ndarray, alpha: float = 10.0) -> "LinearStateProbe":
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Z, targets)
        return cls(alpha=alpha, _model=model)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self._model.predict(Z)

    def score(self, Z: np.ndarray, targets: np.ndarray) -> float:
        return r_squared(self.predict(Z), targets)
