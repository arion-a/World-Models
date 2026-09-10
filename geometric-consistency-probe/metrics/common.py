"""Small, well-defined numeric metrics shared by the probe/metric modules.

Every function here has a textbook definition and is unit tested
(tests/test_metrics.py) precisely because the project principle is
"every metric must have a clear mathematical interpretation" -- nothing
exotic, nothing that needs a paragraph of hand-waving to explain.
"""

from __future__ import annotations

import numpy as np


def _as_samples_by_features(x: np.ndarray) -> np.ndarray:
    """Normalize to (N, D). A 1-D input of length N is treated as N samples
    of a scalar feature, i.e. reshaped to (N, 1) -- NOT via `np.atleast_2d`,
    which would instead produce (1, N) and silently make every downstream
    row-wise computation broadcast into an (N, N) mess. This distinction
    matters here because scalar physical-state targets (e.g. a single
    camera angle) are exactly the 1-D case these metrics must handle
    correctly.
    """
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for (N, D) arrays -> (N,) in [-1, 1]."""
    a = _as_samples_by_features(a)
    b = _as_samples_by_features(b)
    num = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    denom = np.where(denom == 0, 1e-12, denom)
    return num / denom


def relative_l2_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Row-wise ||pred - target|| / ||target||, for (N, D) arrays -> (N,)."""
    pred = _as_samples_by_features(pred)
    target = _as_samples_by_features(target)
    num = np.linalg.norm(pred - target, axis=-1)
    denom = np.linalg.norm(target, axis=-1)
    denom = np.where(denom == 0, 1e-12, denom)
    return num / denom


def r_squared(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination, aggregated over all output dimensions.

    R^2 = 1 - SS_res / SS_tot, where SS_tot uses the *training-set-free*
    per-column mean of `target` itself (standard multi-output R^2, as in
    sklearn's default 'uniform_average' with variance-weighted skipped --
    here we use the simple total-sum-of-squares form, which is exact for
    a single held-out evaluation set and does not require re-deriving the
    training mean).
    """
    pred = _as_samples_by_features(pred)
    target = _as_samples_by_features(target)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)
