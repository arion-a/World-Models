"""Task 7B metrics (contract Sec. 3): multi-output R^2 with an explicit
"undefined" sentinel (never silently coerced to zero), MSE, relative L2
error, cosine similarity, and bootstrap confidence intervals over scenes.

R^2's mathematical form is identical to metrics/common.py's r_squared
(SS_res / SS_tot using the EVALUATION set's own mean) -- reimplemented
here, not imported, only so this module can return the literal string
"undefined" (JSON-safe) instead of metrics/common.py's float('nan')
sentinel, per the contract's explicit "report undefined, never silently
coerce it to zero" requirement. The underlying arithmetic is the same
formula and is cross-checked against metrics/common.py in
tests/test_task7b_metrics.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNDEFINED = "undefined"


def multi_output_r2(pred: np.ndarray, target: np.ndarray) -> float | str:
    """R^2 = 1 - sum_i||Z'_i - Zhat'_i||^2 / sum_i||Z'_i - mean_eval(Z')||^2
    (contract Sec. 3's exact formula). Returns the string "undefined" when
    the denominator is (numerically) zero, never 0.0 or nan silently."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mean_eval = target.mean(axis=0, keepdims=True)
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - mean_eval) ** 2))
    if ss_tot < 1e-12:
        return UNDEFINED
    return 1.0 - ss_res / ss_tot


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))


def relative_l2_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean, over samples, of ||pred_i - target_i|| / ||target_i||."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    num = np.linalg.norm(pred - target, axis=-1)
    den = np.linalg.norm(target, axis=-1)
    den = np.where(den < 1e-12, 1e-12, den)
    return float(np.mean(num / den))


def cosine_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    num = np.sum(pred * target, axis=-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1)
    den = np.where(den < 1e-12, 1e-12, den)
    return float(np.mean(num / den))


@dataclass
class MetricBundle:
    r2: float | str
    mse: float
    relative_l2_error: float
    cosine_similarity: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "r2": self.r2,
            "mse": self.mse,
            "relative_l2_error": self.relative_l2_error,
            "cosine_similarity": self.cosine_similarity,
            "n_samples": self.n_samples,
        }


def evaluate(pred: np.ndarray, target: np.ndarray) -> MetricBundle:
    return MetricBundle(
        r2=multi_output_r2(pred, target),
        mse=mse(pred, target),
        relative_l2_error=relative_l2_error(pred, target),
        cosine_similarity=cosine_similarity(pred, target),
        n_samples=int(np.asarray(target).shape[0]),
    )


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 0
) -> dict:
    """95% bootstrap CI (percentile method) over a list of ALREADY-COMPUTED,
    independent scalar point estimates -- e.g. one R^2 value per seed
    (learning_curve.aggregate_over_seeds's use). `values` must exclude any
    "undefined" entries (caller's responsibility -- an "undefined" R^2 has
    no numeric value to bootstrap).

    NOT for per-scene R^2 values: multi_output_r2 on a single sample is
    mathematically always "undefined" (its own baseline mean equals
    itself, so ss_tot==0 identically) -- there is no such thing as a
    valid "per-scene R^2" to resample here. Use bootstrap_ci_over_scenes
    below instead, which resamples SCENES and recomputes the multi-output
    metric on each resampled set."""
    arr = np.asarray([v for v in values if isinstance(v, (int, float))], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": UNDEFINED, "low": UNDEFINED, "high": UNDEFINED, "n": 0}
    rng = np.random.default_rng(seed)
    resample_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)])
    alpha = (1.0 - ci) / 2
    low, high = np.quantile(resample_means, [alpha, 1 - alpha])
    return {"mean": float(arr.mean()), "low": float(low), "high": float(high), "n": int(len(arr))}


def bootstrap_ci_over_scenes(
    pred: np.ndarray, target: np.ndarray, n_resamples: int = 2000, ci: float = 0.95, seed: int = 0,
) -> dict:
    """95% bootstrap CI over SCENES for the multi-output R^2 metric
    (contract Sec. 3's "95% bootstrap CIs over seeds/scenes"). Resamples
    scene rows (with replacement) and recomputes multi_output_r2 on each
    resampled (pred, target) set -- unlike a per-scene R^2 (which cannot
    exist: a single-sample R^2 always has a zero-variance baseline, hence
    always "undefined"), this recomputes the SAME set-level statistic
    reported as the point estimate, just over resampled scene sets, which
    is the standard, well-defined way to bootstrap a metric that requires
    more than one sample."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = target.shape[0]
    point = multi_output_r2(pred, target)
    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        v = multi_output_r2(pred[idx], target[idx])
        if isinstance(v, (int, float)):
            resampled.append(v)
    if not resampled:
        return {"point": point, "mean": UNDEFINED, "low": UNDEFINED, "high": UNDEFINED, "n_resamples_defined": 0}
    arr = np.asarray(resampled, dtype=np.float64)
    alpha = (1.0 - ci) / 2
    low, high = np.quantile(arr, [alpha, 1 - alpha])
    return {"point": point, "mean": float(arr.mean()), "low": float(low), "high": float(high), "n_resamples_defined": int(len(arr))}
