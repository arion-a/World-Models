"""Task 10: one consolidated, uniformly-applied entry point for the full
DESIGN.md Sec 12 baseline set.

Tasks 6-9 each called baseline logic individually (persistence, mean,
shuffled-pairing) and DESIGN.md Sec 12 additionally specifies two
baselines -- the non-learned pixel-statistics encoder and the
matched-architecture randomly-initialized encoder -- that were not wired
into any Task 6-9 experiment. This module formalizes and extends
`baselines/identity_baseline.py`, `baselines/mean_baseline.py`, and
`baselines/shuffled_pairing_baseline.py` (their math is untouched here --
this module only calls them) and adds the two missing baselines, so
Tasks 11+ can invoke one function instead of re-wiring baseline calls by
hand each time (research/RESEARCH_INVARIANTS.md invariant 14).

Two entry points, by design:

- `run_baselines_core_three` -- persistence + mean + shuffled-pairing,
  needing only the PRIMARY encoder's own (Z, Z') arrays. Used by call
  sites that cannot supply a fair pixel-statistics/random-encoder
  comparison (e.g. a synthetic-Z unit test with no rendered frames to
  re-encode, or the Task 7 forensic-audit scripts that intentionally
  reproduce Task 7's exact original three-baseline computation).
- `run_all_baselines` -- the full five. `pixel_*`/`random_*` arrays are
  required keyword-only arguments with NO default, so omitting them is a
  loud `TypeError` at the call site, never a silently-incomplete result
  (research/CANONICAL_RESEARCH_PROTOCOL.md TASK 10's "Failure
  conditions": "a missing baseline silently skipped instead of
  raising"). A caller that cannot supply them must call
  `run_baselines_core_three` directly instead -- never call this
  function with placeholder/fabricated pixel or random arrays.

The pixel-statistics/random-encoder baselines (DESIGN.md Sec 12 items
3-4) re-run the IDENTICAL fit-rho(T)-and-evaluate protocol
(`metrics.equivariance.evaluate_equivariance`, same ridge alpha, same
train/test split) the primary encoder's "learned W_T" result uses, just
fed a different encoder's (Z, Z') arrays. Those arrays must themselves
come from re-encoding the SAME rendered frames the primary encoder saw
(never ground-truth `SceneState` coordinates) -- that is the caller's
responsibility (see `experiments/geometric_consistency_lib.py`'s
`encode_all_pairs_pooled`), not something this module can check from
arrays alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.identity_baseline import evaluate_identity_baseline
from baselines.mean_baseline import evaluate_mean_baseline
from baselines.shuffled_pairing_baseline import evaluate_shuffled_pairing_baseline
from metrics.equivariance import evaluate_equivariance

CORE_BASELINE_NAMES = ("persistence", "mean", "shuffled_pairing")
FULL_BASELINE_NAMES = CORE_BASELINE_NAMES + ("pixel_statistics", "random_encoder")


@dataclass
class AlternateEncoderBaselineResult:
    """Same field names/types as `IdentityBaselineResult`/
    `MeanBaselineResult`/`ShuffledPairingBaselineResult` -- the
    pixel-statistics/randomly-initialized-encoder baselines report
    through the identical shape so downstream code (result JSON,
    report tables) treats all five baselines uniformly.
    """

    transform_name: str
    r2: float
    mean_cosine_similarity: float
    mean_relative_l2_error: float


def run_baselines_core_three(
    transform_name: str,
    Z_train: np.ndarray,
    Zp_train: np.ndarray,
    Z_test: np.ndarray,
    Zp_test: np.ndarray,
    alpha: float = 10.0,
    seed: int = 0,
) -> dict[str, object]:
    """persistence + mean + shuffled-pairing baselines, unmodified math
    (see each module's own docstring for why each one exists). Returns
    exactly `{"persistence": ..., "mean": ..., "shuffled_pairing": ...}`,
    each value the same dataclass instance calling the underlying
    function directly would produce.
    """
    return {
        "persistence": evaluate_identity_baseline(transform_name, Z_test, Zp_test),
        "mean": evaluate_mean_baseline(transform_name, Zp_train, Zp_test),
        "shuffled_pairing": evaluate_shuffled_pairing_baseline(
            transform_name, Z_train, Zp_train, Z_test, Zp_test, alpha=alpha, seed=seed
        ),
    }


def run_all_baselines(
    Z_train: np.ndarray,
    Zp_train: np.ndarray,
    Z_test: np.ndarray,
    Zp_test: np.ndarray,
    transform_name: str,
    alpha: float = 10.0,
    seed: int = 0,
    *,
    pixel_Z_train: np.ndarray,
    pixel_Zp_train: np.ndarray,
    pixel_Z_test: np.ndarray,
    pixel_Zp_test: np.ndarray,
    random_Z_train: np.ndarray,
    random_Zp_train: np.ndarray,
    random_Z_test: np.ndarray,
    random_Zp_test: np.ndarray,
) -> dict[str, object]:
    """The full DESIGN.md Sec 12 baseline set -- persistence, mean,
    shuffled-pairing, non-learned pixel-statistics encoder,
    matched-architecture randomly-initialized encoder -- run uniformly
    under one entry point and returned as
    `{"persistence": ..., "mean": ..., "shuffled_pairing": ...,
      "pixel_statistics": ..., "random_encoder": ...}`, always all five.

    Every baseline is evaluated with the identical `transform_name`,
    `alpha` (ridge regularization strength), and train/test split the
    primary representation's own result used -- see this module's
    docstring for the pixel/random-array fair-comparison requirement.
    """
    core = run_baselines_core_three(transform_name, Z_train, Zp_train, Z_test, Zp_test, alpha=alpha, seed=seed)

    pixel_equiv, _ = evaluate_equivariance(
        transform_name, pixel_Z_train, pixel_Zp_train, pixel_Z_test, pixel_Zp_test, alpha=alpha
    )
    random_equiv, _ = evaluate_equivariance(
        transform_name, random_Z_train, random_Zp_train, random_Z_test, random_Zp_test, alpha=alpha
    )

    return {
        **core,
        "pixel_statistics": AlternateEncoderBaselineResult(
            transform_name=transform_name,
            r2=pixel_equiv.r2,
            mean_cosine_similarity=pixel_equiv.mean_cosine_similarity,
            mean_relative_l2_error=pixel_equiv.mean_relative_l2_error,
        ),
        "random_encoder": AlternateEncoderBaselineResult(
            transform_name=transform_name,
            r2=random_equiv.r2,
            mean_cosine_similarity=random_equiv.mean_cosine_similarity,
            mean_relative_l2_error=random_equiv.mean_relative_l2_error,
        ),
    }


def run_probe_baselines(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    predict_fn,
    permutation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Task 8-style probe baselines (Z -> y, not Z -> Z'): mean-prediction
    and shuffled-label control, generalizing `mean_baseline.py`'s/
    `shuffled_pairing_baseline.py`'s ideas from Z'-targets to
    physical-state-label targets. Returns raw PREDICTIONS only
    (`{"mean": ..., "shuffled_label": ...}`, each shape `y_test.shape`)
    -- scoring (R^2/MAE/RMSE, plus Task 8's circular-angle decoding for
    camera_azimuth_deg) is left to the caller, per this task's "must not
    hardcode one metric family" requirement.

    - "mean": `mean(y_train)` broadcast to every test row, ignoring Z
      entirely, exactly `mean_baseline.py`'s prediction rule.
    - "shuffled_label": `predict_fn` (the caller's own fit+predict step,
      e.g. a ridge probe) run on `(Z_train, y_train[permutation])` --
      the true `Z_train` paired with DELIBERATELY WRONG labels, exactly
      `shuffled_pairing_baseline.py`'s idea. `permutation` is the
      caller's own genuine (identity-permutation-guarded) permutation of
      `range(len(y_train))` -- generating it is the caller's
      responsibility so the exact permutation used stays independently
      inspectable/testable (see
      `experiments/task8_physical_state.py`'s `shuffled_label_permutation`).
    """
    mean_pred = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
    shuffled_pred = predict_fn(Z_train, y_train[permutation], Z_test)
    return {"mean": mean_pred, "shuffled_label": shuffled_pred}
