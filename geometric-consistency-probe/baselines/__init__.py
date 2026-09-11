"""Baselines/controls for the geometric-consistency probe (DESIGN.md Sec 12).

Task 10 consolidates these into one uniformly-applied entry point --
`baselines.run_all_baselines` / `baselines.run_baselines_core_three` --
see `baselines/run_all_baselines.py` for the full rationale.
"""

from __future__ import annotations

from baselines.identity_baseline import IdentityBaselineResult, evaluate_identity_baseline
from baselines.mean_baseline import MeanBaselineResult, evaluate_mean_baseline
from baselines.run_all_baselines import (
    CORE_BASELINE_NAMES,
    FULL_BASELINE_NAMES,
    AlternateEncoderBaselineResult,
    run_all_baselines,
    run_baselines_core_three,
    run_probe_baselines,
)
from baselines.shuffled_pairing_baseline import ShuffledPairingBaselineResult, evaluate_shuffled_pairing_baseline

__all__ = [
    "CORE_BASELINE_NAMES",
    "FULL_BASELINE_NAMES",
    "AlternateEncoderBaselineResult",
    "IdentityBaselineResult",
    "MeanBaselineResult",
    "ShuffledPairingBaselineResult",
    "evaluate_identity_baseline",
    "evaluate_mean_baseline",
    "evaluate_shuffled_pairing_baseline",
    "run_all_baselines",
    "run_baselines_core_three",
    "run_probe_baselines",
]
