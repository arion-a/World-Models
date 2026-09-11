"""Sections 12 & 13: synthetic sanity tests of the regression/evaluation
pipeline, completely independent of V-JEPA/bpy -- uses the EXACT SAME
evaluate_transform / evaluate_equivariance / LinearRepTransform / metric
code Task 7 used, fed synthetic Z with a KNOWN linear relationship.

If the pipeline is implemented correctly:
  - Z' = A Z (known random A, no noise): the learned W_T should recover
    the mapping with high test R^2 (near 1.0), clearly beating every
    trivial baseline, given a train sample size sufficient for the
    problem's dimensionality (tested at both a "hard" N=32,D=1024 --
    the real experiment's exact shape -- and an "easy" low-D shape, to
    separate "is the fitting code broken" from "is 32 samples simply
    not enough for D=1024").
  - Z' = Z (identity): the persistence baseline should score ~perfectly
    (R^2 ~ 1), since it directly predicts Z as Z'.
  - Z' = c (constant, independent of Z): the mean baseline should score
    ~perfectly, and the learned W_T should NOT beat it (nothing in Z
    predicts a target that doesn't depend on Z).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import experiments.geometric_consistency_lib as gclib

OUT_DIR = Path(__file__).resolve().parent


def make_split(n, n_train, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return idx[:n_train], idx[n_train:]


def run_linear_recovery_test(n_total, d, n_train, alpha, noise_std, seed, label):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_total, d)).astype(np.float64)
    A = rng.normal(size=(d, d)).astype(np.float64) / np.sqrt(d)  # scaled so A Z has O(1) norm like Z
    b = rng.normal(size=(d,)) * 0.1
    noise = rng.normal(size=(n_total, d)) * noise_std
    Zp = Z @ A.T + b + noise

    train_idx, test_idx = make_split(n_total, n_train, seed=seed + 1)
    Z_train, Zp_train = Z[train_idx], Zp[train_idx]
    Z_test, Zp_test = Z[test_idx], Zp[test_idx]

    result = gclib.evaluate_transform(f"synthetic_{label}", Z_train, Zp_train, Z_test, Zp_test, alpha, shuffled_pairing_seed=0)
    print(f"\n--- {label}: N_total={n_total} D={d} N_train={n_train} alpha={alpha} noise_std={noise_std} ---")
    for method, block in result.items():
        print(f"  {method}: r2={block['r2']:.4f} cos={block['mean_cosine_similarity']:.4f} rel_l2={block['mean_relative_l2_error']:.4f}")
    return result


def run_identity_test(n_total, d, n_train, alpha, seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_total, d))
    Zp = Z.copy()  # Z' = Z exactly
    train_idx, test_idx = make_split(n_total, n_train, seed=seed + 1)
    result = gclib.evaluate_transform("synthetic_identity", Z[train_idx], Zp[train_idx], Z[test_idx], Zp[test_idx], alpha, shuffled_pairing_seed=0)
    print(f"\n--- identity (Z'=Z): N_total={n_total} D={d} N_train={n_train} ---")
    for method, block in result.items():
        print(f"  {method}: r2={block['r2']:.4f}")
    return result


def run_constant_test(n_total, d, n_train, alpha, seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_total, d))
    c = rng.normal(size=(d,))
    Zp = np.broadcast_to(c, (n_total, d)).copy()  # Z' independent of Z
    train_idx, test_idx = make_split(n_total, n_train, seed=seed + 1)
    result = gclib.evaluate_transform("synthetic_constant", Z[train_idx], Zp[train_idx], Z[test_idx], Zp[test_idx], alpha, shuffled_pairing_seed=0)
    print(f"\n--- constant (Z' independent of Z): N_total={n_total} D={d} N_train={n_train} ---")
    for method, block in result.items():
        print(f"  {method}: r2={block['r2']:.4f}")
    return result


def main():
    results = {}

    # Same exact shape as the real Task 7 experiment: N_train=32, D=1024.
    results["real_shape_low_noise_alpha10"] = run_linear_recovery_test(
        n_total=40, d=1024, n_train=32, alpha=10.0, noise_std=0.01, seed=0, label="real_shape_low_noise_alpha10"
    )
    results["real_shape_low_noise_alpha0.01"] = run_linear_recovery_test(
        n_total=40, d=1024, n_train=32, alpha=0.01, noise_std=0.01, seed=0, label="real_shape_low_noise_alpha0.01"
    )
    results["real_shape_moderate_noise_alpha10"] = run_linear_recovery_test(
        n_total=40, d=1024, n_train=32, alpha=10.0, noise_std=0.5, seed=1, label="real_shape_moderate_noise_alpha10"
    )
    # Easy control: low-dimensional, well-determined problem (D << N_train)
    # -- isolates "is the fitting/eval code correct" from "is N=32,D=1024
    # simply an underdetermined regime."
    results["easy_shape_low_d"] = run_linear_recovery_test(
        n_total=40, d=8, n_train=32, alpha=1.0, noise_std=0.01, seed=2, label="easy_shape_low_d"
    )
    results["identity"] = run_identity_test(n_total=40, d=1024, n_train=32, alpha=10.0, seed=3)
    results["constant"] = run_constant_test(n_total=40, d=1024, n_train=32, alpha=10.0, seed=4)

    (OUT_DIR / "synthetic_sanity.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
