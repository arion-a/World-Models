"""Diagnostic D: does mean-pooling over ALL 512 tokens (2 temporal
tubelets x 16x16 spatial patches) destroy information relevant to
geometric transformations?

Compares three representations, all derived from the SAME raw
(512, 1024) token sequence, fit/evaluated with the SAME ridge protocol
(fixed alpha, train-only fitting) as the original Task 7 experiment:

  1. global_mean  -- the ORIGINAL Task 7 representation: mean over all
                     512 tokens -> (1024,).
  2. spatial_grid_mean -- mean over the 2 TEMPORAL tubelets only, keeping
                     the 16x16=256 SPATIAL grid intact -> (256, 1024)
                     flattened to (262144,). This is the direct,
                     minimal, scientifically justified test of "is
                     spatial layout the information mean-pooling
                     destroys": if a geometric transform's effect is a
                     rearrangement across the spatial grid, keeping the
                     grid (and only averaging out the very short, largely
                     redundant temporal axis -- these are near-static
                     clips, see forensic audit's render-determinism
                     finding that all 4 frames of a clip are pixel-
                     identical) should recover comparatively more signal
                     than collapsing the grid too.
  3. quadrant_mean -- a coarser middle ground: average over temporal AND
                     over each of 4 spatial quadrants (2x2 pooling of
                     the 16x16 grid) -> (4, 1024) flattened to (4096,).
                     Retains coarse spatial layout without the full
                     262144-dim blowup of option 2.

This is explicitly a DIAGNOSTIC, not a new equivariant model: it reuses
the exact same Ridge(fit_intercept=True) + evaluate_equivariance-style
scoring already used for the original comparison, only changing what
"Z" and "Z'" mean before fitting.

Only camera_rotation (the transform this audit chain has focused on --
worst original R^2, largest representation-shift magnitude per the
forensic audit) and object_translation (one of the better original
performers, as a cross-check) are re-encoded here to get raw tokens --
the forensic audit's cached recomputed_representations.npz only stored
the pooled (1024,) vectors, not the raw (512,1024) token sequences, so
this diagnostic re-encodes those two transforms' 40 scenes x 2 sides
from the already-rendered, already-cached rgb.npy files (no new
rendering).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
RENDER_DIR = REPO_ROOT / "experiments" / "geometric_consistency"

GRID = 16  # crop_size(256) // patch_size(16)
TEMPORAL_TUBELETS = 2  # num_frames(4) // tubelet_size(2)


def r_squared(pred, target):
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def cosine_sim(pred, target):
    num = np.sum(pred * target, axis=-1)
    denom = np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1)
    denom = np.where(denom == 0, 1e-12, denom)
    return float(np.mean(num / denom))


def global_mean_pool(tokens: np.ndarray) -> np.ndarray:
    """(512, 1024) -> (1024,) -- the ORIGINAL Task 7 pooling."""
    return tokens.mean(axis=0)


def spatial_grid_mean_pool(tokens: np.ndarray) -> np.ndarray:
    """(512, 1024) -> (256, 1024) -> flatten (262144,). Mean over the 2
    temporal tubelets only; the 16x16 spatial grid (in row-major token
    order, per VJEPA2's patch embedding conv -> flatten(2)) is kept."""
    t = tokens.reshape(TEMPORAL_TUBELETS, GRID * GRID, tokens.shape[-1])
    grid = t.mean(axis=0)  # (256, 1024)
    return grid.reshape(-1)


def quadrant_mean_pool(tokens: np.ndarray) -> np.ndarray:
    """(512, 1024) -> mean over temporal AND 2x2 spatial quadrants -> (4, 1024) -> flatten (4096,)."""
    t = tokens.reshape(TEMPORAL_TUBELETS, GRID, GRID, tokens.shape[-1]).mean(axis=0)  # (16, 16, 1024)
    half = GRID // 2
    quadrants = [
        t[:half, :half].mean(axis=(0, 1)),
        t[:half, half:].mean(axis=(0, 1)),
        t[half:, :half].mean(axis=(0, 1)),
        t[half:, half:].mean(axis=(0, 1)),
    ]
    return np.stack(quadrants).reshape(-1)


def dual_ridge_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    """Mathematically identical to sklearn.linear_model.Ridge(alpha,
    fit_intercept=True).fit(X_train, Y_train).predict(X_test), but solved
    in the dual (N x N) form -- W = X^T (X X^T + alpha I)^-1 Y -- rather
    than ever forming the (D_in, D_out) primal coefficient matrix.
    Required here because spatial_grid_mean produces D_in=D_out=262144:
    sklearn's own multi-output Ridge would materialize a
    (262144, 262144) coef_ array (~549 GB), which is infeasible; the
    dual form's largest object is an (N_train, D_out) intermediate,
    trivial at N_train=32."""
    x_mean = X_train.mean(axis=0)
    y_mean = Y_train.mean(axis=0)
    Xc = X_train - x_mean
    Yc = Y_train - y_mean
    Xtc = X_test - x_mean
    n = Xc.shape[0]
    K = Xc @ Xc.T  # (n, n)
    alpha_term = np.linalg.solve(K + alpha * np.eye(n), Yc)  # (n, D_out)
    pred_centered = (Xtc @ Xc.T) @ alpha_term  # (n_test, D_out)
    return pred_centered + y_mean


def evaluate(Z_train, Zp_train, Z_test, Zp_test, alpha):
    pred = dual_ridge_predict(Z_train, Zp_train, Z_test, alpha)
    return {"r2": r_squared(pred, Zp_test), "cosine": cosine_sim(pred, Zp_test), "dim": Z_train.shape[1]}


def main():
    recorded = json.loads((REPO_ROOT / "state" / "task_07_result.json").read_text())
    train_ids = recorded["dataset"]["train_scene_ids"]
    test_ids = recorded["dataset"]["test_scene_ids"]
    all_ids = sorted(set(train_ids) | set(test_ids))
    ridge_alpha = recorded["config"]["ridge_alpha"]

    import experiments.geometric_consistency_lib as gclib
    encoder = gclib.build_encoder(pretrained=True, checkpoint=None, device=None, fallback_seed=0)

    results = {}
    for transform_name in ["camera_rotation", "object_translation"]:
        print(f"\n=== {transform_name} ===")
        t0 = time.time()
        raw_tokens = {"orig": {}, "trans": {}}
        for sid in all_ids:
            pair_dir = RENDER_DIR / transform_name / sid
            rgb_o = np.load(pair_dir / "original" / "rgb.npy")
            rgb_t = np.load(pair_dir / "transformed" / "rgb.npy")
            raw_tokens["orig"][sid] = encoder.encode(rgb_o)  # (512, 1024)
            raw_tokens["trans"][sid] = encoder.encode(rgb_t)
        print(f"  encoded {len(all_ids)} scenes x 2 sides in {time.time() - t0:.0f}s")

        pooling_methods = {
            "global_mean": global_mean_pool,
            "spatial_grid_mean": spatial_grid_mean_pool,
            "quadrant_mean": quadrant_mean_pool,
        }
        entry = {}
        for method_name, pool_fn in pooling_methods.items():
            Z = {sid: pool_fn(raw_tokens["orig"][sid]) for sid in all_ids}
            Zp = {sid: pool_fn(raw_tokens["trans"][sid]) for sid in all_ids}
            Z_train = np.stack([Z[sid] for sid in train_ids])
            Zp_train = np.stack([Zp[sid] for sid in train_ids])
            Z_test = np.stack([Z[sid] for sid in test_ids])
            Zp_test = np.stack([Zp[sid] for sid in test_ids])
            metrics = evaluate(Z_train, Zp_train, Z_test, Zp_test, ridge_alpha)
            entry[method_name] = metrics
            print(f"  {method_name:20s} dim={metrics['dim']:7d}  r2={metrics['r2']:.4f}  cosine={metrics['cosine']:.4f}")

        results[transform_name] = entry

    (OUT_DIR / "pooling_diagnostic.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
