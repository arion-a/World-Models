"""Diagnostic C: temporal context sweep.

Compares camera_rotation equivariance at num_frames in {4, 16, 32, 64}
(4 already exists as Task 7's own real cached data; 16/32/64 are
rendered fresh here), holding the scene set, transform config, and
train/test split identical across all four conditions -- only the
number of frames rendered/encoded per clip changes.

Compute-budget note (recorded here, not hidden): rendering+encoding a
64-frame clip costs roughly 6x a 4-frame clip's wall time (measured:
~112s/side at 64 frames vs ~38s/side at 4 frames on this machine's CPU
-- see diagnostic_report.md's compute-budget section), and this
diagnostic's purpose (isolating temporal context, not sample size) does
not need the full 40-scene set to be informative. This sweep therefore
uses a SUBSET of TEMPORAL_SUBSET_SIZE scenes -- a fixed sub-split of the
original Task 7 train/test scene_ids, not a freshly drawn one -- rather
than silently reducing scope without saying so.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
ORIGINAL_RENDER_DIR = REPO_ROOT / "experiments" / "geometric_consistency" / "camera_rotation"
DIAGNOSTIC_RENDER_DIR = OUT_DIR / "rendered_temporal"

TRANSFORM_NAME = "camera_rotation"
FRAME_COUNTS = [4, 16, 32, 64]
TEMPORAL_SUBSET_TRAIN = 9   # subset of the original 32 train scenes
TEMPORAL_SUBSET_TEST = 3    # subset of the original 8 test scenes
RIDGE_ALPHA = 10.0


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


def dual_ridge_predict(X_train, Y_train, X_test, alpha):
    x_mean = X_train.mean(axis=0)
    y_mean = Y_train.mean(axis=0)
    Xc, Yc, Xtc = X_train - x_mean, Y_train - y_mean, X_test - x_mean
    n = Xc.shape[0]
    K = Xc @ Xc.T
    alpha_term = np.linalg.solve(K + alpha * np.eye(n), Yc)
    return (Xtc @ Xc.T) @ alpha_term + y_mean


def main():
    import experiments.geometric_consistency_lib as gclib
    from encoders.vjepa import mean_pool
    from transforms.pairs import generate_pair, load_pair
    from transforms.scene_transform import TransformConfig

    recorded = json.loads((REPO_ROOT / "state" / "task_07_result.json").read_text())
    cfg = recorded["config"]
    train_subset = recorded["dataset"]["train_scene_ids"][:TEMPORAL_SUBSET_TRAIN]
    test_subset = recorded["dataset"]["test_scene_ids"][:TEMPORAL_SUBSET_TEST]
    all_subset = train_subset + test_subset
    print(f"Using a {len(train_subset)}-train / {len(test_subset)}-test FIXED subset of Task 7's original scenes: {all_subset}")

    scenes = gclib.sample_scenes(40, base_seed=cfg["base_seed"], num_objects_min=cfg["num_objects_min"], num_objects_max=cfg["num_objects_max"])
    scenes_by_id = {s.scene_id: s for s in scenes}
    transform_cfg = TransformConfig()
    encoder = gclib.build_encoder(pretrained=True, checkpoint=None, device=None, fallback_seed=cfg["fallback_seed"])

    results = {}
    total_start = time.time()
    for nf in FRAME_COUNTS:
        Z, Zp = {}, {}
        t0 = time.time()
        for sid in all_subset:
            if nf == 4:
                pair_dir = ORIGINAL_RENDER_DIR / sid  # reuse Task 7's own real cached 4-frame render, read-only
            else:
                pair_dir = DIAGNOSTIC_RENDER_DIR / f"nf{nf}" / sid
                if not (pair_dir / "transformation.json").exists():
                    generate_pair(
                        scenes_by_id[sid], TRANSFORM_NAME, pair_dir,
                        transform_cfg=transform_cfg, num_frames=nf, fps=cfg["fps"], resolution=cfg["resolution"],
                    )
            loaded = load_pair(pair_dir)
            _, orig_rgb, _, _ = loaded["original"]
            _, trans_rgb, _, _ = loaded["transformed"]
            assert orig_rgb.shape[0] == nf, f"{pair_dir}: expected {nf} frames, got {orig_rgb.shape[0]}"
            Z[sid] = mean_pool(encoder.encode(orig_rgb))
            Zp[sid] = mean_pool(encoder.encode(trans_rgb))
        elapsed = time.time() - t0

        Z_train = np.stack([Z[sid] for sid in train_subset])
        Zp_train = np.stack([Zp[sid] for sid in train_subset])
        Z_test = np.stack([Z[sid] for sid in test_subset])
        Zp_test = np.stack([Zp[sid] for sid in test_subset])
        pred = dual_ridge_predict(Z_train, Zp_train, Z_test, RIDGE_ALPHA)
        r2 = r_squared(pred, Zp_test)
        cos = cosine_sim(pred, Zp_test)

        # also record persistence baseline at this frame count, for reference
        persistence_r2 = r_squared(Z_test, Zp_test)

        results[nf] = {
            "num_frames": nf,
            "n_train": len(train_subset),
            "n_test": len(test_subset),
            "wall_time_seconds": elapsed,
            "learned_W_T_r2": r2,
            "learned_W_T_cosine": cos,
            "persistence_baseline_r2": persistence_r2,
        }
        print(f"num_frames={nf:3d}  r2={r2:+.4f}  cosine={cos:.4f}  persistence_r2={persistence_r2:+.4f}  "
              f"(this frame count took {elapsed:.0f}s, total_elapsed={time.time() - total_start:.0f}s)")

        (OUT_DIR / "temporal_context.json").write_text(json.dumps({
            "transform": TRANSFORM_NAME,
            "train_scene_ids": train_subset,
            "test_scene_ids": test_subset,
            "ridge_alpha": RIDGE_ALPHA,
            "results_by_num_frames": results,
        }, indent=2))

    print(f"\nTotal wall time: {time.time() - total_start:.0f}s")


if __name__ == "__main__":
    main()
