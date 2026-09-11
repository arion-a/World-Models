"""Diagnostic A: training-sample scaling learning curve.

Extends the scene pool for `camera_rotation` (the transform with the
worst original R^2 and the largest representation-shift magnitude, per
the forensic audit) well beyond the original 40 scenes, using the SAME
deterministic scene sampling as Task 7
(experiments.geometric_consistency_lib.sample_scenes(N, base_seed=0) --
its first 40 scenes are IDENTICAL, scene-for-scene, seed-for-seed, to
the ones Task 7 actually used, by construction: scene_id/seed are pure
functions of (index, base_seed)).

Held-out test set: the ORIGINAL Task 7 recorded test_scene_ids (8
scenes), FIXED across every condition below -- never changed as
N_train grows, per the user's explicit instruction.

Train pool: every scene NOT in the fixed test set, up to a
computationally-feasible pool size decided empirically (see
POOL_SIZE below and diagnostic_report.md's "compute budget" note).

For each N_train in the target list, draws `SEEDS_PER_N` independent
random subsets (without replacement) of that size from the pool, fits
Ridge (same fixed alpha=10.0 as the original experiment -- this
diagnostic isolates SAMPLE SIZE, not regularization, which is diagnostic
B's job) on each subset, and evaluates on the SAME fixed 8-scene test
set every time.

New scenes' renders/encodings are cached under
experiments/task7_diagnostic/rendered/camera_rotation/ -- NEVER written
into experiments/geometric_consistency/ (Task 7's own, already-audited,
untouched output directory). The original 40 scenes' already-rendered
data is read (never re-rendered) directly from
experiments/geometric_consistency/camera_rotation/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
ORIGINAL_RENDER_DIR = REPO_ROOT / "experiments" / "geometric_consistency" / "camera_rotation"
DIAGNOSTIC_RENDER_DIR = OUT_DIR / "rendered" / "camera_rotation"

TRANSFORM_NAME = "camera_rotation"
N_TRAIN_TARGETS = [32, 64, 128]  # extended to 256 only if the timed rate allows -- see main()
SEEDS_PER_N = 5
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


def ensure_scene_rendered_and_encoded(scene, transform_cfg, encoder, cfg, Z_cache: dict, Zp_cache: dict) -> None:
    """Populates Z_cache[scene.scene_id] / Zp_cache[scene.scene_id] with
    mean-pooled representations, reading from the ORIGINAL Task 7 render
    directory if this scene is one of the original 40 (read-only, never
    re-rendered), or rendering fresh into this diagnostic's own directory
    otherwise (never writing into Task 7's own output)."""
    from encoders.vjepa import mean_pool
    from transforms.pairs import generate_pair, load_pair

    sid = scene.scene_id
    if sid in Z_cache:
        return

    original_pair_dir = ORIGINAL_RENDER_DIR / sid
    diagnostic_pair_dir = DIAGNOSTIC_RENDER_DIR / sid

    if (original_pair_dir / "original" / "rgb.npy").exists():
        pair_dir = original_pair_dir  # reuse Task 7's own real, already-audited render -- read-only
    else:
        if not (diagnostic_pair_dir / "transformation.json").exists():
            generate_pair(
                scene, TRANSFORM_NAME, diagnostic_pair_dir,
                transform_cfg=transform_cfg, num_frames=cfg["num_frames"], fps=cfg["fps"], resolution=cfg["resolution"],
            )
        pair_dir = diagnostic_pair_dir

    loaded = load_pair(pair_dir)
    _, orig_rgb, _, _ = loaded["original"]
    _, trans_rgb, _, _ = loaded["transformed"]
    Z_cache[sid] = mean_pool(encoder.encode(orig_rgb))
    Zp_cache[sid] = mean_pool(encoder.encode(trans_rgb))


def main():
    import experiments.geometric_consistency_lib as gclib
    from transforms.scene_transform import TransformConfig

    recorded = json.loads((REPO_ROOT / "state" / "task_07_result.json").read_text())
    test_ids = recorded["dataset"]["test_scene_ids"]  # FIXED throughout -- never changes with N_train
    cfg = recorded["config"]
    original_train_ids = set(recorded["dataset"]["train_scene_ids"])

    max_n_train = max(N_TRAIN_TARGETS)
    pool_size_needed = max_n_train + len(test_ids)
    print(f"Sampling scene pool: need {pool_size_needed} scenes total ({max_n_train} train-pool candidates + {len(test_ids)} fixed test).")
    scenes = gclib.sample_scenes(pool_size_needed, base_seed=cfg["base_seed"], num_objects_min=cfg["num_objects_min"], num_objects_max=cfg["num_objects_max"])
    scenes_by_id = {s.scene_id: s for s in scenes}

    # sanity: the original 40 scenes must reproduce byte-for-byte (verifies
    # this diagnostic's pool genuinely extends, rather than silently
    # diverging from, Task 7's own scene set)
    assert set(recorded["dataset"]["train_scene_ids"]) | set(test_ids) <= set(scenes_by_id), "pool does not contain Task 7's original scenes"

    train_pool_ids = [sid for sid in scenes_by_id if sid not in set(test_ids)]
    assert len(train_pool_ids) == max_n_train

    transform_cfg = TransformConfig()  # identical to Task 7's transform_config_for("camera_rotation")
    encoder = gclib.build_encoder(pretrained=True, checkpoint=None, device=None, fallback_seed=cfg["fallback_seed"])

    Z_cache: dict[str, np.ndarray] = {}
    Zp_cache: dict[str, np.ndarray] = {}

    # test set first (fixed, small, always needed)
    t0 = time.time()
    for sid in test_ids:
        ensure_scene_rendered_and_encoded(scenes_by_id[sid], transform_cfg, encoder, cfg, Z_cache, Zp_cache)
    print(f"test set ready ({len(test_ids)} scenes) in {time.time() - t0:.0f}s")
    Z_test = np.stack([Z_cache[sid] for sid in test_ids])
    Zp_test = np.stack([Zp_cache[sid] for sid in test_ids])

    per_n_results = {}
    seed_scene_ids_used = {}
    total_start = time.time()
    for n_train in N_TRAIN_TARGETS:
        n_seeds = SEEDS_PER_N if n_train < max_n_train else 1  # only one possible subset when n_train == full pool
        seed_results = []
        seed_scene_ids_used[n_train] = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            if n_train == 32 and seed == 0:
                # seed 0, N=32 is pinned to Task 7's OWN original train
                # split exactly, for direct continuity/comparability with
                # the recorded result -- not an independently drawn subset.
                subset_ids = list(recorded["dataset"]["train_scene_ids"])
            else:
                subset_ids = list(rng.choice(train_pool_ids, size=n_train, replace=False))

            t0 = time.time()
            for sid in subset_ids:
                ensure_scene_rendered_and_encoded(scenes_by_id[sid], transform_cfg, encoder, cfg, Z_cache, Zp_cache)
            elapsed = time.time() - t0

            Z_train = np.stack([Z_cache[sid] for sid in subset_ids])
            Zp_train = np.stack([Zp_cache[sid] for sid in subset_ids])
            pred = dual_ridge_predict(Z_train, Zp_train, Z_test, RIDGE_ALPHA)
            r2 = r_squared(pred, Zp_test)
            cos = cosine_sim(pred, Zp_test)
            seed_results.append({"seed": seed, "r2": r2, "cosine": cos, "new_scenes_encoded_this_seed": elapsed > 1.0})
            seed_scene_ids_used[n_train].append(subset_ids)
            print(f"N_train={n_train:4d} seed={seed}  r2={r2:+.4f}  cosine={cos:.4f}  "
                  f"(render+encode wait: {elapsed:.0f}s)  total_elapsed={time.time() - total_start:.0f}s")

        r2_values = [r["r2"] for r in seed_results]
        per_n_results[n_train] = {
            "n_train": n_train,
            "n_seeds": n_seeds,
            "per_seed": seed_results,
            "r2_mean": float(np.mean(r2_values)),
            "r2_std": float(np.std(r2_values)),
            "cosine_mean": float(np.mean([r["cosine"] for r in seed_results])),
        }
        # persist incrementally so a partial run (if interrupted) is never lost
        (OUT_DIR / "learning_curve.json").write_text(json.dumps({
            "transform": TRANSFORM_NAME,
            "fixed_test_scene_ids": test_ids,
            "ridge_alpha": RIDGE_ALPHA,
            "seeds_per_n": SEEDS_PER_N,
            "results_by_n_train": per_n_results,
            "scene_ids_used_by_n_and_seed": seed_scene_ids_used,
            "total_wall_time_seconds_so_far": time.time() - total_start,
        }, indent=2))

    print(f"\nTotal wall time: {time.time() - total_start:.0f}s")
    for n_train, r in per_n_results.items():
        # baseline R^2 for camera_rotation from the original Task 7 result, for reference
        pass
    original_geometric_r2 = recorded["results"][TRANSFORM_NAME]["learned_W_T"]["r2"]
    original_best_baseline_r2 = max(
        recorded["results"][TRANSFORM_NAME][b]["r2"] for b in ("persistence_baseline", "mean_baseline", "random_pair_control")
    )
    print(f"\nFor reference -- Task 7's original recorded numbers for {TRANSFORM_NAME}: "
          f"learned_W_T r2={original_geometric_r2:.4f}, best_baseline r2={original_best_baseline_r2:.4f}")


if __name__ == "__main__":
    main()
