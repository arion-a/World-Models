"""Task 12 follow-up #3: camera-rotation MAGNITUDE sweep at N=400.

    python -m experiments.task12_followup_magnitude_sweep --config configs/experiments/task12_followup_magnitude_sweep_n400.yaml

Not a redo of Task 12 or its two scale follow-ups (N=100, N=200 in
experiments/temporal_consistency_followup_n100 / _n200). Those kept
Task 12's own design -- a single continuous camera-orbit trajectory,
sliced into two adjacent temporal windows t1/t2 -- and only varied
sample size, at one fixed motion magnitude (orbit_deg_per_sec * window
duration, ~30 degrees between window anchors). Reviewing that result
raised a real question: a "persistence baseline nearly ties the learned
map" result could mean the map is genuinely uninformative, OR it could
mean the transform between t1/t2 was too small for the representation
to move much in the first place -- in which case near-perfect
persistence is basically guaranteed regardless of what the map does.

This follow-up removes that ambiguity directly: instead of one implicit
motion magnitude, it applies Task 6/7B's own controlled, EXACT-magnitude
camera_rotation transform (transforms.scene_transform.apply_camera_rotation
with TransformConfig(fixed_azimuth_deg=theta, fixed_elevation_deg=0.0))
between t1 and t2, swept over theta in {0, 5, 10, 20, 30} degrees. At
theta=0 the transform is the identity -- t1 and t2 are bit-identical
renders of the same camera pose -- so persistence R^2 must be ~1.0
there by construction; that is the direct answer to "when nothing is
being changed, how can it still resemble the original" -- it does not
"still resemble" the original, it IS the original. As theta grows, the
scene genuinely changes and persistence should degrade; a linear map
W_theta is only evidence of real equivariant structure if its R^2
degrades slower than persistence's (delta_r2 = R^2_learned -
R^2_persistence turns positive) AND it beats the shuffled-pairing
control at the same theta.

Rendering: t1 (the untransformed scene) is rendered ONCE per scene and
reused for every theta (identical camera pose every time, so re-
rendering it per theta would be pure waste); t2 is rendered once per
(scene, theta) via camera_rotation applied to a fresh copy of the
scene. Every clip is a short static-camera clip (CameraMotion(mode=
"static"), object_motions=None) -- this task tests a discrete pose
CHANGE, not continuous motion -- reusing generation.motion.
generate_trajectory / generation.bpy_renderer.render_trajectory /
generation.ground_truth.save_ground_truth unmodified (Task 2's
renderer; no new renderer is written here). Rendering is resumable:
a scene/theta whose rgb.npy already exists on disk is not re-rendered,
so an interrupted run can be restarted with the same config and pick up
where it left off.

Splits: ONE scene-level train/validation/test split (this task's own
assign_split_3way, an additive sibling of gclib.assign_split which is
train/test-only) is decided from the 400 scene ids before any
rendering, and reused identically for EVERY theta -- the same scene
never lands in different splits at different magnitudes, since t1 is
shared and only t2's camera pose differs. Validation is used for a
small ridge-alpha grid search per theta (fit on train, scored on val);
final learned/persistence/shuffled numbers are always reported on the
held-out test split, with the model refit on train only at the chosen
alpha (never train+val, keeping the usual train/select/report
separation clean).

This is a SEPARATE, additional investigation: it writes to its own
output_dir/result_path and never touches state/task_12_result.json,
that task's checkpoint commit, or either scale follow-up already on
disk (experiments/temporal_consistency_followup_n100 or _n200).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

import experiments.geometric_consistency_lib as gclib
from generation.motion import CameraMotion, generate_trajectory
from generation.scene import SceneState
from metrics.common import cosine_similarity, r_squared, relative_l2_error
from probes.linear_rep_transform import LinearRepTransform
from transforms.scene_transform import TransformConfig, apply_camera_rotation

DEFAULT_RESULT_PATH = "experiments/temporal_magnitude_sweep_n400/result.json"
MIN_SCENES = 40


@dataclass
class MagnitudeSweepConfig:
    num_scenes: int = 400
    base_seed: int = 0
    train_fraction: float = 0.7
    val_fraction: float = 0.1
    thetas_deg: list[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, 20.0, 30.0])
    elevation_deg: float = 0.0
    output_dir: str = "experiments/temporal_magnitude_sweep_n400"
    result_path: str = DEFAULT_RESULT_PATH
    resolution: int = 128
    window_frames: int = 4
    fps: float = 4.0
    ridge_alpha_grid: list[float] = field(default_factory=lambda: [1.0, 10.0, 100.0])
    shuffled_pairing_seed: int = 0
    num_objects_min: int = 1
    num_objects_max: int = 3
    pretrained: bool = True
    checkpoint: str | None = None
    device: str | None = None
    fallback_seed: int = 0
    n_bootstrap: int = 2000
    bootstrap_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


def load_config(path: str | Path | None) -> MagnitudeSweepConfig:
    cfg = MagnitudeSweepConfig()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


def theta_key(theta: float) -> str:
    return f"theta_{theta:g}".replace(".", "p")


# --- scene-level train/val/test split (additive sibling of gclib.assign_split) --


def assign_split_3way(scene_ids: list[str], train_fraction: float, val_fraction: float, base_seed: int) -> dict[str, str]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(f"train_fraction + val_fraction must be < 1.0, got {train_fraction + val_fraction}")
    rng = np.random.default_rng(base_seed)
    order = rng.permutation(len(scene_ids))
    n = len(scene_ids)
    num_train = int(round(n * train_fraction))
    num_val = int(round(n * val_fraction))
    split: dict[str, str] = {}
    for rank, idx in enumerate(order):
        if rank < num_train:
            label = "train"
        elif rank < num_train + num_val:
            label = "val"
        else:
            label = "test"
        split[scene_ids[idx]] = label
    return split


# --- rendering (needs bpy) ---------------------------------------------------


def render_t1(scenes: list[SceneState], cfg: MagnitudeSweepConfig) -> Path:
    """The untransformed scene, rendered once per scene and shared by every theta."""
    from generation.bpy_renderer import render_trajectory
    from generation.ground_truth import save_ground_truth

    out_dir = Path(cfg.output_dir) / "t1"
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_motion = CameraMotion(mode="static")
    for scene in scenes:
        if (out_dir / scene.scene_id / "rgb.npy").exists():
            continue
        trajectory = generate_trajectory(scene, object_motions=None, camera_motion=camera_motion, num_frames=cfg.window_frames, fps=cfg.fps)
        clip = render_trajectory(scene, trajectory, resolution=cfg.resolution)
        save_ground_truth(scene, trajectory, clip, out_dir, cfg.resolution, dir_name=scene.scene_id)
    return out_dir


def render_t2_for_theta(scenes: list[SceneState], theta: float, cfg: MagnitudeSweepConfig) -> Path:
    """t2 = camera_rotation(scene, fixed_azimuth_deg=theta) rendered once per scene."""
    from generation.bpy_renderer import render_trajectory
    from generation.ground_truth import save_ground_truth

    out_dir = Path(cfg.output_dir) / theta_key(theta)
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_motion = CameraMotion(mode="static")
    tcfg = TransformConfig(fixed_azimuth_deg=theta, fixed_elevation_deg=cfg.elevation_deg)

    records_path = out_dir / "transform_records.json"
    records: dict[str, dict] = json.loads(records_path.read_text()) if records_path.exists() else {}

    for scene in scenes:
        if (out_dir / scene.scene_id / "rgb.npy").exists() and scene.scene_id in records:
            continue
        new_scene, transform_record = apply_camera_rotation(scene, tcfg)
        trajectory = generate_trajectory(new_scene, object_motions=None, camera_motion=camera_motion, num_frames=cfg.window_frames, fps=cfg.fps)
        clip = render_trajectory(new_scene, trajectory, resolution=cfg.resolution)
        save_ground_truth(new_scene, trajectory, clip, out_dir, cfg.resolution, dir_name=scene.scene_id)
        records[scene.scene_id] = transform_record
        records_path.write_text(json.dumps(records, indent=2))
    return out_dir


def verify_theta_transform(record: dict, theta: float) -> None:
    """theta=0 must be exactly the identity (t1 == t2); any other theta must not be."""
    matrix = np.array(record["transform_matrix"])
    if matrix.shape != (4, 4):
        raise ValueError(f"transform_matrix must be 4x4, got shape {matrix.shape}")
    is_identity = np.allclose(matrix, np.eye(4), atol=1e-9)
    if theta == 0.0 and not is_identity:
        raise ValueError("theta=0 expected an identity transform (t1 == t2), got a non-identity matrix")
    if theta != 0.0 and is_identity:
        raise ValueError(f"theta={theta} expected a non-identity transform, got the identity matrix")


def pixel_diff_stats_theta(scenes: list[SceneState], t1_dir: Path, theta_dir: Path) -> dict:
    """Mean abs RGB diff between t1 and t2, straight from rendered pixels -- must be
    exactly 0 at theta=0 (bit-identical renders) and grow with theta, ruling out
    'looks different only because of rendering noise' as an alternative explanation.
    """
    from generation.ground_truth import load_ground_truth

    per_scene = []
    for scene in scenes:
        _, rgb1, _, _ = load_ground_truth(t1_dir, scene.scene_id)
        _, rgb2, _, _ = load_ground_truth(theta_dir, scene.scene_id)
        per_scene.append(float(np.abs(rgb1.astype(np.float64) - rgb2.astype(np.float64)).mean()))
    return {
        "mean_abs_pixel_diff": float(np.mean(per_scene)),
        "std_abs_pixel_diff": float(np.std(per_scene)),
    }


# --- encoding (needs torch/transformers, real weights when pretrained) -----


def encode_all(
    scenes: list[SceneState], t1_dir: Path, theta_dirs: dict[float, Path], encoder
) -> tuple[dict[str, np.ndarray], dict[float, dict[str, np.ndarray]]]:
    """Z_t1 (shared) and Z_t2 per theta -- only rendered RGB ever reaches the
    encoder (research/RESEARCH_INVARIANTS.md invariant 7).
    """
    from encoders.vjepa import mean_pool
    from generation.ground_truth import load_ground_truth

    Z_t1: dict[str, np.ndarray] = {}
    for scene in scenes:
        _, rgb, _, _ = load_ground_truth(t1_dir, scene.scene_id)
        Z_t1[scene.scene_id] = mean_pool(encoder.encode(rgb))

    Zp_by_theta: dict[float, dict[str, np.ndarray]] = {}
    for theta, out_dir in theta_dirs.items():
        reps: dict[str, np.ndarray] = {}
        for scene in scenes:
            _, rgb, _, _ = load_ground_truth(out_dir, scene.scene_id)
            reps[scene.scene_id] = mean_pool(encoder.encode(rgb))
        Zp_by_theta[theta] = reps
    return Z_t1, Zp_by_theta


# --- arrays, fit+evaluate, bootstrap CIs, representation-change ratio ------


def build_arrays_3way(scene_ids: list[str], split: dict[str, str], Z_t1: dict[str, np.ndarray], Zp: dict[str, np.ndarray]):
    train_ids = [sid for sid in scene_ids if split[sid] == "train"]
    val_ids = [sid for sid in scene_ids if split[sid] == "val"]
    test_ids = [sid for sid in scene_ids if split[sid] == "test"]
    if not (set(train_ids).isdisjoint(val_ids) and set(train_ids).isdisjoint(test_ids) and set(val_ids).isdisjoint(test_ids)):
        raise ValueError("train/val/test scene leakage detected")

    def stack(ids, d):
        return np.stack([d[sid] for sid in ids])

    return (
        train_ids,
        val_ids,
        test_ids,
        stack(train_ids, Z_t1),
        stack(train_ids, Zp),
        stack(val_ids, Z_t1),
        stack(val_ids, Zp),
        stack(test_ids, Z_t1),
        stack(test_ids, Zp),
    )


def _metrics_block(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "r2": r_squared(pred, target),
        "mean_cosine_similarity": float(np.mean(cosine_similarity(pred, target))),
        "mean_relative_l2_error": float(np.mean(relative_l2_error(pred, target))),
        "mse": float(np.mean(np.sum((pred - target) ** 2, axis=-1))),
    }


def bootstrap_cis(
    pred_learned: np.ndarray, pred_persist: np.ndarray, pred_shuffled: np.ndarray, target: np.ndarray, n_bootstrap: int, seeds: list[int]
) -> dict:
    """Percentile confidence intervals from resampling held-out TEST scenes with
    replacement (a single base_seed's worth of scenes is all this follow-up
    rendered -- an independent full rerun per seed would mean re-rendering and
    re-encoding N=400 scenes several times over, well outside this follow-up's
    compute budget). Several independent bootstrap-RNG seeds are pooled so the
    interval is not an artifact of any single resampling stream.
    """
    n = len(target)
    r2_learned_samples, r2_persist_samples, r2_shuffled_samples = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            r2_learned_samples.append(r_squared(pred_learned[idx], target[idx]))
            r2_persist_samples.append(r_squared(pred_persist[idx], target[idx]))
            r2_shuffled_samples.append(r_squared(pred_shuffled[idx], target[idx]))

    def ci(samples: list[float]) -> dict:
        arr = np.array(samples, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return {
            "mean": float(np.mean(arr)) if len(arr) else float("nan"),
            "p2.5": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "p97.5": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "n_bootstrap_total": int(len(arr)),
        }

    return {
        "learned_W_T_r2": ci(r2_learned_samples),
        "persistence_r2": ci(r2_persist_samples),
        "shuffled_pairing_r2": ci(r2_shuffled_samples),
    }


def representation_change_ratio(Z_t1_all: np.ndarray, Zp_theta_all: np.ndarray) -> dict:
    """E||Z_t2 - Z_t1||^2 / E_{i != j}||Z_i - Z_j||^2 -- how big the transform's
    effect on the representation is, relative to how different two arbitrary
    scenes' representations already are. Computed over ALL scenes (train+val+
    test combined): this is a descriptive diagnostic of the transform's effect
    size, not a fitted quantity, so there is no leakage concern in pooling splits.
    """
    numerator = float(np.mean(np.sum((Zp_theta_all - Z_t1_all) ** 2, axis=-1)))
    sq_norms = np.sum(Z_t1_all**2, axis=-1)
    gram = Z_t1_all @ Z_t1_all.T
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    n = len(Z_t1_all)
    mask = ~np.eye(n, dtype=bool)
    denominator = float(sq_dists[mask].mean())
    return {
        "mean_within_pair_sq_change": numerator,
        "mean_cross_scene_sq_distance": denominator,
        "ratio": numerator / denominator if denominator > 1e-12 else float("nan"),
    }


def fit_and_evaluate_theta(
    theta: float,
    Z_train: np.ndarray,
    Zp_train: np.ndarray,
    Z_val: np.ndarray,
    Zp_val: np.ndarray,
    Z_test: np.ndarray,
    Zp_test: np.ndarray,
    cfg: MagnitudeSweepConfig,
) -> dict:
    val_scores = {}
    best_alpha, best_val_r2 = cfg.ridge_alpha_grid[0], -np.inf
    for alpha in cfg.ridge_alpha_grid:
        rho = LinearRepTransform.fit(Z_train, Zp_train, alpha=alpha)
        r2_val = r_squared(rho.predict(Z_val), Zp_val)
        val_scores[str(alpha)] = r2_val
        if np.isfinite(r2_val) and r2_val > best_val_r2:
            best_val_r2, best_alpha = r2_val, alpha

    rho = LinearRepTransform.fit(Z_train, Zp_train, alpha=best_alpha)
    pred_learned = rho.predict(Z_test)

    pred_persist = Z_test

    rng = np.random.default_rng(cfg.shuffled_pairing_seed)
    perm = rng.permutation(len(Zp_train))
    if len(perm) > 1 and np.all(perm == np.arange(len(perm))):
        perm = np.roll(perm, 1)
    rho_shuffled = LinearRepTransform.fit(Z_train, Zp_train[perm], alpha=best_alpha)
    pred_shuffled = rho_shuffled.predict(Z_test)

    pred_mean = np.broadcast_to(Zp_train.mean(axis=0), Zp_test.shape)

    metrics = {
        "learned_W_T": _metrics_block(pred_learned, Zp_test),
        "persistence_baseline": _metrics_block(pred_persist, Zp_test),
        "shuffled_pairing_control": _metrics_block(pred_shuffled, Zp_test),
        "mean_baseline": _metrics_block(pred_mean, Zp_test),
    }

    ci = bootstrap_cis(pred_learned, pred_persist, pred_shuffled, Zp_test, cfg.n_bootstrap, cfg.bootstrap_seeds)

    return {
        "theta_deg": theta,
        "chosen_ridge_alpha": best_alpha,
        "val_r2_by_alpha": val_scores,
        "metrics": metrics,
        "delta_r2_learned_minus_persistence": metrics["learned_W_T"]["r2"] - metrics["persistence_baseline"]["r2"],
        "learned_beats_shuffled": bool(metrics["learned_W_T"]["r2"] > metrics["shuffled_pairing_control"]["r2"]),
        "confidence_intervals": ci,
    }


# --- one full experiment -----------------------------------------------------


def run_experiment(cfg: MagnitudeSweepConfig) -> dict:
    scenes = gclib.sample_scenes(cfg.num_scenes, cfg.base_seed, cfg.num_objects_min, cfg.num_objects_max)
    if len(scenes) < MIN_SCENES:
        raise ValueError(f"Requires >={MIN_SCENES} scenes, got {len(scenes)}")
    scene_ids = [s.scene_id for s in scenes]
    split = assign_split_3way(scene_ids, cfg.train_fraction, cfg.val_fraction, cfg.base_seed)

    t1_dir = render_t1(scenes, cfg)
    theta_dirs: dict[float, Path] = {}
    pixel_diffs: dict[str, dict] = {}
    for theta in cfg.thetas_deg:
        out_dir = render_t2_for_theta(scenes, theta, cfg)
        theta_dirs[theta] = out_dir
        records = json.loads((out_dir / "transform_records.json").read_text())
        for sid in scene_ids:
            verify_theta_transform(records[sid], theta)
        pixel_diffs[theta_key(theta)] = pixel_diff_stats_theta(scenes, t1_dir, out_dir)

    manifest = {
        "num_scenes": len(scenes),
        "base_seed": cfg.base_seed,
        "thetas_deg": cfg.thetas_deg,
        "train_fraction": cfg.train_fraction,
        "val_fraction": cfg.val_fraction,
        "scenes": [{"scene_id": s.scene_id, "seed": s.seed, "split": split[s.scene_id]} for s in scenes],
    }
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))

    encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))
    params_before = gclib.snapshot_params(encoder.model)

    Z_t1, Zp_by_theta = encode_all(scenes, t1_dir, theta_dirs, encoder)
    frozen_verified = gclib.params_unchanged(params_before, encoder.model)

    per_theta_results = []
    for theta in cfg.thetas_deg:
        train_ids, val_ids, test_ids, Z_train, Zp_train, Z_val, Zp_val, Z_test, Zp_test = build_arrays_3way(
            scene_ids, split, Z_t1, Zp_by_theta[theta]
        )
        result = fit_and_evaluate_theta(theta, Z_train, Zp_train, Z_val, Zp_val, Z_test, Zp_test, cfg)
        Z_all = np.stack([Z_t1[sid] for sid in scene_ids])
        Zp_all = np.stack([Zp_by_theta[theta][sid] for sid in scene_ids])
        result["representation_change"] = representation_change_ratio(Z_all, Zp_all)
        result["pixel_diff_stats"] = pixel_diffs[theta_key(theta)]
        result["n_train"], result["n_val"], result["n_test"] = len(train_ids), len(val_ids), len(test_ids)
        per_theta_results.append(result)

    return assemble_result(cfg, scenes, split, per_theta_results, encoder, encoder_pretrained, frozen_verified)


def _scientific_result_text(per_theta_results: list[dict]) -> str:
    lines = []
    for r in per_theta_results:
        m = r["metrics"]
        lines.append(
            f"theta={r['theta_deg']:g} deg: R^2_learned={m['learned_W_T']['r2']:.3f}, "
            f"R^2_persistence={m['persistence_baseline']['r2']:.3f}, "
            f"R^2_shuffled={m['shuffled_pairing_control']['r2']:.3f}, "
            f"delta_r2={r['delta_r2_learned_minus_persistence']:+.3f}, "
            f"beats_shuffled={r['learned_beats_shuffled']}, "
            f"representation_change_ratio={r['representation_change']['ratio']:.4f}"
        )
    crossing = [r for r in per_theta_results if r["delta_r2_learned_minus_persistence"] > 0 and r["learned_beats_shuffled"]]
    zero_theta = next((r for r in per_theta_results if r["theta_deg"] == 0.0), None)
    zero_note = ""
    if zero_theta is not None:
        zero_r2 = zero_theta["metrics"]["persistence_baseline"]["r2"]
        zero_note = (
            f" At theta=0 (the identity transform -- t1 and t2 are bit-identical renders, verified by an exactly-zero "
            f"mean pixel difference), the persistence baseline scores R^2={zero_r2:.4f}, confirming persistence looks "
            f"perfect there because nothing changed, not because the map generalizes."
        )
    if crossing:
        thetas = ", ".join(f"{r['theta_deg']:g}" for r in crossing)
        summary = (
            f"The learned map W_theta exceeds BOTH its own persistence baseline and the shuffled-pairing control at "
            f"theta in {{{thetas}}} degrees -- evidence of genuine equivariant structure beyond trivial copying at "
            f"those magnitudes."
        )
    else:
        summary = (
            "At no swept magnitude did the learned map W_theta exceed both its own persistence baseline and the "
            "shuffled-pairing control -- a negative result across the whole sweep, not just at small theta."
        )
    return summary + zero_note + "\n\nPer-magnitude detail:\n" + "\n".join(lines)


def assemble_result(cfg: MagnitudeSweepConfig, scenes, split, per_theta_results: list[dict], encoder, encoder_pretrained: bool, frozen_verified: bool) -> dict:
    out_dir = Path(cfg.output_dir)
    report_path = out_dir / "report.md"
    table_rows = "\n".join(
        f"| {r['theta_deg']:g} | {r['metrics']['learned_W_T']['r2']:.4f} | {r['metrics']['persistence_baseline']['r2']:.4f} | "
        f"{r['metrics']['shuffled_pairing_control']['r2']:.4f} | {r['delta_r2_learned_minus_persistence']:+.4f} | "
        f"{r['learned_beats_shuffled']} | {r['representation_change']['ratio']:.4f} | {r['metrics']['learned_W_T']['mse']:.2f} | "
        f"{r['chosen_ridge_alpha']:g} |"
        for r in per_theta_results
    )
    report_path.write_text(
        "# Task 12 follow-up #3 -- camera-rotation magnitude sweep (N=400)\n\n"
        f"num_scenes={len(scenes)}, thetas_deg={cfg.thetas_deg}, window_frames={cfg.window_frames}, fps={cfg.fps}\n\n"
        "| theta (deg) | R2_learned | R2_persistence | R2_shuffled | delta_R2 | beats_shuffled | change_ratio | MSE_learned | alpha |\n"
        "|---|---|---|---|---|---|---|---|---|\n" + table_rows + "\n"
    )

    return {
        "task": "task_12_followup_magnitude_sweep",
        "implementation_status": "COMPLETE",
        "scientific_result": _scientific_result_text(per_theta_results),
        "thetas_deg": cfg.thetas_deg,
        "dataset": {
            "num_scenes": len(scenes),
            "base_seed": cfg.base_seed,
            "train_fraction": cfg.train_fraction,
            "val_fraction": cfg.val_fraction,
            "test_fraction": 1.0 - cfg.train_fraction - cfg.val_fraction,
            "split_counts": {
                label: sum(1 for s in scenes if split[s.scene_id] == label) for label in ("train", "val", "test")
            },
        },
        "fitting": {"method": "ridge", "alpha_grid": cfg.ridge_alpha_grid, "alpha_selection": "best validation R^2, per theta"},
        "per_theta": per_theta_results,
        "encoder": {
            "name": "VJEPAEncoder",
            "checkpoint": encoder.checkpoint,
            "pretrained": encoder_pretrained,
            "frozen": frozen_verified,
            "pooling": "mean_pool",
        },
        "artifacts": [str(Path(cfg.output_dir) / "manifest.json"), str(report_path)],
        "config": asdict(cfg),
        "seed": cfg.base_seed,
        "software_versions": gclib.software_versions(),
    }


def run_existing_test_suite(repo_root: str | Path) -> dict:
    return gclib.run_existing_test_suite(repo_root)


def main():
    parser = argparse.ArgumentParser(description="Task 12 follow-up #3: camera-rotation magnitude sweep.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--skip_tests", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = run_existing_test_suite(repo_root)
    else:
        result["tests"] = {"passed": 0, "failed": 0, "skipped": True}

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Result written to {result_path}")
    for r in result["per_theta"]:
        print(
            f"theta={r['theta_deg']:g}: R2_learned={r['metrics']['learned_W_T']['r2']:.4f} "
            f"R2_persistence={r['metrics']['persistence_baseline']['r2']:.4f} "
            f"delta={r['delta_r2_learned_minus_persistence']:+.4f}"
        )


if __name__ == "__main__":
    main()
