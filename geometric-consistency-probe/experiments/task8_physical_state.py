"""Task 8: physical-state accessibility.

    python -m experiments.task8_physical_state --config configs/experiments/task8_physical_state.yaml

Pipeline (see tasks/08_physical_state.md and
research/CANONICAL_RESEARCH_PROTOCOL.md's "TASK 8" section, which this
module implements verbatim -- nothing here invents a new protocol):

    Reuse Tasks 6-7's exact scene-sampling convention (>=40 scenes,
    base_seed 0, 1-3 objects/scene) and the identical scene-level
    train/test split, decided the same way as
    experiments.geometric_consistency_lib.assign_split
    -> reuse the ALREADY-RENDERED "original" side of each scene from
       Task 7's cached artifacts (experiments/geometric_consistency/<a
       transform>/<scene_id>/original/), falling back to a fresh render
       via transforms.pairs.generate_pair if that cache is not present
       in this checkout
    -> encode every scene's original render with the SAME frozen
       VJEPAEncoder wrapper Tasks 6-7 use, mean_pool to a (1024,) vector
    -> for each of four physical-state variables
       (generation.state_features.{camera_azimuth_deg,
       camera_elevation_deg, camera_distance, primary_object_position_xy}),
       fit a linear ridge probe Z -> y on TRAIN only, evaluate held-out
       R^2/MAE/RMSE on TEST, and the two required controls (shuffled-label,
       mean-prediction)
    -> write state/task_08_result.json

camera_azimuth_deg is probed via its (sin, cos) components, not raw
degrees, because generation/scene_sampler.py samples it uniformly over
the full (0, 360) range -- a genuine wraparound discontinuity a linear
probe cannot handle directly (Implementation requirement 2). The other
three variables are not sampled across a wraparound discontinuity
(camera elevation is confined to (20, 50) degrees, camera distance to
(6, 8) scene units, object position to +-1.6 scene units) so they are
probed as raw scalars/vectors.

This module deliberately does not modify generation/, transforms/,
encoders/, representations/, probes/, metrics/, or baselines/ -- it only
calls their existing public functions (and
experiments/geometric_consistency_lib.py's shared sampling/splitting/
rendering/encoding helpers), per this task's "Relationship to previous
tasks" section.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Ridge

import experiments.geometric_consistency_lib as gclib
from baselines.run_all_baselines import run_probe_baselines
from generation import state_features
from generation.ground_truth import load_ground_truth
from generation.scene import SceneState
from metrics.common import r_squared
from transforms.scene_transform import TransformConfig

DEFAULT_RESULT_PATH = "state/task_08_result.json"
# Same scene-count discipline as Tasks 6-7 (tasks/08_physical_state.md's
# "Dataset requirements": reuse Tasks 6-7's scene set/count floor).
MIN_SCENES = 40

# camera_azimuth_deg is required for continuity with V0 (Acceptance
# criterion 1); the other three are generation.state_features's
# remaining existing feature_fns, all named explicitly in this task's
# "Relationship to previous tasks" section (DESIGN.md Sec 10) --
# probing all of them (rather than only the required minimum of two)
# is a deliberate choice made BEFORE looking at any result, so it
# cannot be read as picking-the-best-after-the-fact (Prohibited
# shortcuts).
VARIABLES = ("camera_azimuth_deg", "camera_elevation_deg", "camera_distance", "primary_object_position_xy")

RAW_FEATURE_FNS = {
    "camera_azimuth_deg": state_features.camera_azimuth_deg,
    "camera_elevation_deg": state_features.camera_elevation_deg,
    "camera_distance": state_features.camera_distance,
    "primary_object_position_xy": state_features.primary_object_position_xy,
}

# The one circular variable in VARIABLES -- see module docstring.
CIRCULAR_VARIABLES = ("camera_azimuth_deg",)

POSE_CONVENTION_NOTE = (
    "World frame is right-handed, Z-up (Blender convention; see "
    "generation/COORDINATE_SYSTEM.md). camera_azimuth_deg = "
    "degrees(atan2(camera.position.y, camera.position.x)), i.e. the "
    "angle of the camera's position in the world XY-plane, range "
    "(-180, 180]; camera_elevation_deg = degrees(atan2(z, "
    "hypot(x, y))); camera_distance = ||camera.position||_2, scene "
    "units; primary_object_position_xy = (x, y) of objects[0] (the "
    "object always present regardless of per-scene object count), "
    "scene units. camera_azimuth_deg is sampled uniformly over the "
    "FULL (0, 360) degree range (generation/scene_sampler.py's "
    "camera_azimuth_range_deg default), i.e. genuinely wraps around --"
    " it is therefore probed via its (sin, cos) components "
    "(sin(radians(azimuth)), cos(radians(azimuth))), decoded back with "
    "atan2(sin, cos) only to report an interpretable angular error in "
    "degrees. The other three variables are sampled over ranges with "
    "no wraparound (elevation (20, 50) deg, distance (6, 8) scene "
    "units, object position +-1.6 scene units) and are probed directly "
    "as raw scalars/vectors."
)


def _probe_target(name: str, raw_value) -> np.ndarray:
    """The array actually fed to/predicted by the ridge probe for `name`."""
    if name in CIRCULAR_VARIABLES:
        rad = np.radians(float(raw_value))
        return np.array([np.sin(rad), np.cos(rad)], dtype=np.float32)
    return np.atleast_1d(np.asarray(raw_value, dtype=np.float32))


def _decode_angle_deg(sin_cos: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(sin_cos[:, 0], sin_cos[:, 1]))


def _circular_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Smallest signed difference between two angles in degrees, magnitude in [0, 180]."""
    return np.abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


@dataclass
class Task8Config:
    # Identical defaults to Task 7's (num_scenes, train_fraction,
    # base_seed, num_objects_min/max) so gclib.sample_scenes/assign_split
    # reproduce the EXACT same scene set and split as Tasks 6-7 (Dataset
    # requirements / Train/test protocol: "ideally identical").
    num_scenes: int = 40
    train_fraction: float = 0.8
    base_seed: int = 0
    result_path: str = DEFAULT_RESULT_PATH
    # Directory of already-rendered {scene_id}/original/{rgb.npy,...}
    # from Task 7 -- reused as-is when present so Task 8 probes the
    # IDENTICAL rendered pixels Task 7 used, not merely the same
    # SceneState resampled. Any one of Task 7's six transform
    # subdirectories carries an identical "original" side (the
    # transform only touches "transformed"), so the choice of which is
    # arbitrary; camera_rotation is Task 6/7's first and best-exercised.
    reuse_render_dir: str = "experiments/geometric_consistency/camera_rotation"
    # Used only as a fallback when reuse_render_dir's cache is absent
    # (e.g. a checkout that has not run Task 7's render step) --
    # renders a fresh original/transformed pair per scene via the same
    # transforms.pairs.generate_pair Task 7 uses, keeping only the
    # "original" side.
    own_render_dir: str = "experiments/physical_state/original_renders"
    own_render_transform: str = "camera_rotation"
    resolution: int = 128
    num_frames: int = 4
    fps: float = 4.0
    ridge_alpha: float = 10.0
    shuffled_label_seed: int = 0
    num_objects_min: int = 1
    num_objects_max: int = 3
    pretrained: bool = True
    checkpoint: str | None = None  # None -> encoders.vjepa.DEFAULT_CHECKPOINT
    device: str | None = None
    fallback_seed: int = 0
    # Tolerance for the scene/render alignment cross-check (Required
    # scientific-validity tests: "the label computed from exactly the
    # same SceneState whose render produced the probed Z").
    alignment_tolerance: float = 1e-4


def load_task8_config(path: str | Path | None) -> Task8Config:
    cfg = Task8Config()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


# --- 1-2. scene sampling + scene-level train/test split (no bpy, no torch) --


def sample_scenes(cfg: Task8Config) -> list[SceneState]:
    return gclib.sample_scenes(cfg.num_scenes, cfg.base_seed, cfg.num_objects_min, cfg.num_objects_max)


def assign_split(scene_ids: list[str], train_fraction: float, base_seed: int) -> dict[str, str]:
    return gclib.assign_split(scene_ids, train_fraction, base_seed)


# --- 3. original-side renders: reuse Task 7's cache, or render fresh --------


def _has_cached_originals(render_dir: Path, scenes: list[SceneState]) -> bool:
    return render_dir.exists() and all((render_dir / s.scene_id / "original" / "rgb.npy").exists() for s in scenes)


def ensure_original_renders(scenes: list[SceneState], cfg: Task8Config) -> tuple[Path, str]:
    """Return (render_dir, source) where render_dir/{scene_id}/original/
    holds a rendered clip for every scene in `scenes`. `source` records
    which path was taken, for the result JSON's provenance.
    """
    reuse_dir = Path(cfg.reuse_render_dir)
    if _has_cached_originals(reuse_dir, scenes):
        return reuse_dir, "reused_task7_cache"

    out_dir = Path(cfg.own_render_dir)
    gclib.render_transform_pairs(
        scenes, cfg.own_render_transform, TransformConfig(), out_dir, cfg.num_frames, cfg.fps, cfg.resolution
    )
    return out_dir, "freshly_rendered"


# --- 4. encoding + alignment check (needs torch/transformers) --------------


def build_encoder(cfg: Task8Config):
    return gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)


def check_scene_alignment(scene: SceneState, metadata: dict, tolerance: float) -> float:
    """Required scientific-validity check: the label computed from `scene`
    must describe the SAME physical configuration that produced the
    rendered clip whose metadata is `metadata` -- cross-checked directly
    on camera position rather than assumed from matching scene_ids.
    Returns the max abs component difference (for reporting); raises if
    it exceeds `tolerance`.
    """
    rendered_position = np.array(metadata["camera"]["per_frame_pose"][0]["position"], dtype=np.float64)
    state_position = np.array(scene.camera.position, dtype=np.float64)
    max_diff = float(np.max(np.abs(rendered_position - state_position)))
    if max_diff > tolerance:
        raise ValueError(
            f"scene/render misalignment for {scene.scene_id}: camera position differs by {max_diff} "
            f"(tolerance {tolerance}) -- the probed Z and the label do not describe the same SceneState"
        )
    return max_diff


def encode_originals(
    scenes: list[SceneState], render_dir: Path, encoder, alignment_tolerance: float
) -> tuple[dict[str, np.ndarray], float]:
    """Z per scene: mean_pool(encoder.encode(rgb)) of the original render.

    Only the rendered RGB video ever reaches `encoder.encode` -- no
    ground-truth pose data is passed in (research/RESEARCH_INVARIANTS.md
    invariant 7); metadata.json is read here only to cross-check
    alignment, never to compute Z.
    """
    from encoders.vjepa import mean_pool

    reps: dict[str, np.ndarray] = {}
    max_alignment_diff = 0.0
    for scene in scenes:
        metadata, rgb, _depth, _seg = load_ground_truth(render_dir, f"{scene.scene_id}/original")
        max_alignment_diff = max(max_alignment_diff, check_scene_alignment(scene, metadata, alignment_tolerance))
        reps[scene.scene_id] = mean_pool(encoder.encode(rgb))
    return reps, max_alignment_diff


# --- 5-6. assemble labels + arrays, enforcing scene-level disjointness -----


def build_variable_arrays(
    scenes: list[SceneState], split: dict[str, str], reps: dict[str, np.ndarray], variable: str
) -> dict:
    feature_fn = RAW_FEATURE_FNS[variable]
    train_ids = [s.scene_id for s in scenes if split[s.scene_id] == "train"]
    test_ids = [s.scene_id for s in scenes if split[s.scene_id] == "test"]
    if not set(train_ids).isdisjoint(test_ids):
        raise ValueError("train/test scene leakage detected")

    by_id = {s.scene_id: s for s in scenes}

    def _collect(ids: list[str]):
        Z = np.stack([reps[i] for i in ids])
        raw = np.array([feature_fn(by_id[i]) for i in ids], dtype=np.float64)
        targets = np.stack([_probe_target(variable, feature_fn(by_id[i])) for i in ids])
        return Z, raw, targets

    Z_train, raw_train, y_train = _collect(train_ids)
    Z_test, raw_test, y_test = _collect(test_ids)

    return {
        "train_scene_ids": train_ids,
        "test_scene_ids": test_ids,
        "Z_train": Z_train,
        "Z_test": Z_test,
        "raw_train": raw_train,
        "raw_test": raw_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# --- shuffled-label control (mirrors baselines/shuffled_pairing_baseline.py) -


def shuffled_label_permutation(n: int, seed: int) -> np.ndarray:
    """A genuine permutation of `n` indices -- guards against an identity
    permutation for small `n`, exactly like
    baselines/shuffled_pairing_baseline.py's existing guard, so the
    control cannot silently degenerate into "no shuffling at all"
    (Required software tests / Leakage checks).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n > 1 and np.all(perm == np.arange(n)):
        perm = np.roll(perm, 1)
    return perm


# --- fit + evaluate one variable --------------------------------------------


def _mae_rmse(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def _fit_and_predict(Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray, alpha: float) -> np.ndarray:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Z_train, y_train)
    return model.predict(Z_test)


def evaluate_variable(variable: str, arrays: dict, ridge_alpha: float, shuffled_label_seed: int) -> dict:
    Z_train, y_train = arrays["Z_train"], arrays["y_train"]
    Z_test, y_test = arrays["Z_test"], arrays["y_test"]
    raw_train, raw_test = arrays["raw_train"], arrays["raw_test"]
    is_circular = variable in CIRCULAR_VARIABLES

    def _block(pred: np.ndarray) -> dict:
        mae, rmse = _mae_rmse(pred, y_test)
        block = {"r2": r_squared(pred, y_test), "mae": mae, "rmse": rmse}
        if is_circular:
            decoded = _decode_angle_deg(np.atleast_2d(pred))
            actual = raw_test if raw_test.ndim == 1 else raw_test[:, 0]
            block["angular_error_deg"] = float(np.mean(_circular_diff_deg(decoded, actual)))
        return block

    # Real probe: fit on true (Z_train, y_train), evaluate on TEST only.
    real_pred = _fit_and_predict(Z_train, y_train, Z_test, ridge_alpha)
    real_block = _block(real_pred)

    # Shuffled-label control + mean-prediction baseline: Task 10's
    # consolidated baselines.run_all_baselines.run_probe_baselines
    # (same math as before the Task 10 refactor -- see that function's
    # docstring; this call site owns generating the permutation and the
    # fit+predict step, run_probe_baselines only broadcasts/applies them).
    perm = shuffled_label_permutation(len(y_train), shuffled_label_seed)
    probe_baselines = run_probe_baselines(
        Z_train, y_train, Z_test, y_test,
        predict_fn=lambda zt, yt, zte: _fit_and_predict(zt, yt, zte, ridge_alpha),
        permutation=perm,
    )
    shuffled_block = _block(probe_baselines["shuffled_label"])
    mean_block = _block(probe_baselines["mean"])

    label_diversity = {
        "min": float(np.min(raw_train.tolist() + raw_test.tolist()) if raw_train.ndim == 1 else np.min(np.vstack([raw_train, raw_test]))),
        "max": float(np.max(raw_train.tolist() + raw_test.tolist()) if raw_train.ndim == 1 else np.max(np.vstack([raw_train, raw_test]))),
        "std": float(np.std(np.concatenate([raw_train, raw_test], axis=0) if raw_train.ndim > 1 else np.concatenate([raw_train, raw_test]))),
    }

    return {
        "probe_encoding": "sin_cos" if is_circular else "raw",
        "target_dim": int(y_train.shape[1]),
        "real_probe": real_block,
        "shuffled_label_control": shuffled_block,
        "mean_baseline": mean_block,
        "label_diversity": label_diversity,
    }


# --- scientific-result text -------------------------------------------------


def _exceeds_baselines(metrics_block: dict) -> bool:
    real_r2 = metrics_block["real_probe"]["r2"]
    best_baseline_r2 = max(metrics_block["shuffled_label_control"]["r2"], metrics_block["mean_baseline"]["r2"])
    return real_r2 > best_baseline_r2


def _scientific_result_text(metrics: dict) -> str:
    lines = []
    positive = []
    for name in VARIABLES:
        m = metrics[name]
        best_baseline_r2 = max(m["shuffled_label_control"]["r2"], m["mean_baseline"]["r2"])
        verdict = "exceeded" if _exceeds_baselines(m) else "did not exceed"
        if _exceeds_baselines(m):
            positive.append(name)
        extra = f", angular_error_deg={m['real_probe']['angular_error_deg']:.2f}" if "angular_error_deg" in m["real_probe"] else ""
        lines.append(
            f"{name} ({m['probe_encoding']}): real probe R^2={m['real_probe']['r2']:.3f}{extra} "
            f"{verdict} best of {{shuffled-label, mean}} baselines R^2={best_baseline_r2:.3f}"
        )

    azimuth_positive = "camera_azimuth_deg" in positive
    other_positive = [n for n in positive if n != "camera_azimuth_deg"]
    hypothesis_confirmed = azimuth_positive and len(other_positive) >= 1

    summary = (
        f"Across the {len(VARIABLES)} physical-state variables probed ({', '.join(VARIABLES)}), evaluated with an "
        f"identical scene set, scene-level train/test split, encoder, pooling, and linear ridge probe: "
        f"{len(positive)}/{len(VARIABLES)} variables showed the real probe's held-out R^2 exceeding the best of the "
        f"required shuffled-label and mean-prediction baselines ({', '.join(positive) or 'none'}). "
        + (
            "This confirms the hypothesis that at least camera azimuth and one additional physical-state variable "
            "are linearly decodable from Z above baseline level, under this scene distribution and pooling scheme "
            "-- an accessibility finding, not a claim that the encoder 'understands' these quantities (DESIGN.md Sec 13)."
            if hypothesis_confirmed
            else "This does NOT confirm the hypothesis as stated (camera azimuth plus at least one additional "
            "variable both linearly decodable above baseline) -- a valid negative/mixed result: a low linear-probe "
            "score here is evidence the quantity is not LINEARLY accessible under this setup, not evidence it is "
            "absent from Z in some non-linearly-decodable form (DESIGN.md Sec 13; tasks/08_physical_state.md's "
            "interpretation limits). Reported plainly rather than adjusted (e.g. via a nonlinear probe) to force a "
            "positive-looking result."
        )
    )
    return summary + "\n\nPer-variable detail:\n" + "\n".join(f"- {line}" for line in lines)


def _comparison_table_markdown(metrics: dict) -> str:
    header = "| variable | encoding | real probe R^2 | shuffled-label R^2 | mean-baseline R^2 |\n"
    header += "|---|---|---|---|---|\n"
    rows = []
    for name in VARIABLES:
        m = metrics[name]
        rows.append(
            f"| {name} | {m['probe_encoding']} | {m['real_probe']['r2']:.4f} | "
            f"{m['shuffled_label_control']['r2']:.4f} | {m['mean_baseline']['r2']:.4f} |"
        )
    return header + "\n".join(rows) + "\n"


def run_experiment(cfg: Task8Config) -> dict:
    scenes = sample_scenes(cfg)
    if len(scenes) < MIN_SCENES:
        raise ValueError(f"Task 8 requires >={MIN_SCENES} scenes, got {len(scenes)}")
    scene_ids = [s.scene_id for s in scenes]

    # Split decided the same documented way as Tasks 6-7 (research/
    # RESEARCH_INVARIANTS.md invariants 4, 5; tasks/08_physical_state.md's
    # Train/test protocol: "ideally identical to Tasks 6-7's").
    split = assign_split(scene_ids, cfg.train_fraction, cfg.base_seed)

    render_dir, render_source = ensure_original_renders(scenes, cfg)

    encoder = build_encoder(cfg)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))
    params_before = gclib.snapshot_params(encoder.model)

    reps, max_alignment_diff = encode_originals(scenes, render_dir, encoder, cfg.alignment_tolerance)

    frozen_verified = gclib.params_unchanged(params_before, encoder.model)

    metrics: dict[str, dict] = {}
    reference_train_ids = reference_test_ids = None
    for variable in VARIABLES:
        arrays = build_variable_arrays(scenes, split, reps, variable)
        if reference_train_ids is None:
            reference_train_ids = arrays["train_scene_ids"]
            reference_test_ids = arrays["test_scene_ids"]
        elif arrays["train_scene_ids"] != reference_train_ids or arrays["test_scene_ids"] != reference_test_ids:
            raise ValueError(f"variable '{variable}' does not share the identical scene split with the others")
        metrics[variable] = evaluate_variable(variable, arrays, cfg.ridge_alpha, cfg.shuffled_label_seed)

    comparison_table = _comparison_table_markdown(metrics)
    out_dir = Path("experiments/physical_state")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text(
        "# Task 8 -- physical-state accessibility\n\n"
        f"- scenes: {len(scenes)} (train={len(reference_train_ids)}, test={len(reference_test_ids)})\n"
        f"- render source: {render_source} ({render_dir})\n"
        f"- encoder pretrained: {encoder_pretrained}\n"
        f"- frozen verified (params bit-identical before/after encoding): {frozen_verified}\n"
        f"- max scene/render alignment diff: {max_alignment_diff}\n\n"
        "## R^2 comparison (real probe vs. required controls)\n\n" + comparison_table + "\n## Pose/angle convention\n\n"
        f"{POSE_CONVENTION_NOTE}\n"
    )

    manifest = {
        "num_scenes": len(scenes),
        "base_seed": cfg.base_seed,
        "train_fraction": cfg.train_fraction,
        "scenes": [{"scene_id": s.scene_id, "seed": s.seed, "split": split[s.scene_id]} for s in scenes],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    result = {
        "task": 8,
        "implementation_status": "COMPLETE",
        "scientific_result": _scientific_result_text(metrics),
        "variables_probed": list(VARIABLES),
        "pose_convention": POSE_CONVENTION_NOTE,
        "dataset": {
            "num_scenes": len(scenes),
            "train_scene_ids": reference_train_ids,
            "test_scene_ids": reference_test_ids,
            "train_fraction": cfg.train_fraction,
            "base_seed": cfg.base_seed,
            "reused_task6_7_scene_sampling_convention": True,
            "render_source": render_source,
            "render_dir": str(render_dir),
        },
        "encoder": {
            "name": "VJEPAEncoder",
            "checkpoint": encoder.checkpoint,
            "pretrained": encoder_pretrained,
            "frozen": frozen_verified,
            "pooling": "mean_pool",
        },
        "fitting": {"method": "ridge", "alpha": cfg.ridge_alpha},
        "metrics": metrics,
        "scene_render_alignment_check": {
            "verified": True,
            "max_camera_position_abs_diff": max_alignment_diff,
            "tolerance": cfg.alignment_tolerance,
            "note": (
                "Camera position from each scene's own SceneState (used to compute the probe label) compared "
                "directly against the camera position recorded in that scene's rendered clip's metadata.json "
                "(used to compute Z) -- confirms the label and the representation describe the same physical "
                "configuration, not merely the same scene_id."
            ),
        },
        "comparison_table_markdown": comparison_table,
        "shuffled_label_seed": cfg.shuffled_label_seed,
        "tests": {"passed": 0, "failed": 0},
        "artifacts": [str(manifest_path), str(report_path)],
        "config": asdict(cfg),
        "seed": cfg.base_seed,
        "software_versions": gclib.software_versions(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the Task 8 physical-state accessibility experiment.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument(
        "--skip_tests", action="store_true", help="Skip running the existing test suite before writing the result"
    )
    args = parser.parse_args()

    cfg = load_task8_config(args.config)
    result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = gclib.run_existing_test_suite(repo_root)

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Task 8 result written to {result_path}")
    print(result["comparison_table_markdown"])


if __name__ == "__main__":
    main()
