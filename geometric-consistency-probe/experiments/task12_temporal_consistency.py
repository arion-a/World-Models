"""Task 12: temporal geometric consistency.

    python -m experiments.task12_temporal_consistency --config configs/experiments/task12_temporal_consistency.yaml

Pipeline (see tasks/12_temporal_consistency.md and
research/CANONICAL_RESEARCH_PROTOCOL.md's "TASK 12" section, which this
module implements verbatim):

    for each of two INDEPENDENT conditions (disjoint scene sets, decided
    by two different base_seed values):
        real_motion:     camera orbits at a fixed deg/sec rate
        static_control:  camera and every object stay motionless

    >=40 sampled SceneStates per condition (generation.scene_sampler
    .sample_scene, via experiments.geometric_consistency_lib.sample_scenes,
    unmodified) -> scene-level train/test split, decided BEFORE any
    rendering -> for every scene, render ONE continuous
    2*window_frames-frame trajectory (generation.motion.generate_trajectory
    + generation.bpy_renderer.render_trajectory -- genuinely new
    infrastructure usage for this task, per "Relationship to previous
    tasks") -> split the single clip into two non-overlapping temporal
    windows (frames [0, window_frames) and [window_frames, 2*window_frames))
    -> encode each window separately with the frozen VJEPAEncoder,
    mean_pool to one (1024,) vector per window -> fit W_T (via
    experiments.geometric_consistency_lib.evaluate_transform, Task 6's
    exact fit/evaluate/baseline logic, unmodified) on TRAIN only ->
    evaluate W_T and the three required controls (persistence, mean,
    shuffled-pairing) on the identical TEST split -> combine both
    conditions' results, compute the required real-motion-vs-static-
    control margin, and write state/task_12_result.json.

This module deliberately does not modify generation/, transforms/,
encoders/, or experiments/geometric_consistency_lib.py -- it only calls
their existing public functions, reusing Task 6/7's shared library
(`gclib`) for everything that is NOT temporal-window-specific (scene
sampling, scene-level splitting, encoder construction, fit+evaluate+
baselines, provenance/reproducibility helpers, the pytest-summary-
parsing test runner).

Because this task doubles the total rendered-frame count per scene
relative to Task 6/7/10/11 (one continuous 2*window_frames-frame clip
per scene per condition, vs. those tasks' 2*num_frames per scene for one
transform pair) and runs TWO full conditions, the real (non-test) run of
this module is split into three CLI-invokable steps
(--condition real_motion / static_control / merge) so each expensive
step fits inside one foreground command's wall-clock budget; run_experiment()
(used by tests and by `--condition all`) still runs both conditions in a
single call when that is fast enough (e.g. a tiny pretrained=False dev
config).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

import experiments.geometric_consistency_lib as gclib
from generation.motion import CameraMotion, Trajectory, generate_trajectory
from generation.scene import SceneState
from transforms.se3 import compose, inverse_rigid, pose_matrix

DEFAULT_RESULT_PATH = "state/task_12_result.json"
# Matches Task 6/7/10/11's own scene-count floor (tasks/06_camera_rotation.md's
# acceptance criterion), applied independently to EACH of this task's two
# conditions (tasks/12_temporal_consistency.md's Dataset requirements).
MIN_SCENES = 40
CONDITIONS = ("real_motion", "static_control")


@dataclass
class Task12Config:
    num_scenes: int = 40
    train_fraction: float = 0.8
    # Two DIFFERENT base seeds -> gclib.sample_scenes's
    # `seed = base_seed * 1_000_003 + i` convention makes the two
    # conditions' scene seed ranges provably disjoint (see
    # test_base_seeds_produce_disjoint_scene_sets) -- the static-clip
    # control is run on its OWN, entirely separate scene set, not a
    # motion-disabled variant of the real_motion scenes (tasks/
    # 12_temporal_consistency.md's Leakage checks: "kept disjoint ... or
    # explicitly documented as sharing scenes" -- this task takes the
    # "kept disjoint" branch).
    base_seed_real_motion: int = 0
    base_seed_static_control: int = 1
    output_dir: str = "experiments/temporal_consistency"
    result_path: str = DEFAULT_RESULT_PATH
    resolution: int = 128
    # Windowing scheme (tasks/12_temporal_consistency.md's Inputs section:
    # "the exact windowing scheme decided and documented explicitly, not
    # left implicit"): each clip is ONE continuous trajectory of
    # 2*window_frames frames; window t1 = frames [0, window_frames),
    # window t2 = frames [window_frames, 2*window_frames) -- adjacent,
    # non-overlapping, contiguous, fixed before any run (never chosen
    # after seeing which split scores best -- a prohibited shortcut).
    window_frames: int = 4
    fps: float = 4.0
    # Chosen so the ground-truth rotation between the two windows' anchor
    # frames (window_frames / fps = 1.0s apart) is exactly 30 degrees --
    # the same fixed rotation magnitude Task 6/7/10/11 use for
    # camera_rotation, for direct comparability.
    orbit_deg_per_sec: float = 30.0
    ridge_alpha: float = 10.0
    shuffled_pairing_seed: int = 0
    num_objects_min: int = 1
    num_objects_max: int = 3
    pretrained: bool = True
    checkpoint: str | None = None  # None -> encoders.vjepa.DEFAULT_CHECKPOINT
    device: str | None = None
    fallback_seed: int = 0


def load_task12_config(path: str | Path | None) -> Task12Config:
    cfg = Task12Config()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


# --- windowing scheme + temporal ground truth (pure numpy, no bpy) ----------


def camera_motion_for(condition: str, cfg: Task12Config) -> CameraMotion:
    if condition == "real_motion":
        return CameraMotion(mode="orbit", orbit_deg_per_sec=cfg.orbit_deg_per_sec)
    if condition == "static_control":
        return CameraMotion(mode="static")
    raise ValueError(f"Unknown condition {condition!r}. Known: {CONDITIONS}")


def compute_relative_camera_transform(trajectory: Trajectory, window_frames: int) -> np.ndarray:
    """Ground-truth SE(3) transform between the two windows' anchor
    frames (frame 0 = window t1's first frame, frame `window_frames` =
    window t2's first frame), read directly off the trajectory's own
    recorded per-frame poses (not re-derived from motion parameters) --
    tasks/12_temporal_consistency.md's Inputs section requirement.
    """
    anchor1 = trajectory.frames[0].camera
    anchor2 = trajectory.frames[window_frames].camera
    pose1 = pose_matrix(anchor1.position, anchor1.rotation_euler)
    pose2 = pose_matrix(anchor2.position, anchor2.rotation_euler)
    return compose(pose2, inverse_rigid(pose1))


def verify_temporal_ground_truth(trajectory: Trajectory, window_frames: int, condition: str) -> None:
    """Required scientific-validity / software check, enforced at render
    time (not just in a separate test): a `condition == "real_motion"`
    trajectory must show the camera actually having moved between the
    two windows' anchor frames, a `condition == "static_control"`
    trajectory must show EVERY frame (camera and every object) bit-for-
    bit unchanged, and in both conditions no object may have moved (this
    task tests camera orbit motion only; object motion is out of scope
    -- tasks/12_temporal_consistency.md's "What this task does NOT
    establish").
    """
    if len(trajectory.frames) != 2 * window_frames:
        raise ValueError(
            f"{condition}: expected {2 * window_frames} frames (2 windows of {window_frames}), "
            f"got {len(trajectory.frames)}"
        )
    frames = trajectory.frames

    ref_objects = frames[0].objects
    for frame in frames[1:]:
        for ref_o, o in zip(ref_objects, frame.objects):
            if not np.allclose(ref_o.position, o.position) or not np.allclose(ref_o.rotation_euler, o.rotation_euler):
                raise ValueError(
                    f"{condition}: object instance {o.instance_id} moved at frame {frame.frame_index} -- "
                    "this task tests camera motion only; object_motions must be empty"
                )

    anchor1, anchor2 = frames[0].camera, frames[window_frames].camera
    camera_moved_between_anchors = not (
        np.allclose(anchor1.position, anchor2.position) and np.allclose(anchor1.rotation_euler, anchor2.rotation_euler)
    )

    if condition == "real_motion":
        if not camera_moved_between_anchors:
            raise ValueError("real_motion: camera did not move between window anchors -- motion_config produced no motion")
    elif condition == "static_control":
        ref_cam = frames[0].camera
        for frame in frames[1:]:
            if not np.allclose(ref_cam.position, frame.camera.position) or not np.allclose(
                ref_cam.rotation_euler, frame.camera.rotation_euler
            ):
                raise ValueError(f"static_control: camera moved at frame {frame.frame_index} -- expected a fully static clip")
    else:
        raise ValueError(f"Unknown condition {condition!r}. Known: {CONDITIONS}")


# --- rendering (needs bpy) ---------------------------------------------------


def render_condition_clips(scenes: list[SceneState], condition: str, cfg: Task12Config) -> Path:
    """Render ONE continuous trajectory per scene for `condition`, verify
    its ground truth, and save the temporal transform record alongside
    the usual ground truth (generation.ground_truth.save_ground_truth,
    reused unmodified).
    """
    from generation.bpy_renderer import render_trajectory
    from generation.ground_truth import save_ground_truth

    out_dir = Path(cfg.output_dir) / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_motion = camera_motion_for(condition, cfg)
    num_frames = 2 * cfg.window_frames

    for scene in scenes:
        trajectory = generate_trajectory(scene, object_motions=None, camera_motion=camera_motion, num_frames=num_frames, fps=cfg.fps)
        verify_temporal_ground_truth(trajectory, cfg.window_frames, condition)
        clip = render_trajectory(scene, trajectory, resolution=cfg.resolution)
        scene_dir = save_ground_truth(scene, trajectory, clip, out_dir, cfg.resolution, dir_name=scene.scene_id)

        transform_matrix = compute_relative_camera_transform(trajectory, cfg.window_frames)
        record = {
            "condition": condition,
            "scene_id": scene.scene_id,
            "window_frames": cfg.window_frames,
            "num_frames": num_frames,
            "anchor_t1_frame_index": 0,
            "anchor_t2_frame_index": cfg.window_frames,
            "transform_matrix": transform_matrix.tolist(),
            "camera_motion": asdict(camera_motion),
        }
        (scene_dir / "temporal_transform.json").write_text(json.dumps(record, indent=2))
    return out_dir


def verify_saved_temporal_transform(scene_dir: Path, condition: str) -> None:
    """Independent re-check of what render_condition_clips wrote to disk,
    mirroring gclib.verify_transform_ground_truth's "re-verify the
    recorded ground truth, don't just trust the writer" pattern.
    """
    record = json.loads((scene_dir / "temporal_transform.json").read_text())
    if record.get("condition") != condition:
        raise ValueError(f"{scene_dir}: expected condition {condition!r}, got {record.get('condition')!r}")
    matrix = np.array(record["transform_matrix"])
    if matrix.shape != (4, 4):
        raise ValueError(f"{scene_dir}: transform_matrix must be 4x4, got shape {matrix.shape}")
    is_identity = np.allclose(matrix, np.eye(4))
    if condition == "real_motion" and is_identity:
        raise ValueError(f"{scene_dir}: real_motion condition recorded an identity transform (no motion)")
    if condition == "static_control" and not is_identity:
        raise ValueError(f"{scene_dir}: static_control condition recorded a non-identity transform (unexpected motion)")


# --- encoding (needs torch/transformers, real weights when pretrained) -----


def encode_condition_windows(scenes: list[SceneState], out_dir: Path, encoder, window_frames: int) -> dict[str, dict[str, np.ndarray]]:
    """Z_t1, Z_t2 per scene: mean_pool(encoder.encode(window_rgb)) for
    each of the two non-overlapping windows sliced out of the single
    saved clip -- only rendered RGB ever reaches the encoder (research/
    RESEARCH_INVARIANTS.md invariant 7); ground-truth poses live only in
    temporal_transform.json / metadata.json, read here never for
    encoding. Keyed "Z"/"Z_prime" (not "Z_t1"/"Z_t2") so the result
    plugs directly into gclib.build_arrays/evaluate_transform unmodified.
    """
    from encoders.vjepa import mean_pool
    from generation.ground_truth import load_ground_truth

    reps: dict[str, dict[str, np.ndarray]] = {}
    for scene in scenes:
        _, rgb, _, _ = load_ground_truth(out_dir, scene.scene_id)
        window1 = rgb[:window_frames]
        window2 = rgb[window_frames : 2 * window_frames]
        reps[scene.scene_id] = {
            "Z": mean_pool(encoder.encode(window1)),
            "Z_prime": mean_pool(encoder.encode(window2)),
        }
    return reps


def temporal_pixel_diff_stats(scenes: list[SceneState], out_dir: Path, window_frames: int) -> dict:
    """Mean/std absolute per-pixel RGB difference between window t1 and
    window t2, per scene -- the required scientific-QA check ruling out
    "simple adjacent-frame visual similarity ... as an alternative
    explanation for any positive result" (tasks/12_temporal_consistency.md's
    Required scientific-validity tests), computed straight from rendered
    RGB, never from the encoder's representation. For static_control this
    must be (near-)zero by construction (verify_temporal_ground_truth
    already enforces bit-identical frames); reported for both conditions
    so that guarantee is visible in the result, not just assumed.
    """
    from generation.ground_truth import load_ground_truth

    per_scene = []
    for scene in scenes:
        _, rgb, _, _ = load_ground_truth(out_dir, scene.scene_id)
        w1 = rgb[:window_frames].astype(np.float64)
        w2 = rgb[window_frames : 2 * window_frames].astype(np.float64)
        per_scene.append(float(np.abs(w1 - w2).mean()))
    return {
        "mean_abs_pixel_diff": float(np.mean(per_scene)),
        "std_abs_pixel_diff": float(np.std(per_scene)),
        "per_scene_mean_abs_pixel_diff": per_scene,
    }


# --- one condition end to end -----------------------------------------------


def run_condition(condition: str, cfg: Task12Config) -> dict:
    base_seed = cfg.base_seed_real_motion if condition == "real_motion" else cfg.base_seed_static_control
    scenes = gclib.sample_scenes(cfg.num_scenes, base_seed, cfg.num_objects_min, cfg.num_objects_max)
    if len(scenes) < MIN_SCENES:
        raise ValueError(f"Task 12 requires >={MIN_SCENES} scenes per condition, got {len(scenes)} for {condition}")
    scene_ids = [s.scene_id for s in scenes]

    # Split decided BEFORE any rendering (research/RESEARCH_INVARIANTS.md
    # invariants 4, 5; Global Invariants 7-10).
    split = gclib.assign_split(scene_ids, cfg.train_fraction, base_seed)

    out_dir = render_condition_clips(scenes, condition, cfg)
    for scene in scenes:
        verify_saved_temporal_transform(out_dir / scene.scene_id, condition)

    manifest = {
        "condition": condition,
        "num_scenes": len(scenes),
        "base_seed": base_seed,
        "train_fraction": cfg.train_fraction,
        "window_frames": cfg.window_frames,
        "motion_config": asdict(camera_motion_for(condition, cfg)),
        "scenes": [{"scene_id": s.scene_id, "seed": s.seed, "split": split[s.scene_id]} for s in scenes],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    diff_stats = temporal_pixel_diff_stats(scenes, out_dir, cfg.window_frames)

    encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))
    params_before = gclib.snapshot_params(encoder.model)

    reps = encode_condition_windows(scenes, out_dir, encoder, cfg.window_frames)
    frozen_verified = gclib.params_unchanged(params_before, encoder.model)

    train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test = gclib.build_arrays(scenes, split, reps)

    metrics_block = gclib.evaluate_transform(
        f"temporal_{condition}", Z_train, Zp_train, Z_test, Zp_test, cfg.ridge_alpha, cfg.shuffled_pairing_seed
    )

    return {
        "condition": condition,
        "base_seed": base_seed,
        "num_scenes": len(scenes),
        "scene_ids": scene_ids,
        "scene_seeds": [s.seed for s in scenes],
        "train_scene_ids": train_ids,
        "test_scene_ids": test_ids,
        "train_fraction": cfg.train_fraction,
        "encoder": {
            "name": "VJEPAEncoder",
            "checkpoint": encoder.checkpoint,
            "pretrained": encoder_pretrained,
            "frozen": frozen_verified,
            "pooling": "mean_pool",
        },
        "metrics": metrics_block,
        "pixel_diff_stats": diff_stats,
        "out_dir": str(out_dir),
        "manifest_path": str(out_dir / "manifest.json"),
    }


# --- combine both conditions --------------------------------------------


def _scientific_result_text(real: dict, static: dict) -> str:
    real_m, static_m = real["metrics"], static["metrics"]
    real_learned, static_learned = real_m["learned_W_T"], static_m["learned_W_T"]
    real_best_baseline_r2 = max(
        real_m["persistence_baseline"]["r2"], real_m["mean_baseline"]["r2"], real_m["random_pair_control"]["r2"]
    )
    margin_over_baselines = real_learned["r2"] - real_best_baseline_r2
    margin_over_static_control = real_learned["r2"] - static_learned["r2"]

    positive = margin_over_baselines > 0 and margin_over_static_control > 0
    if positive:
        return (
            f"On held-out test scenes of the real_motion condition (camera orbiting at a fixed rate), the linear map "
            f"W_T fit between temporal window t1 and window t2 achieved R^2={real_learned['r2']:.3f} predicting the "
            f"frozen encoder's mean-pooled representation at window t2 from window t1, exceeding the best of the "
            f"three Task-6-style controls (R^2={real_best_baseline_r2:.3f}) by {margin_over_baselines:.3f}, AND "
            f"exceeding the identical protocol's static-clip control (no real motion, same windowing scheme, "
            f"R^2={static_learned['r2']:.3f}) by {margin_over_static_control:.3f}. Because the margin over the "
            f"static-clip control -- not just over the standard baselines -- is positive, this is evidence the "
            f"representation-transition is predictable from genuine camera motion, beyond whatever apparent "
            f"consistency temporal windowing or adjacent-frame visual similarity alone would produce. This is "
            f"evidence of consistency under one motion type (camera orbit) and one windowing scheme only -- it does "
            f"not establish general temporal-dynamics understanding (see DESIGN.md Sec 13)."
        )
    if margin_over_baselines > 0 >= margin_over_static_control:
        return (
            f"On held-out test scenes of the real_motion condition, the linear map W_T fit between temporal window "
            f"t1 and window t2 achieved R^2={real_learned['r2']:.3f}, exceeding the best of the three Task-6-style "
            f"controls (R^2={real_best_baseline_r2:.3f}) by {margin_over_baselines:.3f}, but did NOT exceed the "
            f"identical protocol's static-clip control (R^2={static_learned['r2']:.3f}; margin={margin_over_static_control:.3f}). "
            f"Per tasks/12_temporal_consistency.md, beating the standard baselines is not sufficient here -- a "
            f"positive result requires beating the static-clip control specifically, since the static-clip control is "
            f"the primary tool for ruling out that temporal windowing/adjacent-frame smoothness alone (not genuine "
            f"motion) produced this apparent consistency. This is a valid negative result for the "
            f"'beyond-the-static-control' hypothesis: no measurable temporal consistency attributable to camera "
            f"motion itself was found beyond what windowing alone already gives, under this motion type and "
            f"windowing scheme."
        )
    return (
        f"On held-out test scenes of the real_motion condition, the linear map W_T fit between temporal window t1 "
        f"and window t2 achieved R^2={real_learned['r2']:.3f}, which did not exceed the best of the three "
        f"Task-6-style controls (R^2={real_best_baseline_r2:.3f}; margin={margin_over_baselines:.3f}) and did not "
        f"exceed the static-clip control (R^2={static_learned['r2']:.3f}; margin={margin_over_static_control:.3f}). "
        f"Under this protocol, no measurable linear predictive temporal consistency beyond trivial controls was "
        f"found for camera-orbit motion with this windowing scheme -- a valid negative result, not a task failure."
    )


def assemble_result(cfg: Task12Config, real: dict, static: dict) -> dict:
    real_seeds, static_seeds = set(real["scene_seeds"]), set(static["scene_seeds"])
    if not real_seeds.isdisjoint(static_seeds):
        raise ValueError(
            "real_motion and static_control scene sets are not disjoint -- this would leak train/test assignment "
            "across the two conditions (tasks/12_temporal_consistency.md's Leakage checks)"
        )

    real_m, static_m = real["metrics"], static["metrics"]
    real_best_baseline_r2 = max(
        real_m["persistence_baseline"]["r2"], real_m["mean_baseline"]["r2"], real_m["random_pair_control"]["r2"]
    )
    margins = {
        "real_motion_learned_W_T_minus_own_best_baseline": real_m["learned_W_T"]["r2"] - real_best_baseline_r2,
        "real_motion_learned_W_T_minus_static_control_learned_W_T": real_m["learned_W_T"]["r2"] - static_m["learned_W_T"]["r2"],
    }

    out_dir = Path(cfg.output_dir)
    report_path = out_dir / "report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Task 12 -- temporal geometric consistency\n\n"
        f"- window_frames={cfg.window_frames}, fps={cfg.fps}, orbit_deg_per_sec={cfg.orbit_deg_per_sec}\n"
        f"- real_motion: {real['num_scenes']} scenes (train={len(real['train_scene_ids'])}, test={len(real['test_scene_ids'])})\n"
        f"- static_control: {static['num_scenes']} scenes (train={len(static['train_scene_ids'])}, test={len(static['test_scene_ids'])})\n\n"
        "| condition | method | R^2 | mean cosine sim | mean rel. L2 err |\n"
        "|---|---|---|---|---|\n"
        + "".join(
            f"| {cond_name} | {method} | {block['r2']:.4f} | {block['mean_cosine_similarity']:.4f} | {block['mean_relative_l2_error']:.4f} |\n"
            for cond_name, m in (("real_motion", real_m), ("static_control", static_m))
            for method, block in m.items()
        )
        + f"\nmargins: {json.dumps(margins, indent=2)}\n"
    )

    return {
        "task": 12,
        "implementation_status": "COMPLETE",
        "scientific_result": _scientific_result_text(real, static),
        "windowing_scheme": {
            "description": (
                "Each clip is ONE continuous 2*window_frames-frame trajectory; window t1 = frames "
                "[0, window_frames), window t2 = frames [window_frames, 2*window_frames) -- non-overlapping, "
                "contiguous, adjacent, fixed before any run (never chosen after seeing which split scores best)."
            ),
            "window_frames": cfg.window_frames,
            "num_frames": 2 * cfg.window_frames,
            "fps": cfg.fps,
            "anchor_t1_frame_index": 0,
            "anchor_t2_frame_index": cfg.window_frames,
        },
        "motion_config": {
            "real_motion": asdict(camera_motion_for("real_motion", cfg))
            | {"expected_azimuth_deg_between_window_anchors": cfg.orbit_deg_per_sec * cfg.window_frames / cfg.fps},
            "static_control": asdict(camera_motion_for("static_control", cfg)),
            "object_motions": None,
        },
        "dataset": {
            "real_motion": {
                "num_scenes": real["num_scenes"],
                "base_seed": real["base_seed"],
                "train_scene_ids": real["train_scene_ids"],
                "test_scene_ids": real["test_scene_ids"],
                "train_fraction": cfg.train_fraction,
            },
            "static_control": {
                "num_scenes": static["num_scenes"],
                "base_seed": static["base_seed"],
                "train_scene_ids": static["train_scene_ids"],
                "test_scene_ids": static["test_scene_ids"],
                "train_fraction": cfg.train_fraction,
            },
            "conditions_scene_disjointness": (
                "real_motion and static_control use two entirely different sampled scene sets (different base_seed "
                "-> disjoint scene-seed ranges under gclib.sample_scenes's seed = base_seed*1_000_003 + i convention), "
                "verified pairwise disjoint at assembly time -- the static-clip control is NOT a motion-disabled "
                "variant of the real_motion scenes, so no scene can appear as train in one condition and test in "
                "the other."
            ),
        },
        "fitting": {"method": "ridge", "alpha": cfg.ridge_alpha},
        "metrics": {"real_motion": real_m, "static_clip_control": static_m},
        "static_clip_control": static_m,
        "margins": margins,
        "pixel_diff_stats": {"real_motion": real["pixel_diff_stats"], "static_control": static["pixel_diff_stats"]},
        "encoder": real["encoder"],
        "tests": {"passed": 0, "failed": 0},
        "artifacts": [real["manifest_path"], static["manifest_path"], str(report_path)],
        "config": asdict(cfg),
        "seed": cfg.base_seed_real_motion,
        "software_versions": gclib.software_versions(),
    }


def run_experiment(cfg: Task12Config) -> dict:
    real = run_condition("real_motion", cfg)
    static = run_condition("static_control", cfg)
    return assemble_result(cfg, real, static)


def run_existing_test_suite(repo_root: str | Path) -> dict:
    return gclib.run_existing_test_suite(repo_root)


def main():
    parser = argparse.ArgumentParser(description="Run the Task 12 temporal geometric consistency experiment.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--skip_tests", action="store_true", help="Skip running the existing test suite before writing the result")
    parser.add_argument(
        "--condition",
        choices=["all", "real_motion", "static_control", "merge"],
        default="all",
        help="Run one expensive condition, merge previously-computed partials, or (default) do everything in one call.",
    )
    parser.add_argument("--partial_dir", type=str, default=None, help="Where per-condition partial JSON results are read/written")
    args = parser.parse_args()

    cfg = load_task12_config(args.config)
    partial_dir = Path(args.partial_dir or cfg.output_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    real_partial_path = partial_dir / "real_motion_partial.json"
    static_partial_path = partial_dir / "static_control_partial.json"

    real = static = None
    if args.condition in ("all", "real_motion"):
        real = run_condition("real_motion", cfg)
        real_partial_path.write_text(json.dumps(real, indent=2))
        print(f"real_motion partial written to {real_partial_path}")
        if args.condition == "real_motion":
            return
    if args.condition in ("all", "static_control"):
        static = run_condition("static_control", cfg)
        static_partial_path.write_text(json.dumps(static, indent=2))
        print(f"static_control partial written to {static_partial_path}")
        if args.condition == "static_control":
            return
    if args.condition == "merge":
        real = json.loads(real_partial_path.read_text())
        static = json.loads(static_partial_path.read_text())

    result = assemble_result(cfg, real, static)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = run_existing_test_suite(repo_root)

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Task 12 result written to {result_path}")
    print(json.dumps(result["margins"], indent=2))


if __name__ == "__main__":
    main()
