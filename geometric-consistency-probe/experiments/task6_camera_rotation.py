"""Task 6: first camera-rotation geometric consistency experiment.

    python -m experiments.task6_camera_rotation --config configs/experiments/task6_camera_rotation.yaml

Pipeline (see tasks/06_camera_rotation.md and
research/CANONICAL_RESEARCH_PROTOCOL.md's "TASK 6" section, which this
module implements verbatim -- nothing here invents a new protocol):

    >=40 sampled SceneStates (generation.scene_sampler.sample_scene)
    -> scene-level train/test split, decided BEFORE any rendering
    -> for every scene, render the original/camera_rotation-transformed
       pair at a FIXED azimuth magnitude (transforms.pairs.generate_pair)
    -> encode both sides with the frozen VJEPAEncoder, mean_pool to a
       single (1024,) vector per clip
    -> fit W_T (probes.linear_rep_transform.LinearRepTransform, via
       metrics.equivariance.evaluate_equivariance) on TRAIN only
    -> evaluate W_T, and the three required controls (persistence, mean,
       shuffled-pairing), on the identical TEST split
    -> write state/task_06_result.json

This module deliberately does not modify generation/, transforms/, or
encoders/ -- it only calls their existing public functions, per Task 6's
"Relationship to previous tasks" section.

As of Task 7 ("Multiple geometric transformations"), the actual
sampling / splitting / rendering+verification / encoding / array /
fit+evaluate logic lives in `experiments/geometric_consistency_lib.py`
(a shared library, generalized to any of the six transforms) and this
module calls into it -- Task 7's "Task 6's script must be refactored
into a per-transform function if it was not already, rather than
copy-pasted per transform." The public names below are kept stable so
this module's own behavior and `tests/test_task6_camera_rotation.py`
are unaffected by the refactor.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

import experiments.geometric_consistency_lib as gclib
from generation.scene import SceneState
from transforms.scene_transform import TransformConfig

TRANSFORM_NAME = "camera_rotation"
DEFAULT_RESULT_PATH = "state/task_06_result.json"
# tasks/06_camera_rotation.md's acceptance criterion #1 ("scene count
# below 40" is an explicit failure condition) -- a module-level constant
# (rather than a literal inline in run_experiment) so tests can exercise
# the full pipeline on a smaller scene count via monkeypatch without
# duplicating run_experiment's own logic.
MIN_SCENES = 40


@dataclass
class Task6Config:
    num_scenes: int = 40
    train_fraction: float = 0.8
    base_seed: int = 0
    output_dir: str = "experiments/camera_rotation"
    result_path: str = DEFAULT_RESULT_PATH
    resolution: int = 128
    num_frames: int = 4
    fps: float = 4.0
    # Fixed-magnitude azimuth (degrees); sign is still randomized per
    # scene by transforms.scene_transform.apply_camera_rotation, so the
    # ROTATION MAGNITUDE is fixed, not the exact SE(3) matrix -- see this
    # module's docstring and tasks/06_camera_rotation.md's Inputs section
    # ("e.g. 30 degrees azimuth ... set explicitly via TransformConfig's
    # existing camera_rotation_azimuth_deg_range field").
    camera_rotation_azimuth_deg: float = 30.0
    ridge_alpha: float = 10.0
    shuffled_pairing_seed: int = 0
    num_objects_min: int = 1
    num_objects_max: int = 3
    pretrained: bool = True
    checkpoint: str | None = None  # None -> encoders.vjepa.DEFAULT_CHECKPOINT
    device: str | None = None
    fallback_seed: int = 0


def load_task6_config(path: str | Path | None) -> Task6Config:
    cfg = Task6Config()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


# --- 1-2. scene sampling + scene-level train/test split (no bpy, no torch) --


def sample_scenes(cfg: Task6Config) -> list[SceneState]:
    """>=40 scenes, one distinct seed each -- delegates to
    experiments.geometric_consistency_lib.sample_scenes (shared with
    Task 7) using the same base_seed*1_000_003 + i convention Task 2
    uses, so the same (num_scenes, base_seed, num_objects_*) reproduces
    exactly the same scene set here as it would there.
    """
    return gclib.sample_scenes(cfg.num_scenes, cfg.base_seed, cfg.num_objects_min, cfg.num_objects_max)


def assign_split(scene_ids: list[str], train_fraction: float, base_seed: int) -> dict[str, str]:
    """Scene-level train/test split, decided before any rendering --
    delegates to experiments.geometric_consistency_lib.assign_split
    (shared with Task 7, so the identical split is reproduced given the
    identical (scene_ids, train_fraction, base_seed))."""
    return gclib.assign_split(scene_ids, train_fraction, base_seed)


# --- 3. rendering (needs bpy) ------------------------------------------------


def render_all_pairs(scenes: list[SceneState], split: dict[str, str], cfg: Task6Config) -> Path:
    """Render the original/camera_rotation pair for every scene (delegates
    to experiments.geometric_consistency_lib.render_transform_pairs,
    shared with Task 7, which also runs _verify_transform_ground_truth's
    generic equivalent per pair) and write a top-level manifest.json
    recording the scene-level split decided in assign_split() -- BEFORE
    this function is ever called.
    """
    out_dir = Path(cfg.output_dir)
    transform_cfg = TransformConfig(
        camera_rotation_azimuth_deg_range=(cfg.camera_rotation_azimuth_deg, cfg.camera_rotation_azimuth_deg)
    )
    gclib.render_transform_pairs(
        scenes, TRANSFORM_NAME, transform_cfg, out_dir, cfg.num_frames, cfg.fps, cfg.resolution
    )

    manifest = {
        "transform": TRANSFORM_NAME,
        "num_scenes": len(scenes),
        "base_seed": cfg.base_seed,
        "train_fraction": cfg.train_fraction,
        "camera_rotation_azimuth_deg": cfg.camera_rotation_azimuth_deg,
        "scenes": [{"scene_id": s.scene_id, "seed": s.seed, "split": split[s.scene_id]} for s in scenes],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def _verify_transform_ground_truth(pair_dir: Path) -> None:
    """Required scientific-validity check, enforced at run time (not just
    in a separate test): the rendered pair's own transformation.json must
    show EXACTLY a camera rotation (camera.position + camera.rotation_euler
    changed, a real 4x4 matrix, nothing else touched) -- guards against a
    future refactor of transforms/scene_transform.py silently changing
    what camera_rotation does out from under this experiment. Thin
    camera_rotation-specific wrapper around
    experiments.geometric_consistency_lib.verify_transform_ground_truth
    (num_objects is irrelevant to camera_rotation's expected variable
    set, so it is passed as 0).
    """
    gclib.verify_transform_ground_truth(pair_dir, TRANSFORM_NAME, num_objects=0)


# --- 4. encoding (needs torch/transformers, real weights when pretrained) ---


def build_encoder(cfg: Task6Config):
    return gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)


def encode_all_pairs(scenes: list[SceneState], out_dir: Path, encoder) -> dict[str, dict[str, np.ndarray]]:
    """Z, Z' per scene: mean_pool(encoder.encode(rgb)) for each side.

    Only the rendered RGB video ever reaches `encoder.encode` -- no
    ground-truth pose/rotation-matrix data is passed in (research/
    RESEARCH_INVARIANTS.md invariant 7); those live only in
    transformation.json / metadata.json, read here only for provenance
    bookkeeping elsewhere, never for encoding. Delegates to
    experiments.geometric_consistency_lib.encode_all_pairs (shared with
    Task 7).
    """
    return gclib.encode_all_pairs(scenes, out_dir, encoder)


# --- assemble train/test arrays, enforcing scene-level disjointness --------


def build_arrays(
    scenes: list[SceneState], split: dict[str, str], reps: dict[str, dict[str, np.ndarray]]
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return gclib.build_arrays(scenes, split, reps)


def _software_versions() -> dict:
    return gclib.software_versions()


def _snapshot_params(model) -> list:
    return gclib.snapshot_params(model)


def _params_unchanged(before: list, model) -> bool:
    return gclib.params_unchanged(before, model)


def _scientific_result_text(learned: dict, persistence: dict, mean_b: dict, random_pair: dict) -> str:
    best_control_r2 = max(persistence["r2"], mean_b["r2"], random_pair["r2"])
    if learned["r2"] > best_control_r2:
        margin = learned["r2"] - best_control_r2
        return (
            f"On held-out test scenes, the linear map W_T fit on train scenes achieved R^2={learned['r2']:.3f} "
            f"predicting the frozen encoder's mean-pooled representation after a fixed-magnitude camera "
            f"rotation, exceeding the best of the three required controls (persistence R^2={persistence['r2']:.3f}, "
            f"mean-transformed-representation R^2={mean_b['r2']:.3f}, shuffled-pairing R^2={random_pair['r2']:.3f}) "
            f"by {margin:.3f}. This indicates the representation exhibits measurable predictive consistency under "
            f"the tested camera rotation, under this scene distribution, this pooling scheme (mean_pool), and this "
            f"rotation magnitude -- it does not establish 3D scene comprehension or an internal camera model in "
            f"the encoder (see DESIGN.md Sec 13)."
        )
    return (
        f"On held-out test scenes, the linear map W_T fit on train scenes achieved R^2={learned['r2']:.3f} "
        f"predicting the frozen encoder's mean-pooled representation after a fixed-magnitude camera rotation, "
        f"which did not exceed the best of the three required controls (persistence R^2={persistence['r2']:.3f}, "
        f"mean-transformed-representation R^2={mean_b['r2']:.3f}, shuffled-pairing R^2={random_pair['r2']:.3f}). "
        f"Under this protocol, no measurable linear predictive consistency beyond trivial controls was found for "
        f"camera rotation -- a valid negative result, not a task failure."
    )


def run_experiment(cfg: Task6Config) -> dict:
    scenes = sample_scenes(cfg)
    if len(scenes) < MIN_SCENES:
        raise ValueError(f"Task 6 requires >={MIN_SCENES} scenes, got {len(scenes)}")
    scene_ids = [s.scene_id for s in scenes]

    # Split decided BEFORE any rendering (research/RESEARCH_INVARIANTS.md
    # invariants 4, 5; tasks/06_camera_rotation.md's Train/test protocol).
    split = assign_split(scene_ids, cfg.train_fraction, cfg.base_seed)

    out_dir = render_all_pairs(scenes, split, cfg)

    encoder = build_encoder(cfg)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))
    params_before = _snapshot_params(encoder.model)

    reps = encode_all_pairs(scenes, out_dir, encoder)

    frozen_verified = _params_unchanged(params_before, encoder.model)

    train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test = build_arrays(scenes, split, reps)

    metrics_block = gclib.evaluate_transform(
        TRANSFORM_NAME, Z_train, Zp_train, Z_test, Zp_test, cfg.ridge_alpha, cfg.shuffled_pairing_seed
    )
    learned = metrics_block["learned_W_T"]
    persistence = metrics_block["persistence_baseline"]
    mean_b = metrics_block["mean_baseline"]
    random_pair = metrics_block["random_pair_control"]

    report_path = out_dir / "report.md"
    report_path.write_text(
        "# Task 6 -- camera_rotation geometric consistency\n\n"
        f"- scenes: {len(scenes)} (train={len(train_ids)}, test={len(test_ids)})\n"
        f"- fixed azimuth magnitude: {cfg.camera_rotation_azimuth_deg} deg\n"
        f"- encoder pretrained: {encoder_pretrained}\n"
        f"- frozen verified (params bit-identical before/after encoding): {frozen_verified}\n\n"
        "| method | R^2 | mean cosine sim | mean rel. L2 err |\n"
        "|---|---|---|---|\n"
        f"| learned W_T | {learned['r2']:.4f} | {learned['mean_cosine_similarity']:.4f} | {learned['mean_relative_l2_error']:.4f} |\n"
        f"| persistence | {persistence['r2']:.4f} | {persistence['mean_cosine_similarity']:.4f} | {persistence['mean_relative_l2_error']:.4f} |\n"
        f"| mean baseline | {mean_b['r2']:.4f} | {mean_b['mean_cosine_similarity']:.4f} | {mean_b['mean_relative_l2_error']:.4f} |\n"
        f"| random-pair control | {random_pair['r2']:.4f} | {random_pair['mean_cosine_similarity']:.4f} | {random_pair['mean_relative_l2_error']:.4f} |\n"
    )

    result = {
        "task": 6,
        "implementation_status": "COMPLETE",
        "scientific_result": _scientific_result_text(learned, persistence, mean_b, random_pair),
        "transform": TRANSFORM_NAME,
        "transform_params": {
            "camera_rotation_azimuth_deg_range": [cfg.camera_rotation_azimuth_deg, cfg.camera_rotation_azimuth_deg],
            "camera_rotation_elevation_deg_range": list(TransformConfig().camera_rotation_elevation_deg_range),
            "note": "azimuth magnitude is fixed; sign is still randomized per scene by apply_camera_rotation",
        },
        "encoder": {
            "name": "VJEPAEncoder",
            "checkpoint": encoder.checkpoint,
            "pretrained": encoder_pretrained,
            "frozen": frozen_verified,
            "pooling": "mean_pool",
        },
        "dataset": {
            "num_scenes": len(scenes),
            "train_scene_ids": train_ids,
            "test_scene_ids": test_ids,
            "train_fraction": cfg.train_fraction,
            "base_seed": cfg.base_seed,
        },
        "fitting": {"method": "ridge", "alpha": cfg.ridge_alpha},
        "metrics": {
            "learned_W_T": learned,
            "persistence_baseline": persistence,
            "mean_baseline": mean_b,
            "random_pair_control": random_pair,
        },
        "tests": {"passed": 0, "failed": 0},
        "artifacts": [str(out_dir / "manifest.json"), str(report_path)],
        "config": asdict(cfg),
        "seed": cfg.base_seed,
        "software_versions": _software_versions(),
    }
    return result


def _parse_pytest_summary(output: str) -> tuple[int, int]:
    """Parse pytest -q's final summary line -- delegates to
    experiments.geometric_consistency_lib.parse_pytest_summary (shared
    with Task 7)."""
    return gclib.parse_pytest_summary(output)


def run_existing_test_suite(repo_root: str | Path) -> dict:
    """Run `pytest -m "not slow" -q` from the repo root and report honest
    pass/fail counts -- required by this task's own instructions, though
    (per those same instructions) an independent process re-runs this
    after Claude finishes and is the actual basis for acceptance, not
    this self-report. Delegates to
    experiments.geometric_consistency_lib.run_existing_test_suite
    (shared with Task 7).
    """
    return gclib.run_existing_test_suite(repo_root)


def main():
    parser = argparse.ArgumentParser(description="Run the Task 6 camera_rotation geometric consistency experiment.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--skip_tests", action="store_true", help="Skip running the existing test suite before writing the result")
    args = parser.parse_args()

    cfg = load_task6_config(args.config)
    result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = run_existing_test_suite(repo_root)

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Task 6 result written to {result_path}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
