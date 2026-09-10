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
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from baselines.identity_baseline import evaluate_identity_baseline
from baselines.mean_baseline import evaluate_mean_baseline
from baselines.shuffled_pairing_baseline import evaluate_shuffled_pairing_baseline
from generation.scene import SceneState
from generation.scene_sampler import SceneSamplerConfig, sample_scene
from metrics.equivariance import evaluate_equivariance
from transforms.pairs import generate_pair, load_pair
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
    """>=40 scenes, one distinct seed each -- see generate_dataset() in
    generation/generate.py for the same base_seed*1_000_003 + i convention,
    reused here so Task 6 scene ids/seeds are derived the same documented
    way as Task 2's.
    """
    sampler_cfg = SceneSamplerConfig(num_objects_min=cfg.num_objects_min, num_objects_max=cfg.num_objects_max)
    scenes = []
    for i in range(cfg.num_scenes):
        scene_id = f"scene_{i:04d}"
        seed = cfg.base_seed * 1_000_003 + i
        scenes.append(sample_scene(scene_id, seed, sampler_cfg))
    return scenes


def assign_split(scene_ids: list[str], train_fraction: float, base_seed: int) -> dict[str, str]:
    """Scene-level train/test split, decided before any rendering.

    A permutation seeded by `base_seed` (not scene order) decides which
    scenes are train vs. test; every variant of one scene_id (there is
    exactly one here -- original + camera_rotation-transformed) shares
    the resulting label by construction, since callers key everything by
    scene_id, never by (scene_id, variant).
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    rng = np.random.default_rng(base_seed)
    order = rng.permutation(len(scene_ids))
    num_train = int(round(len(scene_ids) * train_fraction))
    split: dict[str, str] = {}
    for rank, idx in enumerate(order):
        split[scene_ids[idx]] = "train" if rank < num_train else "test"
    return split


# --- 3. rendering (needs bpy) ------------------------------------------------


def render_all_pairs(scenes: list[SceneState], split: dict[str, str], cfg: Task6Config) -> Path:
    """Render the original/camera_rotation pair for every scene via
    transforms.pairs.generate_pair (Task 3's renderer, reused unmodified)
    and write a top-level manifest.json recording the scene-level split
    decided in assign_split() -- BEFORE this function is ever called.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform_cfg = TransformConfig(
        camera_rotation_azimuth_deg_range=(cfg.camera_rotation_azimuth_deg, cfg.camera_rotation_azimuth_deg)
    )

    for scene in scenes:
        pair_dir = out_dir / scene.scene_id
        generate_pair(
            scene,
            TRANSFORM_NAME,
            pair_dir,
            transform_cfg=transform_cfg,
            num_frames=cfg.num_frames,
            fps=cfg.fps,
            resolution=cfg.resolution,
        )
        _verify_transform_ground_truth(pair_dir)

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
    what camera_rotation does out from under this experiment.
    """
    transformation = json.loads((pair_dir / "transformation.json").read_text())
    if transformation.get("type") != TRANSFORM_NAME or transformation.get("transform_name") != TRANSFORM_NAME:
        raise ValueError(f"{pair_dir}: expected transform '{TRANSFORM_NAME}', got {transformation.get('type')!r}")
    if transformation.get("transform_matrix") is None:
        raise ValueError(f"{pair_dir}: camera_rotation must be a rigid transform with a real SE(3) matrix")
    changed = set(transformation.get("changed_variables", []))
    if changed != {"camera.position", "camera.rotation_euler"}:
        raise ValueError(f"{pair_dir}: camera_rotation changed unexpected variables: {sorted(changed)}")


# --- 4. encoding (needs torch/transformers, real weights when pretrained) ---


def build_encoder(cfg: Task6Config):
    from encoders.vjepa import DEFAULT_CHECKPOINT, VJEPAEncoder

    return VJEPAEncoder(
        checkpoint=cfg.checkpoint or DEFAULT_CHECKPOINT,
        pretrained=cfg.pretrained,
        device=cfg.device,
        fallback_seed=cfg.fallback_seed,
    )


def encode_all_pairs(scenes: list[SceneState], out_dir: Path, encoder) -> dict[str, dict[str, np.ndarray]]:
    """Z, Z' per scene: mean_pool(encoder.encode(rgb)) for each side.

    Only the rendered RGB video ever reaches `encoder.encode` -- no
    ground-truth pose/rotation-matrix data is passed in (research/
    RESEARCH_INVARIANTS.md invariant 7); those live only in
    transformation.json / metadata.json, read here only for provenance
    bookkeeping elsewhere, never for encoding.
    """
    from encoders.vjepa import mean_pool

    reps: dict[str, dict[str, np.ndarray]] = {}
    for scene in scenes:
        loaded = load_pair(out_dir / scene.scene_id)
        _, orig_rgb, _, _ = loaded["original"]
        _, trans_rgb, _, _ = loaded["transformed"]
        z = mean_pool(encoder.encode(orig_rgb))
        z_prime = mean_pool(encoder.encode(trans_rgb))
        reps[scene.scene_id] = {"Z": z, "Z_prime": z_prime}
    return reps


# --- assemble train/test arrays, enforcing scene-level disjointness --------


def build_arrays(
    scenes: list[SceneState], split: dict[str, str], reps: dict[str, dict[str, np.ndarray]]
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ids = [s.scene_id for s in scenes if split[s.scene_id] == "train"]
    test_ids = [s.scene_id for s in scenes if split[s.scene_id] == "test"]
    if not set(train_ids).isdisjoint(test_ids):
        raise ValueError("train/test scene leakage detected")

    Z_train = np.stack([reps[sid]["Z"] for sid in train_ids])
    Zp_train = np.stack([reps[sid]["Z_prime"] for sid in train_ids])
    Z_test = np.stack([reps[sid]["Z"] for sid in test_ids])
    Zp_test = np.stack([reps[sid]["Z_prime"] for sid in test_ids])
    return train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test


def _software_versions() -> dict:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _snapshot_params(model) -> list:
    return [p.detach().clone() for p in model.parameters()]


def _params_unchanged(before: list, model) -> bool:
    import torch

    after = list(model.parameters())
    if len(before) != len(after):
        return False
    return all(torch.equal(b, a) for b, a in zip(before, after))


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

    equiv_result, _rho = evaluate_equivariance(
        TRANSFORM_NAME, Z_train, Zp_train, Z_test, Zp_test, alpha=cfg.ridge_alpha
    )
    persistence_result = evaluate_identity_baseline(TRANSFORM_NAME, Z_test, Zp_test)
    mean_result = evaluate_mean_baseline(TRANSFORM_NAME, Zp_train, Zp_test)
    random_pair_result = evaluate_shuffled_pairing_baseline(
        TRANSFORM_NAME, Z_train, Zp_train, Z_test, Zp_test, alpha=cfg.ridge_alpha, seed=cfg.shuffled_pairing_seed
    )

    learned = {
        "r2": equiv_result.r2,
        "mean_cosine_similarity": equiv_result.mean_cosine_similarity,
        "mean_relative_l2_error": equiv_result.mean_relative_l2_error,
    }
    persistence = {
        "r2": persistence_result.r2,
        "mean_cosine_similarity": persistence_result.mean_cosine_similarity,
        "mean_relative_l2_error": persistence_result.mean_relative_l2_error,
    }
    mean_b = {
        "r2": mean_result.r2,
        "mean_cosine_similarity": mean_result.mean_cosine_similarity,
        "mean_relative_l2_error": mean_result.mean_relative_l2_error,
    }
    random_pair = {
        "r2": random_pair_result.r2,
        "mean_cosine_similarity": random_pair_result.mean_cosine_similarity,
        "mean_relative_l2_error": random_pair_result.mean_relative_l2_error,
    }

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
    """Parse pytest -q's final summary line ('12 passed in 0.34s',
    '3 failed, 9 passed in 1.02s', 'no tests ran in 0.00s') into
    (passed, failed) counts. Returns (0, 0) if no recognizable summary
    line is found, rather than guessing.
    """
    import re

    passed = failed = 0
    for line in reversed(output.splitlines()):
        m_passed = re.search(r"(\d+) passed", line)
        m_failed = re.search(r"(\d+) failed", line)
        if m_passed or m_failed:
            if m_passed:
                passed = int(m_passed.group(1))
            if m_failed:
                failed = int(m_failed.group(1))
            break
    return passed, failed


def run_existing_test_suite(repo_root: str | Path) -> dict:
    """Run `pytest -m "not slow" -q` from the repo root and report honest
    pass/fail counts -- required by this task's own instructions, though
    (per those same instructions) an independent process re-runs this
    after Claude finishes and is the actual basis for acceptance, not
    this self-report.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["pytest", "-m", "not slow", "-q"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"passed": 0, "failed": 0, "error": f"could not run test suite: {exc!r}"}

    output = proc.stdout + proc.stderr
    passed, failed = _parse_pytest_summary(output)
    return {"passed": passed, "failed": failed, "returncode": proc.returncode}


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
