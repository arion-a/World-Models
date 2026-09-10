"""Task 2 entry point: controlled 3D scene generation with complete ground truth.

    python -m generation.generate --config configs/generation.yaml

Exposes exactly the four functions the task specifies:

    generate_scene()      -- sample a scene's static physical state S
    generate_trajectory()  -- compute a per-frame trajectory for that scene
    render_scene()         -- render RGB + depth + segmentation for that trajectory
    save_ground_truth()    -- write everything to disk in a documented, machine-readable form

No physical transformation `T` (S -> S' = T(S), the subject of a later
task) appears anywhere in this module -- this module only ever produces
one trajectory per scene, from its own sampled motion, never a
transformed *copy* of another scene.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from configs.config import GenerationConfig, load_generation_config
from generation.bpy_renderer import ClipGroundTruth, render_trajectory
from generation.ground_truth import save_ground_truth as _save_ground_truth
from generation.motion import CameraMotion, ObjectMotion, Trajectory, generate_trajectory as _generate_trajectory
from generation.scene import SceneState
from generation.scene_sampler import SceneSamplerConfig, generate_scene as _generate_scene

# Re-exported under the exact names Task 2 asks for.
generate_scene = _generate_scene
generate_trajectory = _generate_trajectory
save_ground_truth = _save_ground_truth


def _rng_for(seed: int, tag: str) -> np.random.Generator:
    # Same stable-hash approach as transforms/scene_transform.py's
    # _rng_for, and for the same reason: Python's built-in hash() of a
    # str is randomized per process, which would silently break
    # reproducibility across runs.
    return np.random.default_rng((seed, zlib.crc32(tag.encode("utf-8"))))


def sample_motions(scene: SceneState, cfg: GenerationConfig) -> tuple[dict[int, ObjectMotion], CameraMotion]:
    """Sample this scene's (deterministic, seed-derived) motion parameters.

    Kept separate from generate_trajectory() itself: generate_trajectory()
    takes explicit motion parameters and is pure/deterministic given them
    (see generation/motion.py); *choosing* those parameters at random is a
    generation-time policy, not something that belongs in the trajectory
    math itself.
    """
    object_motions: dict[int, ObjectMotion] = {}
    if cfg.object_motion.enabled:
        for obj in scene.objects:
            rng = _rng_for(scene.seed, f"object_motion_{obj.instance_id}")
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            speed = float(rng.uniform(*cfg.object_motion.linear_speed_range))
            angular_speed = float(rng.uniform(*cfg.object_motion.angular_speed_deg_range)) * float(rng.choice([-1.0, 1.0]))
            object_motions[obj.instance_id] = ObjectMotion(
                linear_velocity=(float(direction[0] * speed), float(direction[1] * speed), 0.0),
                angular_velocity_axis=(0.0, 0.0, 1.0),
                angular_velocity_deg_per_sec=angular_speed,
            )

    if cfg.camera_motion.mode == "orbit":
        rng = _rng_for(scene.seed, "camera_motion")
        rate = float(rng.uniform(*cfg.camera_motion.orbit_deg_per_sec_range)) * float(rng.choice([-1.0, 1.0]))
        camera_motion = CameraMotion(mode="orbit", orbit_deg_per_sec=rate)
    elif cfg.camera_motion.mode == "static":
        camera_motion = CameraMotion(mode="static")
    else:
        raise ValueError(f"Unknown camera_motion.mode '{cfg.camera_motion.mode}'. Known: 'static', 'orbit'.")

    return object_motions, camera_motion


def render_scene(scene: SceneState, trajectory: Trajectory, cfg: GenerationConfig) -> ClipGroundTruth:
    return render_trajectory(
        scene,
        trajectory,
        resolution=cfg.resolution,
        depth_near=cfg.depth_near,
        depth_far=cfg.depth_far,
        max_instances=cfg.max_instances,
    )


def generate_dataset(cfg: GenerationConfig) -> list[str]:
    """Generate cfg.num_scenes scenes end-to-end, writing each one's ground
    truth to `cfg.output_dir/{scene_id}/` and a top-level manifest.json.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler_cfg = SceneSamplerConfig(num_objects_min=cfg.num_objects_min, num_objects_max=cfg.num_objects_max)

    scene_ids = []
    for i in range(cfg.num_scenes):
        scene_id = f"scene_{i:04d}"
        seed = cfg.base_seed * 1_000_003 + i

        scene = generate_scene(scene_id, seed, sampler_cfg)
        object_motions, camera_motion = sample_motions(scene, cfg)
        trajectory = generate_trajectory(
            scene,
            object_motions=object_motions,
            camera_motion=camera_motion,
            num_frames=cfg.num_frames,
            fps=cfg.fps,
        )
        clip = render_scene(scene, trajectory, cfg)
        save_ground_truth(scene, trajectory, clip, out_dir, cfg.resolution)

        scene_ids.append(scene_id)
        print(f"[{i + 1}/{cfg.num_scenes}] generated {scene_id} ({len(scene.objects)} objects)")

    manifest = {
        "num_scenes": cfg.num_scenes,
        "base_seed": cfg.base_seed,
        "resolution": cfg.resolution,
        "num_frames": cfg.num_frames,
        "fps": cfg.fps,
        "scene_ids": scene_ids,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return scene_ids


def main():
    parser = argparse.ArgumentParser(description="Generate the GCP Task 2 controlled 3D scene dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config overriding configs/generation.yaml")
    args = parser.parse_args()

    default_path = Path(__file__).resolve().parent.parent / "configs" / "generation.yaml"
    config_path = args.config or (default_path if default_path.exists() else None)
    cfg = load_generation_config(config_path)

    generate_dataset(cfg)


if __name__ == "__main__":
    main()
