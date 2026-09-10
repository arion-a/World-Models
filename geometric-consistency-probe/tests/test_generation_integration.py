"""End-to-end tests for Task 2: generate_scene -> generate_trajectory ->
render_scene -> save_ground_truth, and the CLI that chains them.

Marked slow (real renders); skipped if bpy is unavailable.
"""

import json

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from configs.config import GenerationConfig
from generation.generate import generate_dataset, generate_scene, generate_trajectory, render_scene, sample_motions, save_ground_truth
from generation.ground_truth import load_ground_truth
from generation.scene_sampler import SceneSamplerConfig

RES = 32


def _tiny_cfg(tmp_path, **overrides) -> GenerationConfig:
    cfg = GenerationConfig(
        num_scenes=2,
        base_seed=7,
        output_dir=str(tmp_path / "gen"),
        resolution=RES,
        num_frames=3,
        fps=3.0,
        num_objects_min=1,
        num_objects_max=2,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.slow
def test_generate_dataset_creates_expected_files(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    scene_ids = generate_dataset(cfg)
    assert scene_ids == ["scene_0000", "scene_0001"]

    out_dir = tmp_path / "gen"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["scene_ids"] == scene_ids
    assert manifest["num_frames"] == cfg.num_frames

    for scene_id in scene_ids:
        scene_dir = out_dir / scene_id
        assert (scene_dir / "metadata.json").exists()
        assert (scene_dir / "rgb.npy").exists()
        assert (scene_dir / "depth.npy").exists()
        assert (scene_dir / "segmentation.npy").exists()


@pytest.mark.slow
def test_frame_counts_match_metadata(tmp_path):
    cfg = _tiny_cfg(tmp_path, num_frames=4)
    generate_dataset(cfg)
    metadata, rgb, depth, seg = load_ground_truth(tmp_path / "gen", "scene_0000")
    assert metadata["num_frames"] == 4
    assert rgb.shape[0] == 4
    assert depth.shape[0] == 4
    assert seg.shape[0] == 4
    assert len(metadata["camera"]["per_frame_pose"]) == 4
    for obj_meta in metadata["objects"]:
        assert len(obj_meta["per_frame_pose"]) == 4


@pytest.mark.slow
def test_metadata_poses_correspond_to_generated_trajectory(tmp_path):
    """The exact trajectory used to render must be recoverable, unchanged,
    from the saved metadata -- i.e. saving does not silently perturb it."""
    cfg = _tiny_cfg(tmp_path)
    sampler_cfg = SceneSamplerConfig(num_objects_min=cfg.num_objects_min, num_objects_max=cfg.num_objects_max)
    scene = generate_scene("scene_check", seed=42, cfg=sampler_cfg)
    object_motions, camera_motion = sample_motions(scene, cfg)
    trajectory = generate_trajectory(scene, object_motions, camera_motion, num_frames=cfg.num_frames, fps=cfg.fps)
    clip = render_scene(scene, trajectory, cfg)
    scene_dir = save_ground_truth(scene, trajectory, clip, cfg.output_dir, cfg.resolution)

    metadata = json.loads((scene_dir / "metadata.json").read_text())
    for frame, saved in zip(trajectory.frames, metadata["camera"]["per_frame_pose"]):
        assert list(frame.camera.position) == pytest.approx(saved["position"])
        assert list(frame.camera.rotation_euler) == pytest.approx(saved["rotation_euler"])
    for obj, obj_meta in zip(scene.objects, metadata["objects"]):
        assert obj_meta["instance_id"] == obj.instance_id
        for frame, saved_pose in zip(trajectory.frames, obj_meta["per_frame_pose"]):
            true_pose = next(p for p in frame.objects if p.instance_id == obj.instance_id)
            assert list(true_pose.position) == pytest.approx(saved_pose["position"])
            assert list(true_pose.linear_velocity) == pytest.approx(saved_pose["linear_velocity"])


@pytest.mark.slow
def test_generation_is_reproducible_across_runs(tmp_path):
    cfg1 = _tiny_cfg(tmp_path / "run1")
    cfg1.output_dir = str(tmp_path / "run1")
    cfg2 = _tiny_cfg(tmp_path / "run2")
    cfg2.output_dir = str(tmp_path / "run2")

    generate_dataset(cfg1)
    generate_dataset(cfg2)

    m1, rgb1, depth1, seg1 = load_ground_truth(tmp_path / "run1", "scene_0000")
    m2, rgb2, depth2, seg2 = load_ground_truth(tmp_path / "run2", "scene_0000")

    assert m1["seed"] == m2["seed"]
    assert np.array_equal(rgb1, rgb2)
    assert np.array_equal(depth1, depth2)
    assert np.array_equal(seg1, seg2)


@pytest.mark.slow
def test_camera_intrinsics_recorded_and_consistent_with_resolution(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    generate_dataset(cfg)
    metadata, *_ = load_ground_truth(cfg.output_dir, "scene_0000")
    intrinsics = metadata["camera"]["intrinsics"]
    assert intrinsics["cx"] == pytest.approx(cfg.resolution / 2.0)
    assert intrinsics["cy"] == pytest.approx(cfg.resolution / 2.0)
    assert intrinsics["fx"] == pytest.approx(intrinsics["fy"])
    assert intrinsics["fx"] > 0


@pytest.mark.slow
def test_object_instance_ids_are_unique_and_nonzero(tmp_path):
    cfg = _tiny_cfg(tmp_path, num_objects_min=3, num_objects_max=3)
    generate_dataset(cfg)
    metadata, *_ = load_ground_truth(cfg.output_dir, "scene_0000")
    ids = [o["instance_id"] for o in metadata["objects"]]
    assert len(ids) == len(set(ids))
    assert all(i > 0 for i in ids)
