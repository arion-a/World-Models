"""Task 3: tests for the paired original/transformed sample generator
(transforms/pairs.py). Marked slow (real renders); skipped if bpy is
unavailable.
"""

import json

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from generation.motion import CameraMotion, ObjectMotion
from generation.scene_sampler import SceneSamplerConfig, generate_scene
from transforms.pairs import generate_pair, load_pair
from transforms.scene_transform import TRANSFORM_NAMES

RES = 40


@pytest.fixture
def scene():
    return generate_scene("pairs_test", seed=21, cfg=SceneSamplerConfig(num_objects_min=2, num_objects_max=2))


@pytest.mark.slow
def test_pair_directory_structure(scene, tmp_path):
    pair_dir = generate_pair(scene, "camera_rotation", tmp_path / "pair", num_frames=1, resolution=RES)
    assert (pair_dir / "original" / "metadata.json").exists()
    assert (pair_dir / "original" / "rgb.npy").exists()
    assert (pair_dir / "original" / "depth.npy").exists()
    assert (pair_dir / "original" / "segmentation.npy").exists()
    assert (pair_dir / "transformed" / "metadata.json").exists()
    assert (pair_dir / "transformed" / "rgb.npy").exists()
    assert (pair_dir / "transformation.json").exists()


@pytest.mark.slow
def test_transformation_json_matches_apply_transform_output(scene, tmp_path):
    from transforms.scene_transform import TransformConfig, apply_transform

    pair_dir = generate_pair(scene, "camera_rotation", tmp_path / "pair", num_frames=1, resolution=RES)
    saved = json.loads((pair_dir / "transformation.json").read_text())
    _, expected = apply_transform(scene, "camera_rotation", TransformConfig())
    assert saved == expected


@pytest.mark.slow
@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_original_and_transformed_renders_differ(scene, name, tmp_path):
    """Every transform type must actually change the rendered pixels --
    a pair whose two renders are identical would mean the transform
    silently did nothing."""
    pair_dir = generate_pair(scene, name, tmp_path / "pair", num_frames=1, resolution=RES)
    original = np.load(pair_dir / "original" / "rgb.npy")
    transformed = np.load(pair_dir / "transformed" / "rgb.npy")
    assert not np.array_equal(original, transformed)


@pytest.mark.slow
def test_geometric_transform_changes_depth_but_appearance_transform_need_not(scene, tmp_path):
    """A sanity cross-check tying the rendered depth map to the
    transform's own changed_variables claim: moving the camera changes
    the depth map (different viewpoint -> different distances), while a
    lighting change is not guaranteed to (same geometry, same distances)."""
    cam_pair = generate_pair(scene, "camera_translation", tmp_path / "cam", num_frames=1, resolution=RES)
    cam_depth_orig = np.load(cam_pair / "original" / "depth.npy")
    cam_depth_transformed = np.load(cam_pair / "transformed" / "depth.npy")
    assert not np.array_equal(cam_depth_orig, cam_depth_transformed)

    light_pair = generate_pair(scene, "lighting_change", tmp_path / "light", num_frames=1, resolution=RES)
    light_depth_orig = np.load(light_pair / "original" / "depth.npy")
    light_depth_transformed = np.load(light_pair / "transformed" / "depth.npy")
    assert np.array_equal(light_depth_orig, light_depth_transformed)


@pytest.mark.slow
def test_pair_generation_is_reproducible(scene, tmp_path):
    pair1 = generate_pair(scene, "object_rotation", tmp_path / "run1", num_frames=1, resolution=RES)
    pair2 = generate_pair(scene, "object_rotation", tmp_path / "run2", num_frames=1, resolution=RES)

    t1 = json.loads((pair1 / "transformation.json").read_text())
    t2 = json.loads((pair2 / "transformation.json").read_text())
    assert t1 == t2

    for side in ("original", "transformed"):
        rgb1 = np.load(pair1 / side / "rgb.npy")
        rgb2 = np.load(pair2 / side / "rgb.npy")
        assert np.array_equal(rgb1, rgb2)


@pytest.mark.slow
def test_pair_with_motion_produces_varying_frames_on_both_sides(scene, tmp_path):
    """Passing object_motions/camera_motion must make BOTH sides genuine
    multi-frame videos (frames differ within a clip), not just a
    transform applied to an otherwise-static pair -- and the two sides
    must share the same motion, differing only by what T did at frame 0."""
    object_motions = {scene.objects[0].instance_id: ObjectMotion(linear_velocity=(0.3, 0.1, 0.0))}
    camera_motion = CameraMotion(mode="static")

    pair_dir = generate_pair(
        scene, "object_rotation", tmp_path / "pair",
        num_frames=4, fps=4.0, resolution=RES,
        object_motions=object_motions, camera_motion=camera_motion,
    )
    for side in ("original", "transformed"):
        rgb = np.load(pair_dir / side / "rgb.npy")
        assert rgb.shape[0] == 4
        assert not np.array_equal(rgb[0], rgb[-1])  # real motion within the clip

    orig_meta = json.loads((pair_dir / "original" / "metadata.json").read_text())
    trans_meta = json.loads((pair_dir / "transformed" / "metadata.json").read_text())
    moved_id = scene.objects[0].instance_id
    orig_obj = next(o for o in orig_meta["objects"] if o["instance_id"] == moved_id)
    trans_obj = next(o for o in trans_meta["objects"] if o["instance_id"] == moved_id)
    # same velocity recorded on both sides -- the motion itself was not altered by T
    assert orig_obj["per_frame_pose"][0]["linear_velocity"] == trans_obj["per_frame_pose"][0]["linear_velocity"]


@pytest.mark.slow
def test_load_pair_round_trips(scene, tmp_path):
    generate_pair(scene, "texture_change", tmp_path / "pair", num_frames=1, resolution=RES)
    loaded = load_pair(tmp_path / "pair")
    assert loaded["transformation"]["type"] == "texture_change"
    orig_meta, orig_rgb, orig_depth, orig_seg = loaded["original"]
    trans_meta, trans_rgb, trans_depth, trans_seg = loaded["transformed"]
    assert orig_rgb.shape == trans_rgb.shape
    assert orig_meta["scene_id"] == trans_meta["scene_id"]
