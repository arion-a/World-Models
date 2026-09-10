"""Renderer-level tests for Task 2's multi-pass (RGB + depth + segmentation)
rendering. Marked slow (invokes real Blender/Cycles renders); skipped if
bpy is unavailable.
"""

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from generation.bpy_renderer import render_trajectory
from generation.motion import CameraMotion, ObjectMotion, generate_trajectory
from generation.scene_sampler import SceneSamplerConfig, generate_scene

RES = 32


@pytest.fixture
def scene_and_trajectory():
    scene = generate_scene("bpy_gt_test", seed=11, cfg=SceneSamplerConfig(num_objects_min=2, num_objects_max=2))
    motions = {scene.objects[0].instance_id: ObjectMotion(linear_velocity=(0.2, 0.0, 0.0))}
    traj = generate_trajectory(scene, motions, num_frames=3, fps=3.0)
    return scene, traj


@pytest.mark.slow
def test_render_trajectory_shapes_match_request(scene_and_trajectory):
    scene, traj = scene_and_trajectory
    clip = render_trajectory(scene, traj, resolution=RES)
    assert clip.rgb.shape == (3, RES, RES, 3)
    assert clip.depth.shape == (3, RES, RES)
    assert clip.segmentation.shape == (3, RES, RES)
    assert clip.rgb.dtype == np.uint8
    assert clip.depth.dtype == np.float32
    assert clip.segmentation.dtype == np.uint8


@pytest.mark.slow
def test_render_trajectory_is_reproducible(scene_and_trajectory):
    scene, traj = scene_and_trajectory
    clip1 = render_trajectory(scene, traj, resolution=RES)
    clip2 = render_trajectory(scene, traj, resolution=RES)
    assert np.array_equal(clip1.rgb, clip2.rgb)
    assert np.array_equal(clip1.depth, clip2.depth)
    assert np.array_equal(clip1.segmentation, clip2.segmentation)


@pytest.mark.slow
def test_segmentation_values_match_scene_instance_ids(scene_and_trajectory):
    scene, traj = scene_and_trajectory
    clip = render_trajectory(scene, traj, resolution=RES)
    expected_ids = {0} | {obj.instance_id for obj in scene.objects}
    found_ids = set(np.unique(clip.segmentation).tolist())
    # Every value present must be a real id (0=background or a real
    # object); not every object need be visible in every tiny test render,
    # so this is a subset check, not equality.
    assert found_ids <= expected_ids
    assert 0 in found_ids  # the floor should always be visible somewhere


@pytest.mark.slow
def test_depth_within_configured_range(scene_and_trajectory):
    scene, traj = scene_and_trajectory
    depth_near, depth_far = 0.1, 25.0
    clip = render_trajectory(scene, traj, resolution=RES, depth_near=depth_near, depth_far=depth_far)
    assert clip.depth.min() >= depth_near - 1e-3
    assert clip.depth.max() <= depth_far + 1e-3


@pytest.mark.slow
def test_moving_object_changes_pixels_across_frames(scene_and_trajectory):
    """The object with nonzero velocity should visibly move; a scene
    rendered with all-static motion should not (sanity check that the
    renderer actually reflects generate_trajectory()'s output, not a
    cached/static frame)."""
    scene, traj = scene_and_trajectory
    clip = render_trajectory(scene, traj, resolution=RES)
    assert not np.array_equal(clip.rgb[0], clip.rgb[-1])
    assert not np.array_equal(clip.segmentation[0], clip.segmentation[-1])


@pytest.mark.slow
def test_static_trajectory_renders_identical_frames():
    """Inverse of the above: zero motion must render pixel-identical frames --
    a transform (here, "no transform") must not silently introduce change."""
    scene = generate_scene("bpy_static_test", seed=12, cfg=SceneSamplerConfig())
    traj = generate_trajectory(scene, num_frames=3, fps=2.0)  # no motions passed -> everything stationary
    clip = render_trajectory(scene, traj, resolution=RES)
    assert np.array_equal(clip.rgb[0], clip.rgb[1])
    assert np.array_equal(clip.rgb[0], clip.rgb[2])
    assert np.array_equal(clip.depth[0], clip.depth[2])
    assert np.array_equal(clip.segmentation[0], clip.segmentation[2])


@pytest.mark.slow
def test_depth_pass_is_planar_not_euclidean():
    """Verifies generation/COORDINATE_SYSTEM.md's depth claim directly:
    a large flat plane viewed face-on (camera looking straight down)
    must render with spatially UNIFORM depth. If Blender's Z-pass were
    true Euclidean ray distance instead of camera-space Z, off-center
    pixels (viewed at an oblique angle) would read a larger depth than
    the frame center.
    """
    from generation.scene import CameraState, LightState, ObjectState, SceneState

    scene = SceneState(
        scene_id="depth_semantics_test",
        seed=0,
        objects=(),
        camera=CameraState(position=(0.0, 0.0, 5.0), rotation_euler=(0.0, 0.0, 0.0)),
        light=LightState(position=(5.0, 5.0, 10.0), energy=3.0, color=(1.0, 1.0, 1.0)),
        floor_color=(0.5, 0.5, 0.5),
    )
    traj = generate_trajectory(scene, num_frames=1, fps=1.0)
    clip = render_trajectory(scene, traj, resolution=64, depth_near=0.1, depth_far=20.0)
    depth = clip.depth[0]
    center = depth[32, 32]
    corner = depth[2, 2]
    assert np.isclose(center, 5.0, atol=0.05)
    assert np.isclose(corner, 5.0, atol=0.05)
    assert np.isclose(center, corner, atol=1e-4)
