"""Pure-math tests for generate_trajectory() -- no bpy/rendering involved.

Covers the Task 2 requirements that don't need a real render: scenes/
trajectories are reproducible, frame counts match what was asked for,
and a transform to one part of the scene does not silently change an
unrelated part of it.
"""

import numpy as np
import pytest

from generation.motion import CameraMotion, ObjectMotion, generate_trajectory
from generation.scene_sampler import SceneSamplerConfig, generate_scene


@pytest.fixture
def scene():
    return generate_scene("motion_test", seed=3, cfg=SceneSamplerConfig(num_objects_min=2, num_objects_max=2))


def test_generate_trajectory_is_deterministic(scene):
    motions = {scene.objects[0].instance_id: ObjectMotion(linear_velocity=(0.1, 0.2, 0.0))}
    cam_motion = CameraMotion(mode="orbit", orbit_deg_per_sec=10.0)
    t1 = generate_trajectory(scene, motions, cam_motion, num_frames=6, fps=5.0)
    t2 = generate_trajectory(scene, motions, cam_motion, num_frames=6, fps=5.0)
    assert t1.to_dict() == t2.to_dict()


def test_frame_count_and_timestamps_match_request(scene):
    traj = generate_trajectory(scene, num_frames=7, fps=2.0)
    assert traj.num_frames == 7
    assert len(traj.frames) == 7
    for i, frame in enumerate(traj.frames):
        assert frame.frame_index == i
        assert np.isclose(frame.timestamp, i / 2.0)


def test_static_scene_has_unchanging_poses(scene):
    traj = generate_trajectory(scene, num_frames=5, fps=3.0)
    for frame in traj.frames:
        assert frame.camera.position == scene.camera.position
        assert frame.camera.rotation_euler == scene.camera.rotation_euler
        for pose, obj in zip(frame.objects, scene.objects):
            assert pose.position == obj.position
            assert pose.rotation_euler == obj.rotation_euler


def test_constant_velocity_object_position_is_exact(scene):
    v = (0.3, -0.1, 0.05)
    motions = {scene.objects[0].instance_id: ObjectMotion(linear_velocity=v)}
    fps = 4.0
    traj = generate_trajectory(scene, motions, num_frames=5, fps=fps)
    p0 = np.array(scene.objects[0].position)
    for frame in traj.frames:
        t = frame.frame_index / fps
        expected = p0 + np.array(v) * t
        actual = np.array(frame.objects[0].position)
        assert np.allclose(actual, expected, atol=1e-10)


def test_object_motion_does_not_move_other_objects_or_camera(scene):
    """A transform applied to object 0 must not silently perturb object 1 or the camera."""
    motions = {scene.objects[0].instance_id: ObjectMotion(linear_velocity=(1.0, 1.0, 1.0), angular_velocity_deg_per_sec=45.0)}
    traj = generate_trajectory(scene, motions, num_frames=4, fps=2.0)
    for frame in traj.frames:
        assert frame.camera.position == scene.camera.position
        assert frame.camera.rotation_euler == scene.camera.rotation_euler
        other = frame.objects[1]
        assert other.position == scene.objects[1].position
        assert other.rotation_euler == scene.objects[1].rotation_euler


def test_camera_motion_does_not_move_objects(scene):
    cam_motion = CameraMotion(mode="orbit", orbit_deg_per_sec=30.0)
    traj = generate_trajectory(scene, camera_motion=cam_motion, num_frames=4, fps=2.0)
    for frame in traj.frames:
        for pose, obj in zip(frame.objects, scene.objects):
            assert pose.position == obj.position
            assert pose.rotation_euler == obj.rotation_euler
    # but the camera itself DID move (sanity: the test setup is not vacuous)
    positions = {frame.camera.position for frame in traj.frames}
    assert len(positions) > 1


def test_camera_orbit_preserves_radius_and_elevation(scene):
    cam_motion = CameraMotion(mode="orbit", orbit_deg_per_sec=40.0)
    traj = generate_trajectory(scene, camera_motion=cam_motion, num_frames=6, fps=3.0)
    pivot = np.zeros(3)
    radius0 = np.linalg.norm(np.array(scene.camera.position) - pivot)
    z0 = scene.camera.position[2]
    for frame in traj.frames:
        pos = np.array(frame.camera.position)
        assert np.isclose(np.linalg.norm(pos - pivot), radius0, atol=1e-8)
        assert np.isclose(pos[2], z0, atol=1e-8)


def test_zero_angular_velocity_leaves_rotation_unchanged(scene):
    motions = {scene.objects[0].instance_id: ObjectMotion(linear_velocity=(0.5, 0.0, 0.0), angular_velocity_deg_per_sec=0.0)}
    traj = generate_trajectory(scene, motions, num_frames=4, fps=2.0)
    for frame in traj.frames:
        assert frame.objects[0].rotation_euler == scene.objects[0].rotation_euler


def test_invalid_fps_or_num_frames_raise():
    scene0 = generate_scene("bad_args", seed=1, cfg=SceneSamplerConfig())
    with pytest.raises(ValueError):
        generate_trajectory(scene0, num_frames=0, fps=10.0)
    with pytest.raises(ValueError):
        generate_trajectory(scene0, num_frames=5, fps=0.0)


def test_unknown_camera_motion_mode_raises(scene):
    with pytest.raises(ValueError):
        generate_trajectory(scene, camera_motion=CameraMotion(mode="teleport"), num_frames=3, fps=1.0)
