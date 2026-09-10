import numpy as np
import pytest

from generation.scene_sampler import SceneSamplerConfig, sample_scene
from transforms.scene_transform import (
    CONTROL_TRANSFORMS,
    GEOMETRIC_TRANSFORMS,
    TRANSFORM_NAMES,
    TransformConfig,
    apply_transform,
)
from transforms.se3 import euler_to_matrix, matrix_to_euler, orbit_position, rotation_about_axis


def test_euler_matrix_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        euler = tuple(rng.uniform(-np.pi, np.pi, size=3))
        mat = euler_to_matrix(euler)
        recovered = matrix_to_euler(mat)
        mat2 = euler_to_matrix(recovered)
        # Compare rotation matrices, not raw angles (Euler angles are not unique).
        assert np.allclose(mat, mat2, atol=1e-6)


def test_rotation_about_axis_is_a_rotation():
    rot = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.pi / 3)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-8)


def test_orbit_position_preserves_radius():
    pivot = np.zeros(3)
    pos = np.array([5.0, 0.0, 2.0])
    radius = np.linalg.norm(pos - pivot)
    new_pos = orbit_position(pos, pivot, np.radians(20.0), np.radians(-5.0))
    assert np.isclose(np.linalg.norm(new_pos - pivot), radius, atol=1e-6)
    assert not np.allclose(new_pos, pos)


@pytest.fixture
def base_scene():
    return sample_scene("scene_test", seed=42, cfg=SceneSamplerConfig())


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_transform_is_deterministic(base_scene, name):
    cfg = TransformConfig()
    s1, p1 = apply_transform(base_scene, name, cfg)
    s2, p2 = apply_transform(base_scene, name, cfg)
    assert s1.to_dict() == s2.to_dict()
    assert p1 == p2


@pytest.mark.parametrize("name", GEOMETRIC_TRANSFORMS)
def test_geometric_transforms_change_geometry(base_scene, name):
    new_scene, _ = apply_transform(base_scene, name, TransformConfig())
    cam_moved = base_scene.camera != new_scene.camera
    objects_moved = base_scene.objects != new_scene.objects
    assert cam_moved or objects_moved


@pytest.mark.parametrize("name", CONTROL_TRANSFORMS)
def test_control_transforms_preserve_geometry(base_scene, name):
    new_scene, _ = apply_transform(base_scene, name, TransformConfig())
    assert base_scene.camera.position == new_scene.camera.position
    assert base_scene.camera.rotation_euler == new_scene.camera.rotation_euler
    for old_obj, new_obj in zip(base_scene.objects, new_scene.objects):
        assert old_obj.position == new_obj.position
        assert old_obj.rotation_euler == new_obj.rotation_euler


def test_transforms_use_independent_random_streams(base_scene):
    cfg = TransformConfig()
    _, p_rot = apply_transform(base_scene, "camera_rotation", cfg)
    _, p_trans = apply_transform(base_scene, "camera_translation", cfg)
    # Different transform types must not accidentally share a random draw.
    assert p_rot["azimuth_deg"] != p_trans["magnitude"]


def test_different_seeds_give_different_scenes():
    s1 = sample_scene("a", seed=1)
    s2 = sample_scene("a", seed=2)
    assert s1.to_dict() != s2.to_dict()


def test_same_seed_gives_identical_scene():
    s1 = sample_scene("a", seed=7)
    s2 = sample_scene("a", seed=7)
    assert s1.to_dict() == s2.to_dict()
