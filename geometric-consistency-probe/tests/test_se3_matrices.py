"""Task 3: unit tests for the SE(3) matrix engine itself (transforms/se3.py
and generation/scene.py's camera-pose wrappers), independent of scenes or
rendering.

Covers exactly the properties the task calls out: identity, composition,
inverse recovery, and the world-to-camera vs. camera-to-world distinction
not being interchangeable.
"""

import numpy as np
import pytest

from generation.scene import CameraState, camera_to_world_matrix, world_to_camera_matrix
from transforms.se3 import (
    compose,
    euler_to_matrix,
    identity_matrix,
    inverse_rigid,
    look_at_euler,
    matrix_to_pose,
    orbit_rotation_matrix,
    pose_matrix,
    rotate_about_point_matrix,
    rotation_about_axis,
    rotation_only_matrix,
    translation_matrix,
)


def _random_pose(rng):
    position = tuple(rng.uniform(-5, 5, size=3).tolist())
    rotation_euler = tuple(rng.uniform(-np.pi, np.pi, size=3).tolist())
    return position, rotation_euler


def test_identity_matrix_leaves_pose_unchanged():
    rng = np.random.default_rng(0)
    position, rotation_euler = _random_pose(rng)
    M = pose_matrix(position, rotation_euler)
    result = identity_matrix() @ M
    assert np.allclose(result, M, atol=1e-12)
    new_position, new_rotation = matrix_to_pose(result)
    assert new_position == pytest.approx(position)
    # Compare via rotation matrix, not raw Euler angles: Euler triples are
    # not a unique representation of a rotation (see
    # tests/test_transforms.py::test_euler_matrix_roundtrip), so two
    # different-looking triples can be the same physical orientation.
    assert euler_to_matrix(new_rotation) == pytest.approx(euler_to_matrix(rotation_euler), abs=1e-10)


def test_pose_matrix_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        position, rotation_euler = _random_pose(rng)
        M = pose_matrix(position, rotation_euler)
        recovered_position, recovered_rotation = matrix_to_pose(M)
        assert recovered_position == pytest.approx(position, abs=1e-10)
        assert euler_to_matrix(recovered_rotation) == pytest.approx(euler_to_matrix(rotation_euler), abs=1e-10)


def test_composition_is_associative_and_matches_manual_matmul():
    rng = np.random.default_rng(2)
    A = pose_matrix(*_random_pose(rng))
    B = pose_matrix(*_random_pose(rng))
    C = pose_matrix(*_random_pose(rng))
    assert np.allclose(compose(A, B, C), A @ B @ C, atol=1e-10)
    assert np.allclose(compose(A, B) @ C, A @ compose(B, C), atol=1e-10)


def test_translation_then_inverse_translation_is_identity():
    delta = (1.5, -2.0, 0.3)
    T = translation_matrix(delta)
    T_inv = translation_matrix(tuple(-d for d in delta))
    assert np.allclose(T @ T_inv, identity_matrix(), atol=1e-12)
    assert np.allclose(T_inv @ T, identity_matrix(), atol=1e-12)


def test_inverse_rigid_recovers_original_pose():
    rng = np.random.default_rng(3)
    for _ in range(20):
        M = pose_matrix(*_random_pose(rng))
        assert np.allclose(inverse_rigid(M) @ M, identity_matrix(), atol=1e-8)
        assert np.allclose(M @ inverse_rigid(M), identity_matrix(), atol=1e-8)


def test_inverse_rigid_is_not_naive_negate_translation():
    """Regression test for the classic SE(3) bug: [R^T, -t] is NOT the
    rigid inverse of [R, t] unless R is the identity. If this test ever
    passes with the naive formula, something has silently regressed."""
    R = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.radians(37))
    t = np.array([3.0, -1.0, 2.0])
    from transforms.se3 import rt_to_matrix

    M = rt_to_matrix(R, t)
    correct_inverse = inverse_rigid(M)
    naive_wrong_inverse = rt_to_matrix(R.T, -t)
    assert not np.allclose(correct_inverse, naive_wrong_inverse, atol=1e-6)
    assert np.allclose(correct_inverse @ M, identity_matrix(), atol=1e-8)
    assert not np.allclose(naive_wrong_inverse @ M, identity_matrix(), atol=1e-6)


def test_world_to_camera_and_camera_to_world_are_inverses_but_not_equal():
    camera = CameraState(position=(4.0, -3.0, 2.5), rotation_euler=(1.0, 0.2, -0.7))
    c2w = camera_to_world_matrix(camera)
    w2c = world_to_camera_matrix(camera)
    assert np.allclose(c2w @ w2c, identity_matrix(), atol=1e-8)
    assert np.allclose(w2c @ c2w, identity_matrix(), atol=1e-8)
    # The two matrices themselves must not be confused for each other.
    assert not np.allclose(c2w, w2c, atol=1e-3)


def test_world_to_camera_maps_camera_origin_to_world_position():
    """A sanity check with an unambiguous physical meaning: the camera's
    own origin (in camera-local coordinates, [0,0,0]) maps to its world
    position under camera_to_world, and the camera's world position maps
    back to the camera-local origin under world_to_camera."""
    camera = CameraState(position=(2.0, 5.0, 1.0), rotation_euler=(0.3, -0.4, 1.1))
    c2w = camera_to_world_matrix(camera)
    w2c = world_to_camera_matrix(camera)
    local_origin = np.array([0.0, 0.0, 0.0, 1.0])
    world_point = np.array([*camera.position, 1.0])
    assert np.allclose(c2w @ local_origin, world_point, atol=1e-10)
    assert np.allclose(w2c @ world_point, local_origin, atol=1e-10)


def test_rotate_about_point_matrix_leaves_pivot_fixed():
    pivot = (1.0, 2.0, 0.5)
    R = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.radians(50))
    T = rotate_about_point_matrix(R, pivot)
    pivot_h = np.array([*pivot, 1.0])
    assert np.allclose(T @ pivot_h, pivot_h, atol=1e-10)


def test_rotate_about_point_reduces_to_rotation_only_at_origin():
    R = rotation_about_axis(np.array([1.0, 0.0, 0.0]), np.radians(20))
    assert np.allclose(rotate_about_point_matrix(R, (0.0, 0.0, 0.0)), rotation_only_matrix(R), atol=1e-12)


def test_orbit_rotation_matrix_matches_look_at_orientation_update():
    """The single rotation matrix used for camera_rotation must correctly
    transform BOTH position and orientation of an orbiting, pivot-facing
    camera -- verified against independently re-deriving the orientation
    with look_at_euler from scratch at the new position."""
    rng = np.random.default_rng(4)
    for _ in range(50):
        radius = rng.uniform(4, 10)
        az0 = rng.uniform(0, 2 * np.pi)
        el0 = np.radians(rng.uniform(10, 70))
        position = radius * np.array(
            [np.cos(el0) * np.cos(az0), np.cos(el0) * np.sin(az0), np.sin(el0)]
        )
        old_euler = look_at_euler(position, np.zeros(3))
        az_delta = np.radians(rng.uniform(-40, 40))
        el_delta = np.radians(rng.uniform(-15, 15))

        R = orbit_rotation_matrix(position, az_delta, el_delta)
        new_position = R @ position
        new_rotation_via_matrix = R @ euler_to_matrix(old_euler)
        new_rotation_via_lookat = euler_to_matrix(look_at_euler(new_position, np.zeros(3)))

        assert np.allclose(new_rotation_via_matrix, new_rotation_via_lookat, atol=1e-8)


def test_orbit_rotation_matrix_matches_orbit_position():
    from transforms.se3 import orbit_position

    rng = np.random.default_rng(5)
    for _ in range(50):
        radius = rng.uniform(4, 10)
        az0 = rng.uniform(0, 2 * np.pi)
        el0 = np.radians(rng.uniform(10, 70))
        position = radius * np.array(
            [np.cos(el0) * np.cos(az0), np.cos(el0) * np.sin(az0), np.sin(el0)]
        )
        az_delta = np.radians(rng.uniform(-40, 40))
        el_delta = np.radians(rng.uniform(-15, 15))

        expected = orbit_position(position, np.zeros(3), az_delta, el_delta)
        R = orbit_rotation_matrix(position, az_delta, el_delta)
        assert np.allclose(R @ position, expected, atol=1e-8)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
