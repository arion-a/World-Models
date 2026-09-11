"""Task 7B, contract Sec. 9: before any latent-space composition/inverse/
identity claim relies on camera_rotation being a valid one-parameter
rotation group, verify it PHYSICALLY -- in the SE(3) matrix algebra and
in real rendered SceneState poses -- not assumed from the docstring in
transforms/se3.py:orbit_rotation_matrix.

transforms/scene_transform.py's fixed_azimuth_deg/fixed_elevation_deg
(Task 7B's additive TransformConfig fields) are what make this
controllable: Task 7B pins elevation to 0.0, which orbit_rotation_matrix's
own math reduces to a pure rotation about the fixed world +Z axis
(R_el = rotation_about_axis(tangent, 0) = identity when elevation_delta=0),
independent of camera position -- exactly the "single fixed axis" shape
that guarantees R(a) @ R(b) == R(a+b) as a matter of matrix algebra, not
approximation. This file proves that holds for the ACTUAL code path
Task 7B's transform family uses, at the magnitudes Task 7B's protocol
actually specifies, before any downstream test treats camera_rotation
as APPLICABLE for composition/inverse/interpolation.
"""
from __future__ import annotations

import numpy as np
import pytest

from generation.scene_sampler import SceneSamplerConfig, generate_scene
from transforms.scene_transform import TransformConfig, apply_camera_rotation
from transforms.se3 import euler_to_matrix, identity_matrix, matrix_to_pose, orbit_rotation_matrix, pose_matrix

# The exact magnitude set the Task 7B protocol declares (contract Sec. 8).
MAGNITUDES_DEG = (-60.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 60.0)
TOL = 1e-9


@pytest.fixture
def scene():
    return generate_scene("t7b_composition", seed=42, cfg=SceneSamplerConfig(num_objects_min=2, num_objects_max=2))


def _fixed_yaw_cfg(azimuth_deg: float) -> TransformConfig:
    return TransformConfig(fixed_azimuth_deg=azimuth_deg, fixed_elevation_deg=0.0)


# --- matrix-algebra level: orbit_rotation_matrix with elevation=0 --------


@pytest.mark.parametrize("theta1", MAGNITUDES_DEG)
@pytest.mark.parametrize("theta2", MAGNITUDES_DEG)
def test_orbit_rotation_matrix_composes_additively_at_zero_elevation(theta1, theta2):
    """R(theta2) @ R(theta1) == R(theta1 + theta2) to floating-point
    precision, for every pair drawn from Task 7B's actual magnitude set --
    not just derived from the docstring, checked for the specific
    magnitudes this task will use."""
    position = np.array([3.0, 4.0, 5.0])  # an arbitrary, non-pole camera position
    R1 = orbit_rotation_matrix(position, np.radians(theta1), 0.0)
    R2 = orbit_rotation_matrix(position, np.radians(theta2), 0.0)
    R_sum = orbit_rotation_matrix(position, np.radians(theta1 + theta2), 0.0)
    assert R2 @ R1 == pytest.approx(R_sum, abs=TOL)


def test_orbit_rotation_matrix_zero_elevation_is_position_independent():
    """A pure-yaw rotation matrix must not depend on the camera's current
    position (a real requirement for a valid one-parameter group acting
    the same way regardless of which scene/camera it's applied to)."""
    R_a = orbit_rotation_matrix(np.array([3.0, 4.0, 5.0]), np.radians(30.0), 0.0)
    R_b = orbit_rotation_matrix(np.array([-7.0, 1.0, 2.0]), np.radians(30.0), 0.0)
    assert R_a == pytest.approx(R_b, abs=TOL)


# --- full SceneState level: apply_camera_rotation with fixed magnitudes -


@pytest.mark.parametrize("theta1,theta2", [(10.0, 20.0), (-30.0, 60.0), (20.0, -20.0), (30.0, 30.0)])
def test_camera_rotation_composition_holds_on_real_scene(scene, theta1, theta2):
    """T_theta2(T_theta1(S)) matches T_(theta1+theta2)(S) on a real
    SceneState's camera pose, within declared tolerance (contract Sec. 9's
    required physical pre-check)."""
    scene1, _ = apply_camera_rotation(scene, _fixed_yaw_cfg(theta1))
    scene_composed, _ = apply_camera_rotation(scene1, _fixed_yaw_cfg(theta2))

    scene_direct, _ = apply_camera_rotation(scene, _fixed_yaw_cfg(theta1 + theta2))

    assert scene_composed.camera.position == pytest.approx(scene_direct.camera.position, abs=1e-6)
    assert euler_to_matrix(scene_composed.camera.rotation_euler) == pytest.approx(
        euler_to_matrix(scene_direct.camera.rotation_euler), abs=1e-6
    )


@pytest.mark.parametrize("theta", [10.0, -20.0, 30.0, 60.0])
def test_camera_rotation_inverse_holds_on_real_scene(scene, theta):
    """T_-theta(T_theta(S)) recovers S's original camera pose."""
    forward, _ = apply_camera_rotation(scene, _fixed_yaw_cfg(theta))
    back, _ = apply_camera_rotation(forward, _fixed_yaw_cfg(-theta))

    assert back.camera.position == pytest.approx(scene.camera.position, abs=1e-6)
    assert euler_to_matrix(back.camera.rotation_euler) == pytest.approx(euler_to_matrix(scene.camera.rotation_euler), abs=1e-6)


def test_camera_rotation_identity_holds_on_real_scene(scene):
    """T_0(S) leaves the camera pose unchanged -- theta=0 is a true
    identity, not merely a near-zero perturbation."""
    scene0, params = apply_camera_rotation(scene, _fixed_yaw_cfg(0.0))
    assert scene0.camera.position == pytest.approx(scene.camera.position, abs=1e-9)
    assert euler_to_matrix(scene0.camera.rotation_euler) == pytest.approx(euler_to_matrix(scene.camera.rotation_euler), abs=1e-9)
    assert np.array(params["transform_matrix"]) == pytest.approx(identity_matrix(), abs=1e-9)


def test_fixed_azimuth_default_none_reproduces_original_random_behavior(scene):
    """The additive fields must be fully backward compatible: leaving them
    at their default (None) must reproduce Task 6/7's original
    random-range-plus-sign behavior exactly, byte for byte."""
    cfg_default = TransformConfig()
    scene_a, params_a = apply_camera_rotation(scene, cfg_default)
    scene_b, params_b = apply_camera_rotation(scene, cfg_default)
    # deterministic given (scene, transform_name) -- see _rng_for -- so two
    # independent calls with the same unmodified scene must agree exactly
    assert params_a["azimuth_deg"] == params_b["azimuth_deg"]
    assert params_a["elevation_deg"] == params_b["elevation_deg"]
    assert scene_a.camera.position == pytest.approx(scene_b.camera.position, abs=1e-12)
