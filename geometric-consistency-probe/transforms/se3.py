"""Minimal SE(3) helpers: rotation matrices and Euler-angle composition.

Kept separate from any scene or rendering code so the math can be unit
tested in isolation (see tests/test_transforms.py).
"""

from __future__ import annotations

import numpy as np


def euler_to_matrix(euler: tuple[float, float, float]) -> np.ndarray:
    """XYZ-order Euler angles (radians) -> 3x3 rotation matrix (Blender convention)."""
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def matrix_to_euler(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of euler_to_matrix (XYZ order, Blender convention)."""
    sy = -rot[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    ry = np.arcsin(sy)
    if abs(np.cos(ry)) > 1e-6:
        rx = np.arctan2(rot[2, 1], rot[2, 2])
        rz = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        rx = np.arctan2(-rot[1, 2], rot[1, 1])
        rz = 0.0
    return float(rx), float(ry), float(rz)


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula: 3x3 rotation matrix for `angle_rad` about unit `axis`."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def look_at_euler(camera_pos: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Blender-convention XYZ Euler angles for a camera at `camera_pos` looking at `target`.

    Blender's camera looks down its local -Z axis with local +Y "up".
    world_up is world +Z, except when the view direction is itself
    (near-)vertical, in which case world +Y is used instead to avoid a
    degenerate (zero-length) cross product.
    """
    direction = target - camera_pos
    direction = direction / np.linalg.norm(direction)
    forward = -direction
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    rot = np.stack([right, up, forward], axis=1)  # world = rot @ local
    return matrix_to_euler(rot)


def orbit_position(position: np.ndarray, pivot: np.ndarray, azimuth_delta_rad: float, elevation_delta_rad: float) -> np.ndarray:
    """Move `position` around `pivot` by a change in azimuth/elevation on a sphere.

    Used to express "camera rotation" as an orbit of the camera around the
    scene center, which changes viewpoint (a real geometric transform)
    without changing the camera-to-pivot distance.

    Note: elevation is clamped away from the poles (+-pi/2) to avoid the
    azimuth singularity there. This project's configured elevation ranges
    never approach that regime (see orbit_rotation_matrix's docstring for
    why that matters), so the clamp is inactive in practice, but it does
    mean this function is not *exactly* a rotation near the poles -- only
    `orbit_rotation_matrix` below is used when an exact SE(3) matrix is
    required (Task 3).
    """
    rel = position - pivot
    radius = np.linalg.norm(rel)
    az = np.arctan2(rel[1], rel[0])
    el = np.arcsin(np.clip(rel[2] / radius, -1.0, 1.0))
    az2 = az + azimuth_delta_rad
    el2 = np.clip(el + elevation_delta_rad, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
    new_rel = radius * np.array(
        [np.cos(el2) * np.cos(az2), np.cos(el2) * np.sin(az2), np.sin(el2)]
    )
    return pivot + new_rel


def orbit_rotation_matrix(position: np.ndarray, azimuth_delta_rad: float, elevation_delta_rad: float) -> np.ndarray:
    """The single 3x3 rotation (about the origin) equivalent to `orbit_position`
    (with `pivot = origin`) for a position AWAY FROM THE POLES.

    An azimuth change alone is a rotation about world +Z. An elevation
    change alone, *at a fixed azimuth*, is a rotation about the
    horizontal "East" tangent direction at that azimuth
    `(sin(az), -cos(az), 0)` (sign chosen, and verified numerically
    against `orbit_position`, so that a positive elevation_delta raises
    the point). Composing "rotate azimuth, then rotate elevation about
    the tangent at the NEW azimuth" gives a single rotation matrix that:

      1. reproduces `orbit_position(position, origin, az_delta, el_delta)`
         exactly (`R @ position == orbit_position(...)`), and
      2. applied to a camera's full camera-to-world *rotation* (not just
         its position) reproduces exactly the orientation you'd get by
         re-deriving it from scratch with `look_at_euler(new_position,
         origin)` -- i.e. this one matrix correctly transforms the whole
         rigid pose, not just the position.

    Both properties were verified numerically (not just derived) over
    the elevation range this project actually uses; see
    tests/test_se3_matrices.py. This only holds for `pivot = origin`
    (world-frame rotation about the point every orbiting camera pose is
    measured from) and away from the +-90 degree elevation poles, where
    `orbit_position`'s clamping (see its docstring) makes the two
    diverge -- not a concern for this project's configured elevation
    ranges (see configs default camera elevation/rotation ranges), but a
    real limit of this function, not swept under the rug.
    """
    world_z = np.array([0.0, 0.0, 1.0])
    R_az = rotation_about_axis(world_z, azimuth_delta_rad)
    az_old = np.arctan2(position[1], position[0])
    az_new = az_old + azimuth_delta_rad
    tangent = np.array([np.sin(az_new), -np.cos(az_new), 0.0])
    R_el = rotation_about_axis(tangent, elevation_delta_rad)
    return R_el @ R_az


# --- 4x4 homogeneous (SE(3)) matrices -----------------------------------
#
# Convention used throughout this project: a 4x4 matrix M = [[R, t], [0, 1]]
# represents a rigid pose as "local-frame-to-world", i.e. it maps a point
# given in the object/camera's own local coordinates to world coordinates:
# p_world = M @ [p_local; 1]. For a camera this is exactly the
# "camera-to-world" matrix (see generation/scene.py:camera_to_world_matrix);
# its inverse is "world-to-camera" -- see world_to_camera below for why
# these are NOT interchangeable and must not be confused.


def rt_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous matrix from a 3x3 rotation and a translation."""
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def matrix_to_rt(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of rt_to_matrix: (R, t) = (M[:3,:3], M[:3,3])."""
    return M[:3, :3].copy(), M[:3, 3].copy()


def pose_matrix(position: tuple[float, float, float], rotation_euler: tuple[float, float, float]) -> np.ndarray:
    """4x4 local-to-world pose matrix for a (position, XYZ-Euler) pair."""
    return rt_to_matrix(euler_to_matrix(rotation_euler), np.array(position, dtype=float))


def matrix_to_pose(M: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Inverse of pose_matrix: (position, rotation_euler)."""
    R, t = matrix_to_rt(M)
    return tuple(t.tolist()), matrix_to_euler(R)


def identity_matrix() -> np.ndarray:
    return np.eye(4)


def translation_matrix(delta: tuple[float, float, float]) -> np.ndarray:
    """Pure-translation SE(3) matrix (rotation = identity)."""
    return rt_to_matrix(np.eye(3), np.array(delta, dtype=float))


def rotation_only_matrix(R: np.ndarray) -> np.ndarray:
    """Pure-rotation-about-the-origin SE(3) matrix (translation = 0)."""
    return rt_to_matrix(R, np.zeros(3))


def rotate_about_point_matrix(R: np.ndarray, pivot: tuple[float, float, float]) -> np.ndarray:
    """SE(3) matrix for "rotate by R about a fixed world point `pivot`"
    (rather than about the origin): Translate(pivot) @ RotateOnly(R) @
    Translate(-pivot). Applying this to a pose leaves an object exactly
    at `pivot` unchanged in position and only rotates its orientation --
    which is what "object rotation in place" (Task 3, camera_rotation
    uses `pivot = origin` instead, i.e. this reduces to
    `rotation_only_matrix(R)`) means concretely in SE(3) terms.
    """
    pivot = np.array(pivot, dtype=float)
    return translation_matrix(pivot) @ rotation_only_matrix(R) @ translation_matrix(-pivot)


def inverse_rigid(M: np.ndarray) -> np.ndarray:
    """Exact inverse of a rigid (SE(3)) matrix: [R^T, -R^T @ t], NOT [R^T, -t].

    Using `-t` instead of `-R^T @ t` is the single most common SE(3)
    bug and exactly what "do not assume [world-to-camera and
    camera-to-world] are interchangeable" (Task 3) is warning about --
    see generation/scene.py:world_to_camera_matrix and
    tests/test_se3_matrices.py for the regression test against it.
    """
    R, t = matrix_to_rt(M)
    R_inv = R.T
    return rt_to_matrix(R_inv, -R_inv @ t)


def compose(*matrices: np.ndarray) -> np.ndarray:
    """Compose SE(3) matrices left-to-right as they'd be applied: compose(A, B, C) == A @ B @ C."""
    result = np.eye(4)
    for M in matrices:
        result = result @ M
    return result
