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


def orbit_position(position: np.ndarray, pivot: np.ndarray, azimuth_delta_rad: float, elevation_delta_rad: float) -> np.ndarray:
    """Move `position` around `pivot` by a change in azimuth/elevation on a sphere.

    Used to express "camera rotation" as an orbit of the camera around the
    scene center, which changes viewpoint (a real geometric transform)
    without changing the camera-to-pivot distance.
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
