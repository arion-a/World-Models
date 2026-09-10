"""Scalar/vector summaries of a SceneState, used as physical-state-probe targets.

Kept separate from SceneState itself so that "what we choose to probe
for" (a research decision, likely to grow over time) is decoupled from
"what a scene physically contains" (the state representation).
"""

from __future__ import annotations

import numpy as np

from generation.scene import SceneState


def camera_azimuth_deg(state: SceneState) -> float:
    x, y, _ = state.camera.position
    return float(np.degrees(np.arctan2(y, x)))


def camera_elevation_deg(state: SceneState) -> float:
    x, y, z = state.camera.position
    radius_xy = np.hypot(x, y)
    return float(np.degrees(np.arctan2(z, radius_xy)))


def camera_distance(state: SceneState) -> float:
    return float(np.linalg.norm(state.camera.position))


def primary_object_position_xy(state: SceneState) -> np.ndarray:
    """Position of objects[0], the object always present regardless of scene object count."""
    x, y, _ = state.objects[0].position
    return np.array([x, y], dtype=np.float32)
