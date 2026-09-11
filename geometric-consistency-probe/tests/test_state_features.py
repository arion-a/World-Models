"""Tests for generation/state_features.py -- the physical-state-probe
target functions used by Task 8 (see experiments/task8_physical_state.py
and tasks/08_physical_state.md's Required software tests: "correct
labels, verified against known SceneState fixtures").
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from generation.scene import CameraState, LightState, ObjectState, SceneState
from generation.state_features import (
    camera_azimuth_deg,
    camera_distance,
    camera_elevation_deg,
    primary_object_position_xy,
)


def _make_scene(camera_position, object_position=(0.5, -0.3, 0.4)) -> SceneState:
    return SceneState(
        scene_id="fixture",
        seed=0,
        objects=(
            ObjectState(
                shape="cube",
                position=object_position,
                rotation_euler=(0.0, 0.0, 0.0),
                scale=1.0,
                color=(0.5, 0.5, 0.5),
                instance_id=1,
            ),
        ),
        camera=CameraState(position=camera_position, rotation_euler=(0.0, 0.0, 0.0)),
        light=LightState(position=(0.0, 0.0, 5.0), energy=3.0, color=(1.0, 1.0, 1.0)),
    )


def test_camera_azimuth_deg_matches_hand_computed_value_on_axis():
    # Camera directly on +X axis at (5, 0, 0): azimuth = 0 degrees.
    scene = _make_scene((5.0, 0.0, 0.0))
    assert camera_azimuth_deg(scene) == pytest.approx(0.0)

    # Camera directly on +Y axis: azimuth = 90 degrees.
    scene = _make_scene((0.0, 5.0, 0.0))
    assert camera_azimuth_deg(scene) == pytest.approx(90.0)

    # Camera on -X axis: azimuth = +-180 degrees.
    scene = _make_scene((-5.0, 0.0, 0.0))
    assert abs(camera_azimuth_deg(scene)) == pytest.approx(180.0)

    # Camera on -Y axis: azimuth = -90 degrees.
    scene = _make_scene((0.0, -5.0, 0.0))
    assert camera_azimuth_deg(scene) == pytest.approx(-90.0)


def test_camera_azimuth_deg_arbitrary_position_matches_atan2():
    x, y, z = 3.0, 4.0, 2.0
    scene = _make_scene((x, y, z))
    expected = math.degrees(math.atan2(y, x))
    assert camera_azimuth_deg(scene) == pytest.approx(expected)


def test_camera_elevation_deg_matches_hand_computed_value():
    # Camera straight up: elevation = 90 degrees.
    scene = _make_scene((0.0, 0.0, 5.0))
    assert camera_elevation_deg(scene) == pytest.approx(90.0)

    # Camera in the XY-plane: elevation = 0 degrees.
    scene = _make_scene((5.0, 0.0, 0.0))
    assert camera_elevation_deg(scene) == pytest.approx(0.0)

    x, y, z = 3.0, 4.0, 2.0
    scene = _make_scene((x, y, z))
    expected = math.degrees(math.atan2(z, math.hypot(x, y)))
    assert camera_elevation_deg(scene) == pytest.approx(expected)


def test_camera_distance_matches_euclidean_norm():
    scene = _make_scene((3.0, 4.0, 0.0))
    assert camera_distance(scene) == pytest.approx(5.0)

    scene = _make_scene((1.0, 2.0, 2.0))
    assert camera_distance(scene) == pytest.approx(3.0)


def test_primary_object_position_xy_reads_objects_zero():
    scene = _make_scene((5.0, 0.0, 0.0), object_position=(1.5, -2.5, 0.7))
    pos = primary_object_position_xy(scene)
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (2,)
    np.testing.assert_allclose(pos, [1.5, -2.5], atol=1e-6)


def test_primary_object_position_xy_ignores_other_objects():
    scene = _make_scene((5.0, 0.0, 0.0), object_position=(0.1, 0.2, 0.3))
    extra_object = ObjectState(
        shape="cone",
        position=(9.9, 9.9, 9.9),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=1.0,
        color=(0.1, 0.1, 0.1),
        instance_id=2,
    )
    scene = scene.replace(objects=(scene.objects[0], extra_object))
    pos = primary_object_position_xy(scene)
    np.testing.assert_allclose(pos, [0.1, 0.2], atol=1e-6)


def test_feature_functions_return_finite_values():
    rng = np.random.default_rng(0)
    for _ in range(20):
        camera_position = tuple(rng.uniform(-8, 8, size=3).tolist())
        object_position = tuple(rng.uniform(-2, 2, size=3).tolist())
        scene = _make_scene(camera_position, object_position)
        for fn in (camera_azimuth_deg, camera_elevation_deg, camera_distance):
            value = fn(scene)
            assert np.isfinite(value)
        pos = primary_object_position_xy(scene)
        assert np.all(np.isfinite(pos))
