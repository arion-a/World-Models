"""Deterministic sampling of random scenes.

Every scene is generated from a single integer seed so that the whole
dataset is exactly reproducible, and scene identity is defined purely by
that seed (never by wall-clock time, dict ordering, or RNG global state).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generation.scene import CameraState, LightState, ObjectState, SceneState, SHAPES


@dataclass
class SceneSamplerConfig:
    num_objects_min: int = 1
    num_objects_max: int = 3
    object_position_xy_range: float = 1.6
    object_scale_range: tuple[float, float] = (0.5, 1.0)
    min_object_separation: float = 1.1

    camera_radius_range: tuple[float, float] = (6.0, 8.0)
    camera_azimuth_range_deg: tuple[float, float] = (0.0, 360.0)
    camera_elevation_range_deg: tuple[float, float] = (20.0, 50.0)

    light_energy_range: tuple[float, float] = (2.5, 5.0)
    light_azimuth_range_deg: tuple[float, float] = (0.0, 360.0)
    light_elevation_range_deg: tuple[float, float] = (30.0, 70.0)

    floor_color_range: tuple[float, float] = (0.35, 0.75)


def _look_at_euler(camera_pos: np.ndarray, target: np.ndarray = np.zeros(3)) -> tuple[float, float, float]:
    """Euler angles (Blender XYZ) for a camera at camera_pos looking at target."""
    direction = target - camera_pos
    direction = direction / np.linalg.norm(direction)
    # Blender camera looks down local -Z, with local +Y "up".
    forward = -direction
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    rot = np.stack([right, up, forward], axis=1)  # world = rot @ local
    # Extract XYZ Euler angles from rotation matrix (Blender convention).
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


def _sphere_position(radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    x = radius * np.cos(el) * np.cos(az)
    y = radius * np.cos(el) * np.sin(az)
    z = radius * np.sin(el)
    return np.array([x, y, z])


def sample_scene(scene_id: str, seed: int, cfg: SceneSamplerConfig | None = None) -> SceneState:
    """Sample one random SceneState. Deterministic given (scene_id, seed, cfg)."""
    cfg = cfg or SceneSamplerConfig()
    rng = np.random.default_rng(seed)

    n_objects = int(rng.integers(cfg.num_objects_min, cfg.num_objects_max + 1))
    objects = []
    positions_xy: list[np.ndarray] = []
    attempts = 0
    while len(objects) < n_objects and attempts < 200:
        attempts += 1
        xy = rng.uniform(-cfg.object_position_xy_range, cfg.object_position_xy_range, size=2)
        if any(np.linalg.norm(xy - p) < cfg.min_object_separation for p in positions_xy):
            continue
        shape = rng.choice(SHAPES)
        scale = float(rng.uniform(*cfg.object_scale_range))
        z = scale if shape != "monkey" else scale * 0.6
        rotation = tuple(rng.uniform(0, 2 * np.pi, size=3).tolist())
        color = tuple(rng.uniform(0.1, 0.9, size=3).tolist())
        objects.append(
            ObjectState(
                shape=str(shape),
                position=(float(xy[0]), float(xy[1]), float(z)),
                rotation_euler=rotation,
                scale=scale,
                color=color,
            )
        )
        positions_xy.append(xy)

    cam_radius = float(rng.uniform(*cfg.camera_radius_range))
    cam_az = float(rng.uniform(*cfg.camera_azimuth_range_deg))
    cam_el = float(rng.uniform(*cfg.camera_elevation_range_deg))
    cam_pos = _sphere_position(cam_radius, cam_az, cam_el)
    cam_rot = _look_at_euler(cam_pos)
    camera = CameraState(position=tuple(cam_pos.tolist()), rotation_euler=cam_rot)

    light_az = float(rng.uniform(*cfg.light_azimuth_range_deg))
    light_el = float(rng.uniform(*cfg.light_elevation_range_deg))
    light_pos = _sphere_position(10.0, light_az, light_el)
    light_energy = float(rng.uniform(*cfg.light_energy_range))
    light_color = tuple(rng.uniform(0.9, 1.0, size=3).tolist())
    light = LightState(position=tuple(light_pos.tolist()), energy=light_energy, color=light_color)

    floor_shade = float(rng.uniform(*cfg.floor_color_range))
    floor_color = (floor_shade, floor_shade, floor_shade)

    return SceneState(
        scene_id=scene_id,
        seed=seed,
        objects=tuple(objects),
        camera=camera,
        light=light,
        floor_color=floor_color,
    )
