"""Physical scene transformations T: S -> S' = T(S).

Six transformation families are implemented, matching the project spec:

Geometric (expected to move the physical state in a well-defined way):
    camera_translation, camera_rotation, object_translation, object_rotation

Non-geometric controls (should leave 3D geometry unchanged, only appearance):
    lighting_change, texture_change

Every function is deterministic given (scene, seed) so transformed pairs
are exactly reproducible, and every function returns both the new
SceneState S' and a small dict of ground-truth transform parameters
(e.g. the translation vector, the rotation angle) that downstream probes
can use as regression targets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _replace

import numpy as np

from generation.scene import SceneState
from transforms.se3 import euler_to_matrix, matrix_to_euler, orbit_position, rotation_about_axis

TRANSFORM_NAMES = (
    "camera_translation",
    "camera_rotation",
    "object_translation",
    "object_rotation",
    "lighting_change",
    "texture_change",
)

GEOMETRIC_TRANSFORMS = ("camera_translation", "camera_rotation", "object_translation", "object_rotation")
CONTROL_TRANSFORMS = ("lighting_change", "texture_change")


@dataclass
class TransformConfig:
    camera_translation_range: tuple[float, float] = (0.5, 1.5)
    camera_rotation_azimuth_deg_range: tuple[float, float] = (10.0, 35.0)
    camera_rotation_elevation_deg_range: tuple[float, float] = (-10.0, 10.0)
    object_translation_range: tuple[float, float] = (0.4, 1.0)
    object_rotation_deg_range: tuple[float, float] = (30.0, 150.0)
    light_energy_scale_range: tuple[float, float] = (0.3, 2.5)
    light_azimuth_deg_range: tuple[float, float] = (60.0, 180.0)
    texture_color_jitter: float = 0.6


def _rng_for(scene: SceneState, transform_name: str) -> np.random.Generator:
    # Derived, not reused, from the scene seed: each transform type gets an
    # independent stream so applying several transforms to the same scene
    # never correlates their random parameters. Python's built-in hash() of
    # a str is randomized per-process (PYTHONHASHSEED), so we use a stable
    # hash (zlib.crc32) instead -- otherwise the "same seed" would not
    # reproduce the same scene across separate runs/processes.
    import zlib

    name_hash = zlib.crc32(transform_name.encode("utf-8"))
    return np.random.default_rng((scene.seed, name_hash))


def apply_camera_translation(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "camera_translation")
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    magnitude = float(rng.uniform(*cfg.camera_translation_range))
    delta = direction * magnitude
    new_pos = np.array(scene.camera.position) + delta
    new_camera = _replace(scene.camera, position=tuple(new_pos.tolist()))
    new_scene = scene.replace(camera=new_camera)
    params = {"delta_position": delta.tolist(), "magnitude": magnitude}
    return new_scene, params


def apply_camera_rotation(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "camera_rotation")
    az_deg = float(rng.uniform(*cfg.camera_rotation_azimuth_deg_range)) * rng.choice([-1.0, 1.0])
    el_deg = float(rng.uniform(*cfg.camera_rotation_elevation_deg_range))
    pivot = np.zeros(3)
    old_pos = np.array(scene.camera.position)
    new_pos = orbit_position(old_pos, pivot, np.radians(az_deg), np.radians(el_deg))

    # Recompute orientation so the camera keeps looking at the pivot
    # (an orbit, not a translation): this isolates "viewpoint rotation"
    # from "viewpoint translation".
    direction = pivot - new_pos
    direction /= np.linalg.norm(direction)
    forward = -direction
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    rot = np.stack([right, up, forward], axis=1)
    new_euler = matrix_to_euler(rot)

    new_camera = _replace(scene.camera, position=tuple(new_pos.tolist()), rotation_euler=new_euler)
    new_scene = scene.replace(camera=new_camera)
    params = {"azimuth_deg": az_deg, "elevation_deg": el_deg}
    return new_scene, params


def apply_object_translation(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "object_translation")
    idx = int(rng.integers(0, len(scene.objects)))
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    magnitude = float(rng.uniform(*cfg.object_translation_range))
    delta_xy = direction * magnitude
    obj = scene.objects[idx]
    new_pos = (obj.position[0] + delta_xy[0], obj.position[1] + delta_xy[1], obj.position[2])
    new_scene = scene.replace_object(idx, position=new_pos)
    params = {"object_index": idx, "delta_xy": delta_xy.tolist(), "magnitude": magnitude}
    return new_scene, params


def apply_object_rotation(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "object_rotation")
    idx = int(rng.integers(0, len(scene.objects)))
    angle_deg = float(rng.uniform(*cfg.object_rotation_deg_range)) * rng.choice([-1.0, 1.0])
    obj = scene.objects[idx]
    delta_rot = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.radians(angle_deg))
    old_rot = euler_to_matrix(obj.rotation_euler)
    new_rot = delta_rot @ old_rot
    new_euler = matrix_to_euler(new_rot)
    new_scene = scene.replace_object(idx, rotation_euler=new_euler)
    params = {"object_index": idx, "angle_deg": angle_deg, "axis": [0.0, 0.0, 1.0]}
    return new_scene, params


def apply_lighting_change(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "lighting_change")
    scale = float(rng.uniform(*cfg.light_energy_scale_range))
    az_delta_deg = float(rng.uniform(*cfg.light_azimuth_deg_range)) * rng.choice([-1.0, 1.0])
    pos = np.array(scene.light.position)
    new_pos = orbit_position(pos, np.zeros(3), np.radians(az_delta_deg), 0.0)
    new_energy = scene.light.energy * scale
    new_light = _replace(scene.light, position=tuple(new_pos.tolist()), energy=new_energy)
    new_scene = scene.replace(light=new_light)
    params = {"energy_scale": scale, "azimuth_delta_deg": az_delta_deg}
    return new_scene, params


def apply_texture_change(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "texture_change")
    jitter = cfg.texture_color_jitter
    new_objects = []
    deltas = []
    for obj in scene.objects:
        delta = rng.uniform(-jitter, jitter, size=3)
        new_color = tuple(np.clip(np.array(obj.color) + delta, 0.05, 0.95).tolist())
        new_objects.append(obj.__class__(**{**obj.to_dict(), "color": new_color}))
        deltas.append(delta.tolist())
    new_scene = scene.replace(objects=tuple(new_objects))
    params = {"color_deltas": deltas}
    return new_scene, params


_REGISTRY = {
    "camera_translation": apply_camera_translation,
    "camera_rotation": apply_camera_rotation,
    "object_translation": apply_object_translation,
    "object_rotation": apply_object_rotation,
    "lighting_change": apply_lighting_change,
    "texture_change": apply_texture_change,
}


def apply_transform(scene: SceneState, transform_name: str, cfg: TransformConfig | None = None) -> tuple[SceneState, dict]:
    if transform_name not in _REGISTRY:
        raise ValueError(f"Unknown transform '{transform_name}'. Known: {sorted(_REGISTRY)}")
    cfg = cfg or TransformConfig()
    new_scene, params = _REGISTRY[transform_name](scene, cfg)
    params["transform_name"] = transform_name
    return new_scene, params
