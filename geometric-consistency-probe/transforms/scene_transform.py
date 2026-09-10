"""Physical scene transformations T: S -> S' = T(S). (Task 3 engine.)

Six transformation families are implemented, matching the project spec:

Geometric (an explicit SE(3) rigid transform -- see transforms/se3.py):
    camera_translation, camera_rotation, object_translation, object_rotation

Non-geometric controls (should leave 3D geometry unchanged, only appearance):
    lighting_change, texture_change

Every function is deterministic given (scene, seed) so transformed pairs
are exactly reproducible, and every function returns both the new
SceneState S' and a dict of ground-truth transform metadata: the
transform's own parameters (e.g. the translation vector, the rotation
angle), the exact 4x4 SE(3) matrix `T` for the geometric ones (`None`
for the two appearance-only controls, which are not rigid transforms),
and `changed_variables` / `fixed_variables` -- the complete, explicit
list of which parts of `SceneState` this call did and did not touch,
per Task 3's "metadata describing which physical variables changed" /
"held fixed" requirement.

For every geometric transform, the returned pose is computed by
*applying* `T` to the old pose matrix (`T @ old_pose_matrix`), not by
computing the new position/orientation separately and then constructing
a matching-looking matrix afterwards -- there is exactly one
computation, and `transform_matrix` in the returned dict is always
exactly the matrix that produced the result. See
tests/test_scene_transform_matrices.py for the tests that check this
class of claim (composition, inverse, isolation) at the SceneState
level, and tests/test_se3_matrices.py for the underlying matrix algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _replace

import numpy as np

from generation.scene import SceneState
from transforms.se3 import (
    matrix_to_pose,
    orbit_rotation_matrix,
    pose_matrix,
    rotate_about_point_matrix,
    rotation_about_axis,
    rotation_only_matrix,
    translation_matrix,
)

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


# --- variable bookkeeping: "which variables changed / were held fixed" -----


def _all_variable_paths(scene: SceneState) -> list[str]:
    """Every leaf field of `scene` this project's transforms could touch,
    named consistently with SceneState's own structure. This is the
    universe `changed_variables`/`fixed_variables` partitions for every
    transform -- computed from the scene itself (not hand-maintained) so
    it can never drift out of sync with an object count that varies
    scene-to-scene.
    """
    paths = ["camera.position", "camera.rotation_euler", "camera.lens_mm", "camera.sensor_width_mm"]
    for i in range(len(scene.objects)):
        paths += [f"objects[{i}].position", f"objects[{i}].rotation_euler", f"objects[{i}].scale", f"objects[{i}].color", f"objects[{i}].shape"]
    paths += ["light.position", "light.energy", "light.color"]
    paths += ["floor_color"]
    return paths


def _changed_and_fixed(scene: SceneState, changed: list[str]) -> tuple[list[str], list[str]]:
    all_paths = _all_variable_paths(scene)
    changed_set = set(changed)
    unknown = changed_set - set(all_paths)
    if unknown:
        raise ValueError(f"changed_variables contains paths not in the scene: {unknown}")
    fixed = [p for p in all_paths if p not in changed_set]
    return sorted(changed_set), fixed


# --- geometric transforms: each computes an explicit SE(3) matrix T and
#     applies it as new_pose_matrix = T @ old_pose_matrix ------------------


def apply_camera_translation(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "camera_translation")
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    magnitude = float(rng.uniform(*cfg.camera_translation_range))
    delta = direction * magnitude

    T = translation_matrix(tuple(delta.tolist()))
    old_pose = pose_matrix(scene.camera.position, scene.camera.rotation_euler)
    # A pure translation's rotation block is exactly the identity, so the
    # orientation is mathematically unchanged; we only take the position
    # out of the matrix product and leave rotation_euler as the literal
    # original tuple, rather than round-tripping it through
    # matrix_to_pose(). The physical rotation matrix would come back
    # bit-identical either way, but Euler angles are not a unique
    # representation of it (see tests/test_transforms.py's
    # test_euler_matrix_roundtrip), so matrix_to_euler() can legitimately
    # return a different-looking (but equivalent) triple -- which would
    # make "camera.rotation_euler" look changed when physically nothing
    # rotated. Skipping the round trip keeps "fixed_variables" true at
    # the field level, not just up to Euler-representation equivalence.
    new_position, _ = matrix_to_pose(T @ old_pose)

    new_camera = _replace(scene.camera, position=new_position)
    new_scene = scene.replace(camera=new_camera)
    changed, fixed = _changed_and_fixed(scene, ["camera.position"])
    params = {
        "delta_position": delta.tolist(),
        "magnitude": magnitude,
        "transform_matrix": T.tolist(),
        "changed_variables": changed,
        "fixed_variables": fixed,
    }
    return new_scene, params


def apply_camera_rotation(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "camera_rotation")
    az_deg = float(rng.uniform(*cfg.camera_rotation_azimuth_deg_range)) * rng.choice([-1.0, 1.0])
    el_deg = float(rng.uniform(*cfg.camera_rotation_elevation_deg_range))
    pivot = np.zeros(3)
    old_position = np.array(scene.camera.position)

    # An orbit about the world origin is exactly a rotation about the
    # origin (pivot = 0), applied identically to the camera's position
    # AND its full orientation -- see transforms/se3.py:orbit_rotation_matrix
    # for why this reproduces "recompute orientation via look-at" exactly,
    # verified in tests/test_se3_matrices.py.
    R = orbit_rotation_matrix(old_position, np.radians(az_deg), np.radians(el_deg))
    T = rotation_only_matrix(R)
    old_pose = pose_matrix(scene.camera.position, scene.camera.rotation_euler)
    new_position, new_rotation_euler = matrix_to_pose(T @ old_pose)

    new_camera = _replace(scene.camera, position=new_position, rotation_euler=new_rotation_euler)
    new_scene = scene.replace(camera=new_camera)
    changed, fixed = _changed_and_fixed(scene, ["camera.position", "camera.rotation_euler"])
    params = {
        "azimuth_deg": az_deg,
        "elevation_deg": el_deg,
        "pivot": pivot.tolist(),
        "transform_matrix": T.tolist(),
        "changed_variables": changed,
        "fixed_variables": fixed,
    }
    return new_scene, params


def apply_object_translation(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "object_translation")
    idx = int(rng.integers(0, len(scene.objects)))
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    magnitude = float(rng.uniform(*cfg.object_translation_range))
    delta_xy = direction * magnitude
    delta = (float(delta_xy[0]), float(delta_xy[1]), 0.0)

    obj = scene.objects[idx]
    T = translation_matrix(delta)
    old_pose = pose_matrix(obj.position, obj.rotation_euler)
    # See apply_camera_translation's comment: rotation is provably
    # unchanged by a pure translation, so it is left as the literal
    # original tuple rather than round-tripped through matrix_to_pose().
    new_position, _ = matrix_to_pose(T @ old_pose)

    new_scene = scene.replace_object(idx, position=new_position)
    changed, fixed = _changed_and_fixed(scene, [f"objects[{idx}].position"])
    params = {
        "object_index": idx,
        "delta_xy": delta_xy.tolist(),
        "magnitude": magnitude,
        "transform_matrix": T.tolist(),
        "changed_variables": changed,
        "fixed_variables": fixed,
    }
    return new_scene, params


def apply_object_rotation(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    rng = _rng_for(scene, "object_rotation")
    idx = int(rng.integers(0, len(scene.objects)))
    angle_deg = float(rng.uniform(*cfg.object_rotation_deg_range)) * rng.choice([-1.0, 1.0])
    axis = np.array([0.0, 0.0, 1.0])

    obj = scene.objects[idx]
    R = rotation_about_axis(axis, np.radians(angle_deg))
    # Rotate the object about ITS OWN position (pivot), not the world
    # origin -- "object rotation in place", the object stays where it is
    # and only its orientation changes. See
    # transforms/se3.py:rotate_about_point_matrix.
    T = rotate_about_point_matrix(R, obj.position)
    old_pose = pose_matrix(obj.position, obj.rotation_euler)
    # rotate_about_point_matrix leaves the pivot exactly fixed by
    # construction (see its docstring/test), so position is left as the
    # literal original tuple rather than round-tripped through the
    # matrix -- floating-point arithmetic in that round trip would only
    # ever perturb it by a few ULP, but there is no reason to accept even
    # that when the true answer ("unchanged") is already known exactly.
    _, new_rotation_euler = matrix_to_pose(T @ old_pose)

    new_scene = scene.replace_object(idx, rotation_euler=new_rotation_euler)
    changed, fixed = _changed_and_fixed(scene, [f"objects[{idx}].rotation_euler"])
    params = {
        "object_index": idx,
        "angle_degrees": angle_deg,
        "axis": axis.tolist(),
        "pivot": list(obj.position),
        "transform_matrix": T.tolist(),
        "changed_variables": changed,
        "fixed_variables": fixed,
    }
    return new_scene, params


# --- appearance controls: not rigid transforms, so transform_matrix=None ---


def apply_lighting_change(scene: SceneState, cfg: TransformConfig) -> tuple[SceneState, dict]:
    from transforms.se3 import orbit_position

    rng = _rng_for(scene, "lighting_change")
    scale = float(rng.uniform(*cfg.light_energy_scale_range))
    az_delta_deg = float(rng.uniform(*cfg.light_azimuth_deg_range)) * rng.choice([-1.0, 1.0])
    pos = np.array(scene.light.position)
    new_pos = orbit_position(pos, np.zeros(3), np.radians(az_delta_deg), 0.0)
    new_energy = scene.light.energy * scale
    new_light = _replace(scene.light, position=tuple(new_pos.tolist()), energy=new_energy)
    new_scene = scene.replace(light=new_light)
    changed, fixed = _changed_and_fixed(scene, ["light.position", "light.energy"])
    params = {
        "energy_scale": scale,
        "azimuth_delta_deg": az_delta_deg,
        "transform_matrix": None,  # not a rigid transform on scene geometry
        "changed_variables": changed,
        "fixed_variables": fixed,
    }
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
    changed, fixed = _changed_and_fixed(scene, [f"objects[{i}].color" for i in range(len(scene.objects))])
    params = {
        "color_deltas": deltas,
        "transform_matrix": None,  # not a rigid transform on scene geometry
        "changed_variables": changed,
        "fixed_variables": fixed,
    }
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
    params["type"] = transform_name  # Task 3's example metadata schema calls this field "type"
    return new_scene, params
