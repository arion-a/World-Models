"""Physical scene-state representation.

This module defines the physical state ``S`` in the pipeline

    S --renderer R--> V --encoder E--> Z

``SceneState`` is a plain, serializable description of a small tabletop
scene: a handful of rigid objects, a camera, and a light. It carries no
rendering-backend detail (no Blender objects, no OpenGL handles) so that
the renderer is the only module that needs to know how to turn a
``SceneState`` into pixels, and so ``SceneState`` instances can be
diffed, hashed, and stored as JSON for exact reproducibility.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np

Vec3 = tuple[float, float, float]

# Primitive shapes used for V0. All but "sphere" are rotationally
# asymmetric, which matters for the object-rotation transform: rotating a
# sphere in place produces a pixel-identical render, so sphere is excluded
# from the default shape pool (kept here only as a documented option for
# invariance sanity checks).
SHAPES = ("cube", "cone", "cylinder", "monkey")


@dataclass(frozen=True)
class ObjectState:
    shape: str
    position: Vec3
    rotation_euler: Vec3  # radians, XYZ order
    scale: float
    color: tuple[float, float, float]  # linear RGB in [0, 1]
    # Persistent instance id, unique within a scene, used both as the
    # segmentation-mask pixel value (see generation/bpy_renderer.py) and as
    # the key that ties one object's identity across frames/transforms.
    # 0 is reserved for "not an object" (background/floor) -- see
    # generation/COORDINATE_SYSTEM.md. Default 0 only for backward
    # compatibility with code that never assigns one; generate_scene()
    # (Task 2) always assigns 1..N.
    instance_id: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ObjectState":
        return ObjectState(
            shape=d["shape"],
            position=tuple(d["position"]),
            rotation_euler=tuple(d["rotation_euler"]),
            scale=float(d["scale"]),
            color=tuple(d["color"]),
            instance_id=int(d.get("instance_id", 0)),
        )


@dataclass(frozen=True)
class CameraState:
    position: Vec3
    rotation_euler: Vec3  # radians, XYZ order, Blender camera convention
    lens_mm: float = 35.0
    # Blender's camera "sensor width" for a perspective lens -- together
    # with lens_mm and the render resolution this fully determines the
    # pinhole intrinsics matrix K (see camera_intrinsics() below). 32mm is
    # Blender's own default sensor width.
    sensor_width_mm: float = 32.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CameraState":
        return CameraState(
            position=tuple(d["position"]),
            rotation_euler=tuple(d["rotation_euler"]),
            lens_mm=float(d.get("lens_mm", 35.0)),
            sensor_width_mm=float(d.get("sensor_width_mm", 32.0)),
        )


@dataclass(frozen=True)
class LightState:
    position: Vec3
    energy: float  # Watts, Blender SUN lamp strength proxy
    color: tuple[float, float, float]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "LightState":
        return LightState(
            position=tuple(d["position"]),
            energy=float(d["energy"]),
            color=tuple(d["color"]),
        )


@dataclass(frozen=True)
class SceneState:
    """The full physical state S of one synthetic scene."""

    scene_id: str
    seed: int
    objects: tuple[ObjectState, ...]
    camera: CameraState
    light: LightState
    floor_color: tuple[float, float, float] = (0.6, 0.6, 0.6)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "seed": self.seed,
            "objects": [o.to_dict() for o in self.objects],
            "camera": self.camera.to_dict(),
            "light": self.light.to_dict(),
            "floor_color": list(self.floor_color),
        }

    @staticmethod
    def from_dict(d: dict) -> "SceneState":
        return SceneState(
            scene_id=d["scene_id"],
            seed=int(d["seed"]),
            objects=tuple(ObjectState.from_dict(o) for o in d["objects"]),
            camera=CameraState.from_dict(d["camera"]),
            light=LightState.from_dict(d["light"]),
            floor_color=tuple(d.get("floor_color", (0.6, 0.6, 0.6))),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "SceneState":
        return SceneState.from_dict(json.loads(s))

    def replace(self, **kwargs) -> "SceneState":
        """Return a new SceneState with the given fields replaced (S is immutable)."""
        return dataclasses.replace(self, **kwargs)

    def replace_object(self, index: int, **kwargs) -> "SceneState":
        objs = list(self.objects)
        objs[index] = dataclasses.replace(objs[index], **kwargs)
        return self.replace(objects=tuple(objs))


def camera_forward_vector(camera: CameraState) -> np.ndarray:
    """Unit forward (viewing) direction of the camera in world space.

    Blender cameras look down their local -Z axis with +Y as "up" in
    camera space; rotation_euler is applied in XYZ order.
    """
    rx, ry, rz = camera.rotation_euler
    local_forward = np.array([0.0, 0.0, -1.0])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rot = rot_z @ rot_y @ rot_x
    return rot @ local_forward


def camera_intrinsics(camera: CameraState, resolution_x: int, resolution_y: int) -> dict:
    """Pinhole intrinsics (fx, fy, cx, cy, and the 3x3 matrix K) for `camera`.

    Standard Blender perspective-camera conversion (sensor fit 'AUTO', no
    lens shift): the sensor width in mm maps to the *larger* image
    dimension. This project always renders square frames
    (resolution_x == resolution_y), so fx == fy and the formula is exact
    without needing to special-case the sensor-fit axis; that assumption
    is asserted here rather than silently mishandled for a non-square
    render. See generation/COORDINATE_SYSTEM.md for the full camera model.
    """
    if resolution_x != resolution_y:
        raise NotImplementedError(
            "camera_intrinsics() assumes a square render (sensor fit AUTO maps "
            "sensor_width_mm to the larger dimension); non-square resolutions "
            "need that axis handled explicitly and are not supported yet."
        )
    fx = resolution_x * camera.lens_mm / camera.sensor_width_mm
    fy = fx
    cx = resolution_x / 2.0
    cy = resolution_y / 2.0
    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K}
