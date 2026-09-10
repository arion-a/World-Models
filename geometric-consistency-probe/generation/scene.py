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
        )


@dataclass(frozen=True)
class CameraState:
    position: Vec3
    rotation_euler: Vec3  # radians, XYZ order, Blender camera convention
    lens_mm: float = 35.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CameraState":
        return CameraState(
            position=tuple(d["position"]),
            rotation_euler=tuple(d["rotation_euler"]),
            lens_mm=float(d.get("lens_mm", 35.0)),
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
