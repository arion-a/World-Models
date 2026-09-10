"""Analytic scene trajectories: `generate_trajectory()` (Task 2).

A `Trajectory` is a sequence of per-frame poses derived from a base
`SceneState` and a constant-velocity motion model for the camera and for
each object. "Constant velocity" is the key design choice: every
per-frame position, orientation, and velocity is an exact closed-form
function of elapsed time (`position(t) = position(0) + velocity * t`,
and similarly a fixed-axis constant-rate rotation for orientation), not
something obtained by numerically integrating or by finite-differencing
rendered poses. That is what makes velocity genuine ground truth here
(see DESIGN.md §2's note on when velocity is/isn't defined) rather than
an estimate, and what makes every frame's pose exactly reproducible from
`(scene.seed, motion parameters, frame_index)` alone.

This module contains no rendering code and does not touch `bpy` --
everything here is pure geometry, which is what makes it independently
unit-testable (tests/test_motion.py) without a Blender install.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generation.scene import SceneState, Vec3
from transforms.se3 import euler_to_matrix, look_at_euler, matrix_to_euler, orbit_position, rotation_about_axis


@dataclass(frozen=True)
class ObjectMotion:
    """Constant-velocity motion for one object, for the whole trajectory."""

    linear_velocity: Vec3 = (0.0, 0.0, 0.0)  # world-frame, scene-units/second
    angular_velocity_axis: Vec3 = (0.0, 0.0, 1.0)  # unit axis, world frame, through the object's own center
    angular_velocity_deg_per_sec: float = 0.0


@dataclass(frozen=True)
class CameraMotion:
    """Motion model for the camera, for the whole trajectory."""

    mode: str = "static"  # "static" | "orbit"
    # Used only when mode == "orbit": constant azimuth angular rate around
    # the world origin; the camera is re-oriented every frame to keep
    # looking at the origin (radius and elevation stay at their initial
    # values) -- the same "orbit" definition as camera_rotation in
    # transforms/scene_transform.py, just continuous in time here instead
    # of a single discrete before/after pair.
    orbit_deg_per_sec: float = 0.0


@dataclass(frozen=True)
class FrameObjectPose:
    instance_id: int
    position: Vec3
    rotation_euler: Vec3
    linear_velocity: Vec3
    angular_velocity_deg_per_sec: float
    angular_velocity_axis: Vec3

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "position": list(self.position),
            "rotation_euler": list(self.rotation_euler),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity_deg_per_sec": self.angular_velocity_deg_per_sec,
            "angular_velocity_axis": list(self.angular_velocity_axis),
        }


@dataclass(frozen=True)
class FrameCameraPose:
    position: Vec3
    rotation_euler: Vec3

    def to_dict(self) -> dict:
        return {"position": list(self.position), "rotation_euler": list(self.rotation_euler)}


@dataclass(frozen=True)
class TrajectoryFrame:
    frame_index: int
    timestamp: float
    camera: FrameCameraPose
    objects: tuple[FrameObjectPose, ...]

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "camera": self.camera.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
        }


@dataclass(frozen=True)
class Trajectory:
    fps: float
    num_frames: int
    frames: tuple[TrajectoryFrame, ...]

    def to_dict(self) -> dict:
        return {"fps": self.fps, "num_frames": self.num_frames, "frames": [f.to_dict() for f in self.frames]}


def _camera_pose_at(scene: SceneState, motion: CameraMotion, t: float) -> FrameCameraPose:
    if motion.mode == "static":
        return FrameCameraPose(position=scene.camera.position, rotation_euler=scene.camera.rotation_euler)
    if motion.mode == "orbit":
        pivot = np.zeros(3)
        pos0 = np.array(scene.camera.position)
        az_delta_rad = np.radians(motion.orbit_deg_per_sec * t)
        new_pos = orbit_position(pos0, pivot, az_delta_rad, 0.0)
        new_euler = look_at_euler(new_pos, pivot)
        return FrameCameraPose(position=tuple(new_pos.tolist()), rotation_euler=new_euler)
    raise ValueError(f"Unknown camera motion mode '{motion.mode}'. Known: 'static', 'orbit'.")


def _object_pose_at(obj_position: Vec3, obj_rotation_euler: Vec3, instance_id: int, motion: ObjectMotion, t: float) -> FrameObjectPose:
    new_pos = np.array(obj_position) + np.array(motion.linear_velocity) * t
    angle_rad = np.radians(motion.angular_velocity_deg_per_sec * t)
    if angle_rad != 0.0:
        delta_rot = rotation_about_axis(np.array(motion.angular_velocity_axis), angle_rad)
        new_rot_matrix = delta_rot @ euler_to_matrix(obj_rotation_euler)
        new_euler = matrix_to_euler(new_rot_matrix)
    else:
        new_euler = obj_rotation_euler
    return FrameObjectPose(
        instance_id=instance_id,
        position=tuple(new_pos.tolist()),
        rotation_euler=new_euler,
        linear_velocity=motion.linear_velocity,
        angular_velocity_deg_per_sec=motion.angular_velocity_deg_per_sec,
        angular_velocity_axis=motion.angular_velocity_axis,
    )


def generate_trajectory(
    scene: SceneState,
    object_motions: dict[int, ObjectMotion] | None = None,
    camera_motion: CameraMotion | None = None,
    num_frames: int = 8,
    fps: float = 12.0,
) -> Trajectory:
    """Compute the per-frame ground-truth trajectory for `scene`.

    `object_motions` maps `instance_id -> ObjectMotion`; any object
    without an entry is stationary (`ObjectMotion()` default: zero linear
    and angular velocity), which is a legitimate, explicit choice, not a
    missing one -- see DESIGN.md §2, "velocity is defined but not always
    observable": here it is always defined and always available, and
    zero is a real value, not a stand-in for "unknown".

    Deterministic given `(scene, object_motions, camera_motion,
    num_frames, fps)`: no randomness is introduced in this function
    itself (any randomness in *choosing* a motion belongs to the caller,
    e.g. a config-driven sampler), so the same inputs always produce
    bit-for-bit the same trajectory.
    """
    camera_motion = camera_motion or CameraMotion()
    object_motions = object_motions or {}
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")

    dt = 1.0 / fps
    frames = []
    for f in range(num_frames):
        t = f * dt
        cam_pose = _camera_pose_at(scene, camera_motion, t)
        obj_poses = tuple(
            _object_pose_at(
                obj.position,
                obj.rotation_euler,
                obj.instance_id,
                object_motions.get(obj.instance_id, ObjectMotion()),
                t,
            )
            for obj in scene.objects
        )
        frames.append(TrajectoryFrame(frame_index=f, timestamp=t, camera=cam_pose, objects=obj_poses))
    return Trajectory(fps=fps, num_frames=num_frames, frames=tuple(frames))
