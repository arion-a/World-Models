"""`save_ground_truth()` (Task 2): write one scene's complete ground truth to disk.

Layout under `out_dir/{scene_id}/`:

    metadata.json     -- scene config, camera intrinsics, per-frame camera
                          and object poses/velocities, encoding notes
    rgb.npy           -- (T, H, W, 3) uint8   the "RGB video"
    depth.npy         -- (T, H, W) float32    scene-unit camera-space depth
    segmentation.npy  -- (T, H, W) uint8      instance id per pixel (0 = background)

See generation/COORDINATE_SYSTEM.md for exactly what every field and
array means. `rgb`/`depth`/`segmentation` are saved as single stacked
arrays rather than per-frame image files: this project's "video" is by
definition just the ordered frame sequence produced by one
`render_trajectory` call, and a single `.npy` is the most direct,
lossless way to store that sequence for programmatic reuse (loading it
back is one `np.load` call) -- see IMPLEMENTATION_NOTES.md if a
per-frame-image or compressed-video format is wanted later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from generation.bpy_renderer import ClipGroundTruth
from generation.motion import Trajectory
from generation.scene import SceneState, camera_intrinsics


def _build_metadata(scene: SceneState, trajectory: Trajectory, clip: ClipGroundTruth, resolution: int) -> dict:
    intrinsics = camera_intrinsics(scene.camera, resolution, resolution)

    camera_per_frame = [
        {
            "frame_index": f.frame_index,
            "timestamp": f.timestamp,
            "position": list(f.camera.position),
            "rotation_euler": list(f.camera.rotation_euler),
        }
        for f in trajectory.frames
    ]

    objects_meta = []
    for obj in scene.objects:
        per_frame = []
        for frame in trajectory.frames:
            pose = next(p for p in frame.objects if p.instance_id == obj.instance_id)
            per_frame.append(
                {
                    "frame_index": frame.frame_index,
                    "timestamp": frame.timestamp,
                    "position": list(pose.position),
                    "rotation_euler": list(pose.rotation_euler),
                    "linear_velocity": list(pose.linear_velocity),
                    "angular_velocity_deg_per_sec": pose.angular_velocity_deg_per_sec,
                    "angular_velocity_axis": list(pose.angular_velocity_axis),
                }
            )
        objects_meta.append(
            {
                "instance_id": obj.instance_id,
                "shape": obj.shape,
                "scale": obj.scale,
                "color": list(obj.color),
                "per_frame_pose": per_frame,
            }
        )

    return {
        "scene_id": scene.scene_id,
        "seed": scene.seed,
        "resolution": resolution,
        "fps": trajectory.fps,
        "num_frames": trajectory.num_frames,
        "camera": {
            "lens_mm": scene.camera.lens_mm,
            "sensor_width_mm": scene.camera.sensor_width_mm,
            "intrinsics": intrinsics,
            "per_frame_pose": camera_per_frame,
        },
        "objects": objects_meta,
        "light": scene.light.to_dict(),
        "floor_color": list(scene.floor_color),
        "depth_encoding": {
            "near": clip.depth_near,
            "far": clip.depth_far,
            "note": (
                "depth.npy already stores decoded scene-unit float32 depth "
                "(camera-space Z, i.e. perpendicular distance to the camera's "
                "image plane -- NOT Euclidean ray distance; see "
                "generation/COORDINATE_SYSTEM.md). Values beyond `far` were "
                "clamped to `far` during rendering, so `far` acts as a "
                "saturation ceiling, not a guarantee that nothing is farther."
            ),
        },
        "segmentation_encoding": {
            "max_instances": clip.max_instances,
            "note": (
                "segmentation.npy stores instance_id directly as a uint8 "
                "label map; 0 = background/floor, i = the object with that "
                "instance_id in `objects`."
            ),
        },
        "coordinate_system": (
            "See generation/COORDINATE_SYSTEM.md for the full specification. "
            "Summary: world frame is right-handed, Z-up (Blender convention); "
            "camera looks down its local -Z axis with local +Y up (Blender "
            "convention -- NOT the OpenCV/computer-vision convention of "
            "looking down +Z with +Y down; a conversion formula is given "
            "there). All rotations are XYZ-order Euler angles in radians."
        ),
    }


def save_ground_truth(scene: SceneState, trajectory: Trajectory, clip: ClipGroundTruth, out_dir: str | Path, resolution: int) -> Path:
    """Write `scene`'s complete ground truth to `out_dir/{scene.scene_id}/`. Returns that path."""
    scene_dir = Path(out_dir) / scene.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    np.save(scene_dir / "rgb.npy", clip.rgb)
    np.save(scene_dir / "depth.npy", clip.depth)
    np.save(scene_dir / "segmentation.npy", clip.segmentation)

    metadata = _build_metadata(scene, trajectory, clip, resolution)
    (scene_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return scene_dir


def load_ground_truth(out_dir: str | Path, scene_id: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of save_ground_truth: returns (metadata, rgb, depth, segmentation)."""
    scene_dir = Path(out_dir) / scene_id
    metadata = json.loads((scene_dir / "metadata.json").read_text())
    rgb = np.load(scene_dir / "rgb.npy")
    depth = np.load(scene_dir / "depth.npy")
    segmentation = np.load(scene_dir / "segmentation.npy")
    return metadata, rgb, depth, segmentation
