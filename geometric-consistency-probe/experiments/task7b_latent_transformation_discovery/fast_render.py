"""Task 7B-local rendering optimization: render ONE frame via Cycles and
repeat it `num_frames` times, instead of `transforms.pairs.generate_pair`'s
default of rendering every frame independently through
`generation.bpy_renderer.render_trajectory`.

This is scoped entirely to Task 7B (a new file under this package) --
the shared `generation/bpy_renderer.py` and `transforms/pairs.py` used by
Task 6/7/the forensic audit/the diagnostic are NOT modified, so nothing
here can affect their behavior or reproducibility.

Why this is safe, not a shortcut: this project's clips are all static
(zero camera/object motion within one clip -- see transforms/pairs.py's
own docstring), so every frame of a `render_trajectory` call is
mathematically guaranteed, and empirically verified, to render
bit-identically:

  - the Task 7 forensic audit's render_determinism.json already recorded
    `within_clip_frames_identical: true` for real camera_rotation/
    lighting_change clips;
  - re-verified here, directly, for Task 7B's own camera_rotation-at-
    fixed-yaw setup, in tests/test_task7b_fast_render.py, which asserts
    this module's rgb/depth/segmentation output is BIT-IDENTICAL to
    generate_pair's (the slow, one-Cycles-render-per-frame path) for the
    same (scene, transform, transform_cfg) -- not just "looks similar."

Measured on this session's hardware: a single-frame render costs
roughly 1/4 to 1/8 of a full num_frames=4 render (each frame is an
independent, equally-expensive Cycles render), so this cuts Stage 1's
rendering-only cost by roughly that factor with ZERO change to the
resulting arrays.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from generation.bpy_renderer import ClipGroundTruth, render_trajectory
from generation.ground_truth import save_ground_truth
from generation.motion import Trajectory, generate_trajectory
from generation.scene import SceneState
from transforms.scene_transform import TransformConfig, apply_transform


def _render_repeated(state: SceneState, num_frames: int, fps: float, resolution: int) -> tuple[Trajectory, ClipGroundTruth]:
    full_trajectory = generate_trajectory(state, num_frames=num_frames, fps=fps)
    one_frame_trajectory = Trajectory(fps=fps, num_frames=1, frames=(full_trajectory.frames[0],))
    clip_one = render_trajectory(state, one_frame_trajectory, resolution=resolution)
    clip_repeated = ClipGroundTruth(
        rgb=clip_one.rgb.repeat(num_frames, axis=0),
        depth=clip_one.depth.repeat(num_frames, axis=0),
        segmentation=clip_one.segmentation.repeat(num_frames, axis=0),
        depth_near=clip_one.depth_near,
        depth_far=clip_one.depth_far,
        max_instances=clip_one.max_instances,
    )
    return full_trajectory, clip_repeated


def generate_pair_fast(
    scene: SceneState,
    transform_name: str,
    out_dir: str | Path,
    transform_cfg: TransformConfig | None = None,
    num_frames: int = 4,
    fps: float = 4.0,
    resolution: int = 128,
    sides: tuple[str, ...] = ("original", "transformed"),
) -> Path:
    """Drop-in equivalent of transforms.pairs.generate_pair for this
    project's zero-motion (static) clips: renders each side's frame 0
    once and repeats it, rather than rendering every frame independently.
    Produces byte-identical rgb/depth/segmentation arrays to
    generate_pair for a static clip -- see this module's docstring and
    tests/test_task7b_fast_render.py.

    `sides` defaults to both (Stage 1's usage: builds the primary Z/Z'
    pair from scratch, needs both). Stage 2/3's multi-magnitude sweep
    only ever needs the transformed side -- the original is
    magnitude-independent and already cached from Stage 1 -- so it
    passes sides=("transformed",) to avoid re-rendering the identical
    original side once per magnitude for no reason."""
    new_scene, transform_record = apply_transform(scene, transform_name, transform_cfg)

    pair_dir = Path(out_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)

    for dir_name, s in [("original", scene), ("transformed", new_scene)]:
        if dir_name not in sides:
            continue
        trajectory, clip = _render_repeated(s, num_frames=num_frames, fps=fps, resolution=resolution)
        save_ground_truth(s, trajectory, clip, pair_dir, resolution, dir_name=dir_name)

    (pair_dir / "transformation.json").write_text(json.dumps(transform_record, indent=2))
    return pair_dir
