"""Task 3: paired original/transformed samples with complete provenance.

    pair_dir/
        original/           -- generation.ground_truth.save_ground_truth() output for S
        transformed/        -- same, for S' = T(S)
        transformation.json -- transform type, parameters, SE(3) matrix (where
                                applicable), and which variables changed/were fixed

Each side is rendered as a short *static* clip (a zero-motion
`Trajectory`, reusing Task 2's `generate_trajectory`/`render_trajectory`)
rather than a single frame, so a pair carries the same RGB + depth +
segmentation ground truth Task 2 produces, not just an RGB frame -- S and
S' each still describe one static instant (see DESIGN.md's "Static-clip
videos" note in IMPLEMENTATION_NOTES.md), Task 3 does not add motion.
"""

from __future__ import annotations

import json
from pathlib import Path

from generation.bpy_renderer import render_trajectory
from generation.ground_truth import save_ground_truth
from generation.motion import generate_trajectory
from generation.scene import SceneState
from transforms.scene_transform import TransformConfig, apply_transform


def generate_pair(
    scene: SceneState,
    transform_name: str,
    out_dir: str | Path,
    transform_cfg: TransformConfig | None = None,
    num_frames: int = 4,
    fps: float = 4.0,
    resolution: int = 128,
) -> Path:
    """Render one original/transformed pair for `transform_name` applied to `scene`.

    Deterministic given `(scene, transform_name, transform_cfg)`: the
    transform's own randomness is seeded from `scene.seed` (see
    transforms/scene_transform.py:_rng_for), and both renders are exactly
    reproducible Blender/Cycles output (see
    tests/test_bpy_renderer.py::test_render_trajectory_is_reproducible).
    """
    new_scene, transform_record = apply_transform(scene, transform_name, transform_cfg)

    pair_dir = Path(out_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)

    for dir_name, s in [("original", scene), ("transformed", new_scene)]:
        trajectory = generate_trajectory(s, num_frames=num_frames, fps=fps)  # static: no object/camera motion
        clip = render_trajectory(s, trajectory, resolution=resolution)
        save_ground_truth(s, trajectory, clip, pair_dir, resolution, dir_name=dir_name)

    (pair_dir / "transformation.json").write_text(json.dumps(transform_record, indent=2))
    return pair_dir


def load_pair(pair_dir: str | Path) -> dict:
    """Inverse of generate_pair: returns {"transformation": ..., "original": ..., "transformed": ...}."""
    from generation.ground_truth import load_ground_truth

    pair_dir = Path(pair_dir)
    transformation = json.loads((pair_dir / "transformation.json").read_text())
    original = load_ground_truth(pair_dir, "original")
    transformed = load_ground_truth(pair_dir, "transformed")
    return {"transformation": transformation, "original": original, "transformed": transformed}
