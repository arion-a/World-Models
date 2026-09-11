"""Proves experiments.task7b_latent_transformation_discovery.fast_render's
optimization (render frame 0 once, repeat) produces BIT-IDENTICAL output
to transforms.pairs.generate_pair (render every frame independently) for
this project's static (zero-motion) clips -- required before Task 7B's
Stage 1 pipeline relies on the faster path for real data, not merely
"looks about the same."
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.task7b_latent_transformation_discovery.fast_render import generate_pair_fast
from generation.ground_truth import load_ground_truth
from generation.scene_sampler import SceneSamplerConfig, generate_scene
from transforms.pairs import generate_pair
from transforms.scene_transform import TransformConfig


@pytest.mark.slow  # needs bpy
@pytest.mark.parametrize("transform_name,cfg", [
    ("camera_rotation", TransformConfig(fixed_azimuth_deg=30.0, fixed_elevation_deg=0.0)),
    ("lighting_change", TransformConfig()),
])
def test_fast_render_matches_slow_render_bit_for_bit(tmp_path, transform_name, cfg):
    scene = generate_scene("fast_render_check", seed=5, cfg=SceneSamplerConfig(num_objects_min=2, num_objects_max=2))

    slow_dir = tmp_path / "slow"
    fast_dir = tmp_path / "fast"
    generate_pair(scene, transform_name, slow_dir, transform_cfg=cfg, num_frames=4, fps=4.0, resolution=128)
    generate_pair_fast(scene, transform_name, fast_dir, transform_cfg=cfg, num_frames=4, fps=4.0, resolution=128)

    for side in ("original", "transformed"):
        _, rgb_slow, depth_slow, seg_slow = load_ground_truth(slow_dir, side)
        _, rgb_fast, depth_fast, seg_fast = load_ground_truth(fast_dir, side)
        assert np.array_equal(rgb_slow, rgb_fast), f"{transform_name}/{side}: rgb mismatch between fast and slow render paths"
        assert np.array_equal(depth_slow, depth_fast), f"{transform_name}/{side}: depth mismatch"
        assert np.array_equal(seg_slow, seg_fast), f"{transform_name}/{side}: segmentation mismatch"

    slow_transformation = (slow_dir / "transformation.json").read_text()
    fast_transformation = (fast_dir / "transformation.json").read_text()
    assert slow_transformation == fast_transformation
