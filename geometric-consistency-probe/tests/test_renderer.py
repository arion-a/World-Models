"""Smoke test for the actual Blender/Cycles renderer.

Marked slow (invokes a real render) but kept tiny (32x32) so the full
suite stays fast; skipped automatically if bpy cannot be imported (e.g.
a machine where the platform-specific bpy wheel isn't installed).
"""

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from generation.scene_sampler import SceneSamplerConfig, sample_scene
from generation.bpy_renderer import render_frame, render_scene


@pytest.mark.slow
def test_render_frame_shape_and_dtype():
    scene = sample_scene("render_test", seed=1, cfg=SceneSamplerConfig())
    frame = render_frame(scene, resolution=32)
    assert frame.shape == (32, 32, 3)
    assert frame.dtype == np.uint8


@pytest.mark.slow
def test_render_scene_repeats_static_frame():
    scene = sample_scene("render_test2", seed=2, cfg=SceneSamplerConfig())
    video = render_scene(scene, num_frames=4, resolution=32)
    assert video.shape == (4, 32, 32, 3)
    assert np.array_equal(video[0], video[1])
    assert np.array_equal(video[0], video[3])


@pytest.mark.slow
def test_different_scenes_render_differently():
    s1 = sample_scene("a", seed=10)
    s2 = sample_scene("b", seed=20)
    f1 = render_frame(s1, resolution=32)
    f2 = render_frame(s2, resolution=32)
    assert not np.array_equal(f1, f2)
