"""Task 10 required scientific-validity check: the non-learned
pixel-statistics baseline (encoders/pixel_baseline.py) must receive only
rendered pixels, never privileged ground-truth SceneState information --
checked structurally (its only input parameter is rendered RGB frames)
and behaviorally (its output depends only on those pixels).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from encoders.base import FrozenEncoder
from encoders.pixel_baseline import PixelStatisticsBaseline


def test_pixel_statistics_baseline_is_a_frozen_encoder():
    assert issubclass(PixelStatisticsBaseline, FrozenEncoder)


def test_encode_video_signature_takes_only_rendered_frames_no_scene_state():
    params = list(inspect.signature(PixelStatisticsBaseline.encode_video).parameters)
    assert params == ["self", "frames"], (
        "the pixel-statistics baseline's encode_video must take exactly one argument (the "
        "rendered RGB array) -- any additional parameter risks a ground-truth SceneState leak"
    )


def test_encode_video_output_depends_only_on_pixel_values():
    enc = PixelStatisticsBaseline(grid_size=4)
    rng = np.random.default_rng(0)
    frames_a = rng.integers(0, 256, size=(3, 16, 16, 3), dtype=np.uint8)
    frames_b = frames_a.copy()

    z_a = enc.encode_video(frames_a)
    z_b = enc.encode_video(frames_b)
    assert np.array_equal(z_a, z_b)  # identical pixels -> identical output, deterministic

    frames_c = np.zeros_like(frames_a)
    z_c = enc.encode_video(frames_c)
    assert not np.array_equal(z_a, z_c)  # different pixels -> (generically) different output


def test_encode_video_output_is_finite_and_matches_declared_output_dim():
    enc = PixelStatisticsBaseline(grid_size=8)
    frames = np.random.default_rng(0).integers(0, 256, size=(4, 128, 128, 3), dtype=np.uint8)
    z = enc.encode_video(frames)
    assert z.shape == (enc.output_dim,)
    assert np.all(np.isfinite(z))


def test_encode_video_handles_non_divisible_resolution():
    enc = PixelStatisticsBaseline(grid_size=5)
    frames = np.random.default_rng(0).integers(0, 256, size=(2, 17, 23, 3), dtype=np.uint8)
    z = enc.encode_video(frames)
    assert z.shape == (enc.output_dim,)
    assert np.all(np.isfinite(z))
