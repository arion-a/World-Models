"""Task 4 smoke tests: VJEPAEncoder + encoders/extract.py, on 2-5 small
synthetic videos. Deliberately does not depend on bpy/rendering -- this
task is about the encoder abstraction, decoupled from where a video
comes from (see encoders/base.py's module docstring).

All tests use `pretrained=False` explicitly: with no real network access
to Hugging Face Hub in this sandbox, `pretrained=True` (the real
default) would still work correctly (falls back automatically -- see
encoders/vjepa.py), but only after a slow, multi-retry connection
timeout. Explicitly requesting the untrained architecture exercises the
same code path (same __init__ logic, same encode()) without that delay,
matching the fast-dev-config pattern already used elsewhere in this
project (configs/experiments/camera_rotation_v0.yaml).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from encoders.extract import extract_and_save, extract_dataset, load_representation
from encoders.vjepa import DEFAULT_FALLBACK_SEED, VJEPAEncoder, mean_pool

CROP = 64
FRAMES = 8


def _synthetic_video(seed: int, num_frames: int = FRAMES, resolution: int = CROP) -> np.ndarray:
    """A small, deterministic pseudo-random RGB video -- stands in for a
    real rendered clip, since this task is about the encoder, not the
    renderer (see module docstring)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(num_frames, resolution, resolution, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def encoder():
    return VJEPAEncoder(pretrained=False, crop_size=CROP, frames_per_clip=FRAMES)


@pytest.fixture(scope="module")
def videos():
    """2-5 small synthetic videos, per the task's smoke-test spec."""
    return {f"video_{i}": _synthetic_video(seed=i) for i in range(4)}


# 1. encoder loads correctly ------------------------------------------------


def test_encoder_loads_correctly(encoder):
    assert encoder.checkpoint  # a non-empty identifier is recorded
    assert encoder.pretrained is False  # requested untrained explicitly
    assert encoder.output_dim == 768
    assert encoder.device in ("cpu", "cuda")
    # frozen: nothing in the model should be trainable
    assert all(not p.requires_grad for p in encoder.model.parameters())
    assert not encoder.model.training  # eval mode


def test_gpu_used_automatically_when_available(monkeypatch):
    """Doesn't require an actual GPU: checks the auto-detection logic
    itself picks 'cuda' when torch reports one available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)  # avoid actually moving to a nonexistent GPU
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)
    enc = VJEPAEncoder(pretrained=False, crop_size=CROP, frames_per_clip=FRAMES, device=None)
    assert enc.device == "cuda"


# T5.3 -- numerical validity (no NaN/Inf) ------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_representation_has_no_nan_or_inf(encoder, seed):
    rep = encoder.encode(_synthetic_video(seed=seed))
    assert np.isfinite(rep).all(), "representation contains NaN or Inf"
    assert not np.isnan(rep).any()
    assert not np.isinf(rep).any()


# T5.5 -- frozen encoder (already partly covered by test_encoder_loads_correctly,
# restated standalone so it can be verified/reported independently) ----------


def test_all_parameters_have_requires_grad_false(encoder):
    params = list(encoder.model.parameters())
    assert len(params) > 0, "sanity check: model actually has parameters"
    assert all(p.requires_grad is False for p in params)


# T5.6 -- no optimizer / no training update -----------------------------------


def test_no_optimizer_exists_anywhere_on_the_encoder(encoder):
    """There is no torch.optim.Optimizer instance attached to the encoder
    or its model -- i.e. there is no object in this codebase capable of
    performing a training step on it."""
    import torch as _torch

    for obj in (encoder, encoder.model):
        for attr_name in dir(obj):
            if attr_name.startswith("__"):
                continue
            try:
                value = getattr(obj, attr_name)
            except Exception:
                continue
            assert not isinstance(value, _torch.optim.Optimizer), f"found an optimizer at {obj}.{attr_name}"


def test_encode_does_not_change_any_parameter_value(encoder):
    """The strongest possible version of 'frozen': run encode() several
    times (including on different videos) and verify every parameter
    tensor is BIT-IDENTICAL to what it was before -- not just that
    requires_grad is False, but that nothing about the weights moved."""
    before = [p.detach().clone() for p in encoder.model.parameters()]

    for seed in range(4):
        encoder.encode(_synthetic_video(seed=seed))

    after = list(encoder.model.parameters())
    assert len(before) == len(after)
    for p_before, p_after in zip(before, after):
        assert torch.equal(p_before, p_after)


# 2. video preprocessing is correct -----------------------------------------


def test_preprocessing_output_shape(encoder):
    video = _synthetic_video(seed=0)
    pixel_values = encoder._preprocess(video)
    assert pixel_values.shape == (1, FRAMES, 3, CROP, CROP)
    assert pixel_values.dtype == torch.float32


def test_preprocessing_normalization_is_exact_for_a_constant_frame(encoder):
    """A constant-color video's normalized value must equal
    (color/255 - mean) / std exactly (up to resize/crop interpolation
    being a no-op on a uniform image), per encoders/vjepa.py's
    documented preprocessing."""
    from encoders.vjepa import IMAGENET_MEAN, IMAGENET_STD

    color = np.array([200, 50, 10], dtype=np.uint8)
    video = np.tile(color, (FRAMES, CROP, CROP, 1)).astype(np.uint8)
    pixel_values = encoder._preprocess(video).numpy()  # (1, T, 3, H, W)

    expected = (color.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    for c in range(3):
        assert np.allclose(pixel_values[0, :, c, :, :], expected[c], atol=1e-5)


def test_preprocessing_config_matches_actual_transform(encoder):
    cfg = encoder.preprocessing_config
    assert cfg.crop_size == CROP
    assert cfg.normalize_mean == tuple(encoder.preprocessing_config.normalize_mean)
    pixel_values = encoder._preprocess(_synthetic_video(seed=1))
    assert pixel_values.shape[-1] == cfg.crop_size


# 3. output shapes are consistent -------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_output_shape_is_consistent_across_videos(encoder, seed):
    rep = encoder.encode(_synthetic_video(seed=seed))
    expected_num_tokens = (FRAMES // 2) * (CROP // 16) ** 2  # tubelet_size=2, patch_size=16
    assert rep.shape == (expected_num_tokens, encoder.output_dim)
    assert rep.dtype == np.float32


def test_output_shape_scales_with_frame_count():
    enc = VJEPAEncoder(pretrained=False, crop_size=CROP, frames_per_clip=4)
    rep = enc.encode(_synthetic_video(seed=0, num_frames=4))
    expected_num_tokens = (4 // 2) * (CROP // 16) ** 2
    assert rep.shape == (expected_num_tokens, enc.output_dim)


def test_mean_pool_reduces_to_hidden_size_vector(encoder):
    rep = encoder.encode(_synthetic_video(seed=2))
    pooled = mean_pool(rep)
    assert pooled.shape == (encoder.output_dim,)
    assert np.allclose(pooled, rep.mean(axis=0))


# 4. representations can be saved and reloaded ------------------------------


def test_extract_and_save_round_trips(encoder, tmp_path):
    video = _synthetic_video(seed=3)
    video_dir = extract_and_save(encoder, video, "vid_a", tmp_path)
    assert (video_dir / "representation.npy").exists()
    assert (video_dir / "metadata.json").exists()

    loaded_rep, loaded_meta = load_representation(tmp_path, "vid_a")
    fresh_rep = encoder.encode(video)
    assert np.array_equal(loaded_rep, fresh_rep)
    assert loaded_meta["video_id"] == "vid_a"
    assert loaded_meta["checkpoint"] == encoder.checkpoint
    assert loaded_meta["pretrained"] == encoder.pretrained
    assert loaded_meta["video_shape"] == list(video.shape)
    assert loaded_meta["representation_shape"] == list(fresh_rep.shape)
    assert loaded_meta["representation_dtype"] == "float32"
    assert loaded_meta["preprocessing_config"]["crop_size"] == CROP


def test_extract_dataset_smoke_with_2_to_5_videos(encoder, videos, tmp_path):
    assert 2 <= len(videos) <= 5
    video_ids = extract_dataset(encoder, videos, tmp_path)
    assert set(video_ids) == set(videos.keys())
    for video_id in video_ids:
        rep, meta = load_representation(tmp_path, video_id)
        assert rep.shape[1] == encoder.output_dim
        assert meta["video_id"] == video_id


def test_extract_and_save_rejects_malformed_video(encoder, tmp_path):
    with pytest.raises(ValueError):
        extract_and_save(encoder, np.zeros((8, 64, 64)), "bad", tmp_path)  # missing channel dim


# 5. the same video produces consistent representations ---------------------


def test_same_video_same_instance_is_bit_identical(encoder):
    video = _synthetic_video(seed=4)
    rep1 = encoder.encode(video)
    rep2 = encoder.encode(video)
    assert np.array_equal(rep1, rep2)


def test_same_video_different_instances_same_seed_is_bit_identical():
    """The untrained-fallback determinism fix (IMPLEMENTATION_NOTES.md's
    Task 4 section): two SEPARATE VJEPAEncoder instances, constructed
    independently, must produce identical representations for the same
    video -- not just within one already-constructed encoder. This is
    the property that was missing from V0's encoders/vjepa2.py."""
    video = _synthetic_video(seed=5)
    enc_a = VJEPAEncoder(pretrained=False, crop_size=CROP, frames_per_clip=FRAMES, fallback_seed=DEFAULT_FALLBACK_SEED)
    enc_b = VJEPAEncoder(pretrained=False, crop_size=CROP, frames_per_clip=FRAMES, fallback_seed=DEFAULT_FALLBACK_SEED)
    assert np.array_equal(enc_a.encode(video), enc_b.encode(video))


def test_different_fallback_seed_gives_different_untrained_weights():
    """Sanity check that fallback_seed is actually doing something --
    otherwise the determinism test above would be trivially true even if
    the model were, say, always zero-initialized."""
    video = _synthetic_video(seed=6)
    enc_a = VJEPAEncoder(pretrained=False, crop_size=CROP, frames_per_clip=FRAMES, fallback_seed=0)
    enc_b = VJEPAEncoder(pretrained=False, crop_size=CROP, frames_per_clip=FRAMES, fallback_seed=1)
    assert not np.array_equal(enc_a.encode(video), enc_b.encode(video))


def test_different_videos_give_different_representations(encoder):
    rep_a = encoder.encode(_synthetic_video(seed=10))
    rep_b = encoder.encode(_synthetic_video(seed=11))
    assert not np.array_equal(rep_a, rep_b)
