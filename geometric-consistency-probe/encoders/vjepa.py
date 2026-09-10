"""Task 4: the frozen V-JEPA 2.1 video encoder.

Upstream model, exactly (see IMPLEMENTATION_NOTES.md's "Task 4" section
for the full account of how this was verified, not assumed):

  * Official source: `facebookresearch/vjepa2` on GitHub. V-JEPA 2.1 is a
    real, distinct release (not a typo for "V-JEPA 2") -- per that
    repo's README, its ViT-B/16 checkpoint is `parameters: 80M`,
    `resolution: 384`, checkpoint file
    `vjepa2_1_vitb_dist_vitG_384.pt` (distilled from a ViT-G teacher),
    trained with config `configs/train_2_1/vitb16`, and is downloaded
    from `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt`.
    Architecturally it is the same JEPA family as V-JEPA 2 (refinements
    are to the training recipe -- dense predictive loss, deep
    self-supervision, distillation -- not the model class), so it loads
    through the same `transformers.VJEPA2Model` used for V-JEPA 2 in
    this project's V0 code (`encoders/vjepa2.py`).
  * As of this writing, no checkpoint under the official `facebook/`
    Hugging Face Hub namespace was found for V-JEPA 2.1 specifically
    (a hosting request for it was open on the upstream GitHub issue
    tracker). A third-party conversion exists at
    `apiantonio/vjepa2.1-vit-base-384`, which that repo's own model card
    claims is bit-exact against Meta's reference implementation; this is
    what `DEFAULT_CHECKPOINT` below points at, clearly NOT an official
    `facebook/`-namespaced release. `checkpoint=` is fully overridable
    once an official Hub repo exists or for anyone who converts the
    original `.pt` checkpoint themselves.
  * This sandbox cannot independently verify any of the above by
    downloading it: both `huggingface.co` and `dl.fbaipublicfiles.com`
    are network-policy-blocked here (see encoders/vjepa2.py's identical
    caveat for V-JEPA 2, confirmed via the proxy's own diagnostic
    endpoint). See "Reproducibility without the real checkpoint" below
    for how this is handled honestly rather than silently.

API verified against the installed `transformers` source (same contract
V0's encoders/vjepa2.py already verified and documents in full):
`pixel_values_videos` is `(batch, num_frames, channels, H, W)`;
`model.get_vision_features(...)` returns the encoder's
`last_hidden_state`, `(batch, num_patches, hidden_size)`, with no
built-in pooling.

Reproducibility without the real checkpoint:
`VJEPA2Model.from_pretrained(checkpoint)` requires Hub access. When it's
unavailable, this class falls back to the same architecture with random
weights, exactly like V0's encoders/vjepa2.py -- but with one fix V0
did not have: the fallback's random initialization is seeded
(`torch.manual_seed(fallback_seed)`, default 0) immediately before
constructing the untrained model, so the SAME "untrained" weights are
produced every time, in every process. Without this, two separate runs
of the fallback path would silently produce two *different* random
encoders, which would make "the same video produces consistent
representations" (Task 4's own smoke-test requirement) true only
*within* one process and false *across* runs -- a real gap this task's
smoke test is designed to catch (see
tests/test_vjepa_encoder.py::test_untrained_fallback_is_reproducible_across_instances).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass

import numpy as np

from encoders.base import VideoEncoder

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# V-JEPA 2.1 ViT-B/16: same patch/hidden/layer conventions as V-JEPA 2's
# ViT-B (encoders/vjepa2.py), at V-JEPA 2.1's native 384 resolution --
# used for the untrained-fallback architecture when pretrained weights
# cannot be loaded (see module docstring).
VITB16_CONFIG_KWARGS = dict(
    patch_size=16,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    pred_hidden_size=384,
    pred_num_attention_heads=12,
    pred_num_hidden_layers=12,
)

# Third-party conversion, NOT an official `facebook/`-namespaced release
# -- see module docstring's "Upstream model" section.
DEFAULT_CHECKPOINT = "apiantonio/vjepa2.1-vit-base-384"
DEFAULT_CROP_SIZE = 384  # V-JEPA 2.1's native resolution (V-JEPA 2 used 256)
DEFAULT_FALLBACK_SEED = 0


@dataclass(frozen=True)
class PreprocessingConfig:
    """Exactly what encode() does to a raw (T, H, W, 3) uint8 video before
    it reaches the model -- recorded verbatim into extraction metadata
    (encoders/extract.py) so a saved representation's provenance is
    self-contained.
    """

    crop_size: int
    resize_size: int  # shortest-edge resize target, before center-crop
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    resize_mode: str = "bilinear"

    def to_dict(self) -> dict:
        return asdict(self)


class VJEPAEncoder(VideoEncoder):
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        pretrained: bool = True,
        crop_size: int = DEFAULT_CROP_SIZE,
        frames_per_clip: int = 8,
        tubelet_size: int = 2,
        device: str | None = None,
        fallback_seed: int = DEFAULT_FALLBACK_SEED,
        set_deterministic_algorithms: bool = True,
    ):
        import torch
        from transformers import VJEPA2Config, VJEPA2Model

        self.torch = torch

        # Automatically use a GPU when one is available; still fully
        # usable on CPU otherwise (Task 4 requirement: "support CPU if
        # feasible for testing" / "automatically use GPU when available").
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        logger.info("VJEPAEncoder using device=%s", resolved_device)

        if set_deterministic_algorithms:
            # Best-effort, process-wide: "deterministic inference where
            # practical" (Task 4). warn_only=True because some CUDA
            # kernels have no deterministic implementation at all; we
            # would rather run with a warning than refuse to run on GPU.
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.crop_size = crop_size
        self.resize_size = int(round(crop_size * 256 / 224))  # same aspect-preserving margin V-JEPA2's own processor uses
        self.preprocessing_config = PreprocessingConfig(
            crop_size=crop_size,
            resize_size=self.resize_size,
            normalize_mean=tuple(IMAGENET_MEAN.tolist()),
            normalize_std=tuple(IMAGENET_STD.tolist()),
        )
        self.pretrained = pretrained
        self.checkpoint = checkpoint
        self.fallback_seed = fallback_seed

        if pretrained:
            try:
                self.model = VJEPA2Model.from_pretrained(checkpoint)
            except Exception as exc:  # network/hub failure, missing checkpoint, etc.
                warnings.warn(
                    f"Could not load pretrained checkpoint '{checkpoint}' ({exc!r}). "
                    "Falling back to a randomly-initialized (but seeded, hence "
                    "reproducible) VJEPA2 model with the same architecture. Results "
                    "are for PIPELINE VALIDATION ONLY and must not be interpreted as "
                    "saying anything about V-JEPA 2.1's real representations. See "
                    "encoders/vjepa.py and IMPLEMENTATION_NOTES.md.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                logger.warning("Falling back to untrained VJEPA2 (pretrained load failed): %s", exc)
                self.pretrained = False
                self.model = self._build_untrained_model(VJEPA2Config, VJEPA2Model, crop_size, frames_per_clip, tubelet_size)
        else:
            self.model = self._build_untrained_model(VJEPA2Config, VJEPA2Model, crop_size, frames_per_clip, tubelet_size)

        self.model.eval().to(resolved_device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.output_dim = self.model.config.hidden_size

    def _build_untrained_model(self, VJEPA2Config, VJEPA2Model, crop_size: int, frames_per_clip: int, tubelet_size: int):
        # Seeded immediately before construction so random init is
        # reproducible across processes -- see module docstring's
        # "Reproducibility without the real checkpoint".
        self.torch.manual_seed(self.fallback_seed)
        config = VJEPA2Config(
            crop_size=crop_size,
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
            **VITB16_CONFIG_KWARGS,
        )
        return VJEPA2Model(config)

    def _preprocess(self, video: np.ndarray) -> "torch.Tensor":
        """(T, H, W, 3) uint8 -> (1, T, 3, crop, crop) normalized float tensor."""
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(video).float() / 255.0  # (T, H, W, 3)
        t = t.permute(0, 3, 1, 2)  # (T, 3, H, W)
        t = F.interpolate(t, size=(self.resize_size, self.resize_size), mode=self.preprocessing_config.resize_mode, align_corners=False)
        offset = (self.resize_size - self.crop_size) // 2
        t = t[:, :, offset:offset + self.crop_size, offset:offset + self.crop_size]
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        t = (t - mean) / std
        return t.unsqueeze(0)  # (1, T, 3, crop, crop)

    def encode(self, video: np.ndarray) -> np.ndarray:
        """video: (T, H, W, 3) uint8 RGB.

        Returns the encoder's native token sequence, shape
        (num_tokens, hidden_size), float32 -- NOT pooled; see
        encoders/REPRESENTATION_FORMAT.md for exactly what a "token"
        corresponds to and encoders/extract.py for how this gets saved.
        Frozen: no gradient is tracked (`torch.no_grad()`) and no model
        parameter has `requires_grad` set (done once in `__init__`).
        """
        pixel_values = self._preprocess(video).to(self.device)
        with self.torch.no_grad():
            tokens = self.model.get_vision_features(pixel_values)  # (1, num_tokens, hidden)
        return tokens.squeeze(0).cpu().numpy().astype(np.float32)


def mean_pool(representation: np.ndarray) -> np.ndarray:
    """Convenience reduction: (num_tokens, hidden_size) -> (hidden_size,).

    Not applied inside `encode()` itself (see encoders/base.py's
    VideoEncoder docstring for why); provided here because "just average
    the tokens" is the obvious first thing most callers will want, and
    should not be silently reimplemented slightly differently in every
    later consumer of these representations.
    """
    return representation.mean(axis=0)
