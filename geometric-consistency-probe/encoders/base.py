"""Common interfaces every frozen encoder implements.

Adding a new encoder (a different video model, a different frozen
backbone) means implementing one method -- nothing else in the repo
needs to change, which is the "keep the system modular" principle from
the project brief.

Two interfaces live here, for two different jobs:

- `FrozenEncoder` (V0): `encode_video(frames) -> pooled vector`. Used by
  the V0 experiment pipeline (`representations/`, `experiments/run.py`),
  which always wants one pooled vector per clip -- pooling happens
  inside the encoder itself.
- `VideoEncoder` (Task 4): `encode(video) -> representation`, with no
  pooling implied by the interface. Task 4 explicitly separates
  *representation extraction* from *evaluation* (pooling is an
  evaluation-time decision, deferred to whichever later task defines
  probes against these representations -- see encoders/vjepa.py and
  encoders/REPRESENTATION_FORMAT.md for what "representation" means
  concretely for `VJEPAEncoder`), so this interface's `encode` returns
  the encoder's native output shape, not a pre-pooled vector.

These are intentionally not unified into one hierarchy: doing so now
would mean deciding, on Task 4's behalf, whether pooling belongs inside
or outside the encoder -- a call Task 4 itself defers. `encoders/vjepa.py`
implements `VideoEncoder`; `encoders/vjepa2.py` (V0) implements
`FrozenEncoder`; see IMPLEMENTATION_NOTES.md's "Task 4" section for how
the two relate and why both exist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FrozenEncoder(ABC):
    """E: video -> representation vector Z. Never trains; no gradients flow."""

    #: dimensionality of the returned representation vector
    output_dim: int

    @abstractmethod
    def encode_video(self, frames: np.ndarray) -> np.ndarray:
        """frames: (T, H, W, 3) uint8 RGB. Returns a 1-D float32 vector of length output_dim."""
        raise NotImplementedError

    def encode_batch(self, videos: list[np.ndarray]) -> np.ndarray:
        """Default batched implementation: encode one at a time. Override for real batching."""
        return np.stack([self.encode_video(v) for v in videos], axis=0)


class VideoEncoder(ABC):
    """E: video -> representation (Task 4's interface; see module docstring
    for how this differs from `FrozenEncoder` above). A "frozen" encoder in
    the same sense as `FrozenEncoder`: no parameter is ever updated by
    `encode`, and every concrete implementation must document exactly how
    it enforces that (see e.g. VJEPAEncoder's `torch.no_grad()` +
    `requires_grad_(False)` + `.eval()`).
    """

    #: model/checkpoint identifier, for provenance in saved metadata
    checkpoint: str
    #: whether `checkpoint` actually loaded pretrained weights (False if
    #: this encoder fell back to a randomly-initialized architecture --
    #: see VJEPAEncoder for why that can happen and how it's still made
    #: reproducible)
    pretrained: bool

    @abstractmethod
    def encode(self, video: np.ndarray) -> np.ndarray:
        """video: (T, H, W, 3) uint8 RGB. Returns the encoder's native
        representation -- shape and dtype are documented per concrete
        encoder (see encoders/REPRESENTATION_FORMAT.md for VJEPAEncoder's).
        Not pooled to a single vector unless the concrete encoder's own
        architecture naturally does that.
        """
        raise NotImplementedError
