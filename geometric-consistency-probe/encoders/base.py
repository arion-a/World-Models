"""Common interface every frozen encoder implements.

Adding a new encoder (a different video model, a different frozen
backbone) means implementing this one method -- nothing else in the
repo needs to change, which is the "keep the system modular" principle
from the project brief.
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
