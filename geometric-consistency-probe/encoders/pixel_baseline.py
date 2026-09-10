"""A trivial, non-learned baseline representation: pooled raw pixels.

This exists purely as a floor/control (project principle: "every
experiment must have a baseline"). If a linear map fit on *raw pixel
statistics* already "explains" a transformation about as well as a
learned encoder's representation does, that is evidence the metric is
picking up low-level pixel correlation rather than anything specific to
the encoder's learned geometry -- exactly the correlation-vs-genuine-
structure distinction the project asks us to keep separate.
"""

from __future__ import annotations

import numpy as np

from encoders.base import FrozenEncoder


class PixelStatisticsBaseline(FrozenEncoder):
    """Downsamples each frame to a grid_size x grid_size grid, averages over
    time and channels-preserving pooling, and flattens to a feature vector.
    """

    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size
        self.output_dim = grid_size * grid_size * 3

    def encode_video(self, frames: np.ndarray) -> np.ndarray:
        # Frames are identical (static scene), so averaging over time is a
        # no-op here but keeps the interface honest for future non-static clips.
        mean_frame = frames.astype(np.float32).mean(axis=0)  # (H, W, 3)
        h, w, _ = mean_frame.shape
        g = self.grid_size
        # Pool into a g x g grid via reshaping (H, W assumed divisible by g;
        # otherwise fall back to simple striding).
        if h % g == 0 and w % g == 0:
            pooled = mean_frame.reshape(g, h // g, g, w // g, 3).mean(axis=(1, 3))
        else:
            ys = np.linspace(0, h, g + 1).astype(int)
            xs = np.linspace(0, w, g + 1).astype(int)
            pooled = np.zeros((g, g, 3), dtype=np.float32)
            for i in range(g):
                for j in range(g):
                    pooled[i, j] = mean_frame[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean(axis=(0, 1))
        return (pooled.flatten() / 255.0).astype(np.float32)
