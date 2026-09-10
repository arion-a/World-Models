"""Run a frozen encoder E over a generated dataset to produce Z = E(V).

Encoding (especially with a real transformer video model) is the
expensive step, so results are cached to disk as a single .npz keyed by
`f"{scene_id}__{variant}"` (variant is "original" or a transform name),
and re-used on a later run unless `--overwrite` is passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from configs.config import Config, apply_override, load_config
from encoders.base import FrozenEncoder


def build_encoder(cfg: Config) -> FrozenEncoder:
    if cfg.encoder.name == "vjepa2":
        from encoders.vjepa2 import VJEPA2Encoder

        return VJEPA2Encoder(
            checkpoint=cfg.encoder.checkpoint,
            pretrained=cfg.encoder.pretrained,
            crop_size=cfg.dataset.resolution,
            frames_per_clip=cfg.dataset.num_frames,
            device=cfg.encoder.device,
        )
    if cfg.encoder.name == "pixel_baseline":
        from encoders.pixel_baseline import PixelStatisticsBaseline

        return PixelStatisticsBaseline(grid_size=cfg.encoder.grid_size)
    raise ValueError(f"Unknown encoder '{cfg.encoder.name}'")


def _load_video(frame_path: Path, num_frames: int) -> np.ndarray:
    frame = iio.imread(frame_path)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.repeat(frame[None], num_frames, axis=0)


def extract_representations(cfg: Config, encoder: FrozenEncoder, overwrite: bool = False) -> dict[str, np.ndarray]:
    out_dir = Path(cfg.dataset.output_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    cache_path = out_dir / f"representations_{cfg.encoder.name}.npz"

    cache: dict[str, np.ndarray] = {}
    if cache_path.exists() and not overwrite:
        cache = dict(np.load(cache_path))
        print(f"Loaded {len(cache)} cached representations from {cache_path}")

    transform_names = manifest["transform_names"]
    n_scenes = len(manifest["scenes"])
    for i, scene in enumerate(manifest["scenes"]):
        scene_id = scene["scene_id"]
        variants = ["original"] + transform_names
        for variant in variants:
            key = f"{scene_id}__{variant}"
            if key in cache:
                continue
            frame_path = out_dir / "frames" / scene_id / f"{variant}.png"
            video = _load_video(frame_path, cfg.dataset.num_frames)
            cache[key] = encoder.encode_video(video)
        print(f"[{i + 1}/{n_scenes}] encoded {scene_id}")

    np.savez(cache_path, **cache)
    print(f"Saved {len(cache)} representations to {cache_path}")
    return cache


def main():
    parser = argparse.ArgumentParser(description="Extract frozen-encoder representations for a generated dataset.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    default_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = load_config(default_path)
    if args.config:
        cfg = apply_override(cfg, args.config)

    encoder = build_encoder(cfg)
    extract_representations(cfg, encoder, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
