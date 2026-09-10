"""End-to-end dataset generation: sample scenes, apply transforms, render, save.

Produces, under `output_dir`:

    manifest.json                        -- full metadata + train/test split
    frames/{scene_id}/original.png       -- V = R(S), one representative frame
    frames/{scene_id}/{transform}.png    -- V' = R(T(S)), one per transform
    states/{scene_id}/original.json      -- S
    states/{scene_id}/{transform}.json   -- S' = T(S)

Train/test split is scene-level: a scene_id is assigned once, and every
transformed variant of that scene inherits the same split. This is what
"strict scene-level train/test separation" and "avoid leakage between
transformed versions of the same scene" (project principles) mean
concretely -- a scene's original and transformed renders must never be
split across train and test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from configs.config import Config, apply_override, load_config
from generation.bpy_renderer import render_frame
from generation.scene import SceneState
from generation.scene_sampler import SceneSamplerConfig, sample_scene
from transforms.scene_transform import TransformConfig, apply_transform


def _train_test_split(num_scenes: int, train_fraction: float, base_seed: int) -> list[str]:
    rng = np.random.default_rng(base_seed)
    perm = rng.permutation(num_scenes)
    n_train = int(round(train_fraction * num_scenes))
    split = ["test"] * num_scenes
    for idx in perm[:n_train]:
        split[idx] = "train"
    return split


def generate_dataset(cfg: Config, transform_names: tuple[str, ...]) -> dict:
    out_dir = Path(cfg.dataset.output_dir)
    frames_dir = out_dir / "frames"
    states_dir = out_dir / "states"
    frames_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)

    sampler_cfg = SceneSamplerConfig(
        num_objects_min=cfg.dataset.num_objects_min,
        num_objects_max=cfg.dataset.num_objects_max,
    )
    transform_cfg = TransformConfig()

    split_labels = _train_test_split(cfg.dataset.num_scenes, cfg.dataset.train_fraction, cfg.dataset.base_seed)

    manifest_scenes = []
    for i in range(cfg.dataset.num_scenes):
        scene_id = f"scene_{i:04d}"
        seed = cfg.dataset.base_seed * 1_000_003 + i
        state = sample_scene(scene_id, seed, sampler_cfg)

        (states_dir / scene_id).mkdir(parents=True, exist_ok=True)
        (frames_dir / scene_id).mkdir(parents=True, exist_ok=True)

        _render_and_save(state, frames_dir / scene_id / "original.png", cfg.dataset.resolution)
        (states_dir / scene_id / "original.json").write_text(state.to_json())

        transform_records = {}
        for name in transform_names:
            new_state, params = apply_transform(state, name, transform_cfg)
            _render_and_save(new_state, frames_dir / scene_id / f"{name}.png", cfg.dataset.resolution)
            (states_dir / scene_id / f"{name}.json").write_text(new_state.to_json())
            transform_records[name] = params

        manifest_scenes.append(
            {
                "scene_id": scene_id,
                "seed": seed,
                "split": split_labels[i],
                "num_objects": len(state.objects),
                "transforms": transform_records,
            }
        )
        print(f"[{i + 1}/{cfg.dataset.num_scenes}] rendered {scene_id} ({len(transform_names)} transforms)")

    manifest = {
        "dataset_config": {
            "num_scenes": cfg.dataset.num_scenes,
            "train_fraction": cfg.dataset.train_fraction,
            "resolution": cfg.dataset.resolution,
            "num_frames": cfg.dataset.num_frames,
            "base_seed": cfg.dataset.base_seed,
        },
        "transform_names": list(transform_names),
        "scenes": manifest_scenes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _render_and_save(state: SceneState, path: Path, resolution: int) -> None:
    frame = render_frame(state, resolution=resolution)
    iio.imwrite(path, frame)


def main():
    parser = argparse.ArgumentParser(description="Generate the GCP V0 synthetic scene dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config overriding configs/default.yaml")
    args = parser.parse_args()

    default_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = load_config(default_path)
    if args.config:
        cfg = apply_override(cfg, args.config)

    generate_dataset(cfg, cfg.experiment.transforms)


if __name__ == "__main__":
    main()
