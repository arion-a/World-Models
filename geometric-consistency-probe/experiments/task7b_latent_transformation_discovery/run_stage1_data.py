"""Task 7B Stage 1 data collection: freeze the protocol, then render and
encode the ENTIRE master population (train_pool + val + test, all at the
primary magnitude, elevation=0) into a disk-persisted representation
cache. This is the long pole of Task 7B's compute budget (~2 hours per
the protocol's compute_budget_note) -- run once, in the background;
learning_curve.py/selection.py/sealed_test.py all read from the cache
this script produces, never re-rendering.

    python -m experiments.task7b_latent_transformation_discovery.run_stage1_data

Writes, under experiments/task7b_latent_transformation_discovery/:
    protocol.json, protocol.md, split_manifest.json  (from protocol.freeze_protocol)
    rendered/camera_rotation_primary/{scene_id}/...  (real render+ground-truth per scene)
    z_cache_primary.npz                              (Z, Z' per scene_id, incrementally saved)
    stage1_data_manifest.json                        (per-scene provenance record)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.task7b_latent_transformation_discovery import data, protocol
from transforms.scene_transform import TransformConfig

OUT_DIR = Path(__file__).resolve().parent
RENDER_DIR = OUT_DIR / "rendered" / "camera_rotation_primary"
Z_CACHE_PATH = OUT_DIR / "z_cache_primary.npz"


def _transform_cfg() -> TransformConfig:
    return TransformConfig(fixed_azimuth_deg=data.PRIMARY_MAGNITUDE_DEG, fixed_elevation_deg=data.FIXED_ELEVATION_DEG)


def _load_cache() -> tuple[dict, dict]:
    Z_cache, Zp_cache = {}, {}
    if Z_CACHE_PATH.exists():
        npz = np.load(Z_CACHE_PATH)
        for key in npz.files:
            if key.startswith("Z__"):
                Z_cache[key[len("Z__"):]] = npz[key]
            elif key.startswith("Zp__"):
                Zp_cache[key[len("Zp__"):]] = npz[key]
    return Z_cache, Zp_cache


def _save_cache(Z_cache: dict, Zp_cache: dict) -> None:
    payload = {f"Z__{k}": v for k, v in Z_cache.items()}
    payload.update({f"Zp__{k}": v for k, v in Zp_cache.items()})
    np.savez(Z_CACHE_PATH, **payload)


def main():
    import experiments.geometric_consistency_lib as gclib
    from encoders.vjepa import mean_pool
    from transforms.pairs import load_pair

    from experiments.task7b_latent_transformation_discovery.fast_render import generate_pair_fast as generate_pair

    frozen, protocol_hash = protocol.freeze_protocol(OUT_DIR, seed=data.BASE_SEED)
    print(f"Protocol frozen: hash={protocol_hash[:16]} population={frozen['master_population_size']} "
          f"train_pool={frozen['split']['train_pool_size']} val={frozen['split']['val_size']} test={frozen['split']['test_size']}")

    all_scene_ids = frozen["split"]["train_pool_ids"] + frozen["split"]["val_ids"] + frozen["split"]["test_ids"]
    assert len(all_scene_ids) == frozen["master_population_size"]

    scenes = data.build_master_population(frozen["master_population_size"], base_seed=frozen["base_seed"])
    scenes_by_id = {s.scene_id: s for s in scenes}

    encoder = gclib.build_encoder(pretrained=True, checkpoint=None, device=None, fallback_seed=protocol.encoder_config()["fallback_seed"])
    cfg = protocol.encoder_config()
    transform_cfg = _transform_cfg()

    Z_cache, Zp_cache = _load_cache()
    print(f"Resuming with {len(Z_cache)} scenes already cached.")

    manifest = []
    t_start = time.time()
    for i, scene_id in enumerate(all_scene_ids):
        pair_dir = RENDER_DIR / scene_id
        if scene_id not in Z_cache:
            if not (pair_dir / "transformation.json").exists():
                generate_pair(
                    scenes_by_id[scene_id], data.TRANSFORM_NAME, pair_dir,
                    transform_cfg=transform_cfg, num_frames=cfg["num_frames"], fps=cfg["fps"], resolution=cfg["resolution"],
                )
            loaded = load_pair(pair_dir)
            _, orig_rgb, _, _ = loaded["original"]
            _, trans_rgb, _, _ = loaded["transformed"]
            Z_cache[scene_id] = mean_pool(encoder.encode(orig_rgb))
            Zp_cache[scene_id] = mean_pool(encoder.encode(trans_rgb))
            if (i + 1) % 20 == 0:
                _save_cache(Z_cache, Zp_cache)
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining = (len(all_scene_ids) - (i + 1)) / rate if rate > 0 else float("inf")
                print(f"  {i + 1}/{len(all_scene_ids)} scenes done, elapsed={elapsed:.0f}s, est. remaining={remaining:.0f}s")

        transformation = json.loads((pair_dir / "transformation.json").read_text())
        manifest.append({
            "scene_id": scene_id,
            "transform_matrix": transformation["transform_matrix"],
            "azimuth_deg": transformation["azimuth_deg"],
            "elevation_deg": transformation["elevation_deg"],
        })

    _save_cache(Z_cache, Zp_cache)
    (OUT_DIR / "stage1_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nStage 1 data collection complete: {len(Z_cache)} scenes cached in {time.time() - t_start:.0f}s total.")


if __name__ == "__main__":
    main()
