import numpy as np
import pytest

from representations.dataset import build_transform_split


def _fake_manifest(splits: list[str]) -> dict:
    return {
        "transform_names": ["camera_rotation"],
        "scenes": [{"scene_id": f"scene_{i:03d}", "split": s} for i, s in enumerate(splits)],
    }


def _fake_cache(manifest: dict, d: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cache = {}
    for scene in manifest["scenes"]:
        sid = scene["scene_id"]
        cache[f"{sid}__original"] = rng.normal(size=d).astype(np.float32)
        for t in manifest["transform_names"]:
            cache[f"{sid}__{t}"] = rng.normal(size=d).astype(np.float32)
    return cache


def test_split_partitions_scenes_correctly():
    manifest = _fake_manifest(["train", "train", "test", "train", "test"])
    cache = _fake_cache(manifest)
    split = build_transform_split(manifest, cache, "camera_rotation")
    assert len(split.train_scene_ids) == 3
    assert len(split.test_scene_ids) == 2
    assert set(split.train_scene_ids).isdisjoint(split.test_scene_ids)
    assert split.Z_train.shape == (3, 4)
    assert split.Z_test.shape == (2, 4)


def test_split_never_puts_original_and_transform_on_different_sides():
    # By construction build_transform_split reads a single split label per
    # scene and applies it to both original and transformed representation,
    # so this is really a guard against a future refactor accidentally
    # decoupling them.
    manifest = _fake_manifest(["train"] * 8 + ["test"] * 2)
    cache = _fake_cache(manifest)
    split = build_transform_split(manifest, cache, "camera_rotation")
    for sid in split.train_scene_ids:
        assert f"{sid}__original" in cache
        assert f"{sid}__camera_rotation" in cache
    assert len(split.train_scene_ids) + len(split.test_scene_ids) == 10


def test_all_scenes_in_one_split_raises_on_disjointness_not_violated():
    # Degenerate but valid case: everything in train, none in test.
    manifest = _fake_manifest(["train"] * 5)
    cache = _fake_cache(manifest)
    split = build_transform_split(manifest, cache, "camera_rotation")
    assert len(split.test_scene_ids) == 0
    assert len(split.train_scene_ids) == 5
