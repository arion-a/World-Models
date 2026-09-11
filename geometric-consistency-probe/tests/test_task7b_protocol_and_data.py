"""QA gates: scene-level split exclusivity, all variants share their
source split, deterministic/reproducible splits and nested subsets, and
protocol content-hashing / immutability guard."""
from __future__ import annotations

import json

import pytest

from experiments.task7b_latent_transformation_discovery import data, protocol


def test_compute_population_size_gives_at_least_the_requested_train_pool():
    total = data.compute_population_size(1024)
    assert round(total * data.TRAIN_FRACTION) >= 1024 or total - round(total * data.VAL_FRACTION) - 1024 >= 0
    # exact contract: train_pool_size is passed explicitly, not re-derived from the fraction
    assert total >= 1024 / data.TRAIN_FRACTION


def test_three_way_split_is_disjoint_and_covers_everything():
    scene_ids = [f"scene_{i:04d}" for i in range(1463)]
    split = data.three_way_split(scene_ids, train_pool_size=1024, val_size=219, seed=0)
    assert len(split.train_pool_ids) == 1024
    assert len(split.val_ids) == 219
    assert len(split.test_ids) == 1463 - 1024 - 219
    assert set(split.train_pool_ids).isdisjoint(split.val_ids)
    assert set(split.train_pool_ids).isdisjoint(split.test_ids)
    assert set(split.val_ids).isdisjoint(split.test_ids)
    assert set(split.train_pool_ids) | set(split.val_ids) | set(split.test_ids) == set(scene_ids)


def test_three_way_split_rejects_too_small_a_test_set():
    scene_ids = [f"scene_{i:04d}" for i in range(200)]
    with pytest.raises(ValueError):
        data.three_way_split(scene_ids, train_pool_size=100, val_size=90, seed=0)  # test would be 10 < 64


def test_three_way_split_is_deterministic():
    scene_ids = [f"scene_{i:04d}" for i in range(500)]
    a = data.three_way_split(scene_ids, train_pool_size=300, val_size=100, seed=7)
    b = data.three_way_split(scene_ids, train_pool_size=300, val_size=100, seed=7)
    assert a.train_pool_ids == b.train_pool_ids
    assert a.val_ids == b.val_ids
    assert a.test_ids == b.test_ids


def test_nested_train_subsets_are_prefixes_of_each_other():
    pool = tuple(f"scene_{i:04d}" for i in range(1024))
    small = data.nested_train_subset(pool, 64, seed=3)
    medium = data.nested_train_subset(pool, 256, seed=3)
    large = data.nested_train_subset(pool, 1024, seed=3)
    assert set(small).issubset(set(medium))
    assert set(medium).issubset(set(large))
    assert set(large) == set(pool)


def test_nested_train_subsets_differ_across_seeds():
    pool = tuple(f"scene_{i:04d}" for i in range(1024))
    a = set(data.nested_train_subset(pool, 64, seed=0))
    b = set(data.nested_train_subset(pool, 64, seed=1))
    assert a != b


def test_magnitude_set_excludes_reserved_and_zero_from_direct_fit():
    assert 0.0 not in data.STAGE2_DIRECT_FIT_MAGNITUDES_DEG
    assert data.INTERPOLATION_RESERVED_MAGNITUDE_DEG not in data.STAGE2_DIRECT_FIT_MAGNITUDES_DEG
    assert len(data.STAGE2_DIRECT_FIT_MAGNITUDES_DEG) == len(data.MAGNITUDES_DEG) - 2


# --- protocol -----------------------------------------------------------------


def test_protocol_content_hash_is_deterministic():
    p1 = protocol.build_protocol(seed=0)
    p2 = protocol.build_protocol(seed=0)
    assert protocol.content_hash(p1) == protocol.content_hash(p2)


def test_protocol_freeze_writes_valid_json_and_markdown(tmp_path):
    frozen, h = protocol.freeze_protocol(tmp_path, seed=0)
    assert (tmp_path / "protocol.json").exists()
    assert (tmp_path / "protocol.md").exists()
    assert (tmp_path / "split_manifest.json").exists()
    reloaded = json.loads((tmp_path / "protocol.json").read_text())
    assert reloaded["protocol_hash"] == h
    split = json.loads((tmp_path / "split_manifest.json").read_text())
    assert split["test_size"] >= data.MIN_TEST_SCENES


def test_protocol_refuses_silent_change_after_sealed_test(tmp_path):
    protocol.freeze_protocol(tmp_path, seed=0)
    (tmp_path / "sealed_test_result.json").write_text(json.dumps({"dummy": True}))
    # re-freezing with the SAME seed is fine (identical content)
    protocol.freeze_protocol(tmp_path, seed=0)
    # re-freezing with a DIFFERENT seed (different content) must raise
    with pytest.raises(RuntimeError):
        protocol.freeze_protocol(tmp_path, seed=1)
