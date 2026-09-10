"""Tests for experiments/geometric_consistency_lib.py -- the shared
library Task 6's script was refactored into and Task 7 (all six
transforms) runs on top of.

Fast tests (no bpy, no torch/transformers, no network) exercise scene
sampling, the scene-level split, the generic ground-truth verifier
(cross-checked against transforms.scene_transform.apply_transform's own
real output for every one of the six transforms, not just camera_rotation),
pixel-diff confound stats, and the fit/evaluate/baseline wiring on
synthetic representation vectors.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import experiments.geometric_consistency_lib as gclib
from generation.scene_sampler import SceneSamplerConfig, sample_scene
from transforms.scene_transform import CONTROL_TRANSFORMS, GEOMETRIC_TRANSFORMS, TRANSFORM_NAMES, TransformConfig, apply_transform


def _scene(seed=1, num_objects_min=2, num_objects_max=2):
    return sample_scene(f"gclib_scene_{seed}", seed=seed, cfg=SceneSamplerConfig(num_objects_min=num_objects_min, num_objects_max=num_objects_max))


# --- sample_scenes / assign_split (pure numpy, no bpy) ----------------------


def test_sample_scenes_returns_at_least_requested_count_with_distinct_ids_and_seeds():
    scenes = gclib.sample_scenes(num_scenes=10, base_seed=3)
    assert len(scenes) == 10
    ids = [s.scene_id for s in scenes]
    seeds = [s.seed for s in scenes]
    assert len(set(ids)) == len(ids)
    assert len(set(seeds)) == len(seeds)


def test_sample_scenes_is_reproducible_given_the_same_arguments():
    a = gclib.sample_scenes(6, base_seed=7)
    b = gclib.sample_scenes(6, base_seed=7)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_sample_scenes_matches_task6s_own_convention_exactly():
    """Task 7 must reuse Task 6's EXACT scene set (same convention) --
    experiments.task6_camera_rotation.sample_scenes now delegates to
    this function, so this is really a regression check that the
    refactor preserved Task 6's scene-id/seed derivation."""
    from experiments.task6_camera_rotation import Task6Config, sample_scenes as task6_sample_scenes

    cfg = Task6Config(num_scenes=8, base_seed=2)
    task6_scenes = task6_sample_scenes(cfg)
    lib_scenes = gclib.sample_scenes(cfg.num_scenes, cfg.base_seed, cfg.num_objects_min, cfg.num_objects_max)
    assert [s.to_dict() for s in task6_scenes] == [s.to_dict() for s in lib_scenes]


def test_assign_split_disjoint_and_covers_everything():
    scene_ids = [f"scene_{i:04d}" for i in range(40)]
    split = gclib.assign_split(scene_ids, train_fraction=0.8, base_seed=0)
    train = {sid for sid, label in split.items() if label == "train"}
    test = {sid for sid, label in split.items() if label == "test"}
    assert train.isdisjoint(test)
    assert train | test == set(scene_ids)
    assert len(train) == 32


def test_assign_split_rejects_invalid_train_fraction():
    with pytest.raises(ValueError):
        gclib.assign_split(["a", "b"], train_fraction=1.0, base_seed=0)


# --- expected_changed_variables / verify_transform_ground_truth ------------
# Cross-checked against apply_transform's own real ground truth for every
# one of the six transforms, not fabricated -- tasks/07_geometric_consistency.md's
# "checked against apply_transform's own changed_variables/fixed_variables/
# transform_matrix ground truth."


@pytest.mark.parametrize("transform_name", TRANSFORM_NAMES)
def test_expected_changed_variables_matches_apply_transforms_real_output(transform_name):
    scene = _scene(seed=TRANSFORM_NAMES.index(transform_name) * 2, num_objects_min=3, num_objects_max=3)
    _, transformation = apply_transform(scene, transform_name, TransformConfig())
    expected = gclib.expected_changed_variables(transform_name, transformation, num_objects=len(scene.objects))
    assert expected == set(transformation["changed_variables"])


@pytest.mark.parametrize("transform_name", TRANSFORM_NAMES)
def test_verify_transform_ground_truth_accepts_every_real_transform(tmp_path, transform_name):
    scene = _scene(seed=TRANSFORM_NAMES.index(transform_name) * 2 + 1, num_objects_min=2, num_objects_max=3)
    _, transformation = apply_transform(scene, transform_name, TransformConfig())
    pair_dir = tmp_path / transform_name
    pair_dir.mkdir()
    (pair_dir / "transformation.json").write_text(json.dumps(transformation))

    gclib.verify_transform_ground_truth(pair_dir, transform_name, num_objects=len(scene.objects))  # must not raise


def test_verify_transform_ground_truth_rejects_wrong_transform_type(tmp_path):
    scene = _scene(seed=99)
    _, transformation = apply_transform(scene, "object_rotation", TransformConfig())
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    (pair_dir / "transformation.json").write_text(json.dumps(transformation))

    with pytest.raises(ValueError):
        gclib.verify_transform_ground_truth(pair_dir, "camera_rotation", num_objects=len(scene.objects))


@pytest.mark.parametrize("transform_name", GEOMETRIC_TRANSFORMS)
def test_verify_transform_ground_truth_rejects_missing_matrix_for_geometric_transforms(tmp_path, transform_name):
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    fake = {
        "type": transform_name,
        "transform_name": transform_name,
        "transform_matrix": None,
        "changed_variables": ["camera.position"],
        "object_index": 0,
    }
    (pair_dir / "transformation.json").write_text(json.dumps(fake))
    with pytest.raises(ValueError):
        gclib.verify_transform_ground_truth(pair_dir, transform_name, num_objects=1)


@pytest.mark.parametrize("transform_name", CONTROL_TRANSFORMS)
def test_verify_transform_ground_truth_rejects_matrix_present_for_control_transforms(tmp_path, transform_name):
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    fake = {
        "type": transform_name,
        "transform_name": transform_name,
        "transform_matrix": np.eye(4).tolist(),
        "changed_variables": ["light.position", "light.energy"] if transform_name == "lighting_change" else ["objects[0].color"],
    }
    (pair_dir / "transformation.json").write_text(json.dumps(fake))
    with pytest.raises(ValueError):
        gclib.verify_transform_ground_truth(pair_dir, transform_name, num_objects=1)


def test_verify_transform_ground_truth_rejects_extra_changed_variables(tmp_path):
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    fake = {
        "type": "camera_translation",
        "transform_name": "camera_translation",
        "transform_matrix": np.eye(4).tolist(),
        "changed_variables": ["camera.position", "light.energy"],
    }
    (pair_dir / "transformation.json").write_text(json.dumps(fake))
    with pytest.raises(ValueError):
        gclib.verify_transform_ground_truth(pair_dir, "camera_translation", num_objects=1)


def test_verify_transform_ground_truth_tracks_which_object_index_object_transforms_touched(tmp_path):
    """object_translation/object_rotation touch a randomly-chosen object
    per scene -- the verifier must read the transform's own recorded
    object_index rather than assuming object 0."""
    scene = _scene(seed=123, num_objects_min=3, num_objects_max=3)
    # Keep sampling scenes until object_translation picks a non-zero index,
    # so this test actually exercises the "not object 0" path.
    for seed in range(123, 200):
        scene = _scene(seed=seed, num_objects_min=3, num_objects_max=3)
        _, transformation = apply_transform(scene, "object_translation", TransformConfig())
        if transformation["object_index"] != 0:
            break
    assert transformation["object_index"] != 0

    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    (pair_dir / "transformation.json").write_text(json.dumps(transformation))
    gclib.verify_transform_ground_truth(pair_dir, "object_translation", num_objects=len(scene.objects))  # must not raise


# --- pixel_diff_stats (monkeypatched load_pair -- no real renders needed) --


def test_pixel_diff_stats_computes_mean_absolute_difference(tmp_path, monkeypatch):
    scenes = [_scene(seed=1), _scene(seed=2)]

    def fake_load_pair(pair_dir):
        orig = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        trans = np.full((2, 4, 4, 3), 10, dtype=np.uint8)
        return {"original": (None, orig, None, None), "transformed": (None, trans, None, None)}

    monkeypatch.setattr(gclib, "load_pair", fake_load_pair)
    stats = gclib.pixel_diff_stats(scenes, tmp_path)
    assert stats["mean_abs_pixel_diff"] == pytest.approx(10.0)
    assert stats["std_abs_pixel_diff"] == pytest.approx(0.0)
    assert len(stats["per_scene_mean_abs_pixel_diff"]) == 2


# --- evaluate_transform (synthetic Z arrays, no torch needed) --------------


def test_evaluate_transform_returns_all_required_keys_and_finite_values():
    rng = np.random.default_rng(0)
    Z_train = rng.normal(size=(20, 8))
    Zp_train = Z_train + rng.normal(scale=0.1, size=(20, 8))
    Z_test = rng.normal(size=(5, 8))
    Zp_test = Z_test + rng.normal(scale=0.1, size=(5, 8))

    block = gclib.evaluate_transform("camera_rotation", Z_train, Zp_train, Z_test, Zp_test, ridge_alpha=10.0, shuffled_pairing_seed=0)

    assert set(block.keys()) == {"learned_W_T", "persistence_baseline", "mean_baseline", "random_pair_control"}
    for method_metrics in block.values():
        assert set(method_metrics.keys()) == {"r2", "mean_cosine_similarity", "mean_relative_l2_error"}
        for value in method_metrics.values():
            assert np.isfinite(value)


# --- provenance helpers -------------------------------------------------


def test_parse_pytest_summary_all_passed():
    assert gclib.parse_pytest_summary("...\n7 passed in 0.1s\n") == (7, 0)


def test_parse_pytest_summary_mixed():
    assert gclib.parse_pytest_summary("...\n2 failed, 5 passed in 0.2s\n") == (5, 2)
