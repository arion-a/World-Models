"""Tests for experiments/task7_geometric_consistency.py.

Fast tests (no bpy, no torch/transformers, no network) exercise config
loading, the "same scene set as Task 6" claim, the shared-split
cross-check, the visual-confound analysis, the Task 6 reproduction
check, and result-text/table generation -- everything that doesn't
require rendering or a model forward pass. A slow group (skipped
without bpy/torch) exercises the real render -> encode -> fit ->
evaluate pipeline for all six transforms end to end on a small number
of scenes with `pretrained=False`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import experiments.geometric_consistency_lib as gclib
from experiments.task7_geometric_consistency import (
    PROTOCOL_NOTES,
    Task7Config,
    _check_shared_split,
    _comparison_table_markdown,
    _confound_analysis,
    _scientific_result_text,
    _task6_reproduction_check,
    assign_split,
    load_task7_config,
    sample_scenes,
    transform_config_for,
)
from transforms.scene_transform import CONTROL_TRANSFORMS, GEOMETRIC_TRANSFORMS, TransformConfig

# --- config loading ----------------------------------------------------------


def test_load_task7_config_defaults():
    cfg = load_task7_config(None)
    assert cfg.num_scenes == 40
    assert cfg.train_fraction == 0.8
    assert cfg.base_seed == 0
    assert cfg.pretrained is True


def test_load_task7_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("num_scenes: 44\npretrained: false\n")
    cfg = load_task7_config(path)
    assert cfg.num_scenes == 44
    assert cfg.pretrained is False
    assert cfg.train_fraction == 0.8


def test_project_config_file_satisfies_task7_minimums():
    cfg = load_task7_config("configs/experiments/task7_geometric_consistency.yaml")
    assert cfg.num_scenes >= 40
    assert cfg.pretrained is True


# --- "same scene set as Task 6" -----------------------------------------


def test_task7_default_config_reuses_task6s_exact_scene_sampling_parameters():
    """tasks/07_geometric_consistency.md's Inputs: 'the same scene set
    used in Task 6.' Task 6's real config (configs/experiments/
    task6_camera_rotation.yaml) uses num_scenes=40, base_seed=0,
    num_objects_min=1, num_objects_max=3 -- Task 7's default config must
    match exactly so the sampled scenes (and hence the derived split)
    are bit-identical."""
    from experiments.task6_camera_rotation import load_task6_config

    task6_cfg = load_task6_config("configs/experiments/task6_camera_rotation.yaml")
    task7_cfg = load_task7_config("configs/experiments/task7_geometric_consistency.yaml")
    assert task7_cfg.num_scenes == task6_cfg.num_scenes
    assert task7_cfg.base_seed == task6_cfg.base_seed
    assert task7_cfg.num_objects_min == task6_cfg.num_objects_min
    assert task7_cfg.num_objects_max == task6_cfg.num_objects_max
    assert task7_cfg.train_fraction == task6_cfg.train_fraction


def test_sample_scenes_reproduces_task6s_scene_set():
    from experiments.task6_camera_rotation import Task6Config, sample_scenes as task6_sample_scenes

    task7_cfg = Task7Config(num_scenes=10, base_seed=0)
    task6_cfg = Task6Config(num_scenes=10, base_seed=0)
    task7_scenes = sample_scenes(task7_cfg)
    task6_scenes = task6_sample_scenes(task6_cfg)
    assert [s.to_dict() for s in task7_scenes] == [s.to_dict() for s in task6_scenes]


def test_assign_split_is_the_same_function_shared_with_task6():
    scene_ids = [f"scene_{i:04d}" for i in range(20)]
    split = assign_split(scene_ids, 0.8, base_seed=0)
    assert split == gclib.assign_split(scene_ids, 0.8, base_seed=0)


# --- transform_config_for: default ranges, uniform across all six ----------


def test_transform_config_for_returns_default_ranges_not_a_fixed_magnitude():
    default = TransformConfig()
    for transform_name in GEOMETRIC_TRANSFORMS + CONTROL_TRANSFORMS:
        cfg = transform_config_for(transform_name)
        assert cfg == default
        # In particular, camera_rotation must NOT be pinned to Task 6's
        # fixed 30-degree azimuth (a single-value range).
        assert cfg.camera_rotation_azimuth_deg_range[0] != cfg.camera_rotation_azimuth_deg_range[1]


def test_protocol_notes_mentions_the_camera_rotation_magnitude_deviation_from_task6():
    text = " ".join(PROTOCOL_NOTES)
    assert "30 degree" in text or "30-degree" in text
    assert "camera_rotation" in text


# --- _check_shared_split ------------------------------------------------


def _fake_results(train_ids, test_ids, transform_names=("camera_rotation", "camera_translation")):
    return {
        name: {"train_scene_ids": list(train_ids), "test_scene_ids": list(test_ids)}
        for name in transform_names
    }


def test_check_shared_split_passes_when_identical():
    results = _fake_results(["a", "b"], ["c"])
    _check_shared_split(results)  # must not raise


def test_check_shared_split_raises_on_mismatched_split():
    results = _fake_results(["a", "b"], ["c"])
    results["camera_translation"]["train_scene_ids"] = ["a"]
    results["camera_translation"]["test_scene_ids"] = ["b", "c"]
    with pytest.raises(ValueError):
        _check_shared_split(results)


def test_check_shared_split_raises_on_leaked_split():
    results = _fake_results(["a", "b"], ["b"])  # 'b' in both train and test
    with pytest.raises(ValueError):
        _check_shared_split(results)


# --- _confound_analysis --------------------------------------------------


def _fake_result_entry(pixel_diff, r2, is_geometric=True):
    return {
        "is_geometric": is_geometric,
        "learned_W_T": {"r2": r2},
        "persistence_baseline": {"r2": 0.0},
        "mean_baseline": {"r2": 0.0},
        "random_pair_control": {"r2": -0.5},
        "visual_confound": {"mean_abs_pixel_diff": pixel_diff},
    }


def test_confound_analysis_reports_strong_positive_correlation_when_perfectly_linked():
    results = {
        "t1": _fake_result_entry(1.0, 0.1),
        "t2": _fake_result_entry(2.0, 0.2),
        "t3": _fake_result_entry(3.0, 0.3),
        "t4": _fake_result_entry(4.0, 0.4),
    }
    analysis = _confound_analysis(results)
    assert analysis["pearson_correlation_pixel_diff_vs_learned_r2"] == pytest.approx(1.0)


def test_confound_analysis_handles_constant_pixel_diff_without_crashing():
    results = {
        "t1": _fake_result_entry(5.0, 0.1),
        "t2": _fake_result_entry(5.0, 0.9),
    }
    analysis = _confound_analysis(results)
    assert analysis["pearson_correlation_pixel_diff_vs_learned_r2"] is None


# --- _task6_reproduction_check --------------------------------------------


def test_task6_reproduction_check_reports_missing_file_gracefully(tmp_path):
    results = {"camera_rotation": _fake_result_entry(1.0, 0.2)}
    check = _task6_reproduction_check(results, tmp_path / "does_not_exist.json")
    assert check["task6_result_found"] is False


def test_task6_reproduction_check_compares_r2_when_file_present(tmp_path):
    task6_path = tmp_path / "task_06_result.json"
    task6_path.write_text(json.dumps({
        "metrics": {"learned_W_T": {"r2": -0.313}},
        "dataset": {"train_scene_ids": ["a"], "test_scene_ids": ["b"]},
    }))
    results = {
        "camera_rotation": {
            **_fake_result_entry(1.0, -0.5),
            "train_scene_ids": ["a"],
            "test_scene_ids": ["b"],
        }
    }
    check = _task6_reproduction_check(results, task6_path)
    assert check["task6_result_found"] is True
    assert check["task6_camera_rotation_r2"] == pytest.approx(-0.313)
    assert check["task7_camera_rotation_r2"] == pytest.approx(-0.5)
    assert check["difference"] == pytest.approx(-0.5 - -0.313)
    assert check["identical_scene_split_as_task6"] is True


# --- text/table generation -------------------------------------------------


def test_scientific_result_text_reports_geometric_vs_control_counts():
    results = {
        "camera_rotation": _fake_result_entry(1.0, 0.5, is_geometric=True),
        "camera_translation": _fake_result_entry(1.0, -0.5, is_geometric=True),
        "object_rotation": _fake_result_entry(1.0, -0.5, is_geometric=True),
        "object_translation": _fake_result_entry(1.0, -0.5, is_geometric=True),
        "lighting_change": _fake_result_entry(1.0, -0.5, is_geometric=False),
        "texture_change": _fake_result_entry(1.0, -0.5, is_geometric=False),
    }
    text = _scientific_result_text(results)
    assert "1/4" in text
    assert "0/2" in text
    assert "camera_rotation" in text


def test_comparison_table_markdown_has_a_row_per_transform():
    results = {
        "camera_rotation": _fake_result_entry(1.0, 0.5),
        "texture_change": _fake_result_entry(1.0, -0.1, is_geometric=False),
    }
    table = _comparison_table_markdown(results)
    assert "camera_rotation" in table
    assert "texture_change" in table
    assert table.count("\n") >= 3  # header x2 + 2 data rows


# --- slow, real end-to-end pipeline tests (bpy + torch/transformers required)
#
# Deliberately NOT a module-level `pytest.importorskip("bpy")` (see
# tests/test_task6_camera_rotation.py's identical rationale): each slow
# test below imports/skips for itself so `pytest -m "not slow"` still
# exercises everything above with zero bpy/torch dependency.


@pytest.mark.slow
def test_full_pipeline_runs_end_to_end_for_all_six_transforms(tmp_path, monkeypatch):
    """Exercises render -> encode -> fit -> evaluate for real, for all
    six transforms, on a small scene count and an untrained
    (pretrained=False) encoder for speed -- NOT a claim about the real
    Task 7 result (which needs >=40 scenes and real pretrained weights
    per configs/experiments/task7_geometric_consistency.yaml), only a
    check that the code path works end to end for every transform
    without error and produces well-shaped, finite, non-leaking output.
    """
    pytest.importorskip("bpy", reason="needs bpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import experiments.task7_geometric_consistency as mod

    monkeypatch.setattr(mod, "MIN_SCENES", 6)

    cfg = Task7Config(
        num_scenes=6,
        train_fraction=0.6667,
        base_seed=0,
        output_dir=str(tmp_path / "geometric_consistency"),
        result_path=str(tmp_path / "task_07_result.json"),
        task6_result_path=str(tmp_path / "nonexistent_task_06_result.json"),
        resolution=32,
        num_frames=2,
        pretrained=False,
    )
    result = mod.run_experiment(cfg)

    assert result["task"] == 7
    assert set(result["transforms_evaluated"]) == set(GEOMETRIC_TRANSFORMS) | set(CONTROL_TRANSFORMS)
    assert result["encoder"]["pretrained"] is False
    assert result["encoder"]["frozen"] is True

    ref_train = set(result["dataset"]["train_scene_ids"])
    ref_test = set(result["dataset"]["test_scene_ids"])
    assert ref_train.isdisjoint(ref_test)
    assert len(ref_train) + len(ref_test) == 6

    for transform_name, entry in result["results"].items():
        assert set(entry["train_scene_ids"]) == ref_train
        assert set(entry["test_scene_ids"]) == ref_test
        for block in ("learned_W_T", "persistence_baseline", "mean_baseline", "random_pair_control"):
            for value in entry[block].values():
                assert np.isfinite(value)
        assert np.isfinite(entry["visual_confound"]["mean_abs_pixel_diff"])
