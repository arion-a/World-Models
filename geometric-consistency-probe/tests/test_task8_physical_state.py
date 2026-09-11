"""Tests for experiments/task8_physical_state.py.

Fast tests (no bpy, no torch/transformers, no network) exercise config
loading, the "same scene set/split as Tasks 6-7" claim, the (sin, cos)
azimuth encode/decode round trip, the shuffled-label permutation guard,
the per-variable fit/evaluate logic on synthetic representations, and
result-text/table generation -- everything that doesn't require
rendering or a model forward pass. A slow group (skipped without
bpy/torch) exercises the real render -> encode -> fit -> evaluate
pipeline end to end on a small number of scenes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import experiments.geometric_consistency_lib as gclib
from experiments.task8_physical_state import (
    RAW_FEATURE_FNS,
    VARIABLES,
    Task8Config,
    _circular_diff_deg,
    _comparison_table_markdown,
    _decode_angle_deg,
    _exceeds_baselines,
    _probe_target,
    _scientific_result_text,
    assign_split,
    build_variable_arrays,
    check_scene_alignment,
    evaluate_variable,
    load_task8_config,
    sample_scenes,
    shuffled_label_permutation,
)

# --- config loading ----------------------------------------------------------


def test_load_task8_config_defaults():
    cfg = load_task8_config(None)
    assert cfg.num_scenes == 40
    assert cfg.train_fraction == 0.8
    assert cfg.base_seed == 0
    assert cfg.pretrained is True


def test_load_task8_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("num_scenes: 44\npretrained: false\n")
    cfg = load_task8_config(path)
    assert cfg.num_scenes == 44
    assert cfg.pretrained is False
    assert cfg.train_fraction == 0.8


def test_project_config_file_satisfies_task8_minimums():
    cfg = load_task8_config("configs/experiments/task8_physical_state.yaml")
    assert cfg.num_scenes >= 40
    assert cfg.pretrained is True


# --- "same scene set / split as Tasks 6-7" ----------------------------------


def test_task8_default_config_reuses_task7s_exact_scene_sampling_parameters():
    from experiments.task7_geometric_consistency import load_task7_config

    task7_cfg = load_task7_config("configs/experiments/task7_geometric_consistency.yaml")
    task8_cfg = load_task8_config("configs/experiments/task8_physical_state.yaml")
    assert task8_cfg.num_scenes == task7_cfg.num_scenes
    assert task8_cfg.base_seed == task7_cfg.base_seed
    assert task8_cfg.num_objects_min == task7_cfg.num_objects_min
    assert task8_cfg.num_objects_max == task7_cfg.num_objects_max
    assert task8_cfg.train_fraction == task7_cfg.train_fraction


def test_sample_scenes_reproduces_task7s_scene_set():
    from experiments.task7_geometric_consistency import Task7Config, sample_scenes as task7_sample_scenes

    task8_cfg = Task8Config(num_scenes=10, base_seed=0)
    task7_cfg = Task7Config(num_scenes=10, base_seed=0)
    task8_scenes = sample_scenes(task8_cfg)
    task7_scenes = task7_sample_scenes(task7_cfg)
    assert [s.to_dict() for s in task8_scenes] == [s.to_dict() for s in task7_scenes]


def test_assign_split_is_the_same_function_shared_with_task7():
    scene_ids = [f"scene_{i:04d}" for i in range(20)]
    split = assign_split(scene_ids, 0.8, base_seed=0)
    assert split == gclib.assign_split(scene_ids, 0.8, base_seed=0)


# --- (sin, cos) azimuth encode/decode ---------------------------------------


def test_probe_target_encodes_azimuth_as_unit_sin_cos():
    target = _probe_target("camera_azimuth_deg", 90.0)
    assert target.shape == (2,)
    np.testing.assert_allclose(target, [1.0, 0.0], atol=1e-6)
    assert np.linalg.norm(target) == pytest.approx(1.0)


def test_probe_target_is_raw_for_non_circular_scalar():
    target = _probe_target("camera_elevation_deg", 33.5)
    assert target.shape == (1,)
    assert target[0] == pytest.approx(33.5)


def test_probe_target_is_raw_for_vector_variable():
    target = _probe_target("primary_object_position_xy", np.array([1.0, -2.0]))
    np.testing.assert_allclose(target, [1.0, -2.0])


@pytest.mark.parametrize("angle_deg", [0.0, 45.0, 90.0, 179.0, -90.0, -179.0, 359.0])
def test_decode_angle_deg_round_trips_probe_target(angle_deg):
    sin_cos = _probe_target("camera_azimuth_deg", angle_deg).reshape(1, 2)
    decoded = _decode_angle_deg(sin_cos)[0]
    assert _circular_diff_deg(np.array([decoded]), np.array([angle_deg]))[0] == pytest.approx(0.0, abs=1e-4)


def test_circular_diff_deg_handles_wraparound():
    # 359 degrees and 1 degree are only 2 degrees apart, not 358.
    diff = _circular_diff_deg(np.array([359.0]), np.array([1.0]))
    assert diff[0] == pytest.approx(2.0)


def test_circular_diff_deg_zero_for_identical_angles():
    diff = _circular_diff_deg(np.array([37.0]), np.array([37.0]))
    assert diff[0] == pytest.approx(0.0)


# --- shuffled-label permutation guard ---------------------------------------


def test_shuffled_label_permutation_is_a_genuine_permutation():
    perm = shuffled_label_permutation(10, seed=0)
    assert sorted(perm.tolist()) == list(range(10))


def test_shuffled_label_permutation_guards_against_identity_for_small_n():
    # Brute-force many seeds; none may produce the identity permutation.
    for seed in range(50):
        perm = shuffled_label_permutation(4, seed=seed)
        assert not np.all(perm == np.arange(4)), f"seed {seed} produced an identity permutation"


def test_shuffled_label_permutation_handles_n_equals_one():
    perm = shuffled_label_permutation(1, seed=0)
    assert perm.tolist() == [0]


# --- evaluate_variable: fit/evaluate/controls on synthetic representations --


def _synthetic_arrays(variable: str, n_train: int = 20, n_test: int = 8, informative: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    dim = 16
    n = n_train + n_test

    if variable == "camera_azimuth_deg":
        raw = rng.uniform(0.0, 360.0, size=n)
        targets = np.stack([_probe_target(variable, v) for v in raw])
    elif variable == "primary_object_position_xy":
        raw = rng.uniform(-1.6, 1.6, size=(n, 2))
        targets = np.stack([_probe_target(variable, v) for v in raw])
    else:
        raw = rng.uniform(20.0, 50.0, size=n)
        targets = np.stack([_probe_target(variable, v) for v in raw])

    if informative:
        # Z linearly encodes the target (plus noise) in its first columns,
        # so a ridge probe should recover it well above baseline.
        w = rng.normal(size=(targets.shape[1], dim))
        Z = targets @ w + 0.01 * rng.normal(size=(n, dim))
    else:
        # Z carries no information about the target at all.
        Z = rng.normal(size=(n, dim))

    scene_ids = [f"scene_{i:04d}" for i in range(n)]
    return {
        "train_scene_ids": scene_ids[:n_train],
        "test_scene_ids": scene_ids[n_train:],
        "Z_train": Z[:n_train],
        "Z_test": Z[n_train:],
        "raw_train": raw[:n_train],
        "raw_test": raw[n_train:],
        "y_train": targets[:n_train],
        "y_test": targets[n_train:],
    }


@pytest.mark.parametrize("variable", VARIABLES)
def test_evaluate_variable_recovers_informative_signal(variable):
    arrays = _synthetic_arrays(variable, informative=True, seed=1)
    result = evaluate_variable(variable, arrays, ridge_alpha=1e-3, shuffled_label_seed=0)
    assert result["real_probe"]["r2"] > 0.9
    assert _exceeds_baselines(result)
    if variable == "camera_azimuth_deg":
        assert result["probe_encoding"] == "sin_cos"
        assert result["real_probe"]["angular_error_deg"] < 5.0
    else:
        assert result["probe_encoding"] == "raw"
        assert "angular_error_deg" not in result["real_probe"]


@pytest.mark.parametrize("variable", VARIABLES)
def test_evaluate_variable_reports_no_nan_or_inf(variable):
    for informative in (True, False):
        arrays = _synthetic_arrays(variable, informative=informative, seed=2)
        result = evaluate_variable(variable, arrays, ridge_alpha=10.0, shuffled_label_seed=0)
        flat = json.dumps(result)
        assert "NaN" not in flat and "Infinity" not in flat


def test_evaluate_variable_is_reproducible():
    arrays = _synthetic_arrays("camera_elevation_deg", informative=True, seed=3)
    r1 = evaluate_variable("camera_elevation_deg", arrays, ridge_alpha=10.0, shuffled_label_seed=0)
    r2 = evaluate_variable("camera_elevation_deg", arrays, ridge_alpha=10.0, shuffled_label_seed=0)
    assert r1 == r2


def test_evaluate_variable_shuffled_control_uses_different_labels_than_real_fit():
    """The shuffled-label control must not silently reduce to the real fit
    -- i.e. it must actually use a different label correspondence."""
    arrays = _synthetic_arrays("camera_distance", informative=True, seed=4)
    result = evaluate_variable("camera_distance", arrays, ridge_alpha=10.0, shuffled_label_seed=0)
    # With a strongly informative signal, the shuffled control should score
    # noticeably worse than the real probe.
    assert result["shuffled_label_control"]["r2"] < result["real_probe"]["r2"]


def test_evaluate_variable_target_dim_matches_encoding():
    arrays = _synthetic_arrays("camera_azimuth_deg", informative=True, seed=5)
    result = evaluate_variable("camera_azimuth_deg", arrays, ridge_alpha=10.0, shuffled_label_seed=0)
    assert result["target_dim"] == 2

    arrays = _synthetic_arrays("camera_elevation_deg", informative=True, seed=5)
    result = evaluate_variable("camera_elevation_deg", arrays, ridge_alpha=10.0, shuffled_label_seed=0)
    assert result["target_dim"] == 1

    arrays = _synthetic_arrays("primary_object_position_xy", informative=True, seed=5)
    result = evaluate_variable("primary_object_position_xy", arrays, ridge_alpha=10.0, shuffled_label_seed=0)
    assert result["target_dim"] == 2


# --- build_variable_arrays: scene-level split disjointness ------------------


class _FakeScene:
    def __init__(self, scene_id):
        self.scene_id = scene_id


def test_build_variable_arrays_produces_disjoint_scene_level_split(monkeypatch):
    scenes = [_FakeScene(f"scene_{i:04d}") for i in range(4)]
    split = {"scene_0000": "train", "scene_0001": "test", "scene_0002": "train", "scene_0003": "test"}

    reps = {s.scene_id: np.zeros(4) for s in scenes}
    monkeypatch.setitem(RAW_FEATURE_FNS, "camera_distance", lambda s: 1.0)
    arrays = build_variable_arrays(scenes, split, reps, "camera_distance")
    assert arrays["train_scene_ids"] == ["scene_0000", "scene_0002"]
    assert arrays["test_scene_ids"] == ["scene_0001", "scene_0003"]
    assert set(arrays["train_scene_ids"]).isdisjoint(arrays["test_scene_ids"])
    assert arrays["y_train"].shape == (2, 1)
    assert arrays["y_test"].shape == (2, 1)


# --- check_scene_alignment ---------------------------------------------------


class _FakeCamera:
    def __init__(self, position):
        self.position = position


class _FakeSceneWithCamera:
    def __init__(self, scene_id, camera_position):
        self.scene_id = scene_id
        self.camera = _FakeCamera(camera_position)


def test_check_scene_alignment_passes_for_matching_position():
    scene = _FakeSceneWithCamera("scene_0000", (1.0, 2.0, 3.0))
    metadata = {"camera": {"per_frame_pose": [{"position": [1.0, 2.0, 3.0]}]}}
    diff = check_scene_alignment(scene, metadata, tolerance=1e-6)
    assert diff == pytest.approx(0.0)


def test_check_scene_alignment_raises_for_mismatched_position():
    scene = _FakeSceneWithCamera("scene_0000", (1.0, 2.0, 3.0))
    metadata = {"camera": {"per_frame_pose": [{"position": [1.0, 2.0, 999.0]}]}}
    with pytest.raises(ValueError, match="misalignment"):
        check_scene_alignment(scene, metadata, tolerance=1e-6)


# --- text/table generation --------------------------------------------------


def _fake_metrics_block(r2, probe_encoding="raw", angular_error=None):
    block = {
        "probe_encoding": probe_encoding,
        "target_dim": 2 if probe_encoding == "sin_cos" else 1,
        "real_probe": {"r2": r2, "mae": 0.1, "rmse": 0.2},
        "shuffled_label_control": {"r2": -0.5, "mae": 1.0, "rmse": 1.5},
        "mean_baseline": {"r2": -0.1, "mae": 0.5, "rmse": 0.7},
        "label_diversity": {"min": 0.0, "max": 1.0, "std": 0.3},
    }
    if angular_error is not None:
        block["real_probe"]["angular_error_deg"] = angular_error
    return block


def test_scientific_result_text_reports_positive_variable_count():
    metrics = {
        "camera_azimuth_deg": _fake_metrics_block(0.8, probe_encoding="sin_cos", angular_error=3.0),
        "camera_elevation_deg": _fake_metrics_block(0.6),
        "camera_distance": _fake_metrics_block(-0.3),
        "primary_object_position_xy": _fake_metrics_block(-0.2),
    }
    text = _scientific_result_text(metrics)
    assert "2/4" in text
    assert "camera_azimuth_deg" in text
    assert "confirms the hypothesis" in text


def test_scientific_result_text_reports_negative_result_when_hypothesis_not_met():
    metrics = {
        "camera_azimuth_deg": _fake_metrics_block(-0.9, probe_encoding="sin_cos", angular_error=90.0),
        "camera_elevation_deg": _fake_metrics_block(-0.3),
        "camera_distance": _fake_metrics_block(-0.3),
        "primary_object_position_xy": _fake_metrics_block(-0.2),
    }
    text = _scientific_result_text(metrics)
    assert "0/4" in text
    assert "does NOT confirm" in text


def test_comparison_table_markdown_has_a_row_per_variable():
    metrics = {name: _fake_metrics_block(0.5) for name in VARIABLES}
    table = _comparison_table_markdown(metrics)
    for name in VARIABLES:
        assert name in table
    assert table.count("\n") >= len(VARIABLES) + 1


# --- slow, real end-to-end pipeline test (bpy + torch/transformers required)


@pytest.mark.slow
def test_full_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    """Exercises render/reuse -> encode -> fit -> evaluate for real, on a
    small scene count and an untrained (pretrained=False) encoder for
    speed -- NOT a claim about the real Task 8 result (which needs
    >=40 scenes and real pretrained weights per
    configs/experiments/task8_physical_state.yaml), only a check that
    the code path works end to end without error and produces
    well-shaped, finite, non-leaking output.
    """
    pytest.importorskip("bpy", reason="needs bpy")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import experiments.task8_physical_state as mod

    monkeypatch.setattr(mod, "MIN_SCENES", 6)

    cfg = Task8Config(
        num_scenes=6,
        train_fraction=0.6667,
        base_seed=0,
        result_path=str(tmp_path / "task_08_result.json"),
        reuse_render_dir=str(tmp_path / "nonexistent_cache"),
        own_render_dir=str(tmp_path / "original_renders"),
        resolution=32,
        num_frames=2,
        pretrained=False,
    )
    result = mod.run_experiment(cfg)

    assert result["task"] == 8
    assert result["dataset"]["render_source"] == "freshly_rendered"
    assert result["encoder"]["pretrained"] is False
    assert result["encoder"]["frozen"] is True

    ref_train = set(result["dataset"]["train_scene_ids"])
    ref_test = set(result["dataset"]["test_scene_ids"])
    assert ref_train.isdisjoint(ref_test)
    assert len(ref_train) + len(ref_test) == 6

    for variable, entry in result["metrics"].items():
        for block in ("real_probe", "shuffled_label_control", "mean_baseline"):
            for value in entry[block].values():
                assert np.isfinite(value)

    assert result["scene_render_alignment_check"]["verified"] is True
    assert result["scene_render_alignment_check"]["max_camera_position_abs_diff"] < 1e-4
