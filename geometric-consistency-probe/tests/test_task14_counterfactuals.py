"""Tests for experiments/task14_counterfactuals.py.

Fast tests (no bpy, no torch/transformers, no network) exercise config
loading, the magnitude-matching math itself (target_rotation_angle_deg /
reference_point_displacement, both directly against transforms.se3
primitives), the chance-accuracy-threshold/binary-accuracy helpers, the
scene-condition array-pairing logic, and result-text assembly. A slow
group (skipped without bpy/torch) exercises the real render -> verify ->
encode -> fit -> evaluate pipeline end to end on a small number of
scenes with `pretrained=False`, matching the fast-dev pattern already
used by tests/test_task13_occlusion.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.task14_counterfactuals import (
    CONDITION_LABEL,
    CONDITIONS,
    CONFOUND_CONDITIONS,
    CONFOUND_LABEL,
    MIN_SCENES,
    Task14Config,
    binary_accuracy,
    build_discrimination_arrays,
    chance_accuracy_threshold,
    load_task14_config,
    reference_point_displacement,
    target_rotation_angle_deg,
    transform_configs_for_scene,
)
from generation.scene import CameraState, LightState, ObjectState, SceneState
from transforms.se3 import rotate_about_point_matrix, rotation_about_axis, translation_matrix

# --- config loading -----------------------------------------------------


def test_load_task14_config_defaults():
    cfg = load_task14_config(None)
    assert cfg.num_scenes == 40
    assert cfg.train_fraction == 0.8
    assert cfg.target_displacement == 0.4
    assert cfg.pretrained is True


def test_load_task14_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("num_scenes: 44\ntarget_displacement: 0.3\npretrained: false\n")
    cfg = load_task14_config(path)
    assert cfg.num_scenes == 44
    assert cfg.target_displacement == 0.3
    assert cfg.pretrained is False
    assert cfg.train_fraction == 0.8  # untouched field keeps its default


def test_project_config_file_satisfies_task14_minimums():
    """The actual config this task's experiment is run with must satisfy
    the >=40-scenes / real pretrained encoder requirements -- catches a
    config regression that would silently violate acceptance criteria."""
    cfg = load_task14_config("configs/experiments/task14_counterfactuals.yaml")
    assert cfg.num_scenes >= MIN_SCENES
    assert cfg.pretrained is True


def test_conditions_and_confound_conditions_are_disjoint_and_distinct():
    assert set(CONDITIONS) == {"object_translation", "object_rotation"}
    assert set(CONFOUND_CONDITIONS) == {"camera_translation", "lighting_change"}
    assert set(CONDITIONS).isdisjoint(CONFOUND_CONDITIONS)
    assert set(CONDITION_LABEL.values()) == {0, 1}
    assert set(CONFOUND_LABEL.values()) == {0, 1}


# --- magnitude-matching math (pure numpy, cross-checked against transforms.se3) --


def test_target_rotation_angle_deg_reaches_the_target_displacement_exactly():
    """Round trip: solve for the angle, build the actual SE(3) matrix via
    transforms.se3 primitives (not the formula again), and confirm the
    reference point really does move by the target distance."""
    target = 0.4
    radius = 0.3
    pivot = (1.0, 2.0, 3.0)
    angle_deg, clipped = target_rotation_angle_deg(target, radius)
    assert clipped is False

    R = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.radians(angle_deg))
    T_rot = rotate_about_point_matrix(R, pivot)
    measured = reference_point_displacement(T_rot, pivot, (radius, 0.0, 0.0))
    assert measured == pytest.approx(target, abs=1e-9)


def test_target_rotation_angle_deg_flags_unreachable_target_as_clipped():
    # max reachable displacement for radius=0.1 is 2*0.1=0.2; target=0.5 is unreachable.
    angle_deg, clipped = target_rotation_angle_deg(0.5, 0.1)
    assert clipped is True
    assert angle_deg == pytest.approx(180.0, abs=1e-6)


def test_reference_point_displacement_of_a_translation_equals_its_magnitude_for_any_offset():
    delta = (0.24, 0.32, 0.0)  # norm 0.4
    T = translation_matrix(delta)
    for offset in [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.7, -0.5)]:
        measured = reference_point_displacement(T, (5.0, 5.0, 5.0), offset)
        assert measured == pytest.approx(0.4, abs=1e-9)


def test_reference_point_displacement_compares_translation_and_rotation_to_the_same_target():
    """The exact 'given two known transform params, correctly computes/
    compares their displacement magnitude' case required by
    tasks/14_counterfactuals.md's Required software tests."""
    target = 0.4
    radius = 0.3
    pivot = (0.0, 0.0, 0.0)

    T_trans = translation_matrix((0.24, 0.32, 0.0))  # norm exactly 0.4
    disp_trans = reference_point_displacement(T_trans, pivot, (radius, 0.0, 0.0))

    angle_deg, _ = target_rotation_angle_deg(target, radius)
    R = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.radians(angle_deg))
    T_rot = rotate_about_point_matrix(R, pivot)
    disp_rot = reference_point_displacement(T_rot, pivot, (radius, 0.0, 0.0))

    assert disp_trans == pytest.approx(0.4, abs=1e-9)
    assert disp_rot == pytest.approx(0.4, abs=1e-9)
    assert disp_trans == pytest.approx(disp_rot, abs=1e-9)


def _make_single_object_scene(scene_id: str, seed: int, scale: float) -> SceneState:
    obj = ObjectState(shape="cube", position=(0.0, 0.0, scale), rotation_euler=(0.0, 0.0, 0.0), scale=scale, color=(0.5, 0.5, 0.5), instance_id=1)
    camera = CameraState(position=(6.0, 0.0, 3.0), rotation_euler=(0.0, 0.0, 0.0))
    light = LightState(position=(5.0, 5.0, 5.0), energy=3.0, color=(1.0, 1.0, 1.0))
    return SceneState(scene_id=scene_id, seed=seed, objects=(obj,), camera=camera, light=light, floor_color=(0.5, 0.5, 0.5))


def test_transform_configs_for_scene_never_clips_for_the_default_scale_range():
    """Sanity check on this task's chosen target_displacement=0.4 against
    generation.scene_sampler's default object_scale_range=(0.5, 1.0)
    (radius in [0.25, 0.5]): the rotation angle must always be reachable
    without clipping, or the magnitude-matching design silently breaks
    for small objects."""
    cfg = Task14Config()
    for scale in [0.5, 0.65, 0.8, 1.0]:
        scene = _make_single_object_scene("scene_0000", 0, scale)
        matched = transform_configs_for_scene(scene, cfg)
        assert matched["rotation_angle_clipped"] is False
        assert 0.0 < matched["rotation_angle_deg"] < 180.0


def test_transform_configs_for_scene_fixes_translation_magnitude_exactly():
    cfg = Task14Config(target_displacement=0.35)
    scene = _make_single_object_scene("scene_0000", 0, 0.7)
    matched = transform_configs_for_scene(scene, cfg)
    lo, hi = matched["object_translation"].object_translation_range
    assert lo == hi == pytest.approx(0.35)


# --- chance threshold / accuracy helpers (pure numpy) ------------------------


def test_chance_accuracy_threshold_is_above_half_and_shrinks_with_n():
    small_n = chance_accuracy_threshold(8)
    large_n = chance_accuracy_threshold(800)
    assert small_n > 0.5
    assert large_n > 0.5
    assert small_n > large_n  # more examples -> tighter bound around chance


def test_binary_accuracy_thresholds_at_half():
    pred = np.array([0.9, 0.1, 0.6, 0.49, 0.51])
    y = np.array([1, 0, 1, 0, 0])
    # predicted labels: 1,0,1,0,1 -> 4/5 correct
    assert binary_accuracy(pred, y) == pytest.approx(0.8)


def test_binary_accuracy_perfect_and_worst_case():
    assert binary_accuracy(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(1.0)
    assert binary_accuracy(np.array([1.0, 0.0]), np.array([0, 1])) == pytest.approx(0.0)


# --- scene/condition array pairing (leakage check: correct correspondence) --


def _fake_reps_cache(tmp_path, cfg, condition, scene_id, before, after):
    from experiments.task14_counterfactuals import _scene_cache_paths

    paths = _scene_cache_paths(cfg, condition, scene_id)
    paths["scene_dir"].mkdir(parents=True, exist_ok=True)
    np.savez(paths["reps"], Z_before=np.array(before, dtype=np.float64), Z_after=np.array(after, dtype=np.float64))
    paths["record"].write_text(json.dumps({"scene_id": scene_id, "condition": condition}))


def test_build_discrimination_arrays_pairs_conditions_and_scenes_correctly(tmp_path):
    cfg = Task14Config(output_dir=str(tmp_path / "counterfactuals"))
    scene_ids = ["scene_0000", "scene_0001", "scene_0002"]
    for i, sid in enumerate(scene_ids):
        _fake_reps_cache(tmp_path, cfg, "object_translation", sid, before=[0.0, 0.0], after=[float(i), 0.0])
        _fake_reps_cache(tmp_path, cfg, "object_rotation", sid, before=[0.0, 0.0], after=[0.0, float(i)])

    X, y = build_discrimination_arrays(cfg, scene_ids, CONDITIONS, CONDITION_LABEL)
    assert X.shape == (6, 2)
    assert list(y) == [0, 1, 0, 1, 0, 1]
    # scene_1's rows are at index 2 (translation) and 3 (rotation):
    np.testing.assert_allclose(X[2], [1.0, 0.0])
    np.testing.assert_allclose(X[3], [0.0, 1.0])
    # scene_2's rows are at index 4/5:
    np.testing.assert_allclose(X[4], [2.0, 0.0])
    np.testing.assert_allclose(X[5], [0.0, 2.0])


def test_build_discrimination_arrays_raises_on_missing_cache(tmp_path):
    cfg = Task14Config(output_dir=str(tmp_path / "counterfactuals"))
    with pytest.raises(ValueError):
        build_discrimination_arrays(cfg, ["scene_0000"], CONDITIONS, CONDITION_LABEL)


# --- scientific-result text (pure dict math) ---------------------------------


def _fake_metrics(real_acc, scrambled_acc, norm_acc, leak_acc, chance=0.6):
    return {
        "discrimination": {
            "real_probe_accuracy": real_acc,
            "scrambled_label_control_accuracy": scrambled_acc,
            "chance_accuracy_threshold": chance,
            "num_train_examples": 64,
            "num_test_examples": 16,
        },
        "representation_norm_only_check": {
            "accuracy": norm_acc,
            "chance_accuracy_threshold": chance,
            "not_trivially_separable": norm_acc <= chance,
        },
        "fixed_variable_leakage_check": {
            "confound_conditions": list(CONFOUND_CONDITIONS),
            "accuracy_on_real_test_set": leak_acc,
            "chance_accuracy_threshold": chance,
            "passed": leak_acc <= chance,
            "num_confound_train_examples": 64,
        },
        "magnitude_matching": {
            "target_displacement": 0.4,
            "object_translation_measured_displacement": {"mean": 0.4, "std": 0.0, "min": 0.4, "max": 0.4},
            "object_rotation_measured_displacement": {"mean": 0.4, "std": 0.0, "min": 0.4, "max": 0.4},
            "mean_absolute_difference_between_conditions": 0.0,
            "any_rotation_angle_clipped": False,
        },
    }


def test_scientific_result_text_positive_case():
    from experiments.task14_counterfactuals import _scientific_result_text

    metrics = _fake_metrics(real_acc=0.85, scrambled_acc=0.5, norm_acc=0.55, leak_acc=0.5)
    text = _scientific_result_text(metrics, hypothesis_supported=True)
    assert "POSITIVE result" in text
    assert "causal or compositional world model" in text.lower() or "not a causal" in text.lower()


def test_scientific_result_text_leakage_failure_is_prominent_and_invalidates():
    from experiments.task14_counterfactuals import _scientific_result_text

    metrics = _fake_metrics(real_acc=0.85, scrambled_acc=0.5, norm_acc=0.55, leak_acc=0.9)
    text = _scientific_result_text(metrics, hypothesis_supported=False)
    assert "leakage check FAILED" in text
    assert "cannot be trusted" in text.lower()


def test_scientific_result_text_norm_trivial_failure_is_reported():
    from experiments.task14_counterfactuals import _scientific_result_text

    metrics = _fake_metrics(real_acc=0.85, scrambled_acc=0.5, norm_acc=0.9, leak_acc=0.5)
    text = _scientific_result_text(metrics, hypothesis_supported=False)
    assert "trivially separable" in text.lower()


def test_scientific_result_text_plain_negative_case():
    from experiments.task14_counterfactuals import _scientific_result_text

    metrics = _fake_metrics(real_acc=0.5, scrambled_acc=0.5, norm_acc=0.5, leak_acc=0.5)
    text = _scientific_result_text(metrics, hypothesis_supported=False)
    assert "NEGATIVE result" in text
    assert "valid, complete outcome" in text.lower()


# --- pytest summary parsing (delegates to gclib, sanity-checked here) -------


def test_run_existing_test_suite_delegates_to_gclib(monkeypatch):
    import experiments.geometric_consistency_lib as gclib
    import experiments.task14_counterfactuals as mod

    monkeypatch.setattr(gclib, "run_existing_test_suite", lambda root: {"passed": 3, "failed": 0})
    assert mod.run_existing_test_suite("unused") == {"passed": 3, "failed": 0}


# --- slow, real end-to-end pipeline test (bpy + torch/transformers required) -


@pytest.mark.slow
def test_full_pipeline_runs_end_to_end_on_a_small_scene_set(tmp_path):
    """Exercises render -> verify -> encode -> fit -> evaluate for both
    conditions AND the confound leakage-check conditions on a small
    scene count and an untrained (pretrained=False) encoder for speed --
    NOT a claim about the real Task 14 result (which needs >=40 scenes
    and real pretrained weights per
    configs/experiments/task14_counterfactuals.yaml), only a check that
    the code path works end to end without error and produces
    well-shaped, finite metrics.
    """
    pytest.importorskip("bpy", reason="needs bpy")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import experiments.task14_counterfactuals as mod

    monkeypatch_min = mod.MIN_SCENES
    mod.MIN_SCENES = 6
    try:
        cfg = mod.Task14Config(
            num_scenes=6,
            train_fraction=0.6667,
            output_dir=str(tmp_path / "counterfactuals"),
            resolution=48,
            num_frames=2,
            fps=4.0,
            pretrained=False,
        )
        result = mod.run_experiment(cfg)
    finally:
        mod.MIN_SCENES = monkeypatch_min

    assert result["task"] == 14
    assert result["implementation_status"] == "COMPLETE"
    assert result["encoder"]["pretrained"] is False
    assert set(result["dataset"]["train_scene_ids"]).isdisjoint(result["dataset"]["test_scene_ids"])

    def _assert_finite(d):
        for v in d.values():
            if isinstance(v, dict):
                _assert_finite(v)
            elif isinstance(v, float):
                assert np.isfinite(v)

    _assert_finite(result["metrics"])
