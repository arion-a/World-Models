"""Tests for experiments/task12_temporal_consistency.py.

Fast tests (no bpy, no torch/transformers, no network) exercise config
loading, the windowing scheme, temporal ground-truth derivation/
verification (generation.motion.generate_trajectory is pure math, so
this is all directly testable), the disjoint-scene-set leakage guard
between conditions, result assembly/margin computation, and scientific-
result text. A slow group (skipped without bpy/torch) exercises the
real render -> encode -> fit -> evaluate pipeline end to end on a small
number of scenes with `pretrained=False`, matching the fast-dev pattern
already used by tests/test_task6_camera_rotation.py and
tests/test_task11_scale.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.task12_temporal_consistency import (
    CONDITIONS,
    MIN_SCENES,
    Task12Config,
    _scientific_result_text,
    assemble_result,
    camera_motion_for,
    compute_relative_camera_transform,
    load_task12_config,
    verify_saved_temporal_transform,
    verify_temporal_ground_truth,
)
from generation.motion import CameraMotion, ObjectMotion, generate_trajectory
from generation.scene_sampler import SceneSamplerConfig, sample_scene
from transforms.se3 import compose, inverse_rigid, pose_matrix

# --- config loading -----------------------------------------------------


def test_load_task12_config_defaults():
    cfg = load_task12_config(None)
    assert cfg.num_scenes == 40
    assert cfg.train_fraction == 0.8
    assert cfg.window_frames == 4
    assert cfg.pretrained is True
    assert cfg.base_seed_real_motion != cfg.base_seed_static_control


def test_load_task12_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("num_scenes: 44\norbit_deg_per_sec: 15.0\npretrained: false\n")
    cfg = load_task12_config(path)
    assert cfg.num_scenes == 44
    assert cfg.orbit_deg_per_sec == 15.0
    assert cfg.pretrained is False
    assert cfg.train_fraction == 0.8  # untouched field keeps its default


def test_project_config_file_satisfies_task12_minimums():
    """The actual config this task's experiment is run with must satisfy
    the >=40-scenes-per-condition / real pretrained encoder / disjoint-
    base-seed requirements -- catches a config regression that would
    silently violate the task's acceptance criteria without a person
    noticing."""
    cfg = load_task12_config("configs/experiments/task12_temporal_consistency.yaml")
    assert cfg.num_scenes >= MIN_SCENES
    assert cfg.pretrained is True
    assert cfg.base_seed_real_motion != cfg.base_seed_static_control
    assert cfg.orbit_deg_per_sec > 0
    assert cfg.window_frames >= 2


# --- disjoint scene sets between conditions (leakage requirement) ---------


def test_base_seeds_produce_disjoint_scene_sets():
    """Required leakage check: the static-clip control must use a
    different, disjoint scene set from the real-motion experiment."""
    import experiments.geometric_consistency_lib as gclib

    cfg = Task12Config()
    real_scenes = gclib.sample_scenes(cfg.num_scenes, cfg.base_seed_real_motion, cfg.num_objects_min, cfg.num_objects_max)
    static_scenes = gclib.sample_scenes(cfg.num_scenes, cfg.base_seed_static_control, cfg.num_objects_min, cfg.num_objects_max)
    real_seeds = {s.seed for s in real_scenes}
    static_seeds = {s.seed for s in static_scenes}
    assert real_seeds.isdisjoint(static_seeds)


def test_assemble_result_raises_if_scene_seeds_overlap_between_conditions():
    """assemble_result must independently re-verify condition disjointness,
    not just trust that the caller used distinct base seeds."""
    real = _fake_condition("real_motion", r2=0.5, seeds=[0, 1, 2])
    static = _fake_condition("static_control", r2=0.1, seeds=[2, 3, 4])  # overlaps at seed 2
    cfg = Task12Config(output_dir="unused")
    with pytest.raises(ValueError, match="not disjoint"):
        assemble_result(cfg, real, static)


# --- windowing scheme + temporal ground truth (pure math, no bpy) ---------


@pytest.fixture
def scene():
    return sample_scene("temporal_test", seed=11, cfg=SceneSamplerConfig(num_objects_min=1, num_objects_max=2))


def test_camera_motion_for_each_condition(scene):
    assert camera_motion_for("real_motion", Task12Config(orbit_deg_per_sec=30.0)).mode == "orbit"
    assert camera_motion_for("static_control", Task12Config()).mode == "static"
    with pytest.raises(ValueError):
        camera_motion_for("bogus", Task12Config())


def test_compute_relative_camera_transform_matches_the_trajectorys_own_poses(scene):
    """The derived T must satisfy T @ pose(anchor1) == pose(anchor2)
    EXACTLY (up to floating point), independently of how T was computed
    -- a direct correctness check on "physical-transition ground truth
    correctly derived from the trajectory"."""
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="orbit", orbit_deg_per_sec=30.0), num_frames=8, fps=4.0)
    window_frames = 4
    T = compute_relative_camera_transform(traj, window_frames)

    anchor1, anchor2 = traj.frames[0].camera, traj.frames[window_frames].camera
    pose1 = pose_matrix(anchor1.position, anchor1.rotation_euler)
    pose2 = pose_matrix(anchor2.position, anchor2.rotation_euler)
    assert np.allclose(T @ pose1, pose2, atol=1e-8)


def test_compute_relative_camera_transform_is_identity_for_a_static_clip(scene):
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="static"), num_frames=8, fps=4.0)
    T = compute_relative_camera_transform(traj, window_frames=4)
    assert np.allclose(T, np.eye(4), atol=1e-10)


def test_compute_relative_camera_transform_uses_frame_0_and_frame_window_frames_not_other_indices(scene):
    """Adjacent/chosen-window pairing correctness: the anchors must be
    exactly (frame 0, frame `window_frames`), not e.g. the clip's last
    two frames or window midpoints."""
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="orbit", orbit_deg_per_sec=30.0), num_frames=8, fps=4.0)
    T = compute_relative_camera_transform(traj, window_frames=4)

    expected = compose(
        pose_matrix(traj.frames[4].camera.position, traj.frames[4].camera.rotation_euler),
        inverse_rigid(pose_matrix(traj.frames[0].camera.position, traj.frames[0].camera.rotation_euler)),
    )
    assert np.allclose(T, expected)
    # Sanity: using the wrong anchor pair gives a different matrix (guards
    # against a future refactor silently hard-coding the wrong index).
    wrong = compose(
        pose_matrix(traj.frames[7].camera.position, traj.frames[7].camera.rotation_euler),
        inverse_rigid(pose_matrix(traj.frames[0].camera.position, traj.frames[0].camera.rotation_euler)),
    )
    assert not np.allclose(T, wrong)


def test_verify_temporal_ground_truth_accepts_a_genuine_orbit_as_real_motion(scene):
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="orbit", orbit_deg_per_sec=30.0), num_frames=8, fps=4.0)
    verify_temporal_ground_truth(traj, window_frames=4, condition="real_motion")  # must not raise


def test_verify_temporal_ground_truth_rejects_orbit_labeled_as_static_control(scene):
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="orbit", orbit_deg_per_sec=30.0), num_frames=8, fps=4.0)
    with pytest.raises(ValueError):
        verify_temporal_ground_truth(traj, window_frames=4, condition="static_control")


def test_verify_temporal_ground_truth_accepts_a_genuine_static_clip(scene):
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="static"), num_frames=8, fps=4.0)
    verify_temporal_ground_truth(traj, window_frames=4, condition="static_control")  # must not raise


def test_verify_temporal_ground_truth_rejects_no_motion_labeled_real_motion(scene):
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="static"), num_frames=8, fps=4.0)
    with pytest.raises(ValueError):
        verify_temporal_ground_truth(traj, window_frames=4, condition="real_motion")


def test_verify_temporal_ground_truth_rejects_object_motion_contamination(scene):
    """This task tests camera motion only; a trajectory with a moving
    object must be rejected under either condition label."""
    motions = {scene.objects[0].instance_id: ObjectMotion(linear_velocity=(0.1, 0.0, 0.0))}
    traj = generate_trajectory(scene, object_motions=motions, camera_motion=CameraMotion(mode="static"), num_frames=8, fps=4.0)
    with pytest.raises(ValueError):
        verify_temporal_ground_truth(traj, window_frames=4, condition="static_control")
    traj2 = generate_trajectory(
        scene, object_motions=motions, camera_motion=CameraMotion(mode="orbit", orbit_deg_per_sec=30.0), num_frames=8, fps=4.0
    )
    with pytest.raises(ValueError):
        verify_temporal_ground_truth(traj2, window_frames=4, condition="real_motion")


def test_verify_temporal_ground_truth_rejects_wrong_frame_count(scene):
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="orbit", orbit_deg_per_sec=30.0), num_frames=6, fps=4.0)
    with pytest.raises(ValueError):
        verify_temporal_ground_truth(traj, window_frames=4, condition="real_motion")


def test_verify_temporal_ground_truth_unknown_condition_raises(scene):
    traj = generate_trajectory(scene, camera_motion=CameraMotion(mode="static"), num_frames=8, fps=4.0)
    with pytest.raises(ValueError):
        verify_temporal_ground_truth(traj, window_frames=4, condition="bogus")


def test_conditions_tuple_matches_the_two_documented_conditions():
    assert set(CONDITIONS) == {"real_motion", "static_control"}


# --- saved-record re-verification (pure json/numpy, no bpy) ---------------


def test_verify_saved_temporal_transform_accepts_real_motion_record(tmp_path):
    scene_dir = tmp_path / "scene_0000"
    scene_dir.mkdir()
    R = np.eye(4)
    R[0, 3] = 1.0  # non-identity
    (scene_dir / "temporal_transform.json").write_text(json.dumps({"condition": "real_motion", "transform_matrix": R.tolist()}))
    verify_saved_temporal_transform(scene_dir, "real_motion")  # must not raise


def test_verify_saved_temporal_transform_rejects_identity_labeled_real_motion(tmp_path):
    scene_dir = tmp_path / "scene_0000"
    scene_dir.mkdir()
    (scene_dir / "temporal_transform.json").write_text(json.dumps({"condition": "real_motion", "transform_matrix": np.eye(4).tolist()}))
    with pytest.raises(ValueError):
        verify_saved_temporal_transform(scene_dir, "real_motion")


def test_verify_saved_temporal_transform_rejects_non_identity_labeled_static_control(tmp_path):
    scene_dir = tmp_path / "scene_0000"
    scene_dir.mkdir()
    R = np.eye(4)
    R[0, 3] = 1.0
    (scene_dir / "temporal_transform.json").write_text(json.dumps({"condition": "static_control", "transform_matrix": R.tolist()}))
    with pytest.raises(ValueError):
        verify_saved_temporal_transform(scene_dir, "static_control")


def test_verify_saved_temporal_transform_rejects_condition_mismatch(tmp_path):
    scene_dir = tmp_path / "scene_0000"
    scene_dir.mkdir()
    (scene_dir / "temporal_transform.json").write_text(json.dumps({"condition": "real_motion", "transform_matrix": np.eye(4).tolist()}))
    with pytest.raises(ValueError):
        verify_saved_temporal_transform(scene_dir, "static_control")


# --- result assembly + scientific-result text (pure dict math) ------------


def _fake_metrics(r2, cos=0.9, l2=0.3):
    return {"r2": r2, "mean_cosine_similarity": cos, "mean_relative_l2_error": l2}


def _fake_condition(condition, r2, seeds, persistence_r2=0.0, mean_r2=0.0, random_pair_r2=-0.5, static_style=False):
    return {
        "condition": condition,
        "base_seed": 0 if condition == "real_motion" else 1,
        "num_scenes": len(seeds),
        "scene_ids": [f"scene_{i:04d}" for i in range(len(seeds))],
        "scene_seeds": seeds,
        "train_scene_ids": [f"scene_{i:04d}" for i in range(len(seeds) - 1)],
        "test_scene_ids": [f"scene_{len(seeds) - 1:04d}"],
        "train_fraction": 0.8,
        "encoder": {"name": "VJEPAEncoder", "checkpoint": "x", "pretrained": False, "frozen": True, "pooling": "mean_pool"},
        "metrics": {
            "learned_W_T": _fake_metrics(r2),
            "persistence_baseline": _fake_metrics(persistence_r2),
            "mean_baseline": _fake_metrics(mean_r2),
            "random_pair_control": _fake_metrics(random_pair_r2),
        },
        "pixel_diff_stats": {"mean_abs_pixel_diff": 0.0, "std_abs_pixel_diff": 0.0, "per_scene_mean_abs_pixel_diff": [0.0] * len(seeds)},
        "out_dir": "unused",
        "manifest_path": "unused/manifest.json",
    }


def test_assemble_result_schema_shape(tmp_path):
    cfg = Task12Config(output_dir=str(tmp_path / "temporal_consistency"))
    real = _fake_condition("real_motion", r2=0.6, seeds=[0, 1, 2, 3])
    static = _fake_condition("static_control", r2=0.1, seeds=[1000, 1001, 1002, 1003])

    result = assemble_result(cfg, real, static)

    assert result["task"] == 12
    assert result["implementation_status"] == "COMPLETE"
    assert set(result["motion_config"].keys()) == {"real_motion", "static_control", "object_motions"}
    assert "static_clip_control" in result
    assert result["static_clip_control"] == static["metrics"]
    assert set(result["metrics"].keys()) == {"real_motion", "static_clip_control"}
    assert result["margins"]["real_motion_learned_W_T_minus_own_best_baseline"] == pytest.approx(0.6 - 0.0)
    assert result["margins"]["real_motion_learned_W_T_minus_static_control_learned_W_T"] == pytest.approx(0.6 - 0.1)
    for value in [
        result["margins"]["real_motion_learned_W_T_minus_own_best_baseline"],
        result["margins"]["real_motion_learned_W_T_minus_static_control_learned_W_T"],
    ]:
        assert np.isfinite(value)


def test_scientific_result_text_positive_case_mentions_both_margins():
    real = _fake_condition("real_motion", r2=0.6, seeds=[0, 1])
    static = _fake_condition("static_control", r2=0.1, seeds=[1000, 1001])
    text = _scientific_result_text(real, static)
    assert "0.600" in text
    assert "static-clip control" in text
    assert "understands" not in text.lower()


def test_scientific_result_text_beats_baselines_but_not_static_control_is_flagged_distinctly():
    real = _fake_condition("real_motion", r2=0.2, seeds=[0, 1], persistence_r2=0.0, mean_r2=0.0, random_pair_r2=-0.2)
    static = _fake_condition("static_control", r2=0.5, seeds=[1000, 1001])
    text = _scientific_result_text(real, static)
    assert "not sufficient" in text
    assert "valid negative result" in text.lower()
    assert "understands" not in text.lower()


def test_scientific_result_text_fully_negative_case():
    real = _fake_condition("real_motion", r2=-0.1, seeds=[0, 1], persistence_r2=0.2, mean_r2=0.1, random_pair_r2=-0.5)
    static = _fake_condition("static_control", r2=0.3, seeds=[1000, 1001])
    text = _scientific_result_text(real, static)
    assert "valid negative result" in text.lower()
    assert "not a task failure" in text


# --- pytest summary parsing (delegates to gclib, sanity-checked here) -----


def test_run_existing_test_suite_delegates_to_gclib(monkeypatch):
    import experiments.geometric_consistency_lib as gclib
    import experiments.task12_temporal_consistency as mod

    monkeypatch.setattr(gclib, "run_existing_test_suite", lambda root: {"passed": 3, "failed": 0})
    assert mod.run_existing_test_suite("unused") == {"passed": 3, "failed": 0}


# --- slow, real end-to-end pipeline tests (bpy + torch/transformers required)


@pytest.mark.slow
def test_full_condition_runs_end_to_end_on_a_small_scene_set(tmp_path):
    """Exercises render -> encode -> fit -> evaluate for one real
    condition, on a small scene count and an untrained (pretrained=False)
    encoder for speed -- NOT a claim about the real Task 12 result (which
    needs >=40 scenes per condition and real pretrained weights per
    configs/experiments/task12_temporal_consistency.yaml), only a check
    that the code path works end to end without error and produces
    well-shaped, finite metrics.
    """
    pytest.importorskip("bpy", reason="needs bpy")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import experiments.task12_temporal_consistency as mod

    monkeypatch_min = mod.MIN_SCENES
    mod.MIN_SCENES = 6
    try:
        cfg = Task12Config(
            num_scenes=6,
            train_fraction=0.6667,
            output_dir=str(tmp_path / "temporal_consistency"),
            resolution=32,
            window_frames=2,
            fps=4.0,
            pretrained=False,
        )
        real = mod.run_condition("real_motion", cfg)
        static = mod.run_condition("static_control", cfg)
    finally:
        mod.MIN_SCENES = monkeypatch_min

    assert real["encoder"]["pretrained"] is False
    assert set(real["train_scene_ids"]).isdisjoint(real["test_scene_ids"])
    assert set(real["scene_seeds"]).isdisjoint(static["scene_seeds"])

    result = mod.assemble_result(cfg, real, static)
    assert result["task"] == 12
    for cond_metrics in result["metrics"].values():
        for block in cond_metrics.values():
            for value in block.values():
                if isinstance(value, float):
                    assert np.isfinite(value)
    # Static control's ground truth camera pose must be exactly unchanged.
    assert static["pixel_diff_stats"]["mean_abs_pixel_diff"] == pytest.approx(0.0, abs=1e-6)
    # Real motion must show a genuine, nonzero pixel difference between windows.
    assert real["pixel_diff_stats"]["mean_abs_pixel_diff"] > 0.0
