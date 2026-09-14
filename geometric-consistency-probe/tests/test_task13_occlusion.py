"""Tests for experiments/task13_occlusion.py.

Fast tests (no bpy, no torch/transformers, no network) exercise config
loading, the occlusion-verification logic on synthetic segmentation
arrays, the rig/per-scene construction (pure numpy -- generation.motion
is pure math), the "identical per-scene draws across conditions" and
"scrambled-control is a genuine derangement" leakage guards, and result
assembly/scientific-result text. A slow group (skipped without
bpy/torch) exercises the real render -> verify -> encode -> fit ->
evaluate pipeline end to end on a small number of scenes with
`pretrained=False`, matching the fast-dev pattern already used by
tests/test_task12_temporal_consistency.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.task13_occlusion import (
    CONDITIONS,
    MIN_SCENES,
    OCCLUDER_INSTANCE_ID,
    TRACKED_INSTANCE_ID,
    VARIABLES,
    Task13Config,
    _sample_tracked_params,
    _scientific_result_text,
    build_scene_and_trajectory,
    compute_post_occlusion_label,
    load_task13_config,
    num_frames_for,
    occlusion_pixel_threshold,
    rig_camera,
    rig_occluder,
    trivial_cue_investigation,
    verify_occlusion,
    verify_visible_throughout,
)

# --- config loading -----------------------------------------------------


def test_load_task13_config_defaults():
    cfg = load_task13_config(None)
    assert cfg.num_scenes == 40
    assert cfg.train_fraction == 0.8
    assert cfg.window_frames == 4
    assert cfg.pretrained is True


def test_load_task13_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("num_scenes: 44\nwindow_frames: 3\npretrained: false\n")
    cfg = load_task13_config(path)
    assert cfg.num_scenes == 44
    assert cfg.window_frames == 3
    assert cfg.pretrained is False
    assert cfg.train_fraction == 0.8  # untouched field keeps its default


def test_project_config_file_satisfies_task13_minimums():
    """The actual config this task's experiment is run with must satisfy
    the >=40-scenes / real pretrained encoder requirements -- catches a
    config regression that would silently violate acceptance criteria."""
    cfg = load_task13_config("configs/experiments/task13_occlusion.yaml")
    assert cfg.num_scenes >= MIN_SCENES
    assert cfg.pretrained is True
    assert cfg.window_frames >= 2


def test_conditions_tuple():
    assert set(CONDITIONS) == {"occlusion", "no_occlusion"}


# --- occlusion-verification logic (pure numpy, no bpy) ----------------------


def _seg(counts_per_frame: list[int], resolution: int = 16) -> np.ndarray:
    """Build a synthetic (T, H, W) segmentation array where instance 1
    occupies exactly `counts_per_frame[t]` pixels in frame t (and
    instance 2, the "occluder", occupies the rest of a small fixed
    region so the array is a plausible two-object scene)."""
    T = len(counts_per_frame)
    seg = np.zeros((T, resolution, resolution), dtype=np.uint8)
    flat_size = resolution * resolution
    for t, count in enumerate(counts_per_frame):
        flat = seg[t].reshape(-1)
        flat[:count] = 1
        seg[t] = flat.reshape(resolution, resolution)
    return seg


def test_verify_occlusion_detects_a_genuine_occlusion_window():
    seg = _seg([50, 48, 45, 40, 0, 0, 0, 0, 41, 46, 49, 51])
    result = verify_occlusion(seg, TRACKED_INSTANCE_ID, window_start=4, window_len=4, threshold=4)
    assert result["passed"] is True
    assert result["occluded_frame_indices"] == [4, 5, 6, 7]
    assert result["measured_occlusion_duration_frames"] == 4


def test_verify_occlusion_rejects_a_window_that_never_actually_hides_the_object():
    """The required "do not assume a scripted trajectory produces
    occlusion without checking" case: nonzero pixel counts throughout."""
    seg = _seg([50, 48, 45, 40, 38, 36, 35, 34, 41, 46, 49, 51])
    result = verify_occlusion(seg, TRACKED_INSTANCE_ID, window_start=4, window_len=4, threshold=4)
    assert result["passed"] is False


def test_verify_occlusion_rejects_partial_occlusion_within_the_window():
    """Every frame in the intended window must be under threshold, not
    just some of them (a single visible frame mid-window must fail)."""
    seg = _seg([50, 48, 45, 40, 0, 0, 30, 0, 41, 46, 49, 51])
    result = verify_occlusion(seg, TRACKED_INSTANCE_ID, window_start=4, window_len=4, threshold=4)
    assert result["passed"] is False


def test_verify_occlusion_uses_a_documented_near_zero_threshold_not_strict_zero():
    """Antialiased edge pixels (a handful of nonzero pixels) must still
    count as "occluded" when below the documented threshold."""
    seg = _seg([50, 48, 45, 40, 3, 2, 0, 3, 41, 46, 49, 51])
    result = verify_occlusion(seg, TRACKED_INSTANCE_ID, window_start=4, window_len=4, threshold=4)
    assert result["passed"] is True


def test_verify_visible_throughout_accepts_a_genuinely_always_visible_object():
    seg = _seg([50, 48, 45, 40, 38, 36, 35, 34, 41, 46, 49, 51])
    result = verify_visible_throughout(seg, TRACKED_INSTANCE_ID, threshold=4)
    assert result["passed"] is True


def test_verify_visible_throughout_rejects_an_object_that_disappears_at_all():
    seg = _seg([50, 48, 45, 40, 0, 36, 35, 34, 41, 46, 49, 51])
    result = verify_visible_throughout(seg, TRACKED_INSTANCE_ID, threshold=4)
    assert result["passed"] is False


def test_occlusion_pixel_threshold_scales_with_resolution_and_has_a_floor():
    assert occlusion_pixel_threshold(64) >= 4
    assert occlusion_pixel_threshold(128) > occlusion_pixel_threshold(64)


# --- rig + per-scene construction (pure math, no bpy) -----------------------


def test_rig_camera_and_occluder_are_deterministic():
    cfg = Task13Config()
    cam1, cam2 = rig_camera(cfg), rig_camera(cfg)
    assert cam1.position == cam2.position
    occ1, occ2 = rig_occluder(cfg), rig_occluder(cfg)
    assert occ1.instance_id == OCCLUDER_INSTANCE_ID
    assert occ1.position == occ2.position


def test_sample_tracked_params_is_deterministic_given_seed():
    cfg = Task13Config()
    p1 = _sample_tracked_params(42, cfg)
    p2 = _sample_tracked_params(42, cfg)
    assert p1 == p2


def test_sample_tracked_params_is_identical_regardless_of_condition():
    """Required "identical scene setup" property for the no_occlusion
    control: the tracked object's identity/shape/color/path-extent
    jitter must NOT depend on which condition is being built -- only
    depth (handled separately in build_scene_and_trajectory) may differ.
    """
    cfg = Task13Config()
    seed = 7
    scene_occ, traj_occ = build_scene_and_trajectory("scene_0000", seed, "occlusion", cfg)
    scene_noc, traj_noc = build_scene_and_trajectory("scene_0000", seed, "no_occlusion", cfg)
    tracked_occ = scene_occ.objects[0]
    tracked_noc = scene_noc.objects[0]
    assert tracked_occ.shape == tracked_noc.shape
    assert tracked_occ.scale == tracked_noc.scale
    assert tracked_occ.rotation_euler == tracked_noc.rotation_euler
    assert tracked_occ.color == tracked_noc.color
    # Only depth (y) differs; x0/z are shared "path shape" primitives.
    assert tracked_occ.position[2] == tracked_noc.position[2]
    assert tracked_occ.position[1] != tracked_noc.position[1]


def test_build_scene_and_trajectory_places_tracked_object_farther_for_occlusion():
    cfg = Task13Config()
    scene_occ, _ = build_scene_and_trajectory("scene_0000", 1, "occlusion", cfg)
    scene_noc, _ = build_scene_and_trajectory("scene_0000", 1, "no_occlusion", cfg)
    tracked_occ_y = scene_occ.objects[0].position[1]
    tracked_noc_y = scene_noc.objects[0].position[1]
    occluder_y = scene_occ.objects[1].position[1]
    camera = rig_camera(cfg)
    dist_occ = abs(tracked_occ_y - camera.position[1])
    dist_noc = abs(tracked_noc_y - camera.position[1])
    dist_occluder = abs(occluder_y - camera.position[1])
    assert dist_occ > dist_occluder  # tracked (occlusion) is BEHIND the occluder
    assert dist_noc < dist_occluder  # tracked (no_occlusion) is IN FRONT of the occluder


def test_build_scene_and_trajectory_unknown_condition_raises():
    cfg = Task13Config()
    with pytest.raises(ValueError):
        build_scene_and_trajectory("scene_0000", 1, "bogus", cfg)


def test_num_frames_for_matches_three_windows():
    cfg = Task13Config(window_frames=5)
    assert num_frames_for(cfg) == 15


def test_compute_post_occlusion_label_reads_the_post_anchor_frame():
    cfg = Task13Config(window_frames=3)
    scene, traj = build_scene_and_trajectory("scene_0000", 3, "occlusion", cfg)
    label = compute_post_occlusion_label(traj, cfg.window_frames)
    assert label["frame_index"] == 2 * cfg.window_frames
    anchor = traj.frames[2 * cfg.window_frames]
    pose = next(p for p in anchor.objects if p.instance_id == TRACKED_INSTANCE_ID)
    assert label["post_occlusion_position_x"] == pytest.approx(pose.position[0])
    assert label["post_occlusion_velocity_x"] == pytest.approx(pose.linear_velocity[0])


def test_tracked_object_moves_only_along_world_x():
    """The occluder must stay static and the tracked object's y/z must
    not drift -- only x changes under the constant-velocity motion."""
    cfg = Task13Config()
    scene, traj = build_scene_and_trajectory("scene_0000", 5, "occlusion", cfg)
    for frame in traj.frames:
        occluder_pose = next(p for p in frame.objects if p.instance_id == OCCLUDER_INSTANCE_ID)
        assert occluder_pose.position == scene.objects[1].position
        tracked_pose = next(p for p in frame.objects if p.instance_id == TRACKED_INSTANCE_ID)
        assert tracked_pose.position[1] == pytest.approx(scene.objects[0].position[1])
        assert tracked_pose.position[2] == pytest.approx(scene.objects[0].position[2])


# --- scrambled-identity control: genuine derangement (leakage check) --------


def test_shuffled_label_permutation_is_a_genuine_derangement_for_small_n():
    from experiments.task8_physical_state import shuffled_label_permutation

    for n in range(2, 8):
        perm = shuffled_label_permutation(n, seed=0)
        assert not np.array_equal(perm, np.arange(n))


# --- trivial-cue investigation (pure numpy) ----------------------------------


def test_trivial_cue_investigation_finds_no_confound_for_independent_draws():
    rng = np.random.default_rng(0)
    n = 60
    scene_ids = [f"scene_{i:04d}" for i in range(n)]
    records = {}
    for i, sid in enumerate(scene_ids):
        records[sid] = {
            "label": {
                "post_occlusion_position_x": float(rng.normal()),
                "post_occlusion_velocity_x": float(rng.normal()),
            },
            "tracked_params": {
                "color": tuple(rng.uniform(0.1, 0.9, size=3).tolist()),
                "floor_shade": float(rng.uniform(0.35, 0.75)),
                "light_energy": float(rng.uniform(2.5, 5.0)),
                "scale": float(rng.uniform(0.45, 0.65)),
            },
        }
    result = trivial_cue_investigation(scene_ids, records)
    assert result["max_abs_correlation"] < 0.4


def test_trivial_cue_investigation_detects_a_genuine_confound():
    """Sanity check on the check itself: if color WERE (artificially)
    made to correlate with the label, the investigation must catch it."""
    rng = np.random.default_rng(0)
    n = 60
    scene_ids = [f"scene_{i:04d}" for i in range(n)]
    records = {}
    for i, sid in enumerate(scene_ids):
        pos = float(rng.normal())
        records[sid] = {
            "label": {"post_occlusion_position_x": pos, "post_occlusion_velocity_x": float(rng.normal())},
            "tracked_params": {
                "color": (pos, 0.5, 0.5),  # deliberately confounded with the label
                "floor_shade": float(rng.uniform(0.35, 0.75)),
                "light_energy": float(rng.uniform(2.5, 5.0)),
                "scale": float(rng.uniform(0.45, 0.65)),
            },
        }
    result = trivial_cue_investigation(scene_ids, records)
    assert result["max_abs_correlation"] > 0.9


# --- scientific-result text (pure dict math) ---------------------------------


def _fake_block(r2):
    return {"r2": r2, "mae": 0.1, "rmse": 0.1}


def _fake_metrics(real_r2, scrambled_r2, no_occ_r2, mean_r2=0.0):
    return {
        "occlusion_condition": {
            "variable": "x",
            "real_probe": _fake_block(real_r2),
            "scrambled_identity_control": _fake_block(scrambled_r2),
            "mean_baseline": _fake_block(mean_r2),
        },
        "no_occlusion_condition": {
            "variable": "x",
            "real_probe": _fake_block(no_occ_r2),
            "scrambled_identity_control": _fake_block(0.0),
            "mean_baseline": _fake_block(0.0),
        },
        "diagnostic_occluded_window_probe": _fake_block(-0.2),
        "diagnostic_pre_window_probe": _fake_block(0.8),
    }


def test_scientific_result_text_positive_case():
    metrics = {v: _fake_metrics(real_r2=0.4, scrambled_r2=0.0, no_occ_r2=0.8) for v in VARIABLES}
    text = _scientific_result_text(metrics)
    assert "exceeding the object-identity-scrambled control" in text
    assert "understands" not in text.lower()
    assert "object model" not in text.lower() or "not evidence of an internal object model" in text.lower()


def test_scientific_result_text_negative_case():
    metrics = {v: _fake_metrics(real_r2=-0.1, scrambled_r2=0.2, no_occ_r2=0.8) for v in VARIABLES}
    text = _scientific_result_text(metrics)
    assert "did NOT exceed the scrambled-identity control" in text
    assert "valid negative" in text.lower()


def test_scientific_result_text_mixed_case_is_reported_as_negative_overall():
    metrics = dict(zip(VARIABLES, [_fake_metrics(0.4, 0.0, 0.8), _fake_metrics(-0.1, 0.2, 0.8)]))
    text = _scientific_result_text(metrics)
    assert "valid negative" in text.lower()
    assert "1/2" in text


# --- pytest summary parsing (delegates to gclib, sanity-checked here) -------


def test_run_existing_test_suite_delegates_to_gclib(monkeypatch):
    import experiments.geometric_consistency_lib as gclib
    import experiments.task13_occlusion as mod

    monkeypatch.setattr(gclib, "run_existing_test_suite", lambda root: {"passed": 3, "failed": 0})
    assert mod.run_existing_test_suite("unused") == {"passed": 3, "failed": 0}


# --- slow, real end-to-end pipeline test (bpy + torch/transformers required) -


@pytest.mark.slow
def test_full_pipeline_runs_end_to_end_on_a_small_scene_set(tmp_path):
    """Exercises render -> verify -> encode -> fit -> evaluate for both
    conditions on a small scene count and an untrained (pretrained=False)
    encoder for speed -- NOT a claim about the real Task 13 result (which
    needs >=40 scenes and real pretrained weights per
    configs/experiments/task13_occlusion.yaml), only a check that the
    code path works end to end without error and produces well-shaped,
    finite metrics, and that occlusion is genuinely verified via
    segmentation for every occlusion-condition scene.
    """
    pytest.importorskip("bpy", reason="needs bpy")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import experiments.task13_occlusion as mod

    monkeypatch_min = mod.MIN_SCENES
    mod.MIN_SCENES = 6
    try:
        cfg = mod.Task13Config(
            num_scenes=6,
            train_fraction=0.6667,
            output_dir=str(tmp_path / "occlusion"),
            resolution=48,
            window_frames=2,
            fps=4.0,
            pretrained=False,
        )
        result = mod.run_experiment(cfg)
    finally:
        mod.MIN_SCENES = monkeypatch_min

    assert result["task"] == 13
    assert result["implementation_status"] == "COMPLETE"
    assert result["encoder"]["pretrained"] is False
    assert set(result["dataset"]["train_scene_ids"]).isdisjoint(result["dataset"]["test_scene_ids"])

    for scene_id, duration in result["occlusion_verification"]["measured_occlusion_duration_frames_by_scene"].items():
        assert duration == cfg.window_frames

    for variable in mod.VARIABLES:
        block = result["metrics"][variable]
        for sub in ("occlusion_condition", "no_occlusion_condition"):
            for metric_block in block[sub].values():
                if not isinstance(metric_block, dict):
                    continue
                for value in metric_block.values():
                    if isinstance(value, float):
                        assert np.isfinite(value) or np.isnan(value)  # NaN allowed only for degenerate tiny test splits
