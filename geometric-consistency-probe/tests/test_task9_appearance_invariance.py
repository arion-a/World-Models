"""Tests for experiments/task9_appearance_invariance.py.

Fast tests (no bpy, no torch/transformers, no network) exercise config
loading, the reused-scene-set check, the appearance physical-equality
re-verification (using transforms.scene_transform.apply_transform's real
output -- pure SceneState math, no rendering), the null-transform ground-
truth verifier, the confound investigation, and text/table generation --
everything that doesn't require rendering or a model forward pass. A slow
group (skipped without bpy/torch) exercises the real render -> encode ->
invariance pipeline end to end, including the null-transform sanity
check, on a small number of scenes with `pretrained=False`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import experiments.geometric_consistency_lib as gclib
from experiments.task9_appearance_invariance import (
    NULL_TRANSFORM_COSINE_THRESHOLD,
    NULL_TRANSFORM_RELATIVE_L2_THRESHOLD,
    Task9Config,
    _check_reused_scene_set_matches,
    _comparison_table_markdown,
    _scientific_result_text,
    confound_investigation,
    load_task9_config,
    sample_scenes,
)
from generation.scene_sampler import SceneSamplerConfig, sample_scene
from transforms.scene_transform import CONTROL_TRANSFORMS, GEOMETRIC_TRANSFORMS, TransformConfig, apply_transform

# --- config loading ----------------------------------------------------------


def test_load_task9_config_defaults():
    cfg = load_task9_config(None)
    assert cfg.num_scenes == 40
    assert cfg.train_fraction == 0.8
    assert cfg.base_seed == 0
    assert cfg.pretrained is True


def test_load_task9_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("num_scenes: 44\npretrained: false\n")
    cfg = load_task9_config(path)
    assert cfg.num_scenes == 44
    assert cfg.pretrained is False
    assert cfg.train_fraction == 0.8


def test_project_config_file_satisfies_task9_minimums():
    cfg = load_task9_config("configs/experiments/task9_appearance_invariance.yaml")
    assert cfg.num_scenes >= 40
    assert cfg.pretrained is True


def test_task9_default_config_reuses_task7s_exact_scene_sampling_parameters():
    """tasks/09_appearance_invariance.md's Dataset requirements: "Same
    scene set as Task 7 where reused." Task 7's real config uses
    num_scenes=40, base_seed=0, num_objects_min=1, num_objects_max=3 --
    Task 9's default config must match exactly so the sampled scenes are
    bit-identical and Task 7's rendered pairs are actually reusable."""
    from experiments.task7_geometric_consistency import load_task7_config

    task7_cfg = load_task7_config("configs/experiments/task7_geometric_consistency.yaml")
    task9_cfg = load_task9_config("configs/experiments/task9_appearance_invariance.yaml")
    assert task9_cfg.num_scenes == task7_cfg.num_scenes
    assert task9_cfg.base_seed == task7_cfg.base_seed
    assert task9_cfg.num_objects_min == task7_cfg.num_objects_min
    assert task9_cfg.num_objects_max == task7_cfg.num_objects_max
    assert task9_cfg.reused_output_dir == task7_cfg.output_dir


def test_sample_scenes_reproduces_task7s_scene_set():
    from experiments.task7_geometric_consistency import Task7Config
    from experiments.task7_geometric_consistency import sample_scenes as task7_sample_scenes

    task9_cfg = Task9Config(num_scenes=10, base_seed=0)
    task7_cfg = Task7Config(num_scenes=10, base_seed=0)
    task9_scenes = sample_scenes(task9_cfg)
    task7_scenes = task7_sample_scenes(task7_cfg)
    assert [s.to_dict() for s in task9_scenes] == [s.to_dict() for s in task7_scenes]


# --- _check_reused_scene_set_matches ----------------------------------------


def test_check_reused_scene_set_matches_raises_when_manifest_missing(tmp_path):
    scenes = [sample_scene("s0", seed=0, cfg=SceneSamplerConfig())]
    with pytest.raises(FileNotFoundError):
        _check_reused_scene_set_matches(scenes, "lighting_change", tmp_path / "does_not_exist")


def test_check_reused_scene_set_matches_raises_on_mismatched_scenes(tmp_path):
    scenes = [sample_scene("s0", seed=0, cfg=SceneSamplerConfig())]
    out_dir = tmp_path / "lighting_change"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({"scenes": [{"scene_id": "some_other_scene"}]}))
    with pytest.raises(ValueError):
        _check_reused_scene_set_matches(scenes, "lighting_change", out_dir)


def test_check_reused_scene_set_matches_passes_when_identical(tmp_path):
    scenes = [sample_scene("s0", seed=0, cfg=SceneSamplerConfig())]
    out_dir = tmp_path / "lighting_change"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({"scenes": [{"scene_id": "s0"}]}))
    _check_reused_scene_set_matches(scenes, "lighting_change", out_dir)  # must not raise


# --- verify_appearance_physical_equality (pure SceneState math, no bpy) -----


def _scene(seed=1, num_objects_min=2, num_objects_max=2):
    return sample_scene(f"t9_scene_{seed}", seed=seed, cfg=SceneSamplerConfig(num_objects_min=num_objects_min, num_objects_max=num_objects_max))


@pytest.mark.parametrize("transform_name", CONTROL_TRANSFORMS)
def test_verify_appearance_physical_equality_accepts_real_control_transforms(tmp_path, transform_name):
    scene = _scene(seed=CONTROL_TRANSFORMS.index(transform_name))
    _, transformation = apply_transform(scene, transform_name, TransformConfig())
    pair_dir = tmp_path / transform_name / scene.scene_id
    pair_dir.mkdir(parents=True)
    (pair_dir / "transformation.json").write_text(json.dumps(transformation))

    gclib.verify_appearance_physical_equality([scene], tmp_path / transform_name, transform_name)  # must not raise


def test_verify_appearance_physical_equality_rejects_geometric_transform_name():
    with pytest.raises(ValueError):
        gclib.verify_appearance_physical_equality([], "somedir", "camera_rotation")


def test_verify_appearance_physical_equality_rejects_pose_change(tmp_path):
    scene = _scene(seed=42)
    pair_dir = tmp_path / "lighting_change" / scene.scene_id
    pair_dir.mkdir(parents=True)
    fake = {
        "type": "lighting_change",
        "transform_name": "lighting_change",
        "transform_matrix": None,
        # Tampered: includes a camera pose change a valid appearance
        # control must never carry.
        "changed_variables": ["light.position", "light.energy", "camera.position"],
    }
    (pair_dir / "transformation.json").write_text(json.dumps(fake))
    with pytest.raises(ValueError):
        gclib.verify_appearance_physical_equality([scene], tmp_path / "lighting_change", "lighting_change")


def test_verify_appearance_physical_equality_rejects_rigid_matrix_present(tmp_path):
    scene = _scene(seed=43)
    pair_dir = tmp_path / "texture_change" / scene.scene_id
    pair_dir.mkdir(parents=True)
    fake = {
        "type": "texture_change",
        "transform_name": "texture_change",
        "transform_matrix": np.eye(4).tolist(),
        "changed_variables": [f"objects[{i}].color" for i in range(len(scene.objects))],
    }
    (pair_dir / "transformation.json").write_text(json.dumps(fake))
    with pytest.raises(ValueError):
        gclib.verify_appearance_physical_equality([scene], tmp_path / "texture_change", "texture_change")


# --- verify_null_transform_ground_truth (pure JSON, no bpy) -----------------


def test_verify_null_transform_ground_truth_accepts_identity_record(tmp_path):
    pair_dir = tmp_path / "scene_0000"
    pair_dir.mkdir()
    record = {
        "type": "null_transform",
        "transform_name": "null_transform",
        "transform_matrix": np.eye(4).tolist(),
        "changed_variables": [],
        "object_index": None,
    }
    (pair_dir / "transformation.json").write_text(json.dumps(record))
    gclib.verify_null_transform_ground_truth(pair_dir)  # must not raise


def test_verify_null_transform_ground_truth_rejects_nonempty_changed_variables(tmp_path):
    pair_dir = tmp_path / "scene_0000"
    pair_dir.mkdir()
    record = {
        "type": "null_transform",
        "transform_name": "null_transform",
        "transform_matrix": np.eye(4).tolist(),
        "changed_variables": ["camera.position"],
    }
    (pair_dir / "transformation.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        gclib.verify_null_transform_ground_truth(pair_dir)


def test_verify_null_transform_ground_truth_rejects_non_identity_matrix(tmp_path):
    pair_dir = tmp_path / "scene_0000"
    pair_dir.mkdir()
    record = {
        "type": "null_transform",
        "transform_name": "null_transform",
        "transform_matrix": (2 * np.eye(4)).tolist(),
        "changed_variables": [],
    }
    (pair_dir / "transformation.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        gclib.verify_null_transform_ground_truth(pair_dir)


# --- compute_invariance_for_transform (fake encoder + monkeypatched I/O) ---


def test_compute_invariance_for_transform_uses_evaluate_invariance_directly(tmp_path, monkeypatch):
    from experiments.task9_appearance_invariance import compute_invariance_for_transform

    scenes = [_scene(seed=1), _scene(seed=2)]
    rng = np.random.default_rng(0)
    rgb_by_scene = {s.scene_id: (rng.integers(0, 255, size=(2, 4, 4, 3), dtype=np.uint8), rng.integers(0, 255, size=(2, 4, 4, 3), dtype=np.uint8)) for s in scenes}

    def fake_load_pair(pair_dir):
        scene_id = pair_dir.name
        orig_rgb, trans_rgb = rgb_by_scene[scene_id]
        return {"original": (None, orig_rgb, None, None), "transformed": (None, trans_rgb, None, None)}

    def fake_mean_pool(representation):
        return representation

    class FakeEncoder:
        def encode(self, video):
            # Deterministic, scene-independent function of the raw video --
            # exercises real array plumbing without needing torch.
            return np.array([float(video.sum())] * 4)

    monkeypatch.setattr(gclib, "load_pair", fake_load_pair)
    monkeypatch.setattr("encoders.vjepa.mean_pool", fake_mean_pool, raising=False)

    entry = compute_invariance_for_transform(scenes, tmp_path, "lighting_change", FakeEncoder())
    assert entry["transform_name"] == "lighting_change"
    assert entry["is_appearance_control"] is True
    assert entry["is_geometric"] is False
    assert entry["n"] == 2
    assert -1.0 <= entry["mean_cosine_similarity"] <= 1.0
    assert entry["mean_relative_l2_error"] >= 0.0
    assert np.isfinite(entry["mean_cosine_similarity"])
    assert np.isfinite(entry["mean_relative_l2_error"])


# --- confound_investigation --------------------------------------------------


def _fake_invariance_entry(cosine, l2, is_geometric):
    return {
        "mean_cosine_similarity": cosine,
        "mean_relative_l2_error": l2,
        "is_geometric": is_geometric,
        "is_appearance_control": not is_geometric,
    }


def test_confound_investigation_reports_clean_separation():
    results = {
        "lighting_change": _fake_invariance_entry(0.99, 0.05, is_geometric=False),
        "texture_change": _fake_invariance_entry(0.98, 0.06, is_geometric=False),
        "camera_translation": _fake_invariance_entry(0.85, 0.3, is_geometric=True),
        "camera_rotation": _fake_invariance_entry(0.80, 0.35, is_geometric=True),
        "object_translation": _fake_invariance_entry(0.90, 0.25, is_geometric=True),
        "object_rotation": _fake_invariance_entry(0.88, 0.28, is_geometric=True),
    }
    analysis = confound_investigation(results)
    assert analysis["clean_separation"] is True
    assert "AGAINST" in analysis["verdict"]


def test_confound_investigation_reports_no_clean_separation_when_overlapping():
    results = {
        "lighting_change": _fake_invariance_entry(0.90, 0.25, is_geometric=False),
        "texture_change": _fake_invariance_entry(0.80, 0.35, is_geometric=False),
        "camera_translation": _fake_invariance_entry(0.85, 0.3, is_geometric=True),
        "camera_rotation": _fake_invariance_entry(0.95, 0.15, is_geometric=True),
        "object_translation": _fake_invariance_entry(0.90, 0.25, is_geometric=True),
        "object_rotation": _fake_invariance_entry(0.88, 0.28, is_geometric=True),
    }
    analysis = confound_investigation(results)
    assert analysis["clean_separation"] is False
    assert "undercuts" in analysis["verdict"]


# --- text/table generation -------------------------------------------------


def _all_six_results():
    results = {}
    for name in CONTROL_TRANSFORMS:
        results[name] = _fake_invariance_entry(0.97, 0.1, is_geometric=False)
    for name in GEOMETRIC_TRANSFORMS:
        results[name] = _fake_invariance_entry(0.85, 0.3, is_geometric=True)
    return results


def test_scientific_result_text_mentions_every_transform_and_null_check():
    results = _all_six_results()
    null_check = {
        "mean_cosine_similarity": 0.9999,
        "mean_relative_l2_error": 0.001,
        "passed_threshold": True,
        "cosine_threshold": NULL_TRANSFORM_COSINE_THRESHOLD,
        "relative_l2_threshold": NULL_TRANSFORM_RELATIVE_L2_THRESHOLD,
    }
    confound = confound_investigation(results)
    text = _scientific_result_text(results, null_check, confound)
    for name in list(CONTROL_TRANSFORMS) + list(GEOMETRIC_TRANSFORMS):
        assert name in text
    assert "PASSED" in text


def test_comparison_table_markdown_has_a_row_per_transform_plus_null():
    results = _all_six_results()
    null_check = {"mean_cosine_similarity": 0.9999, "mean_relative_l2_error": 0.001}
    table = _comparison_table_markdown(results, null_check)
    for name in list(CONTROL_TRANSFORMS) + list(GEOMETRIC_TRANSFORMS):
        assert name in table
    assert "null_transform" in table


# --- slow, real end-to-end pipeline test (bpy + torch/transformers required)


@pytest.mark.slow
def test_full_pipeline_runs_end_to_end_including_null_transform_check(tmp_path, monkeypatch):
    """Exercises render -> encode -> invariance for real, for all six
    reused transforms plus the null-transform sanity check, on a small
    scene count and an untrained (pretrained=False) encoder for speed --
    not a claim about the real Task 9 result (which needs >=40 scenes,
    real pretrained weights, and Task 7's own real rendered pairs), only
    a check that the full code path works end to end without error and
    produces well-shaped, finite, non-leaking output, with the
    null-transform check actually passing its threshold.
    """
    pytest.importorskip("bpy", reason="needs bpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import experiments.task9_appearance_invariance as mod

    reused_dir = tmp_path / "geometric_consistency"
    scenes = gclib.sample_scenes(6, base_seed=0, num_objects_min=1, num_objects_max=3)

    # Fabricate a Task-7-shaped set of rendered pairs for all six transforms
    # (small/fast: resolution 32, 2 frames) so this test doesn't depend on
    # a real Task 7 run having happened first.
    for transform_name in gclib.ALL_TRANSFORMS:
        out_dir = reused_dir / transform_name
        gclib.render_transform_pairs(scenes, transform_name, TransformConfig(), out_dir, num_frames=2, fps=4.0, resolution=32)
        manifest = {
            "transform": transform_name,
            "scenes": [{"scene_id": s.scene_id, "seed": s.seed, "split": "train"} for s in scenes],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest))

    cfg = mod.Task9Config(
        num_scenes=6,
        base_seed=0,
        num_objects_min=1,
        num_objects_max=3,
        reused_output_dir=str(reused_dir),
        null_transform_output_dir=str(tmp_path / "null_transform"),
        result_path=str(tmp_path / "task_09_result.json"),
        resolution=32,
        num_frames=2,
        pretrained=False,
    )
    result = mod.run_experiment(cfg)

    assert result["task"] == 9
    assert result["implementation_status"] == "COMPLETE"
    assert result["encoder"]["pretrained"] is False
    assert result["encoder"]["frozen"] is True
    assert result["null_transform_sanity_check"]["passed_threshold"] is True

    for transform_name, entry in result["results"].items():
        assert entry["n"] == 6
        assert np.isfinite(entry["mean_cosine_similarity"])
        assert np.isfinite(entry["mean_relative_l2_error"])

    assert set(result["appearance_physical_equality_checks"].keys()) == set(CONTROL_TRANSFORMS)
