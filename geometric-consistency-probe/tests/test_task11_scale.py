"""Tests for experiments/task11_scale.py.

Fast tests (no bpy, no torch/transformers, no network) exercise config
loading, ladder validation, base_seed uniqueness, seed aggregation
(mean/std/failed-run handling), and the scale-trend classifier -- all of
which are pure functions of dicts/dataclasses. A slow group (skipped
without bpy/torch) exercises the real render -> encode -> fit -> evaluate
pipeline end to end on a tiny scale ladder with `pretrained=False`,
matching the fast-dev pattern already used by
tests/test_task6_camera_rotation.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.task11_scale import (
    MIN_SCALE_POINTS,
    MIN_TOP_SCALE,
    RESULT_KEYS,
    Task11Config,
    _aggregate_metric,
    _base_seed_for,
    _classify_trend,
    aggregate_scale_point,
    load_task11_config,
    validate_scale_ladder,
)

# --- config loading ------------------------------------------------------


def test_load_task11_config_defaults():
    cfg = load_task11_config(None)
    assert cfg.scale_ladder == [15, 40, 100]
    assert cfg.seeds == [0, 1]
    assert cfg.train_fraction == 0.8
    assert cfg.pretrained is True


def test_load_task11_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("scale_ladder: [10, 20, 30]\nseeds: [0, 1, 2]\npretrained: false\n")
    cfg = load_task11_config(path)
    assert cfg.scale_ladder == [10, 20, 30]
    assert cfg.seeds == [0, 1, 2]
    assert cfg.pretrained is False
    # Fields not present in the override file keep their defaults.
    assert cfg.train_fraction == 0.8


def test_project_config_file_satisfies_task11_minimums():
    """The actual config this task's experiment is run with must satisfy
    the >=3-scale-points / >=100-scene-top / >=2-seeds / real-pretrained
    requirements -- catches a config regression that would silently
    violate the task's acceptance criteria without a person noticing."""
    cfg = load_task11_config("configs/experiments/task11_scale.yaml")
    validate_scale_ladder(cfg.scale_ladder, cfg.seeds)  # must not raise
    assert len(set(cfg.scale_ladder)) >= MIN_SCALE_POINTS
    assert max(cfg.scale_ladder) >= MIN_TOP_SCALE
    assert len(set(cfg.seeds)) >= 2
    assert cfg.pretrained is True


# --- scale ladder validation ----------------------------------------------


def test_validate_scale_ladder_accepts_a_valid_ladder():
    validate_scale_ladder([15, 40, 100], [0, 1])  # must not raise


def test_validate_scale_ladder_rejects_fewer_than_3_points():
    with pytest.raises(ValueError, match=">=3"):
        validate_scale_ladder([40, 100], [0, 1])


def test_validate_scale_ladder_rejects_duplicate_points_that_collapse_below_3():
    with pytest.raises(ValueError):
        validate_scale_ladder([40, 40, 100], [0, 1])


def test_validate_scale_ladder_rejects_ladder_not_reaching_min_top_scale():
    with pytest.raises(ValueError, match="90"):
        validate_scale_ladder([10, 20, 30], [0, 1])


def test_validate_scale_ladder_rejects_a_single_seed():
    with pytest.raises(ValueError, match="seeds"):
        validate_scale_ladder([15, 40, 100], [0])


# --- base_seed derivation: must be fresh/unique per (num_scenes, seed) ----


def test_base_seed_is_unique_across_scale_points_and_seeds():
    combos = [(n, s) for n in (15, 40, 100) for s in (0, 1)]
    base_seeds = [_base_seed_for(n, s) for n, s in combos]
    assert len(set(base_seeds)) == len(base_seeds)


def test_base_seed_is_deterministic():
    assert _base_seed_for(40, 1) == _base_seed_for(40, 1)


def test_base_seed_reproduces_identical_scenes_given_same_inputs():
    import experiments.geometric_consistency_lib as gclib

    base_seed = _base_seed_for(40, 0)
    scenes_a = gclib.sample_scenes(40, base_seed, 1, 3)
    scenes_b = gclib.sample_scenes(40, base_seed, 1, 3)
    assert [s.to_dict() for s in scenes_a] == [s.to_dict() for s in scenes_b]


def test_different_scale_points_same_seed_do_not_trivially_reuse_the_smaller_points_scene_set():
    """A scale point must be a FRESH sample, not literally 'the smaller
    point's scenes plus more' -- tasks/11_scale.md's explicit prohibited
    shortcut ('reusing the exact same train/test split across scale
    points by only adding scenes to one side')."""
    import experiments.geometric_consistency_lib as gclib

    small = gclib.sample_scenes(15, _base_seed_for(15, 0), 1, 3)
    large = gclib.sample_scenes(40, _base_seed_for(40, 0), 1, 3)
    small_seeds = {s.seed for s in small}
    large_seeds = {s.seed for s in large}
    assert small_seeds.isdisjoint(large_seeds)


# --- seed aggregation (mean/std/failed-run handling) -----------------------


def _fake_metrics(r2: float) -> dict:
    return {key: {"r2": r2, "mean_cosine_similarity": 0.5, "mean_relative_l2_error": 0.5} for key in RESULT_KEYS}


def _fake_run(num_scenes, seed, r2, status="ok"):
    if status == "ok":
        return {
            "num_scenes": num_scenes,
            "seed": seed,
            "base_seed": _base_seed_for(num_scenes, seed),
            "status": "ok",
            "error": None,
            "train_scene_ids": [f"scene_{i:04d}" for i in range(num_scenes - 2)],
            "test_scene_ids": [f"scene_{i:04d}" for i in range(num_scenes - 2, num_scenes)],
            "metrics": _fake_metrics(r2),
            "wall_clock_seconds": 1.5,
        }
    return {
        "num_scenes": num_scenes,
        "seed": seed,
        "base_seed": _base_seed_for(num_scenes, seed),
        "status": "failed",
        "error": "boom",
    }


def test_aggregate_metric_computes_mean_and_std_over_successful_runs_only():
    runs = [_fake_run(40, 0, 0.1), _fake_run(40, 1, 0.3), _fake_run(40, 2, r2=0.0, status="failed")]
    agg = _aggregate_metric(runs, "learned_W_T", "r2")
    assert agg["values"] == [0.1, 0.3]
    assert agg["mean"] == pytest.approx(0.2)
    assert agg["std"] == pytest.approx(np.std([0.1, 0.3]))


def test_aggregate_metric_all_failed_returns_none_not_an_exception():
    runs = [_fake_run(40, 0, 0.0, status="failed"), _fake_run(40, 1, 0.0, status="failed")]
    agg = _aggregate_metric(runs, "learned_W_T", "r2")
    assert agg["mean"] is None
    assert agg["values"] == []


def test_aggregate_scale_point_keeps_every_run_failed_or_not():
    """Required software test: a failed run at one scale point must be
    REPORTED, not dropped from the record (even though it is excluded
    from the mean/std)."""
    runs = [_fake_run(40, 0, 0.2), _fake_run(40, 1, 0.0, status="failed")]
    sp = aggregate_scale_point(40, [0, 1], runs)
    assert len(sp["runs"]) == 2
    assert sp["num_seeds_requested"] == 2
    assert sp["num_seeds_succeeded"] == 1
    assert sp["aggregate"]["learned_W_T"]["r2"]["mean"] == pytest.approx(0.2)


def test_aggregate_scale_point_all_baseline_keys_present():
    runs = [_fake_run(40, 0, 0.2), _fake_run(40, 1, 0.3)]
    sp = aggregate_scale_point(40, [0, 1], runs)
    assert set(sp["aggregate"].keys()) == set(RESULT_KEYS)


# --- scale-trend classification --------------------------------------------


def _fake_scale_point(num_scenes, learned_r2, best_baseline_r2, n_seeds_ok=2):
    metrics = {key: {"r2": {"mean": None, "std": 0.0, "values": []}} for key in RESULT_KEYS}
    metrics["learned_W_T"]["r2"]["mean"] = learned_r2
    # Give every baseline a value <= best_baseline_r2, with exactly one
    # baseline hitting best_baseline_r2 exactly (so "best" is well-defined).
    baseline_keys = [k for k in RESULT_KEYS if k != "learned_W_T"]
    for i, k in enumerate(baseline_keys):
        if best_baseline_r2 is None:
            metrics[k]["r2"]["mean"] = None
        else:
            metrics[k]["r2"]["mean"] = best_baseline_r2 if i == 0 else best_baseline_r2 - 0.5
    return {
        "num_scenes": num_scenes,
        "num_seeds_requested": n_seeds_ok,
        "num_seeds_succeeded": n_seeds_ok,
        "aggregate": metrics,
    }


def test_classify_trend_persists_when_margin_is_stable():
    scale_points = [
        _fake_scale_point(15, learned_r2=-0.30, best_baseline_r2=0.70),
        _fake_scale_point(40, learned_r2=-0.30, best_baseline_r2=0.70),
        _fake_scale_point(100, learned_r2=-0.31, best_baseline_r2=0.70),
    ]
    trend = _classify_trend(scale_points)
    assert trend["effect_trend"] == "persists"


def test_classify_trend_strengthens_when_margin_improves_with_scale():
    # No sign crossing (best_baseline_r2 fixed at 0.0, learned_r2 rises
    # from -0.5 to -0.05, staying negative throughout) -- a sign change
    # is classified separately (as "disappears"/"emerges") regardless of
    # magnitude, so this scenario is kept strictly same-sign to isolate
    # the continuous strengthens/weakens/saturates/persists logic.
    scale_points = [
        _fake_scale_point(15, learned_r2=-0.5, best_baseline_r2=0.0),
        _fake_scale_point(40, learned_r2=-0.2, best_baseline_r2=0.0),
        _fake_scale_point(100, learned_r2=-0.05, best_baseline_r2=0.0),
    ]
    trend = _classify_trend(scale_points)
    assert trend["effect_trend"] == "strengthens"


def test_classify_trend_weakens_when_margin_declines_with_scale():
    scale_points = [
        _fake_scale_point(15, learned_r2=0.5, best_baseline_r2=0.0),
        _fake_scale_point(40, learned_r2=0.2, best_baseline_r2=0.0),
        _fake_scale_point(100, learned_r2=0.05, best_baseline_r2=0.4),
    ]
    trend = _classify_trend(scale_points)
    assert trend["effect_trend"] in ("weakens", "disappears")


def test_classify_trend_disappears_when_positive_margin_vanishes():
    scale_points = [
        _fake_scale_point(15, learned_r2=0.5, best_baseline_r2=0.0),
        _fake_scale_point(40, learned_r2=0.1, best_baseline_r2=0.0),
        _fake_scale_point(100, learned_r2=-0.2, best_baseline_r2=0.1),
    ]
    trend = _classify_trend(scale_points)
    assert trend["effect_trend"] == "disappears"


def test_classify_trend_inconclusive_when_a_scale_point_has_no_successful_seed():
    scale_points = [
        _fake_scale_point(15, learned_r2=-0.3, best_baseline_r2=0.7),
        _fake_scale_point(40, learned_r2=None, best_baseline_r2=None),
        _fake_scale_point(100, learned_r2=-0.29, best_baseline_r2=0.69),
    ]
    trend = _classify_trend(scale_points)
    assert trend["effect_trend"] == "inconclusive"


def test_classify_trend_margins_by_scale_keyed_by_num_scenes():
    scale_points = [
        _fake_scale_point(15, learned_r2=-0.3, best_baseline_r2=0.7),
        _fake_scale_point(40, learned_r2=-0.31, best_baseline_r2=0.71),
        _fake_scale_point(100, learned_r2=-0.29, best_baseline_r2=0.69),
    ]
    trend = _classify_trend(scale_points)
    assert set(trend["margins_by_scale"].keys()) == {15, 40, 100}


# --- run_experiment orchestration (mocked combos, no bpy/torch) -----------


def test_run_experiment_reports_failed_runs_without_dropping_them(monkeypatch, tmp_path):
    """A failed (num_scenes, seed) combination must still appear in the
    result's scale_points[*]['runs'], not be silently excluded."""
    import experiments.task11_scale as mod

    cfg = Task11Config(scale_ladder=[15, 40, 100], seeds=[0, 1], output_dir=str(tmp_path / "scale"))

    monkeypatch.setattr(mod.gclib, "build_encoder", lambda *a, **k: type("E", (), {"pretrained": False, "checkpoint": "x", "model": None})())
    monkeypatch.setattr(mod.gclib, "build_pixel_baseline_encoder", lambda: type("P", (), {"grid_size": 4, "output_dim": 48, "encode_video": lambda self, v: v})())
    monkeypatch.setattr(mod.gclib, "build_random_encoder", lambda *a, **k: type("R", (), {"pretrained": False, "checkpoint": "x", "model": None})())
    monkeypatch.setattr(mod.gclib, "software_versions", lambda: {"python": "x"})

    def fake_run_one(num_scenes, seed, cfg_, primary, pixel, random_):
        if num_scenes == 40 and seed == 1:
            raise RuntimeError("simulated render failure")
        return _fake_run(num_scenes, seed, r2=0.1)

    monkeypatch.setattr(mod, "run_one_combination", fake_run_one)

    result = mod.run_experiment(cfg)

    flat_runs = [r for sp in result["scale_points"] for r in sp["runs"]]
    assert len(flat_runs) == 6  # 3 scale points x 2 seeds, none dropped
    failed = [r for r in flat_runs if r["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["num_scenes"] == 40 and failed[0]["seed"] == 1
    assert result["num_failed_runs"] == 1
    assert len(result["scale_points"]) == 3


def test_run_experiment_raises_if_ladder_invalid(monkeypatch):
    import experiments.task11_scale as mod

    cfg = Task11Config(scale_ladder=[40, 100], seeds=[0, 1])
    with pytest.raises(ValueError):
        mod.run_experiment(cfg)


# --- slow, real end-to-end pipeline test (bpy + torch/transformers required)


@pytest.mark.slow
def test_full_pipeline_runs_end_to_end_on_a_tiny_ladder(tmp_path, monkeypatch):
    """Exercises sample -> split -> render -> encode -> fit -> evaluate
    for real, across a tiny scale ladder, with an untrained
    (pretrained=False) encoder for speed -- NOT a claim about the real
    Task 11 result (which needs the ladder in configs/experiments/
    task11_scale.yaml and real pretrained weights), only a check that
    the code path works end to end without error and produces
    well-shaped, finite, disjoint-split metrics at every scale point.
    """
    pytest.importorskip("bpy", reason="needs bpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import experiments.task11_scale as mod

    monkeypatch.setattr(mod, "MIN_TOP_SCALE", 8)

    cfg = Task11Config(
        scale_ladder=[4, 6, 8],
        seeds=[0, 1],
        train_fraction=0.75,
        output_dir=str(tmp_path / "scale"),
        resolution=32,
        num_frames=2,
        pretrained=False,
    )
    result = mod.run_experiment(cfg)

    assert result["task"] == 11
    assert len(result["scale_points"]) == 3
    assert result["num_failed_runs"] == 0

    for sp in result["scale_points"]:
        assert sp["num_seeds_succeeded"] == 2
        for run in sp["runs"]:
            assert run["status"] == "ok"
            assert set(run["train_scene_ids"]).isdisjoint(run["test_scene_ids"])
            for key in RESULT_KEYS:
                for metric_name in ("r2", "mean_cosine_similarity", "mean_relative_l2_error"):
                    assert np.isfinite(run["metrics"][key][metric_name])
        for key in RESULT_KEYS:
            assert sp["aggregate"][key]["r2"]["mean"] is not None

    assert result["trend_analysis"]["effect_trend"] in (
        "persists", "strengthens", "weakens", "saturates", "disappears", "inconclusive",
        "emerges (was absent at small scale, appears at large scale)",
    )
