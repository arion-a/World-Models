"""Task 12 follow-up #4: faster real_motion orbit rate, to fix a
DETECTABILITY problem an effect-size audit found in the `real_motion`
condition, not a scientific-conclusion problem.

    python -m experiments.task12_followup_faster_orbit --config configs/experiments/task12_followup_faster_orbit.yaml

Background (the audit this follow-up answers): the representation-
change ratio

    ratio = E||Z' - Z||^2 / E_{i != j} ||Z_i - Z_j||^2

(same quantity as `task12_followup_magnitude_sweep.representation_change_ratio`,
imported and reused unmodified below) measures how big the encoder's
representation shift between two temporal windows is, RELATIVE to how
different two unrelated scenes' representations already are. Task 12's
original real_motion condition (orbit_deg_per_sec=30.0, see
configs/experiments/task12_temporal_consistency.yaml) produced a ratio
below the audit's calibrated 0.3 threshold: the camera motion between
window anchors was too small, relative to natural cross-scene
representation variance, for ANY linear map -- however good -- to be
reliably detected against that noise floor. A negative R^2 margin at
that ratio is therefore ambiguous (motion-too-small-to-see vs.
genuinely-no-structure), not informative.

This is explicitly a MAGNITUDE fix, not a metric-shopping exercise: the
physical camera-orbit speed is increased BEFORE any equivariance/
persistence/shuffled-pairing evaluation is run, using a small pilot to
find the smallest rate that clears the detectability threshold, then
the full protocol is run once, at N=100, and reported honestly whatever
the R^2 margins come out to be. static_control's motion (none) is
UNCHANGED -- it is a control and is SUPPOSED to show a low ratio; only
real_motion's orbit_deg_per_sec is redesigned here.

This module deliberately does not modify experiments/
task12_temporal_consistency.py, experiments/
task12_followup_magnitude_sweep.py, experiments/
geometric_consistency_lib.py, generation/, transforms/, or encoders/ --
it only imports and reuses their existing public functions:

  * experiments.task12_temporal_consistency: Task12Config, camera_motion_for,
    render_condition_clips, verify_saved_temporal_transform,
    temporal_pixel_diff_stats, assemble_result (Task 12's own fit/evaluate/
    margin/report logic, unmodified) -- but NOT run_condition or
    encode_condition_windows: run_condition does not expose the raw
    per-scene (Z, Z') arrays this follow-up needs for the detectability
    ratio, and encode_condition_windows has no resumability, which this
    follow-up's much longer per-scene encode time (real V-JEPA2 forward
    passes measured at ~90-100s each on this machine's CPU) makes
    necessary. This module's own run_condition_with_ratio() and
    encode_condition_windows_resumable() are additive siblings that call
    the SAME lower-level pieces (gclib.build_full_arrays,
    representation_change_ratio, per-scene on-disk (Z, Z') caching) on
    top of run_condition/encode_condition_windows's own logic, unchanged.
  * experiments.task12_followup_magnitude_sweep.representation_change_ratio:
    the exact detectability-ratio formula the audit used, reused verbatim
    so this follow-up's numbers are computed the identical way.
  * experiments.geometric_consistency_lib (gclib): scene sampling, scene-
    level splitting, encoder construction, array assembly, fit+evaluate+
    baselines, frozen-params verification, software-version provenance,
    the pytest-summary test runner -- all unmodified.

Two phases, both driven by this one config:

  1. PILOT (real_motion only; static_control's rate is never touched):
     a handful of scenes (pilot_num_scenes, default 10) rendered+encoded
     at each of several candidate orbit_deg_per_sec values
     (pilot_candidate_rates_deg_per_sec), and the detectability ratio
     computed for each. The smallest candidate clearing
     `detectability_ratio_threshold` (default 0.3) is chosen; if none
     clears it, `pilot_extra_rates_deg_per_sec` are tried, in order,
     until one clears the threshold or the extra list is exhausted (in
     which case the fastest rate tried is used and the result honestly
     records that no candidate reached the threshold -- this follow-up
     reports what it finds, it does not manufacture a positive).
  2. FULL RUN at num_scenes (default 100) for BOTH real_motion (at the
     chosen rate) and static_control (at its original, untouched
     mode="static"), using the identical fit/evaluate/baseline/margin
     logic Task 12 itself uses (gclib.evaluate_transform,
     task12_temporal_consistency.assemble_result), plus the
     detectability ratio for both conditions.

Uses base_seed_real_motion=102 / base_seed_static_control=103 by
default -- disjoint (per gclib.sample_scenes's
seed = base_seed * 1_000_003 + i convention) from every base_seed this
project's other Task 12 (0/1), scale follow-ups (0/1), and magnitude-
sweep follow-ups (42, 999) already use, and from Task 7/Task 14's
Category-A follow-up seeds. The pilot re-uses base_seed_real_motion
too, at a smaller num_scenes, so pilot scenes are a deterministic,
bit-identical NESTED SUBSET of the full run's real_motion scenes (same
`sample_scenes` seeding-by-index convention every other follow-up in
this project relies on) -- not an independently-drawn pilot sample that
could itself introduce a confound.

Writes ONLY to experiments/task12_followup_faster_orbit_n100/ (or
whatever `output_dir` the config sets) and state/
task12_followup_faster_orbit_result.json (or whatever `result_path` the
config sets) -- never to state/task_12_result.json, that task's
checkpoint, configs/experiments/task12_temporal_consistency.yaml, or
any existing experiments/temporal_consistency* / temporal_magnitude_sweep*
directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import yaml

import experiments.geometric_consistency_lib as gclib
from experiments.task12_followup_magnitude_sweep import representation_change_ratio
from experiments.task12_temporal_consistency import (
    MIN_SCENES,
    Task12Config,
    assemble_result,
    camera_motion_for,
    render_condition_clips,
    temporal_pixel_diff_stats,
    verify_saved_temporal_transform,
)

DEFAULT_RESULT_PATH = "state/task12_followup_faster_orbit_result.json"
DEFAULT_OUTPUT_DIR = "experiments/task12_followup_faster_orbit_n100"
# Must not collide with any existing base_seed this project's other Task 12
# runs/follow-ups use (0, 1, 42, 999) -- see module docstring.
DEFAULT_BASE_SEED_REAL_MOTION = 102
DEFAULT_BASE_SEED_STATIC_CONTROL = 103


@dataclass
class FasterOrbitConfig(Task12Config):
    # Overridden defaults (still all overridable from YAML): N=100 per
    # condition, new disjoint seeds, new output locations -- see module
    # docstring's "Writes ONLY to ..." paragraph.
    num_scenes: int = 100
    base_seed_real_motion: int = DEFAULT_BASE_SEED_REAL_MOTION
    base_seed_static_control: int = DEFAULT_BASE_SEED_STATIC_CONTROL
    output_dir: str = DEFAULT_OUTPUT_DIR
    result_path: str = DEFAULT_RESULT_PATH

    # Detectability-ratio design parameters (this follow-up's own; not
    # part of Task 12's original Task12Config). real_motion's rate is
    # chosen from these; static_control is never swept and always uses
    # Task12Config's own mode="static" (orbit_deg_per_sec is irrelevant
    # for it -- see task12_temporal_consistency.camera_motion_for).
    detectability_ratio_threshold: float = 0.3
    pilot_num_scenes: int = 10
    pilot_candidate_rates_deg_per_sec: list[float] = field(default_factory=lambda: [60.0, 90.0, 120.0])
    pilot_extra_rates_deg_per_sec: list[float] = field(default_factory=lambda: [180.0, 240.0])
    pilot_output_subdir: str = "_pilot"


def load_config(path: str | Path | None) -> FasterOrbitConfig:
    cfg = FasterOrbitConfig()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


# --- encoding, with per-scene on-disk caching -------------------------------
#
# task12_temporal_consistency.encode_condition_windows itself has no
# resumability (it re-encodes every scene on every call) -- fine for
# Task 12's own N<=40-per-call scale, but this follow-up's real V-JEPA2
# forward passes on this machine's CPU take ~90-100s EACH (measured), so
# a single condition's N=100 encoding pass is many hours of wall-clock
# compute. This additive helper -- new to this module, does not touch
# encode_condition_windows -- caches each scene's (Z, Z_prime) to
# `<out_dir>/_encode_cache/<scene_id>.npz` as soon as it is computed, so
# an interrupted run resumes from the last cached scene instead of
# losing all encoding progress. Semantics (mean_pool(encoder.encode(window)),
# same window slicing) are identical to encode_condition_windows.


def encode_condition_windows_resumable(scenes, out_dir: Path, encoder, window_frames: int, cache_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    from encoders.vjepa import mean_pool
    from generation.ground_truth import load_ground_truth

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    reps: dict[str, dict[str, np.ndarray]] = {}
    num_cached = 0
    for scene in scenes:
        cache_file = cache_dir / f"{scene.scene_id}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            reps[scene.scene_id] = {"Z": data["Z"], "Z_prime": data["Z_prime"]}
            num_cached += 1
            continue
        _, rgb, _, _ = load_ground_truth(out_dir, scene.scene_id)
        window1 = rgb[:window_frames]
        window2 = rgb[window_frames : 2 * window_frames]
        Z = mean_pool(encoder.encode(window1))
        Zp = mean_pool(encoder.encode(window2))
        np.savez(cache_file, Z=Z, Z_prime=Zp)
        reps[scene.scene_id] = {"Z": Z, "Z_prime": Zp}
        print(f"    encoded {out_dir}/{scene.scene_id} ({len(reps)}/{len(scenes)})", flush=True)
    if num_cached:
        print(f"    resumed {num_cached}/{len(scenes)} scenes from {cache_dir}", flush=True)
    return reps


# --- pilot: detectability ratio at a handful of candidate orbit rates ------


def run_pilot_rate(rate: float, cfg: FasterOrbitConfig, encoder) -> dict:
    """Render+encode `cfg.pilot_num_scenes` real_motion scenes at
    orbit_deg_per_sec=`rate` and compute the detectability ratio.

    Uses the SAME base_seed_real_motion as the full run (nested-subset
    convention, see module docstring) so pilot scenes 0..pilot_num_scenes-1
    are bit-identical to the full run's first `pilot_num_scenes` scenes
    at whatever rate is ultimately chosen.
    """
    pilot_cfg = replace(
        cfg,
        num_scenes=cfg.pilot_num_scenes,
        orbit_deg_per_sec=rate,
        output_dir=str(Path(cfg.output_dir) / cfg.pilot_output_subdir / f"rate_{rate:g}"),
    )
    scenes = gclib.sample_scenes(pilot_cfg.num_scenes, pilot_cfg.base_seed_real_motion, pilot_cfg.num_objects_min, pilot_cfg.num_objects_max)
    out_dir = render_condition_clips(scenes, "real_motion", pilot_cfg)
    for scene in scenes:
        verify_saved_temporal_transform(out_dir / scene.scene_id, "real_motion")

    reps = encode_condition_windows_resumable(scenes, out_dir, encoder, pilot_cfg.window_frames, out_dir / "_encode_cache")
    _, Z_all, Zp_all = gclib.build_full_arrays(scenes, reps)
    ratio_block = representation_change_ratio(Z_all, Zp_all)

    return {
        "orbit_deg_per_sec": rate,
        "num_scenes": pilot_cfg.num_scenes,
        "expected_azimuth_deg_between_window_anchors": rate * pilot_cfg.window_frames / pilot_cfg.fps,
        "representation_change": ratio_block,
        "cleared_threshold": bool(ratio_block["ratio"] >= cfg.detectability_ratio_threshold),
    }


def run_pilot(cfg: FasterOrbitConfig) -> dict:
    """Sweep pilot_candidate_rates_deg_per_sec (and, if needed,
    pilot_extra_rates_deg_per_sec) and pick the SMALLEST rate that
    clears detectability_ratio_threshold. Builds the encoder ONCE and
    reuses it across every candidate rate (rendering is what changes,
    not the frozen encoder).
    """
    encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))

    tried: list[dict] = []
    chosen_rate = None
    for rate in cfg.pilot_candidate_rates_deg_per_sec:
        result = run_pilot_rate(rate, cfg, encoder)
        tried.append(result)
        print(f"pilot rate={rate:g} deg/s -> ratio={result['representation_change']['ratio']:.4f}")
        if chosen_rate is None and result["cleared_threshold"]:
            chosen_rate = rate

    extra_tried: list[dict] = []
    if chosen_rate is None:
        for rate in cfg.pilot_extra_rates_deg_per_sec:
            result = run_pilot_rate(rate, cfg, encoder)
            extra_tried.append(result)
            tried.append(result)
            print(f"pilot EXTRA rate={rate:g} deg/s -> ratio={result['representation_change']['ratio']:.4f}")
            if chosen_rate is None and result["cleared_threshold"]:
                chosen_rate = rate
                break

    threshold_cleared_by_any = any(r["cleared_threshold"] for r in tried)
    if chosen_rate is None:
        # Prohibited-shortcuts guardrail: never silently substitute a
        # rate that didn't clear the threshold as if it had. Use the
        # fastest rate actually tried, and say plainly that no candidate
        # cleared the threshold -- this is a valid, reportable outcome,
        # not a failure to paper over (tasks/12_temporal_consistency.md's
        # sibling "negative results are valid" principle applies here too).
        chosen_rate = tried[-1]["orbit_deg_per_sec"]

    return {
        "detectability_ratio_threshold": cfg.detectability_ratio_threshold,
        "pilot_num_scenes": cfg.pilot_num_scenes,
        "candidates_tried": tried,
        "extra_candidates_tried": extra_tried,
        "chosen_orbit_deg_per_sec": chosen_rate,
        "threshold_cleared_by_chosen_rate": bool(
            next(r for r in tried if r["orbit_deg_per_sec"] == chosen_rate)["cleared_threshold"]
        ),
        "threshold_cleared_by_any_candidate": threshold_cleared_by_any,
        "encoder_pretrained": encoder_pretrained,
    }


# --- full run: additive sibling of task12_temporal_consistency.run_condition,
# identical logic + the detectability ratio -----------------------------------


def run_condition_with_ratio(condition: str, cfg: FasterOrbitConfig) -> dict:
    base_seed = cfg.base_seed_real_motion if condition == "real_motion" else cfg.base_seed_static_control
    scenes = gclib.sample_scenes(cfg.num_scenes, base_seed, cfg.num_objects_min, cfg.num_objects_max)
    if len(scenes) < MIN_SCENES:
        raise ValueError(f"Requires >={MIN_SCENES} scenes per condition, got {len(scenes)} for {condition}")
    scene_ids = [s.scene_id for s in scenes]

    split = gclib.assign_split(scene_ids, cfg.train_fraction, base_seed)

    out_dir = render_condition_clips(scenes, condition, cfg)
    for scene in scenes:
        verify_saved_temporal_transform(out_dir / scene.scene_id, condition)

    manifest = {
        "condition": condition,
        "num_scenes": len(scenes),
        "base_seed": base_seed,
        "train_fraction": cfg.train_fraction,
        "window_frames": cfg.window_frames,
        "motion_config": asdict(camera_motion_for(condition, cfg)),
        "scenes": [{"scene_id": s.scene_id, "seed": s.seed, "split": split[s.scene_id]} for s in scenes],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    diff_stats = temporal_pixel_diff_stats(scenes, out_dir, cfg.window_frames)

    encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))
    params_before = gclib.snapshot_params(encoder.model)

    reps = encode_condition_windows_resumable(scenes, out_dir, encoder, cfg.window_frames, out_dir / "_encode_cache")
    frozen_verified = gclib.params_unchanged(params_before, encoder.model)

    train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test = gclib.build_arrays(scenes, split, reps)
    metrics_block = gclib.evaluate_transform(
        f"temporal_{condition}", Z_train, Zp_train, Z_test, Zp_test, cfg.ridge_alpha, cfg.shuffled_pairing_seed
    )

    _, Z_all, Zp_all = gclib.build_full_arrays(scenes, reps)
    ratio_block = representation_change_ratio(Z_all, Zp_all)

    return {
        "condition": condition,
        "base_seed": base_seed,
        "num_scenes": len(scenes),
        "scene_ids": scene_ids,
        "scene_seeds": [s.seed for s in scenes],
        "train_scene_ids": train_ids,
        "test_scene_ids": test_ids,
        "train_fraction": cfg.train_fraction,
        "encoder": {
            "name": "VJEPAEncoder",
            "checkpoint": encoder.checkpoint,
            "pretrained": encoder_pretrained,
            "frozen": frozen_verified,
            "pooling": "mean_pool",
        },
        "metrics": metrics_block,
        "pixel_diff_stats": diff_stats,
        "representation_change": ratio_block,
        "out_dir": str(out_dir),
        "manifest_path": str(out_dir / "manifest.json"),
    }


def run_full(cfg: FasterOrbitConfig) -> tuple[dict, dict]:
    real = run_condition_with_ratio("real_motion", cfg)
    static = run_condition_with_ratio("static_control", cfg)
    return real, static


def run_existing_test_suite(repo_root: str | Path) -> dict:
    return gclib.run_existing_test_suite(repo_root)


def run_experiment(cfg: FasterOrbitConfig) -> dict:
    pilot_report = run_pilot(cfg)
    cfg = replace(cfg, orbit_deg_per_sec=pilot_report["chosen_orbit_deg_per_sec"])

    real, static = run_full(cfg)

    base_result = assemble_result(cfg, real, static)
    base_result["task"] = "task_12_followup_faster_orbit"
    base_result["orbit_rate_selection"] = pilot_report
    base_result["representation_change"] = {
        "real_motion": real["representation_change"],
        "static_control": static["representation_change"],
        "detectability_ratio_threshold": cfg.detectability_ratio_threshold,
        "real_motion_cleared_threshold": bool(
            real["representation_change"]["ratio"] >= cfg.detectability_ratio_threshold
        ),
    }
    base_result["scientific_result"] = (
        f"Detectability fix (orbit rate {cfg.orbit_deg_per_sec:g} deg/sec, chosen by pilot sweep -- see "
        f"orbit_rate_selection): real_motion's representation-change ratio E||Z'-Z||^2 / "
        f"E_cross-scene||Zi-Zj||^2 = {real['representation_change']['ratio']:.4f} at N={real['num_scenes']} "
        f"(static_control, UNCHANGED motion, = {static['representation_change']['ratio']:.4f}). "
        + base_result["scientific_result"]
    )
    return base_result


def main():
    parser = argparse.ArgumentParser(description="Task 12 follow-up #4: faster real_motion orbit rate (detectability fix).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--skip_tests", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = run_existing_test_suite(repo_root)
    else:
        result["tests"] = {"passed": 0, "failed": 0, "skipped": True}

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Result written to {result_path}")
    print(json.dumps(result["orbit_rate_selection"], indent=2, default=str))
    print(json.dumps(result["representation_change"], indent=2))
    print(json.dumps(result["margins"], indent=2))


if __name__ == "__main__":
    main()
