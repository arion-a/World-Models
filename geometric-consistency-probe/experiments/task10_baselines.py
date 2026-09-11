"""Task 10: baseline framework -- consolidation + the expanded, five-baseline
flagship comparison.

    python -m experiments.task10_baselines --config configs/experiments/task10_baselines.yaml

Pipeline (see tasks/10_baselines.md and
research/CANONICAL_RESEARCH_PROTOCOL.md's "TASK 10" section, which this
module implements verbatim):

    Reuse Task 6's EXACT scene set, scene-level train/test split, and
    already-rendered camera_rotation pairs (experiments/camera_rotation/,
    Task 6's flagship transform, shared with Task 7) -- no new scene
    generation, no new rendering.
    -> re-encode every scene's original/transformed pair with THREE
       encoders reading the identical rendered frames: the primary
       frozen pretrained VJEPAEncoder (Task 6's own), the non-learned
       pixel-statistics baseline (encoders/pixel_baseline.py), and a
       matched-architecture, randomly-initialized VJEPAEncoder
       (encoders.vjepa.VJEPAEncoder(pretrained=False))
    -> fit W_T and evaluate the full baselines.run_all_baselines() set
       (persistence, mean, shuffled-pairing, pixel-statistics,
       random-encoder) for the primary encoder, on the identical TEST
       split Task 6 used
    -> verify NUMERICAL PARITY: the primary encoder's re-computed
       learned_W_T / persistence / mean / random-pair numbers must
       match state/task_06_result.json's recorded numbers (same seed,
       same data, same code path through the Task-10-refactored
       experiments.geometric_consistency_lib.evaluate_transform)
    -> write state/task_10_result.json with the full five-baseline
       comparison table and the required "does pixel-statistics/
       random-encoder perform comparably to the primary encoder?"
       scientific-QA verdict.

This module deliberately does not modify generation/, transforms/, or
encoders/ -- it only calls their existing public functions (and
experiments/geometric_consistency_lib.py's shared helpers, several of
which this task adds: encode_all_pairs_pooled, build_pixel_baseline_encoder,
build_random_encoder).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

import experiments.geometric_consistency_lib as gclib
from baselines.run_all_baselines import FULL_BASELINE_NAMES
from experiments.task6_camera_rotation import Task6Config, load_task6_config

DEFAULT_RESULT_PATH = "state/task_10_result.json"
TASK6_RESULT_PATH = "state/task_06_result.json"
TASK6_CONFIG_PATH = "configs/experiments/task6_camera_rotation.yaml"
TRANSFORM_NAME = "camera_rotation"  # Task 6/7's shared flagship transform
# Tolerance for the numerical-parity check (Experimental protocol step 4)
# -- both runs are deterministic (frozen encoder, no dropout, CPU/GPU
# determinism best-effort per encoders/vjepa.py) so this is tight, not a
# loose "roughly similar" bound.
PARITY_ATOL = 1e-5


@dataclass
class Task10Config:
    # Reuses Task 6's own config values so the scene set/split/render
    # directory line up exactly -- see load_task10_config below, which
    # loads Task 6's own config file by default.
    result_path: str = DEFAULT_RESULT_PATH
    task6_config_path: str = TASK6_CONFIG_PATH
    task6_result_path: str = TASK6_RESULT_PATH


def load_task10_config(path: str | Path | None) -> Task10Config:
    cfg = Task10Config()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


def _load_task6_result(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _relative_diff(a: float, b: float) -> float:
    return abs(a - b)


def run_experiment(cfg: Task10Config) -> dict:
    task6_result = _load_task6_result(cfg.task6_result_path)
    task6_cfg: Task6Config = load_task6_config(cfg.task6_config_path)

    # 1-2. Reproduce Task 6's EXACT scene set + split (pure functions of
    # (num_scenes, base_seed, num_objects_min/max, train_fraction) --
    # no rendering here, so this is cheap and needs no bpy).
    scenes = gclib.sample_scenes(task6_cfg.num_scenes, task6_cfg.base_seed, task6_cfg.num_objects_min, task6_cfg.num_objects_max)
    scene_ids = [s.scene_id for s in scenes]
    split = gclib.assign_split(scene_ids, task6_cfg.train_fraction, task6_cfg.base_seed)

    task6_dataset = task6_result["dataset"]
    if sorted(scene_ids) != sorted(task6_dataset["train_scene_ids"] + task6_dataset["test_scene_ids"]):
        raise ValueError("Task 10: re-derived scene set does not match state/task_06_result.json's recorded scenes")
    if [sid for sid in scene_ids if split[sid] == "train"] != task6_dataset["train_scene_ids"]:
        raise ValueError("Task 10: re-derived split does not match state/task_06_result.json's recorded split")

    render_dir = Path(task6_cfg.output_dir)
    if not render_dir.exists():
        raise FileNotFoundError(
            f"Task 10 requires Task 6's already-rendered {render_dir} -- 'no new scene generation required "
            "unless a baseline's fair-comparison requirement cannot otherwise be met' (this one can)."
        )

    # 3. Three encoders, same rendered frames --------------------------------
    primary_encoder = gclib.build_encoder(task6_cfg.pretrained, task6_cfg.checkpoint, task6_cfg.device, task6_cfg.fallback_seed)
    primary_encoder_pretrained = bool(getattr(primary_encoder, "pretrained", False))
    primary_params_before = gclib.snapshot_params(primary_encoder.model)
    primary_reps = gclib.encode_all_pairs(scenes, render_dir, primary_encoder)
    primary_frozen_verified = gclib.params_unchanged(primary_params_before, primary_encoder.model)

    pixel_encoder = gclib.build_pixel_baseline_encoder()
    pixel_reps = gclib.encode_all_pairs_pooled(scenes, render_dir, pixel_encoder.encode_video)

    random_encoder = gclib.build_random_encoder(task6_cfg.checkpoint, task6_cfg.device, task6_cfg.fallback_seed)
    random_params_before = gclib.snapshot_params(random_encoder.model)
    random_reps = gclib.encode_all_pairs(scenes, render_dir, random_encoder)
    random_frozen_verified = gclib.params_unchanged(random_params_before, random_encoder.model)

    # Assemble train/test arrays, enforcing scene-level disjointness for
    # all three encoders' representations (research/RESEARCH_INVARIANTS.md
    # invariants 4, 5, 16).
    train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test = gclib.build_arrays(scenes, split, primary_reps)
    _, _, pZ_train, pZp_train, pZ_test, pZp_test = gclib.build_arrays(scenes, split, pixel_reps)
    _, _, rZ_train, rZp_train, rZ_test, rZp_test = gclib.build_arrays(scenes, split, random_reps)

    # 4-5. Fit + evaluate the full five-baseline comparison via the
    # Task-10-refactored evaluate_transform / baselines.run_all_baselines.
    metrics_block = gclib.evaluate_transform(
        TRANSFORM_NAME,
        Z_train, Zp_train, Z_test, Zp_test,
        task6_cfg.ridge_alpha, task6_cfg.shuffled_pairing_seed,
        pixel_Z_train=pZ_train, pixel_Zp_train=pZp_train, pixel_Z_test=pZ_test, pixel_Zp_test=pZp_test,
        random_Z_train=rZ_train, random_Zp_train=rZp_train, random_Z_test=rZ_test, random_Zp_test=rZp_test,
    )

    # Numerical-parity check against Task 6's previously recorded numbers
    # (Experimental protocol step 4): the primary-encoder core-three +
    # learned_W_T numbers must reproduce, since nothing about the primary
    # encoder's data/code path changed -- only baselines/ was consolidated.
    task6_metrics = task6_result["metrics"]
    parity_checks = {}
    parity_ok = True
    for key in ("learned_W_T", "persistence_baseline", "mean_baseline", "random_pair_control"):
        for metric_name in ("r2", "mean_cosine_similarity", "mean_relative_l2_error"):
            recomputed = metrics_block[key][metric_name]
            recorded = task6_metrics[key][metric_name]
            diff = _relative_diff(recomputed, recorded)
            parity_checks[f"{key}.{metric_name}"] = {"recomputed": recomputed, "recorded": recorded, "abs_diff": diff}
            if diff > PARITY_ATOL:
                parity_ok = False

    comparison_table = (
        "| baseline | R^2 | mean cosine sim | mean rel. L2 err |\n"
        "|---|---|---|---|\n"
        f"| learned W_T (primary, pretrained) | {metrics_block['learned_W_T']['r2']:.4f} | "
        f"{metrics_block['learned_W_T']['mean_cosine_similarity']:.4f} | {metrics_block['learned_W_T']['mean_relative_l2_error']:.4f} |\n"
        f"| persistence | {metrics_block['persistence_baseline']['r2']:.4f} | "
        f"{metrics_block['persistence_baseline']['mean_cosine_similarity']:.4f} | {metrics_block['persistence_baseline']['mean_relative_l2_error']:.4f} |\n"
        f"| mean | {metrics_block['mean_baseline']['r2']:.4f} | "
        f"{metrics_block['mean_baseline']['mean_cosine_similarity']:.4f} | {metrics_block['mean_baseline']['mean_relative_l2_error']:.4f} |\n"
        f"| shuffled-pairing | {metrics_block['random_pair_control']['r2']:.4f} | "
        f"{metrics_block['random_pair_control']['mean_cosine_similarity']:.4f} | {metrics_block['random_pair_control']['mean_relative_l2_error']:.4f} |\n"
        f"| pixel-statistics encoder | {metrics_block['pixel_statistics_baseline']['r2']:.4f} | "
        f"{metrics_block['pixel_statistics_baseline']['mean_cosine_similarity']:.4f} | {metrics_block['pixel_statistics_baseline']['mean_relative_l2_error']:.4f} |\n"
        f"| randomly-initialized encoder | {metrics_block['random_encoder_baseline']['r2']:.4f} | "
        f"{metrics_block['random_encoder_baseline']['mean_cosine_similarity']:.4f} | {metrics_block['random_encoder_baseline']['mean_relative_l2_error']:.4f} |\n"
    )

    learned_r2 = metrics_block["learned_W_T"]["r2"]
    all_baseline_r2 = {
        name: metrics_block[key]["r2"]
        for name, key in [
            ("persistence", "persistence_baseline"),
            ("mean", "mean_baseline"),
            ("shuffled_pairing", "random_pair_control"),
            ("pixel_statistics", "pixel_statistics_baseline"),
            ("random_encoder", "random_encoder_baseline"),
        ]
    }
    best_baseline_name = max(all_baseline_r2, key=all_baseline_r2.get)
    best_baseline_r2 = all_baseline_r2[best_baseline_name]
    beats_full_baseline_set = learned_r2 > best_baseline_r2
    pixel_comparable = all_baseline_r2["pixel_statistics"] >= learned_r2 - 0.05
    random_encoder_comparable = all_baseline_r2["random_encoder"] >= learned_r2 - 0.05

    scientific_result = (
        f"Task 10 consolidation: baselines.run_all_baselines now provides the complete DESIGN.md Sec 12 baseline "
        f"set (persistence, mean, shuffled-pairing, pixel-statistics encoder, randomly-initialized encoder) "
        f"through one entry point, used by experiments.geometric_consistency_lib.evaluate_transform (called by "
        f"Task 6/7's experiment scripts) and by Task 8's probe-baseline computation. Numerical parity with Task "
        f"6's previously recorded persistence/mean/shuffled-pairing/learned_W_T numbers: "
        f"{'CONFIRMED' if parity_ok else 'FAILED'} (max abs diff "
        f"{max(c['abs_diff'] for c in parity_checks.values()):.2e}, tolerance {PARITY_ATOL:.0e}).\n\n"
        f"Expanded five-baseline flagship comparison ({TRANSFORM_NAME}, the transform shared by Tasks 6/7, "
        f"identical {len(train_ids)}/{len(test_ids)} train/test scene split, identical rendered frames for all "
        f"three encoders): the primary pretrained encoder's learned W_T achieved R^2={learned_r2:.4f}, versus "
        + ", ".join(f"{name} R^2={r2:.4f}" for name, r2 in all_baseline_r2.items())
        + f". The best baseline overall is '{best_baseline_name}' (R^2={best_baseline_r2:.4f}); the primary "
        f"representation {'exceeds' if beats_full_baseline_set else 'does NOT exceed'} the full DESIGN.md Sec 12 "
        f"baseline set here.\n\n"
        f"Required scientific-QA finding (Task 10's 'Required scientific-validity tests': explicitly asking "
        f"whether a trivial pixel-statistics representation can match the primary encoder): the pixel-statistics "
        f"baseline scores R^2={all_baseline_r2['pixel_statistics']:.4f} against the primary encoder's "
        f"R^2={learned_r2:.4f} -- "
        + (
            "COMPARABLE (within 0.05 R^2), a major negative finding: this trivial, non-learned pixel-grid "
            "representation performs about as well as the frozen pretrained V-JEPA 2 representation on this "
            "task, undercutting any claim that Task 6/7's apparent geometric-consistency signal is specific to "
            "the learned representation."
            if pixel_comparable
            else "NOT comparable -- the primary representation clears the pixel-statistics floor on this metric."
        )
        + f" Similarly, the randomly-initialized (matched-architecture, untrained) encoder scores "
        f"R^2={all_baseline_r2['random_encoder']:.4f} -- "
        + (
            "COMPARABLE to the pretrained encoder, a major negative finding: the architecture alone (no "
            "self-supervised pretraining) accounts for essentially all of the measured effect here, isolating "
            "pretraining's contribution as negligible for this transform/metric (DESIGN.md Sec 12 item 4)."
            if random_encoder_comparable
            else "NOT comparable -- pretraining measurably contributes beyond the bare architecture here."
        )
        + " As with every Task 6-9 result on this transform, learned_W_T's R^2 is itself negative and below "
        "several trivial baselines, so this comparison should be read primarily as a baseline-completeness/"
        "framework-validity result, not as evidence of strong geometric consistency (see Task 6/7's own "
        "scientific_result fields, unchanged by this task)."
    )

    out_dir = Path("experiments/baselines_task10")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text(
        "# Task 10 -- baseline framework: expanded flagship comparison\n\n"
        f"- transform: {TRANSFORM_NAME} (Task 6/7's flagship)\n"
        f"- scenes: {len(scenes)} (train={len(train_ids)}, test={len(test_ids)})\n"
        f"- primary encoder pretrained: {primary_encoder_pretrained}, frozen verified: {primary_frozen_verified}\n"
        f"- random-encoder frozen verified: {random_frozen_verified}\n"
        f"- numerical parity with state/task_06_result.json: {'CONFIRMED' if parity_ok else 'FAILED'}\n\n"
        "## Five-baseline comparison\n\n" + comparison_table + "\n## Parity check detail\n\n"
        + json.dumps(parity_checks, indent=2) + "\n"
    )

    result = {
        "task": 10,
        "implementation_status": "COMPLETE",
        "scientific_result": scientific_result,
        "protected_files_justification": (
            "None -- no Tasks 1-5 file (generation/, transforms/, encoders/, DESIGN.md, "
            "IMPLEMENTATION_NOTES.md, configs/config.py) was modified. This task's consolidation lives in "
            "baselines/run_all_baselines.py (new), baselines/__init__.py (new exports), "
            "experiments/geometric_consistency_lib.py, experiments/task8_physical_state.py, and this module -- "
            "all Task 6-10 files."
        ),
        "transform": TRANSFORM_NAME,
        "reused_from_task6": {
            "config_path": cfg.task6_config_path,
            "render_dir": str(render_dir),
            "result_path": cfg.task6_result_path,
        },
        "dataset": {
            "num_scenes": len(scenes),
            "train_scene_ids": train_ids,
            "test_scene_ids": test_ids,
            "train_fraction": task6_cfg.train_fraction,
            "base_seed": task6_cfg.base_seed,
        },
        "encoders": {
            "primary": {
                "name": "VJEPAEncoder",
                "checkpoint": primary_encoder.checkpoint,
                "pretrained": primary_encoder_pretrained,
                "frozen": primary_frozen_verified,
                "pooling": "mean_pool",
            },
            "pixel_statistics": {
                "name": "PixelStatisticsBaseline",
                "grid_size": pixel_encoder.grid_size,
                "output_dim": pixel_encoder.output_dim,
                "input": "rendered RGB frames only (encoders/pixel_baseline.py) -- no SceneState/ground-truth access",
            },
            "random_encoder": {
                "name": "VJEPAEncoder",
                "checkpoint": random_encoder.checkpoint,
                "pretrained": bool(getattr(random_encoder, "pretrained", False)),
                "frozen": random_frozen_verified,
                "pooling": "mean_pool",
                "note": "matched architecture (same VITL16_CONFIG_KWARGS), seeded random weights, no network access",
            },
        },
        "fitting": {"method": "ridge", "alpha": task6_cfg.ridge_alpha},
        "metrics": metrics_block,
        "baseline_r2_comparison": all_baseline_r2,
        "numerical_parity_with_task6": {"passed": parity_ok, "tolerance": PARITY_ATOL, "checks": parity_checks},
        "scientific_qa": {
            "pixel_statistics_comparable_to_primary": pixel_comparable,
            "random_encoder_comparable_to_primary": random_encoder_comparable,
            "primary_beats_full_baseline_set": beats_full_baseline_set,
            "note": (
                "'comparable' means within 0.05 R^2 of the primary encoder's learned_W_T R^2 -- a fixed "
                "threshold decided before computing any of these numbers, matching Task 6/7/9's own practice of "
                "documenting thresholds in advance rather than after seeing results."
            ),
        },
        "software": {
            "run_all_baselines_module": "baselines/run_all_baselines.py",
            "baseline_names": list(FULL_BASELINE_NAMES),
            "refactored_call_sites": [
                "experiments/geometric_consistency_lib.py:evaluate_transform (Task 6/7's shared library)",
                "experiments/task8_physical_state.py:evaluate_variable (via baselines.run_all_baselines.run_probe_baselines)",
            ],
            "task9_note": (
                "Task 9 (experiments/task9_appearance_invariance.py) does not fit any map and never called "
                "persistence/mean/shuffled-pairing baseline logic in the first place (it computes direct "
                "Z-vs-Z' invariance over the full scene set, explicitly documented as NOT reusing "
                "persistence_baseline numbers) -- there is nothing in Task 9 for this refactor to consolidate."
            ),
        },
        "tests": {"passed": 0, "failed": 0},
        "artifacts": [str(report_path)],
        "config": asdict(cfg),
        "seed": task6_cfg.base_seed,
        "software_versions": gclib.software_versions(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the Task 10 baseline-framework experiment.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--skip_tests", action="store_true", help="Skip running the existing test suite before writing the result")
    args = parser.parse_args()

    cfg = load_task10_config(args.config)
    result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = gclib.run_existing_test_suite(repo_root)

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Task 10 result written to {result_path}")
    print(json.dumps(result["baseline_r2_comparison"], indent=2))


if __name__ == "__main__":
    main()
