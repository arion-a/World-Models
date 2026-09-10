"""Task 7: multiple geometric transformations + appearance controls.

    python -m experiments.task7_geometric_consistency --config configs/experiments/task7_geometric_consistency.yaml

Pipeline (see tasks/07_geometric_consistency.md and
research/CANONICAL_RESEARCH_PROTOCOL.md's "TASK 7" section, which this
module implements verbatim -- nothing here invents a new protocol):

    Reuse Task 6's scene-sampling convention (>=40 scenes) and identical
    scene-level train/test split, decided ONCE, BEFORE any rendering,
    and reused verbatim across all six transforms
    -> for each transform in transforms.scene_transform.GEOMETRIC_TRANSFORMS
       + CONTROL_TRANSFORMS: render every scene's original/transformed
       pair (transforms.pairs.generate_pair), encode both sides with the
       SAME frozen VJEPAEncoder instance, mean_pool to a (1024,) vector,
       fit W_T on TRAIN only, evaluate W_T + the three required controls
       (persistence, mean, shuffled-pairing) on the identical TEST split
    -> write state/task_07_result.json with a `results: {transform_name:
       {...}}` per-transform breakdown, a 6-transform comparison table,
       and an explicit visual-confound check (pixel-level magnitude of
       change vs. representation-level score, per transform).

This module deliberately does not modify generation/, transforms/,
encoders/, representations/, probes/, metrics/, or baselines/ -- it only
calls their existing public functions (and experiments/task6_camera_rotation.py's
sibling library, experiments/geometric_consistency_lib.py) per Task 7's
"Relationship to previous tasks" section.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

import experiments.geometric_consistency_lib as gclib
from generation.scene import SceneState
from transforms.scene_transform import GEOMETRIC_TRANSFORMS, TransformConfig

DEFAULT_RESULT_PATH = "state/task_07_result.json"
TASK6_RESULT_PATH = "state/task_06_result.json"
# Same scene-count discipline as Task 6 (tasks/07_geometric_consistency.md's
# "Dataset requirements": >=40 scenes) -- a module constant, not a
# literal, for the same reason as task6_camera_rotation.py's MIN_SCENES.
MIN_SCENES = 40


@dataclass
class Task7Config:
    # Identical defaults to configs/experiments/task6_camera_rotation.yaml's
    # (num_scenes, train_fraction, base_seed, num_objects_min/max) so that,
    # given the identical sampling code (experiments.geometric_consistency_lib.sample_scenes),
    # this reproduces Task 6's EXACT scene set -- "The same scene set used
    # in Task 6" (tasks/07_geometric_consistency.md's Inputs), not a fresh one.
    num_scenes: int = 40
    train_fraction: float = 0.8
    base_seed: int = 0
    output_dir: str = "experiments/geometric_consistency"
    result_path: str = DEFAULT_RESULT_PATH
    task6_result_path: str = TASK6_RESULT_PATH
    resolution: int = 128
    num_frames: int = 4
    fps: float = 4.0
    ridge_alpha: float = 10.0
    shuffled_pairing_seed: int = 0
    num_objects_min: int = 1
    num_objects_max: int = 3
    pretrained: bool = True
    checkpoint: str | None = None  # None -> encoders.vjepa.DEFAULT_CHECKPOINT
    device: str | None = None
    fallback_seed: int = 0


def load_task7_config(path: str | Path | None) -> Task7Config:
    cfg = Task7Config()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


# --- 1-2. scene sampling + the ONE scene-level train/test split shared -----
#          across all six transforms (no bpy, no torch) ---------------------


def sample_scenes(cfg: Task7Config) -> list[SceneState]:
    return gclib.sample_scenes(cfg.num_scenes, cfg.base_seed, cfg.num_objects_min, cfg.num_objects_max)


def assign_split(scene_ids: list[str], train_fraction: float, base_seed: int) -> dict[str, str]:
    return gclib.assign_split(scene_ids, train_fraction, base_seed)


def transform_config_for(transform_name: str) -> TransformConfig:
    """The magnitude configuration used for `transform_name`.

    Per tasks/07_geometric_consistency.md's Inputs section ("transform
    magnitudes drawn from TransformConfig's existing ranges unless a
    specific fixed magnitude is scientifically motivated and recorded"):
    every one of the six transforms uses TransformConfig()'s *default,
    unmodified ranges* here, uniformly -- not Task 6's fixed 30-degree
    azimuth for camera_rotation. This is a deliberate, recorded protocol
    choice (not a silent deviation from Task 6): fixing a magnitude for
    only one of the six transforms, while every other transform draws
    from its full sampled range, would itself be a transform-specific
    methodology change and would confound "transform type" with "was
    this transform's magnitude fixed or not" when comparing six results
    side by side -- exactly what tasks/07_geometric_consistency.md's
    Mathematical formulation requires avoiding ("results are comparable
    transform-to-transform, not confounded"). See PROTOCOL_NOTES below
    and this task's result JSON's "protocol_notes" field for the
    consequence: camera_rotation's Task 7 numbers are not expected to
    numerically reproduce Task 6's (Global Invariant 27).
    """
    return TransformConfig()


PROTOCOL_NOTES = [
    (
        "Task 6 fixed camera_rotation's azimuth magnitude at exactly 30 degrees "
        "(configs/experiments/task6_camera_rotation.yaml). Task 7 instead uses "
        "TransformConfig()'s default azimuth RANGE (10-35 degrees) for camera_rotation, "
        "uniformly with every other transform's default range, so that no single "
        "transform among the six is privileged with a fixed magnitude while the "
        "others are not -- see transform_config_for()'s docstring. As a direct "
        "consequence, this task's camera_rotation numbers are NOT expected to "
        "numerically reproduce state/task_06_result.json's recorded camera_rotation "
        "numbers exactly, even though the same >=40 scenes and the same scene-level "
        "train/test split are reused verbatim. This is an intentional, recorded "
        "protocol difference (Global Invariant 27), not a silent contradiction of "
        "Task 6's own result, which is left untouched on disk."
    )
]


# --- per-transform pipeline: render -> verify -> encode -> fit -> evaluate --


def run_single_transform(
    scenes: list[SceneState], split: dict[str, str], transform_name: str, cfg: Task7Config, encoder
) -> dict:
    out_dir = Path(cfg.output_dir) / transform_name
    transform_cfg = transform_config_for(transform_name)

    gclib.render_transform_pairs(
        scenes, transform_name, transform_cfg, out_dir, cfg.num_frames, cfg.fps, cfg.resolution
    )

    reps = gclib.encode_all_pairs(scenes, out_dir, encoder)
    train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test = gclib.build_arrays(scenes, split, reps)

    metrics_block = gclib.evaluate_transform(
        transform_name, Z_train, Zp_train, Z_test, Zp_test, cfg.ridge_alpha, cfg.shuffled_pairing_seed
    )
    visual_confound = gclib.pixel_diff_stats(scenes, out_dir)

    manifest = {
        "transform": transform_name,
        "num_scenes": len(scenes),
        "base_seed": cfg.base_seed,
        "train_fraction": cfg.train_fraction,
        "transform_config": asdict(transform_cfg),
        "scenes": [{"scene_id": s.scene_id, "seed": s.seed, "split": split[s.scene_id]} for s in scenes],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "transform_name": transform_name,
        "is_geometric": transform_name in GEOMETRIC_TRANSFORMS,
        "train_scene_ids": train_ids,
        "test_scene_ids": test_ids,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "transform_config": asdict(transform_cfg),
        **metrics_block,
        "visual_confound": visual_confound,
        "manifest_path": str(out_dir / "manifest.json"),
    }


# --- cross-transform checks required by tasks/07_geometric_consistency.md --


def _check_shared_split(results: dict[str, dict]) -> None:
    """Required research-alignment / leakage check: every transform's
    entry must record the IDENTICAL train_scene_ids/test_scene_ids --
    the split was decided once, before any rendering, and reused
    verbatim (tasks/07_geometric_consistency.md's Leakage checks)."""
    names = list(results.keys())
    ref_train = set(results[names[0]]["train_scene_ids"])
    ref_test = set(results[names[0]]["test_scene_ids"])
    for name in names[1:]:
        train = set(results[name]["train_scene_ids"])
        test = set(results[name]["test_scene_ids"])
        if train != ref_train or test != ref_test:
            raise ValueError(
                f"Shared-split violation: transform '{name}' does not use the identical "
                f"train/test scene split as '{names[0]}' -- this would confound transform "
                f"effect with sample effect (tasks/07_geometric_consistency.md's failure conditions)."
            )
        if not train.isdisjoint(test):
            raise ValueError(f"train/test scene leakage detected for transform '{name}'")


def _confound_analysis(results: dict[str, dict]) -> dict:
    """Required scientific-QA step (tasks/07_geometric_consistency.md's
    Required scientific-validity tests): investigate, explicitly,
    whether cross-transform differences in the learned-W_T R^2 could be
    explained by differences in average pixel-level change magnitude
    rather than by geometry itself. Computed purely from rendered RGB
    (gclib.pixel_diff_stats), never from the encoder's representation,
    and correlated here against each transform's learned_W_T R^2.
    """
    names = list(results.keys())
    pixel_diffs = np.array([results[n]["visual_confound"]["mean_abs_pixel_diff"] for n in names])
    r2s = np.array([results[n]["learned_W_T"]["r2"] for n in names])

    correlation = None
    if len(names) >= 2 and np.std(pixel_diffs) > 0 and np.std(r2s) > 0:
        correlation = float(np.corrcoef(pixel_diffs, r2s)[0, 1])

    return {
        "pixel_diff_by_transform": {n: results[n]["visual_confound"]["mean_abs_pixel_diff"] for n in names},
        "learned_w_t_r2_by_transform": {n: results[n]["learned_W_T"]["r2"] for n in names},
        "pearson_correlation_pixel_diff_vs_learned_r2": correlation,
        "note": (
            "Pearson correlation, across the six transforms, between each transform's mean absolute "
            "per-pixel RGB change (computed directly from rendered clips, not from Z) and its learned "
            "W_T test R^2. A strong positive correlation here would mean that differences in R^2 across "
            "transforms could plausibly be explained by which transform simply perturbs more pixels, "
            "rather than by anything specific to geometry -- reported honestly regardless of sign or "
            "magnitude, per tasks/07_geometric_consistency.md's required scientific-QA step; not treated "
            "as a target to explain away."
        ),
    }


def _task6_reproduction_check(results: dict[str, dict], task6_result_path: str | Path) -> dict:
    """Acceptance criterion 4 / Global Invariant 27: Task 6's own
    recorded result must not be silently contradicted -- compare Task
    7's camera_rotation numbers against state/task_06_result.json's
    recorded ones and record the (expected, per PROTOCOL_NOTES) numeric
    difference explicitly rather than silently accepting or ignoring it.
    Task 6's own file is never read for anything but this comparison,
    and is never written to.
    """
    task6_path = Path(task6_result_path)
    if not task6_path.exists():
        return {"task6_result_found": False, "note": "state/task_06_result.json not found; no comparison performed."}

    task6 = json.loads(task6_path.read_text())
    task6_r2 = task6.get("metrics", {}).get("learned_W_T", {}).get("r2")
    task7_r2 = results.get("camera_rotation", {}).get("learned_W_T", {}).get("r2")
    same_scenes = None
    if "dataset" in task6:
        same_scenes = set(task6["dataset"].get("train_scene_ids", [])) == set(
            results.get("camera_rotation", {}).get("train_scene_ids", [])
        ) and set(task6["dataset"].get("test_scene_ids", [])) == set(
            results.get("camera_rotation", {}).get("test_scene_ids", [])
        )
    return {
        "task6_result_found": True,
        "task6_camera_rotation_r2": task6_r2,
        "task7_camera_rotation_r2": task7_r2,
        "difference": (None if task6_r2 is None or task7_r2 is None else task7_r2 - task6_r2),
        "identical_scene_split_as_task6": same_scenes,
        "note": (
            "A numeric difference here is EXPECTED and intentional -- see this result's "
            "'protocol_notes' field and transform_config_for()'s docstring (Task 7 uses "
            "camera_rotation's default azimuth RANGE for cross-transform comparability, not "
            "Task 6's fixed 30-degree magnitude). Task 6's own state/task_06_result.json is "
            "left unmodified on disk."
        ),
    }


def _scientific_result_text(results: dict[str, dict]) -> str:
    lines = []
    for name, r in results.items():
        learned_r2 = r["learned_W_T"]["r2"]
        best_control_r2 = max(r["persistence_baseline"]["r2"], r["mean_baseline"]["r2"], r["random_pair_control"]["r2"])
        kind = "geometric" if r["is_geometric"] else "control"
        verdict = "exceeded" if learned_r2 > best_control_r2 else "did not exceed"
        lines.append(f"{name} ({kind}): learned W_T R^2={learned_r2:.3f} {verdict} best control R^2={best_control_r2:.3f}")

    geometric_positive = [
        n for n, r in results.items() if r["is_geometric"] and r["learned_W_T"]["r2"] > max(
            r["persistence_baseline"]["r2"], r["mean_baseline"]["r2"], r["random_pair_control"]["r2"]
        )
    ]
    control_positive = [
        n for n, r in results.items() if not r["is_geometric"] and r["learned_W_T"]["r2"] > max(
            r["persistence_baseline"]["r2"], r["mean_baseline"]["r2"], r["random_pair_control"]["r2"]
        )
    ]

    summary = (
        f"Across the four geometric transforms (camera_rotation, camera_translation, object_rotation, "
        f"object_translation) and two appearance-only controls (lighting_change, texture_change), evaluated "
        f"under the identical scene set, scene-level train/test split, encoder, pooling, fitting method, and "
        f"three required controls: {len(geometric_positive)}/4 geometric transforms showed the learned W_T "
        f"exceeding the best of the three baselines on held-out test scenes ({', '.join(geometric_positive) or 'none'}), "
        f"and {len(control_positive)}/2 appearance-only controls did the same ({', '.join(control_positive) or 'none'}). "
        + (
            "This is broadly consistent with the hypothesis that geometric transforms are more linearly "
            "predictable in Z than non-geometric appearance changes, under this frozen encoder, this scene "
            "distribution, and these transform-magnitude ranges -- it does not establish 3D scene comprehension "
            "or a general claim about geometric transformations in video representations (DESIGN.md Sec 13; "
            "tasks/07_geometric_consistency.md's interpretation limits)."
            if len(geometric_positive) > len(control_positive)
            else "This does NOT show a clean separation between geometric transforms and appearance-only "
            "controls under this protocol -- a meaningful negative/mixed finding (weak evidence that Z is not "
            "cleanly distinguishing geometry from any change), reported plainly rather than adjusted until a "
            "cleaner-looking separation appears (tasks/07_geometric_consistency.md's interpretation limits)."
        )
    )
    return summary + "\n\nPer-transform detail:\n" + "\n".join(f"- {line}" for line in lines)


def _comparison_table_markdown(results: dict[str, dict]) -> str:
    header = "| transform | kind | learned W_T R^2 | persistence R^2 | mean R^2 | random-pair R^2 |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = []
    for name, r in results.items():
        kind = "geometric" if r["is_geometric"] else "control"
        rows.append(
            f"| {name} | {kind} | {r['learned_W_T']['r2']:.4f} | {r['persistence_baseline']['r2']:.4f} | "
            f"{r['mean_baseline']['r2']:.4f} | {r['random_pair_control']['r2']:.4f} |"
        )
    return header + "\n".join(rows) + "\n"


def run_experiment(cfg: Task7Config) -> dict:
    scenes = sample_scenes(cfg)
    if len(scenes) < MIN_SCENES:
        raise ValueError(f"Task 7 requires >={MIN_SCENES} scenes, got {len(scenes)}")
    scene_ids = [s.scene_id for s in scenes]

    # Split decided ONCE, BEFORE any rendering, and reused verbatim
    # across all six transforms (research/RESEARCH_INVARIANTS.md
    # invariants 7, 8; tasks/07_geometric_consistency.md's Train/test protocol).
    split = assign_split(scene_ids, cfg.train_fraction, cfg.base_seed)

    # One frozen encoder instance, reused for all six transforms (not
    # rebuilt/reloaded per transform) -- also lets frozen-ness be
    # verified once across the entire six-transform run.
    encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))
    params_before = gclib.snapshot_params(encoder.model)

    results: dict[str, dict] = {}
    for transform_name in gclib.ALL_TRANSFORMS:
        results[transform_name] = run_single_transform(scenes, split, transform_name, cfg, encoder)

    frozen_verified = gclib.params_unchanged(params_before, encoder.model)

    _check_shared_split(results)

    confound_analysis = _confound_analysis(results)
    task6_check = _task6_reproduction_check(results, cfg.task6_result_path)

    comparison_table = _comparison_table_markdown(results)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text(
        "# Task 7 -- multiple geometric transformations vs. appearance controls\n\n"
        f"- scenes: {len(scenes)} (train={len(results[gclib.ALL_TRANSFORMS[0]]['train_scene_ids'])}, "
        f"test={len(results[gclib.ALL_TRANSFORMS[0]]['test_scene_ids'])})\n"
        f"- encoder pretrained: {encoder_pretrained}\n"
        f"- frozen verified (params bit-identical before/after all six transforms' encoding): {frozen_verified}\n\n"
        "## R^2 comparison (learned W_T vs. all three baselines)\n\n"
        + comparison_table
        + "\n## Visual-confound check\n\n"
        f"Pearson correlation between mean absolute pixel diff and learned W_T R^2 across the six transforms: "
        f"{confound_analysis['pearson_correlation_pixel_diff_vs_learned_r2']}\n\n"
        "## Task 6 reproduction check\n\n"
        f"{json.dumps(task6_check, indent=2)}\n"
    )

    result = {
        "task": 7,
        "implementation_status": "COMPLETE",
        "scientific_result": _scientific_result_text(results),
        "transforms_evaluated": list(gclib.ALL_TRANSFORMS),
        "dataset": {
            "num_scenes": len(scenes),
            "train_scene_ids": results[gclib.ALL_TRANSFORMS[0]]["train_scene_ids"],
            "test_scene_ids": results[gclib.ALL_TRANSFORMS[0]]["test_scene_ids"],
            "train_fraction": cfg.train_fraction,
            "base_seed": cfg.base_seed,
            "reused_task6_scene_sampling_convention": True,
        },
        "encoder": {
            "name": "VJEPAEncoder",
            "checkpoint": encoder.checkpoint,
            "pretrained": encoder_pretrained,
            "frozen": frozen_verified,
            "pooling": "mean_pool",
        },
        "fitting": {"method": "ridge", "alpha": cfg.ridge_alpha},
        "results": results,
        # Alias of "results": tasks/07_geometric_consistency.md's own
        # required schema field is "results" (a per-transform breakdown),
        # but orchestrator/qa.py's generic SPLIT_BEARING_TASKS scientific-
        # validity layer looks for a top-level "metrics" block containing
        # a baseline/control comparison for any experiment-type task --
        # both are satisfied by the same underlying data rather than
        # duplicating it under two different shapes.
        "metrics": results,
        "comparison_table_markdown": comparison_table,
        "confound_analysis": confound_analysis,
        "task6_reproduction_check": task6_check,
        "protocol_notes": PROTOCOL_NOTES,
        "tests": {"passed": 0, "failed": 0},
        "artifacts": [str(report_path)] + [results[t]["manifest_path"] for t in gclib.ALL_TRANSFORMS],
        "config": asdict(cfg),
        "seed": cfg.base_seed,
        "software_versions": gclib.software_versions(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the Task 7 multi-transform geometric consistency experiment."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument(
        "--skip_tests", action="store_true", help="Skip running the existing test suite before writing the result"
    )
    args = parser.parse_args()

    cfg = load_task7_config(args.config)
    result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = gclib.run_existing_test_suite(repo_root)

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Task 7 result written to {result_path}")
    print(result["comparison_table_markdown"])


if __name__ == "__main__":
    main()
