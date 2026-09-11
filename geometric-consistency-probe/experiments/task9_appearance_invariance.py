"""Task 9: appearance invariance controls.

    python -m experiments.task9_appearance_invariance --config configs/experiments/task9_appearance_invariance.yaml

Pipeline (see tasks/09_appearance_invariance.md and
research/CANONICAL_RESEARCH_PROTOCOL.md's "TASK 9" section, which this
module implements verbatim):

    Reuse Task 7's already-rendered (Z, Z') pairs, unmodified on disk, for
    the two appearance controls (lighting_change, texture_change) and,
    for direct comparison, the four geometric transforms
    -> for each of those six transforms: re-verify the reused pair's own
       ground truth (for the two controls: explicitly re-check that only
       appearance fields changed, not assumed from Task 7's own check),
       re-encode both sides with the SAME frozen VJEPAEncoder, and compute
       invariance DIRECTLY (metrics.invariance.evaluate_invariance: mean
       cosine similarity + mean relative L2 error between Z and Z', no
       map fit at all) over every scene -- deliberately a different
       computation from Task 6/7/10's "identity baseline" (equivariance
       with rho(T)=Identity), which is R^2-based and test-split-only.
    -> render + verify + encode a fresh "null-transform" sanity check
       (the same SceneState rendered twice, T=identity) and assert its
       invariance score clears a documented-in-advance near-1.0
       threshold; a failure here is a pipeline-determinism bug, not a
       scientific finding, and blocks this task.
    -> write state/task_09_result.json with per-transform invariance
       metrics, the null-transform sanity check, a comparison table, and
       the required scientific-QA investigation into whether Tasks 6-7's
       apparent geometric sensitivity could instead be explained by
       appearance/rendering-pipeline artifacts.

This module deliberately does not modify generation/, transforms/,
encoders/, representations/, probes/, metrics/, or baselines/, and reuses
(without altering) Task 7's own rendered outputs under
experiments/geometric_consistency/ -- it only calls public functions from
those modules and from experiments/geometric_consistency_lib.py (Task 7's
shared library).
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
from metrics.invariance import evaluate_invariance
from transforms.scene_transform import CONTROL_TRANSFORMS, GEOMETRIC_TRANSFORMS

DEFAULT_RESULT_PATH = "state/task_09_result.json"
# Documented BEFORE any invariance number is computed (tasks/09_appearance_invariance.md's
# "Prohibited shortcuts": never tune this after seeing results). Rendering
# is bit-exact deterministic given an identical SceneState+Trajectory
# (tests/test_bpy_renderer.py::test_render_trajectory_is_reproducible), and
# the encoder is frozen/deterministic in eval mode, so two independent
# renders of literally the same, untransformed scene are expected to
# produce representations that are for all practical purposes identical --
# "near-perfect", not exact, to tolerate ordinary floating-point
# non-associativity across two separate forward passes.
NULL_TRANSFORM_COSINE_THRESHOLD = 0.999
NULL_TRANSFORM_RELATIVE_L2_THRESHOLD = 0.05


@dataclass
class Task9Config:
    # Must match Task 7's sampling/scene parameters exactly so
    # "reused_output_dir" actually contains this exact scene set --
    # verified at run time (_check_reused_scene_set_matches), not assumed.
    num_scenes: int = 40
    train_fraction: float = 0.8
    base_seed: int = 0
    num_objects_min: int = 1
    num_objects_max: int = 3
    reused_output_dir: str = "experiments/geometric_consistency"
    null_transform_output_dir: str = "experiments/appearance_invariance/null_transform"
    result_path: str = DEFAULT_RESULT_PATH
    resolution: int = 128
    num_frames: int = 4
    fps: float = 4.0
    pretrained: bool = True
    checkpoint: str | None = None  # None -> encoders.vjepa.DEFAULT_CHECKPOINT
    device: str | None = None
    fallback_seed: int = 0


def load_task9_config(path: str | Path | None) -> Task9Config:
    cfg = Task9Config()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


# --- reuse Task 7's scene set + its already-rendered pairs, unmodified -----


def sample_scenes(cfg: Task9Config) -> list[SceneState]:
    return gclib.sample_scenes(cfg.num_scenes, cfg.base_seed, cfg.num_objects_min, cfg.num_objects_max)


def _check_reused_scene_set_matches(scenes: list[SceneState], transform_name: str, out_dir: Path) -> None:
    """Task 9's Dataset requirements: "Same scene set as Task 7 where
    reused" -- checked against Task 7's own manifest.json, not assumed
    from matching config values alone.
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- Task 7's rendered pairs for '{transform_name}' are required "
            "for Task 9 to reuse (tasks/09_appearance_invariance.md's Inputs)."
        )
    manifest = json.loads(manifest_path.read_text())
    manifest_ids = {s["scene_id"] for s in manifest["scenes"]}
    this_ids = {s.scene_id for s in scenes}
    if manifest_ids != this_ids:
        raise ValueError(
            f"Task 9's sampled scene set does not match Task 7's reused manifest for '{transform_name}': "
            f"{sorted(this_ids - manifest_ids)} missing from manifest, {sorted(manifest_ids - this_ids)} extra."
        )


# --- per-transform invariance (no fit) --------------------------------------


def compute_invariance_for_transform(scenes: list[SceneState], out_dir: Path, transform_name: str, encoder) -> dict:
    reps = gclib.encode_all_pairs(scenes, out_dir, encoder)
    scene_ids, Z, Zp = gclib.build_full_arrays(scenes, reps)
    result = evaluate_invariance(transform_name, Z, Zp)
    return {
        "transform_name": transform_name,
        "is_geometric": transform_name in GEOMETRIC_TRANSFORMS,
        "is_appearance_control": transform_name in CONTROL_TRANSFORMS,
        "n": result.n,
        "scene_ids": scene_ids,
        "mean_cosine_similarity": result.mean_cosine_similarity,
        "mean_relative_l2_error": result.mean_relative_l2_error,
    }


def run_null_transform_sanity_check(scenes: list[SceneState], out_dir: Path, cfg: Task9Config, encoder) -> dict:
    gclib.render_null_transform_pairs(scenes, out_dir, cfg.num_frames, cfg.fps, cfg.resolution)
    for scene in scenes:
        gclib.verify_null_transform_ground_truth(out_dir / scene.scene_id)

    entry = compute_invariance_for_transform(scenes, out_dir, "null_transform", encoder)
    passed = (
        entry["mean_cosine_similarity"] >= NULL_TRANSFORM_COSINE_THRESHOLD
        and entry["mean_relative_l2_error"] <= NULL_TRANSFORM_RELATIVE_L2_THRESHOLD
    )
    entry["is_geometric"] = False
    entry["is_appearance_control"] = False
    entry["is_null_sanity_check"] = True
    entry["cosine_threshold"] = NULL_TRANSFORM_COSINE_THRESHOLD
    entry["relative_l2_threshold"] = NULL_TRANSFORM_RELATIVE_L2_THRESHOLD
    entry["passed_threshold"] = passed
    return entry


# --- required scientific-QA step: could appearance/pipeline artifacts ------
#     explain Tasks 6-7's apparent geometric sensitivity? -------------------


def confound_investigation(results: dict[str, dict]) -> dict:
    """tasks/09_appearance_invariance.md's required scientific-validity
    test: "Investigate, explicitly, whether apparent geometric sensitivity
    in Tasks 6-7 could actually be caused by lighting, texture, material,
    color, or rendering-pipeline artifacts rather than genuine geometric
    change." Answered directly from this task's own invariance numbers:
    if appearance controls are (markedly) MORE invariant than every
    geometric transform, that is evidence against the "it's just
    appearance/pipeline artifacts" explanation; if appearance controls
    move Z about as much as (or more than) the geometric transforms, that
    undercuts it and must be reported prominently, not minimized (per
    this task's "What a negative result means").
    """
    appearance = {n: r for n, r in results.items() if r.get("is_appearance_control")}
    geometric = {n: r for n, r in results.items() if r.get("is_geometric")}

    appearance_cosines = {n: r["mean_cosine_similarity"] for n, r in appearance.items()}
    geometric_cosines = {n: r["mean_cosine_similarity"] for n, r in geometric.items()}
    appearance_l2 = {n: r["mean_relative_l2_error"] for n, r in appearance.items()}
    geometric_l2 = {n: r["mean_relative_l2_error"] for n, r in geometric.items()}

    min_appearance_cosine = min(appearance_cosines.values())
    max_geometric_cosine = max(geometric_cosines.values())
    max_appearance_l2 = max(appearance_l2.values())
    min_geometric_l2 = min(geometric_l2.values())

    clean_separation = min_appearance_cosine > max_geometric_cosine and max_appearance_l2 < min_geometric_l2

    if clean_separation:
        verdict = (
            "Clean separation: both appearance controls are MORE invariant (higher cosine "
            "similarity, lower relative L2 movement) than every one of the four geometric "
            "transforms. This is evidence AGAINST Tasks 6-7's apparent geometric sensitivity "
            "being explainable by appearance/lighting/texture/rendering-pipeline artifacts alone "
            "-- appearance-only changes to the identical scenes move Z markedly less."
        )
    else:
        verdict = (
            "No clean separation: at least one appearance control is at least as invariant-"
            "violating as at least one geometric transform (or vice-versa is not uniformly true). "
            "This is a meaningful negative finding that directly undercuts confidence in Tasks "
            "6-7's geometric interpretation and must be reported prominently, not minimized "
            "(tasks/09_appearance_invariance.md's 'What a negative result means')."
        )

    return {
        "appearance_cosine_similarity_by_transform": appearance_cosines,
        "geometric_cosine_similarity_by_transform": geometric_cosines,
        "appearance_relative_l2_error_by_transform": appearance_l2,
        "geometric_relative_l2_error_by_transform": geometric_l2,
        "min_appearance_cosine_similarity": min_appearance_cosine,
        "max_geometric_cosine_similarity": max_geometric_cosine,
        "max_appearance_relative_l2_error": max_appearance_l2,
        "min_geometric_relative_l2_error": min_geometric_l2,
        "clean_separation": clean_separation,
        "verdict": verdict,
    }


def _scientific_result_text(results: dict[str, dict], null_check: dict, confound: dict) -> str:
    lines = []
    for name in list(CONTROL_TRANSFORMS) + list(GEOMETRIC_TRANSFORMS):
        r = results[name]
        kind = "appearance control" if r["is_appearance_control"] else "geometric"
        lines.append(
            f"{name} ({kind}): mean_cosine_similarity={r['mean_cosine_similarity']:.4f}, "
            f"mean_relative_l2_error={r['mean_relative_l2_error']:.4f}"
        )

    summary = (
        f"Null-transform sanity check (same scene rendered twice, T=identity): "
        f"mean_cosine_similarity={null_check['mean_cosine_similarity']:.6f}, "
        f"mean_relative_l2_error={null_check['mean_relative_l2_error']:.6f}, "
        f"{'PASSED' if null_check['passed_threshold'] else 'FAILED'} its documented threshold "
        f"(cosine>={null_check['cosine_threshold']}, rel_l2<={null_check['relative_l2_threshold']}) -- "
        + (
            "the rendering+encoding pipeline is confirmed deterministic, so the appearance-control "
            "numbers below are not artifacts of pipeline noise.\n\n"
            if null_check["passed_threshold"]
            else "this indicates a PIPELINE DETERMINISM BUG, not a scientific result; see "
            "implementation_status.\n\n"
        )
        + f"{confound['verdict']}\n\nPer-transform invariance (direct Z vs Z' comparison, no map fit):\n"
        + "\n".join(f"- {line}" for line in lines)
    )
    return summary


def _comparison_table_markdown(results: dict[str, dict], null_check: dict) -> str:
    header = "| transform | kind | mean cosine similarity | mean relative L2 error |\n"
    header += "|---|---|---|---|\n"
    rows = []
    for name in list(CONTROL_TRANSFORMS) + list(GEOMETRIC_TRANSFORMS):
        r = results[name]
        kind = "appearance control" if r["is_appearance_control"] else "geometric"
        rows.append(f"| {name} | {kind} | {r['mean_cosine_similarity']:.4f} | {r['mean_relative_l2_error']:.4f} |")
    rows.append(
        f"| null_transform | sanity check | {null_check['mean_cosine_similarity']:.4f} | "
        f"{null_check['mean_relative_l2_error']:.4f} |"
    )
    return header + "\n".join(rows) + "\n"


RESEARCH_ALIGNMENT_NOTE = (
    "This task's per-transform numbers (mean_cosine_similarity, mean_relative_l2_error) are computed "
    "DIRECTLY between Z (original) and Z' (transformed) via metrics.invariance.evaluate_invariance, with "
    "NO map fit and over every scene (no train/test split, since nothing is fit here) -- per DESIGN.md Sec 7. "
    "This is a deliberately different computation from Tasks 6/7/10's 'identity baseline' (rho(T)=Identity), "
    "which is R^2-based, evaluated only on the held-out test split, and answers 'how well does the trivial "
    "map predict change' rather than 'how much change is there in the first place.' Task 7's own "
    "persistence_baseline numbers are not reused here and are not conflated with this task's invariance numbers."
)


def run_experiment(cfg: Task9Config) -> dict:
    scenes = sample_scenes(cfg)
    scene_ids = [s.scene_id for s in scenes]
    # Retained for cross-task consistency (tasks/09_appearance_invariance.md's
    # Train/test protocol: "the underlying scene set nonetheless retains its
    # scene-level split labeling"), even though nothing is fit in this task.
    split = gclib.assign_split(scene_ids, cfg.train_fraction, cfg.base_seed)

    encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))
    params_before = gclib.snapshot_params(encoder.model)

    reused_root = Path(cfg.reused_output_dir)
    all_six = list(CONTROL_TRANSFORMS) + list(GEOMETRIC_TRANSFORMS)

    appearance_equality_checks: dict[str, dict] = {}
    results: dict[str, dict] = {}
    for transform_name in all_six:
        out_dir = reused_root / transform_name
        _check_reused_scene_set_matches(scenes, transform_name, out_dir)
        if transform_name in CONTROL_TRANSFORMS:
            gclib.verify_appearance_physical_equality(scenes, out_dir, transform_name)
            appearance_equality_checks[transform_name] = {
                "verified": True,
                "note": "changed_variables re-checked directly from transformation.json against "
                "expected_changed_variables; confirmed camera pose, every object's pose, and object "
                "identity/count all unchanged.",
            }
        results[transform_name] = compute_invariance_for_transform(scenes, out_dir, transform_name, encoder)

    null_out_dir = Path(cfg.null_transform_output_dir)
    null_check = run_null_transform_sanity_check(scenes, null_out_dir, cfg, encoder)

    frozen_verified = gclib.params_unchanged(params_before, encoder.model)

    if not null_check["passed_threshold"]:
        return {
            "task": 9,
            "implementation_status": "BLOCKED",
            "blocker_category": "SCIENTIFIC_CONFLICT",
            "blocking_issue": (
                "The null-transform sanity check (same SceneState rendered twice, no transform applied) "
                f"scored mean_cosine_similarity={null_check['mean_cosine_similarity']!r}, "
                f"mean_relative_l2_error={null_check['mean_relative_l2_error']!r}, which does not clear the "
                f"documented-in-advance near-1.0 threshold (cosine>={NULL_TRANSFORM_COSINE_THRESHOLD}, "
                f"rel_l2<={NULL_TRANSFORM_RELATIVE_L2_THRESHOLD}). Per tasks/09_appearance_invariance.md's "
                "Experimental protocol step 2, this indicates a rendering/encoding pipeline determinism bug "
                "-- a software-correctness finding, not a valid scientific result -- and must block this task "
                "until fixed, rather than being reported as an appearance-invariance finding."
            ),
            "null_transform_sanity_check": null_check,
            "results": results,
        }

    confound = confound_investigation(results)
    comparison_table = _comparison_table_markdown(results, null_check)
    scientific_result = _scientific_result_text(results, null_check, confound)

    out_dir = Path(cfg.reused_output_dir).parent / "appearance_invariance"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text(
        "# Task 9 -- appearance invariance controls\n\n"
        f"- scenes: {len(scenes)}\n"
        f"- encoder pretrained: {encoder_pretrained}\n"
        f"- frozen verified (params bit-identical before/after all encoding): {frozen_verified}\n\n"
        "## Invariance comparison (direct Z vs Z', no map fit)\n\n"
        + comparison_table
        + "\n## Null-transform sanity check\n\n"
        f"{json.dumps({k: v for k, v in null_check.items() if k != 'scene_ids'}, indent=2)}\n\n"
        "## Geometric-vs-appearance confound investigation\n\n"
        f"{confound['verdict']}\n"
    )

    result = {
        "task": 9,
        "implementation_status": "COMPLETE",
        "scientific_result": scientific_result,
        "transforms_evaluated": all_six + ["null_transform"],
        "dataset": {
            "num_scenes": len(scenes),
            "all_scene_ids": scene_ids,
            "train_scene_ids": [sid for sid in scene_ids if split[sid] == "train"],
            "test_scene_ids": [sid for sid in scene_ids if split[sid] == "test"],
            "train_fraction": cfg.train_fraction,
            "base_seed": cfg.base_seed,
            "reused_task7_scene_sampling_convention": True,
            "note": "No train/test split is used for the invariance computation itself (nothing is fit in "
            "this task); the split is retained here only for cross-task consistency with Tasks 6-7.",
        },
        "encoder": {
            "name": "VJEPAEncoder",
            "checkpoint": encoder.checkpoint,
            "pretrained": encoder_pretrained,
            "frozen": frozen_verified,
            "pooling": "mean_pool",
        },
        "invariance_computation": "metrics.invariance.evaluate_invariance -- no map fit, computed over all scenes",
        "results": results,
        "metrics": results,
        "appearance_physical_equality_checks": appearance_equality_checks,
        "null_transform_sanity_check": null_check,
        "comparison_table_markdown": comparison_table,
        "geometric_vs_appearance_confound_investigation": confound,
        "research_alignment_note": RESEARCH_ALIGNMENT_NOTE,
        "reused_from_task7": {
            "output_dir": str(reused_root),
            "transforms_reused_unmodified": all_six,
            "note": "This task re-encodes Task 7's already-rendered original/transformed RGB clips "
            "(experiments/geometric_consistency/<transform>/<scene_id>/) unmodified; it does not re-render them.",
        },
        "tests": {"passed": 0, "failed": 0},
        "artifacts": [str(report_path)],
        "config": asdict(cfg),
        "seed": cfg.base_seed,
        "software_versions": gclib.software_versions(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the Task 9 appearance-invariance experiment.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument(
        "--skip_tests", action="store_true", help="Skip running the existing test suite before writing the result"
    )
    args = parser.parse_args()

    cfg = load_task9_config(args.config)
    result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = gclib.run_existing_test_suite(repo_root)

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Task 9 result written to {result_path}")
    if "comparison_table_markdown" in result:
        print(result["comparison_table_markdown"])


if __name__ == "__main__":
    main()
