"""Task 14 follow-up (Category A fix): bigger object_translation displacement.

A detectability/effect-size audit of Task 14's object_translation
counterfactual condition (`experiments/counterfactuals/object_translation/`,
`state/task_14_result.json`, target_displacement=0.4) found that condition
moved the frozen encoder's representation too little relative to natural
scene-to-scene variation to trust any positive discrimination result built
on it: the audit's detectability ratio

    ratio = E||Z_after - Z_before||^2 / E_{i != j}||Z_before_i - Z_before_j||^2

(`detectability_ratio()` below) was ~0.079 at target_displacement=0.4 --
reproduced directly from the committed `reps.npz` caches by this module's
own tests/dev run, matching the audit's "too small to see" reference point
(~0.08), well below the calibrated 0.3 threshold ("learned map wins
decisively" reference range is 0.30-0.43).

This module redesigns ONLY `target_displacement` (Task 14's single shared
magnitude-matching config field -- both `object_translation` and, per
scene, `object_rotation`'s solved angle are matched to it, see
`experiments.task14_counterfactuals`'s module docstring, so raising this
one field raises both conditions' achieved displacement in lockstep,
without breaking their magnitude-matching design) and reruns Task 14's
FULL protocol (all four rendered conditions, the discrimination probe, the
representation-norm-only control, and the fixed-variable leakage check) at
N=100 scenes, `base_seed=104` (disjoint from every base_seed already used
elsewhere in this project).

Two-phase design (Experimental protocol, mirrors
tasks/14_counterfactuals.md / this project's other Category-A follow-ups,
e.g. `experiments/task7_followup_object_translation_magnitude.py`):

    Phase 1 (`--stage pilot`): render+encode ONLY `object_translation`,
    for a small `pilot_num_scenes` (e.g. 8) scenes, at each of
    `candidate_displacements` (e.g. [1.0, 2.0]) -- each candidate in its
    own `output_dir` subfolder so no cache ever collides across
    candidates. For each candidate, `detectability_ratio()` is computed
    from the cached Z_before/Z_after, and `check_object_visibility()`
    re-verifies (from the rendered segmentation ground truth, not
    assumed) that the displaced object is still visible in-frame --
    "don't move the object out of camera view" is checked, not assumed.
    `choose_displacement()` then picks the SMALLEST candidate that clears
    BOTH the detectability threshold and the visibility check; if none
    does, it escalates to one larger value (2x the largest candidate
    tried) rather than silently accepting an unreliable result. Pilot
    results (every candidate, not just the winner) are written to
    `<output_dir>_pilot/pilot_result.json` for a full audit trail.

    Phase 2 (`--stage full`): with `target_displacement` in the loaded
    config already set to the phase-1 winner, runs Task 14's own
    `process_primary_scene_range`/`process_confound_scene_range`/
    `run_finalize` (imported unmodified from
    `experiments.task14_counterfactuals` -- this module does not
    reimplement rendering, encoding, probe-fitting, or the discrimination/
    leakage-check logic) over all `cfg.num_scenes` scenes, then adds this
    follow-up's own `object_translation_detectability_ratio` (the SAME
    audit metric, now at N=100, not just the N=pilot check) and
    `object_translation_visibility_check` on top of Task 14's standard
    result dict, and writes the combined result -- unconditionally, from
    THIS module's own `main()` -- to `cfg.result_path`
    (`state/task14_followup_bigger_displacement_result.json` by default),
    never to Task 14's own `state/task_14_result.json`.

This module deliberately does not modify
`experiments/task14_counterfactuals.py` (imported from, not edited) or any
of Task 14's own config/output/result files -- see this module's own
config, `configs/experiments/task14_followup_bigger_displacement.yaml`,
for the (disjoint) paths and seed this follow-up owns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.task14_counterfactuals import (
    CONDITIONS,
    CONFOUND_CONDITIONS,
    Task14Config,
    build_scenes,
    load_task14_config,
    process_confound_scene_range,
    process_primary_scene_range,
    run_finalize,
    scene_ids_for,
)

DEFAULT_CONFIG_PATH = "configs/experiments/task14_followup_bigger_displacement.yaml"

# The audit's calibrated threshold (background given to this follow-up,
# not re-derived here): below this, a positive discrimination/effect-size
# result built on this condition is not trustworthy. Reference points the
# audit also gave: ratio~=0.08 is "too small to see" (Task 14's original
# target_displacement=0.4 measured ~0.079, reproduced below);
# ratio in [0.30, 0.43] is "learned map wins decisively".
DETECTABILITY_THRESHOLD = 0.3
REFERENCE_TOO_SMALL_RATIO = 0.08
REFERENCE_DECISIVE_RATIO_RANGE = (0.30, 0.43)


def detectability_ratio(Z_before: np.ndarray, Z_after: np.ndarray) -> dict:
    """The audit's detectability/effect-size ratio for one counterfactual
    condition:

        ratio = E||Z_after - Z_before||^2 / E_{i != j}||Z_before_i - Z_before_j||^2

    The numerator is the mean squared representation change the transform
    itself causes (within-scene, before vs. after); the denominator is the
    mean squared representation DIFFERENCE BETWEEN DISTINCT SCENES using
    each scene's own (untransformed) `Z_before` -- i.e. the natural
    scene-to-scene variation the transform's effect must be detectable
    against. Both are computed directly from the mean-pooled frozen
    representations cached by Task 14's own pipeline (`reps.npz`'s
    `Z_before`/`Z_after`) -- never from ground-truth pose/displacement
    data.
    """
    Z_before = np.asarray(Z_before, dtype=np.float64)
    Z_after = np.asarray(Z_after, dtype=np.float64)
    if Z_before.shape != Z_after.shape:
        raise ValueError(f"Z_before/Z_after shape mismatch: {Z_before.shape} != {Z_after.shape}")
    n = Z_before.shape[0]
    if n < 2:
        raise ValueError(f"detectability_ratio needs >=2 scenes for a cross-scene baseline, got {n}")

    within = float(np.mean(np.sum((Z_after - Z_before) ** 2, axis=1)))

    off_diag = ~np.eye(n, dtype=bool)
    idx_i, idx_j = np.where(off_diag)
    cross_sq_diffs = np.sum((Z_before[idx_i] - Z_before[idx_j]) ** 2, axis=1)
    cross = float(np.mean(cross_sq_diffs))

    ratio = within / cross if cross > 0 else float("inf")
    return {
        "num_scenes": n,
        "within_pair_mean_sq_diff": within,
        "cross_scene_mean_sq_diff": cross,
        "ratio": ratio,
        "detectability_threshold": DETECTABILITY_THRESHOLD,
        "clears_threshold": bool(ratio >= DETECTABILITY_THRESHOLD),
    }


def _load_condition_reps(cfg: Task14Config, condition: str, scene_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    Zb, Za = [], []
    for sid in scene_ids:
        npz = np.load(Path(cfg.output_dir) / condition / sid / "reps.npz")
        Zb.append(npz["Z_before"])
        Za.append(npz["Z_after"])
    return np.stack(Zb), np.stack(Za)


def check_object_visibility(cfg: Task14Config, condition: str, scene_ids: list[str]) -> dict:
    """Required, run-time (not assumed) re-verification that the
    displaced object is still visible in-frame after the transform --
    reads the segmentation ground truth `render_transform_pairs` already
    produced (no extra rendering), and checks the tracked object's own
    `instance_id` (from `scene.objects[object_index]`, matched against
    the pair's own recorded `object_index`) appears in at least one pixel
    of at least one frame of the TRANSFORMED clip's segmentation map.
    """
    scenes = build_scenes(cfg)
    by_id = {s.scene_id: s for s in scenes}
    per_scene: dict[str, dict] = {}
    all_visible = True
    for sid in scene_ids:
        pair_dir = Path(cfg.output_dir) / condition / sid
        transformation = json.loads((pair_dir / "transformation.json").read_text())
        obj_idx = transformation["object_index"]
        instance_id = by_id[sid].objects[obj_idx].instance_id
        seg = np.load(pair_dir / "transformed" / "segmentation.npy")
        visible_pixel_count = int(np.sum(seg == instance_id))
        visible = visible_pixel_count > 0
        per_scene[sid] = {
            "object_index": obj_idx,
            "instance_id": instance_id,
            "visible_pixel_count": visible_pixel_count,
            "visible": visible,
        }
        all_visible = all_visible and visible
    return {"all_visible": bool(all_visible), "num_scenes": len(scene_ids), "per_scene": per_scene}


def _candidate_tag(displacement: float) -> str:
    return str(displacement).replace(".", "p").replace("-", "neg")


def run_pilot(base_kwargs: dict, candidate_displacements: list[float], num_pilot_scenes: int) -> dict:
    """Phase 1: object_translation only, `num_pilot_scenes` scenes, one
    subfolder of `<output_dir>_pilot/` per candidate so caches never
    collide across candidates (or with the phase-2 full run's own
    `output_dir`).
    """
    base_output_dir = base_kwargs["output_dir"]
    results: dict[str, dict] = {}
    for d in candidate_displacements:
        tag = _candidate_tag(d)
        cfg_kwargs = dict(base_kwargs)
        cfg_kwargs["num_scenes"] = num_pilot_scenes
        cfg_kwargs["target_displacement"] = d
        cfg_kwargs["output_dir"] = f"{base_output_dir}_pilot/disp_{tag}"
        cfg = Task14Config(**cfg_kwargs)

        process_primary_scene_range("object_translation", 0, num_pilot_scenes, cfg)

        scene_ids = scene_ids_for(cfg)
        Zb, Za = _load_condition_reps(cfg, "object_translation", scene_ids)
        ratio_info = detectability_ratio(Zb, Za)
        visibility = check_object_visibility(cfg, "object_translation", scene_ids)

        results[tag] = {
            "target_displacement": d,
            "output_dir": cfg.output_dir,
            "detectability": ratio_info,
            "visibility": visibility,
            "passes": bool(ratio_info["clears_threshold"] and visibility["all_visible"]),
        }
    return results


def choose_displacement(pilot_results: dict, candidate_displacements: list[float]) -> tuple[float, str]:
    """Smallest candidate that clears BOTH the detectability threshold and
    the visibility check; if none does, escalate to 2x the largest
    candidate tried (explicit, logged escalation -- never silently
    accepting an unreliable displacement).
    """
    for d in sorted(candidate_displacements):
        tag = _candidate_tag(d)
        if pilot_results[tag]["passes"]:
            return d, f"smallest candidate ({d}) that cleared both the detectability threshold and the visibility check"
    fallback = max(candidate_displacements) * 2.0
    return fallback, (
        f"no candidate in {candidate_displacements} cleared both checks -- escalated to 2x the largest "
        f"candidate tried ({fallback})"
    )


def run_full_followup(cfg: Task14Config) -> dict:
    """Phase 2: Task 14's own full pipeline (all four rendered
    conditions, real discrimination probe, label-scrambled control,
    representation-norm-only control, fixed-variable leakage check) at
    `cfg.num_scenes` scenes and `cfg.target_displacement`, plus this
    follow-up's own detectability-ratio/visibility re-check on
    `object_translation` at the same N.
    """
    for condition in CONDITIONS:
        process_primary_scene_range(condition, 0, cfg.num_scenes, cfg)
    for condition in CONFOUND_CONDITIONS:
        process_confound_scene_range(condition, 0, cfg.num_scenes, cfg)
    result = run_finalize(cfg)

    scene_ids = scene_ids_for(cfg)
    Zb, Za = _load_condition_reps(cfg, "object_translation", scene_ids)
    result["object_translation_detectability_ratio"] = detectability_ratio(Zb, Za)
    result["object_translation_visibility_check"] = check_object_visibility(cfg, "object_translation", scene_ids)
    _add_magnitude_matching_validity_check(result)
    return result


def _add_magnitude_matching_validity_check(result: dict) -> None:
    """Required, run-time (not assumed) re-check of a precondition this
    follow-up's whole design depends on: raising `target_displacement`
    to clear the object_translation detectability threshold ALSO raises
    the per-scene object_rotation angle Task 14 solves to match it
    (`target_rotation_angle_deg`) -- but that angle saturates at 180
    degrees once `target_displacement > 2 * radius` for a given scene's
    tracked-object radius. Task 14's own `magnitude_matching` block
    already records `any_rotation_angle_clipped` from the REAL rendered
    transform matrices (not assumed); this function reads that flag back
    out and, if it fired, overrides `result["scientific_result"]` with an
    explicit invalidation notice -- per tasks/14_counterfactuals.md's own
    "Failure conditions" ("Comparing two counterfactual edits of wildly
    different, unmatched magnitude and treating easy discrimination as a
    meaningful positive result"), a discrimination probe run on a
    magnitude-MISMATCHED pair cannot be reported as a valid positive
    result, no matter what its raw accuracy number is -- this must never
    be silently left for a reader to notice only by cross-referencing
    `magnitude_matching`'s two displacement means themselves.
    """
    mm = result["metrics"]["magnitude_matching"]
    clipped = bool(mm["any_rotation_angle_clipped"])
    trans_mean = mm["object_translation_measured_displacement"]["mean"]
    rot_mean = mm["object_rotation_measured_displacement"]["mean"]
    validity = {
        "any_rotation_angle_clipped": clipped,
        "object_translation_measured_displacement_mean": trans_mean,
        "object_rotation_measured_displacement_mean": rot_mean,
        "magnitudes_matched": not clipped,
    }
    if clipped:
        validity["explanation"] = (
            "Task 14's magnitude-matching solves object_rotation's per-scene angle from "
            "2*radius*sin(angle/2) = target_displacement (radius = scale/2, and the scene sampler's "
            "object_scale_range=(0.5, 1.0) gives radius in [0.25, 0.5], so no rotation about the object's "
            "own pivot can ever displace a surface point by more than 2*radius <= 1.0). At this run's "
            f"target_displacement={mm['target_displacement']}, that per-scene angle saturated at the "
            "domain-edge 180 degrees (was_clipped=True) for at least one scene -- the achieved rotation "
            f"displacement (mean={rot_mean:.4f}) did NOT match the achieved translation displacement "
            f"(mean={trans_mean:.4f}); they differ by {abs(trans_mean - rot_mean):.4f} scene units on "
            "average. Per tasks/14_counterfactuals.md's explicit failure condition ('Comparing two "
            "counterfactual edits of wildly different, unmatched magnitude and treating easy discrimination "
            "as a meaningful positive result'), this run's discrimination-probe result is NOT a valid Task 14 "
            "positive finding, regardless of its raw accuracy -- the probe may simply be exploiting the "
            "magnitude mismatch rather than any intervention-specific structure."
        )
        result["scientific_result"] = (
            "INVALIDATED result -- magnitude-matching precondition failed (rotation angle clipped for at "
            "least one scene at this target_displacement): " + validity["explanation"] + "\n\nOriginal "
            "(magnitude-mismatch-blind) scientific_result text, for reference only, NOT to be read as a "
            "valid finding:\n" + result["scientific_result"]
        )
    result["magnitude_matching_validity_check"] = validity


def _pilot_result_path(output_dir: str) -> Path:
    return Path(f"{output_dir}_pilot") / "pilot_result.json"


def main():
    parser = argparse.ArgumentParser(
        description="Task 14 follow-up: bigger object_translation displacement (Category A detectability fix)."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--stage", choices=["pilot", "full"], default="full")
    parser.add_argument(
        "--candidate_displacements",
        type=str,
        default="1.0,2.0",
        help="Comma-separated list of candidate target_displacement values for the pilot phase.",
    )
    parser.add_argument("--num_pilot_scenes", type=int, default=8)
    args = parser.parse_args()

    cfg = load_task14_config(args.config)

    if args.stage == "pilot":
        candidates = [float(x) for x in args.candidate_displacements.split(",")]
        base_kwargs = asdict(cfg)
        pilot_results = run_pilot(base_kwargs, candidates, args.num_pilot_scenes)
        chosen, reason = choose_displacement(pilot_results, candidates)
        out = {
            "candidate_displacements": candidates,
            "num_pilot_scenes": args.num_pilot_scenes,
            "results_by_candidate": pilot_results,
            "chosen_displacement": chosen,
            "chosen_reason": reason,
            "detectability_threshold": DETECTABILITY_THRESHOLD,
            "reference_too_small_ratio": REFERENCE_TOO_SMALL_RATIO,
            "reference_decisive_ratio_range": list(REFERENCE_DECISIVE_RATIO_RANGE),
        }
        pilot_path = _pilot_result_path(cfg.output_dir)
        pilot_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        print(f"\nPilot result written to {pilot_path}")
        print(f"Chosen displacement: {chosen} ({reason})")
        return

    # --stage full: cfg.target_displacement (from --config) is used as-is;
    # the pilot's own record (if present) is embedded into the final
    # result for a complete audit trail, but does not control this run's
    # target_displacement -- that is controlled entirely by the config
    # file, exactly like every other Task 14(-family) run.
    result = run_full_followup(cfg)

    pilot_path = _pilot_result_path(cfg.output_dir)
    result["pilot"] = json.loads(pilot_path.read_text()) if pilot_path.exists() else None

    result["task"] = "14_followup_bigger_displacement"
    result["base_task"] = 14
    result["original_task_14_result_path"] = "state/task_14_result.json"
    result["followup_reason"] = (
        "Detectability/effect-size audit found Task 14's object_translation counterfactual condition "
        "(target_displacement=0.4) moved the frozen encoder's representation too little relative to "
        "natural scene-to-scene variation (detectability ratio ~0.08, below the calibrated 0.3 threshold) "
        "for a positive discrimination result built on it to be trustworthy. This follow-up redesigns ONLY "
        "target_displacement (pilot-tested, then applied uniformly to both object_translation and, per "
        "scene, object_rotation's magnitude-matched angle) and reruns Task 14's full protocol at N=100."
    )

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))

    print(f"Task 14 follow-up (bigger displacement) result written to {result_path}")
    print(result["scientific_result"])
    print("\nobject_translation_detectability_ratio:", json.dumps(result["object_translation_detectability_ratio"], indent=2))


if __name__ == "__main__":
    main()
