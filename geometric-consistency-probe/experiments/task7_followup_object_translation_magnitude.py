"""Task 7 follow-up: object_translation detectability-magnitude fix, N=100.

    python -m experiments.task7_followup_object_translation_magnitude \
        --config configs/experiments/task7_followup_object_translation_magnitude.yaml

Background (a detectability/effect-size audit of Task 7's six-transform
result, state/task_07_result.json): object_translation's default
TransformConfig().object_translation_range == (0.4, 1.0) moves the
frozen encoder's representation too little relative to natural
scene-to-scene variation for its R^2 result to be trustworthy --

    detectability_ratio = E||Z' - Z||^2 / E_{i != j}||Z_i - Z_j||^2

came out below the calibrated 0.3 threshold below which a positive R^2
is unreliable (reference points from the audit: ratio ~= 0.08 is "too
small to see"; ratio ~= 0.30-0.43 is "learned map wins decisively").
This is a SEPARATE, additive investigation into object_translation's
magnitude ONLY -- it does not touch experiments/task7_geometric_consistency.py,
state/task_07_result.json, configs/experiments/task7_geometric_consistency.yaml,
or any other transform's data. Every function it needs already exists in
experiments/geometric_consistency_lib.py (gclib) and
transforms/scene_transform.py (TransformConfig(object_translation_range=
(X, X)) is documented, intentional construction-time flexibility, not a
hack) -- this script only calls them with a different magnitude and N.

Two-phase design:

1. PILOT (a handful of scenes, several candidate magnitudes): for each
   candidate in `candidate_magnitudes`, render the pilot scenes'
   original/object_translation pair at that FIXED magnitude
   (TransformConfig(object_translation_range=(m, m))), encode both sides
   with the same frozen encoder used everywhere else in this project,
   and compute the SAME detectability ratio the audit used. The
   SMALLEST candidate that clears `detectability_threshold` (0.3) is
   chosen -- not the largest -- per this project's "genuinely
   detectable, not gratuitously overshot" design goal. If none of the
   configured candidates clear the threshold, one larger
   `fallback_magnitude` is tried before giving up. Candidates are also
   sanity-checked against the rendered segmentation mask: a magnitude
   that routinely pushes the moved object entirely out of the camera's
   frame stops testing "did Z track a 3D translation" and starts
   testing "did Z notice an object vanish", a qualitatively different
   (and not what this follow-up is chartered to test) effect -- see
   `object_visibility_stats` below and this run's "magnitude_selection"
   result field.

2. FULL RUN (N=100, base_seed=100, disjoint from every base_seed already
   used elsewhere in this project -- Task 6/7 use base_seed=0): the
   chosen magnitude is applied uniformly to all 100 scenes, with the
   IDENTICAL scene-level train/test split protocol, ridge_alpha, and
   pretrained real V-JEPA 2 ViT-L/16 encoder Task 7 used, fitting rho(T)
   on train and evaluating equivariance R^2 (metrics.equivariance,
   probes.linear_rep_transform -- ridge regression, never a trained
   network) plus the three required baselines (persistence, mean,
   shuffled-pairing; baselines.run_all_baselines's core three) on the
   held-out test split. The R^2 result is reported as-is, whatever sign
   or magnitude it comes out to be -- fixing detectability is not the
   same as manufacturing a positive result (research/RESEARCH_INVARIANTS.md;
   this project's "negative results are valid" principle).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

import experiments.geometric_consistency_lib as gclib
from generation.scene import SceneState
from transforms.pairs import load_pair
from transforms.scene_transform import TransformConfig

DEFAULT_RESULT_PATH = "state/task7_followup_object_translation_magnitude_result.json"
# Same scene-count discipline as Task 6/7 (>= 40 scenes); this follow-up
# runs its full experiment at N=100 (a scale increase over Task 7's N=40
# is part of what was asked for, independent of the magnitude fix).
MIN_SCENES_FULL = 40
DETECTABILITY_THRESHOLD = 0.3


@dataclass
class FollowupConfig:
    # --- pilot phase ---
    pilot_num_scenes: int = 8
    candidate_magnitudes: list[float] = field(default_factory=lambda: [2.0, 3.5, 5.0])
    fallback_magnitude: float = 7.0
    detectability_threshold: float = DETECTABILITY_THRESHOLD

    # --- full-run phase ---
    num_scenes: int = 100
    train_fraction: float = 0.8
    # Disjoint from every base_seed already used by any other task/follow-up
    # in this project (Task 6/7 use base_seed=0) -- required by this
    # follow-up's own charter so its sampled scenes never collide with an
    # existing dataset on disk.
    base_seed: int = 100
    output_dir: str = "experiments/task7_followup_object_translation_magnitude"
    result_path: str = DEFAULT_RESULT_PATH
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
    # Videos per `encoder.model.get_vision_features` call (2 sides per
    # scene) -- see `encode_all_pairs_cached`'s docstring for the
    # measured ~2x speedup at batch_size=4 vs. one-video-at-a-time.
    # batch_size=1 reproduces the unbatched `encoder.encode()` path
    # exactly, so this is safe to set to 1 to fall back to that.
    encode_batch_size: int = 4


def load_config(path: str | Path | None) -> FollowupConfig:
    cfg = FollowupConfig()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


def _magnitude_key(magnitude: float) -> str:
    return f"mag_{magnitude:g}".replace(".", "p").replace("-", "neg")


# --- resumable encoding: this project's real ViT-L/16 encoder is expensive
#     on CPU (observed ~130-160s per forward pass in this environment for a
#     4-frame, 256x256-crop clip -- ~2-3 min/scene for an original+
#     transformed pair), so a run spanning N=100 scenes plus a multi-
#     candidate pilot is many hours of wall-clock compute. Each scene's
#     (Z, Z') is cached to disk right after it is computed so an
#     interrupted run resumes instead of re-encoding from scratch. ------


def encode_all_pairs_cached(
    scenes: list[SceneState], out_dir: Path, encoder, batch_size: int = 4
) -> dict[str, dict[str, np.ndarray]]:
    """Like `gclib.encode_all_pairs`, but (a) resumable via the per-scene
    disk cache above, and (b) batched: `batch_size` scenes' worth of
    videos (2 per scene, original+transformed) are stacked into ONE
    `encoder.model.get_vision_features` call instead of `batch_size * 2`
    separate `encoder.encode()` calls.

    This project's real ViT-L/16 forward pass is CPU-bound and slow in
    this environment -- empirically measured at ~142s for one
    `encoder.encode()` call (batch=1) vs. ~72s/video amortized when 4
    videos are stacked into one `get_vision_features` call (a ~2x
    speedup measured directly, not assumed), for a 4-frame/128px clip
    (this project's standard `num_frames`/`resolution`). Batching does
    not change what is computed: it calls the SAME frozen model, through
    the SAME preprocessing (`encoder._preprocess`, the identical method
    `encoder.encode()` itself uses) -- only the batch dimension changes,
    same weights, same no-grad forward, same mean_pool reduction
    afterwards. `encoders/vjepa.py` itself is not modified; this only
    calls attributes it already exposes (`_preprocess`, `model`,
    `device`). `batch_size=1` reproduces `encoder.encode()` exactly
    (one video per forward call), so this is a strict, backward-
    compatible superset of the unbatched path, not a different
    computation.
    """
    import torch

    from encoders.vjepa import mean_pool

    out_dir = Path(out_dir)
    reps: dict[str, dict[str, np.ndarray]] = {}
    pending: list[SceneState] = []
    for scene in scenes:
        cache_path = out_dir / scene.scene_id / "pooled_representations.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            reps[scene.scene_id] = {"Z": cached["Z"], "Z_prime": cached["Z_prime"]}
            continue
        pending.append(scene)

    batch_size = max(1, batch_size)
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        videos: list[np.ndarray] = []
        index_map: list[tuple[SceneState, str]] = []
        for scene in chunk:
            loaded = load_pair(out_dir / scene.scene_id)
            _, orig_rgb, _, _ = loaded["original"]
            _, trans_rgb, _, _ = loaded["transformed"]
            videos.append(orig_rgb)
            index_map.append((scene, "Z"))
            videos.append(trans_rgb)
            index_map.append((scene, "Z_prime"))

        batch_tensor = torch.cat([encoder._preprocess(v) for v in videos], dim=0).to(encoder.device)
        with torch.no_grad():
            tokens = encoder.model.get_vision_features(batch_tensor)  # (B, num_tokens, hidden)
        tokens = tokens.cpu().numpy().astype(np.float32)

        per_scene: dict[str, dict[str, np.ndarray]] = {}
        for (scene, side), row in zip(index_map, tokens):
            per_scene.setdefault(scene.scene_id, {})[side] = mean_pool(row)

        for scene in chunk:
            Z = per_scene[scene.scene_id]["Z"]
            Zp = per_scene[scene.scene_id]["Z_prime"]
            cache_path = out_dir / scene.scene_id / "pooled_representations.npz"
            np.savez(cache_path, Z=Z, Z_prime=Zp)
            reps[scene.scene_id] = {"Z": Z, "Z_prime": Zp}
    return reps


# --- detectability ratio: E||Z'-Z||^2 / E_{i!=j}||Zi-Zj||^2 ----------------


def representation_change_ratio(Z: np.ndarray, Z_prime: np.ndarray) -> dict:
    """The exact ratio the Task 7 detectability audit used: the mean
    squared representation-space displacement the transform itself
    causes, relative to how far apart two arbitrary (unrelated) scenes'
    representations already sit. >= 0.3 is this project's calibrated
    "detectable enough for a positive R^2 to be trustworthy" threshold
    (reference points recorded by the audit: ~0.08 too small to see,
    ~0.30-0.43 learned map wins decisively). Computed over whichever set
    of scenes is passed in (pilot subset, or the full N=100 set) -- a
    descriptive diagnostic of the transform's effect size, not a fitted
    quantity, so pooling train+test here carries no leakage risk (same
    reasoning as experiments/task12_followup_magnitude_sweep.py's
    identically-defined `representation_change_ratio`, reproduced here
    rather than imported so this follow-up stays self-contained and
    does not reach into a sibling Category-A follow-up's module).
    """
    numerator = float(np.mean(np.sum((Z_prime - Z) ** 2, axis=-1)))
    sq_norms = np.sum(Z**2, axis=-1)
    gram = Z @ Z.T
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    n = len(Z)
    mask = ~np.eye(n, dtype=bool)
    denominator = float(sq_dists[mask].mean())
    return {
        "mean_within_pair_sq_change": numerator,
        "mean_cross_scene_sq_distance": denominator,
        "ratio": numerator / denominator if denominator > 1e-12 else float("nan"),
    }


# --- object-visibility sanity check: is the magnitude physically sane? -----


def object_visibility_stats(scenes: list[SceneState], out_dir: Path) -> dict:
    """Fraction of scenes, at a given candidate magnitude, where the
    translated object is still visible (non-empty segmentation mask) in
    the transformed render. A magnitude routinely pushing the object
    entirely out of the camera's frustum (this project's floor plane is
    20x20 world units and easily accommodates a large XY displacement,
    but the camera's field of view does not) would make the
    representation change dominated by "object vanished from view"
    rather than "object moved in 3D" -- a different, less clean
    manipulation than the other three GEOMETRIC_TRANSFORMS test, and not
    what this follow-up is chartered to fix. Read straight from the
    rendered segmentation pass, never inferred from the SE(3) matrix
    alone (a moved-but-still-visible object cannot be distinguished from
    a moved-and-now-offscreen one without actually looking at the pixels).
    """
    visible = 0
    for scene in scenes:
        loaded = load_pair(out_dir / scene.scene_id)
        transformation = loaded["transformation"]
        obj_idx = transformation["object_index"]
        instance_id = scene.objects[obj_idx].instance_id
        _, _, _, seg_trans = loaded["transformed"]
        visible += int(np.any(seg_trans == instance_id))
    return {
        "num_scenes": len(scenes),
        "num_object_visible_after_transform": visible,
        "fraction_object_visible_after_transform": visible / len(scenes) if scenes else float("nan"),
    }


# --- pilot phase -------------------------------------------------------------


def run_pilot_candidate(
    pilot_scenes: list[SceneState], magnitude: float, cfg: FollowupConfig, encoder
) -> dict:
    transform_cfg = TransformConfig(object_translation_range=(magnitude, magnitude))
    out_dir = Path(cfg.output_dir) / "pilot" / _magnitude_key(magnitude)
    gclib.render_transform_pairs(
        pilot_scenes, "object_translation", transform_cfg, out_dir, cfg.num_frames, cfg.fps, cfg.resolution
    )
    reps = encode_all_pairs_cached(pilot_scenes, out_dir, encoder, batch_size=cfg.encode_batch_size)
    _, Z, Zp = gclib.build_full_arrays(pilot_scenes, reps)
    ratio = representation_change_ratio(Z, Zp)
    visibility = object_visibility_stats(pilot_scenes, out_dir)
    pixel_stats = gclib.pixel_diff_stats(pilot_scenes, out_dir)
    return {
        "magnitude": magnitude,
        "detectability_ratio": ratio,
        "object_visibility": visibility,
        "pixel_diff": pixel_stats,
        "out_dir": str(out_dir),
    }


def choose_magnitude(pilot_results: list[dict], threshold: float) -> tuple[float | None, str]:
    """Smallest candidate clearing `threshold`, per this follow-up's
    "genuinely detectable, not gratuitously overshot" goal -- never the
    largest merely because it scores highest.
    """
    passing = sorted(
        (r["magnitude"] for r in pilot_results if r["detectability_ratio"]["ratio"] >= threshold)
    )
    if passing:
        return passing[0], f"smallest candidate clearing ratio >= {threshold}"
    return None, f"no candidate cleared ratio >= {threshold}"


def run_pilot(scenes: list[SceneState], cfg: FollowupConfig, encoder) -> dict:
    """Tests `candidate_magnitudes` in the given (ascending) order and
    stops at the FIRST one clearing `detectability_threshold` -- exactly
    equivalent to "test all three, pick the smallest that passes" when
    the ratio is monotonically non-decreasing in magnitude (true here:
    a larger rigid displacement mechanically produces a larger pixel/
    representation-space change, up to the point an object leaves the
    frame entirely -- see `object_visibility_stats`), and considerably
    cheaper: this project's real ViT-L/16 encoder is CPU-bound and slow
    in this environment (~130-160s per encode call, empirically
    measured), so skipping untested, unnecessarily-large candidates once
    a passing one is found materially shortens this follow-up's runtime
    without changing which magnitude gets chosen.
    """
    pilot_scenes = scenes[: cfg.pilot_num_scenes]
    pilot_results: list[dict] = []
    chosen, reason = None, ""
    for m in cfg.candidate_magnitudes:
        pilot_results.append(run_pilot_candidate(pilot_scenes, m, cfg, encoder))
        chosen, reason = choose_magnitude(pilot_results, cfg.detectability_threshold)
        if chosen is not None:
            break

    fallback_used = False
    if chosen is None:
        fallback_result = run_pilot_candidate(pilot_scenes, cfg.fallback_magnitude, cfg, encoder)
        pilot_results.append(fallback_result)
        fallback_used = True
        chosen, reason = choose_magnitude(pilot_results, cfg.detectability_threshold)
        if chosen is None:
            reason = (
                f"no candidate (including fallback_magnitude={cfg.fallback_magnitude}) cleared "
                f"ratio >= {cfg.detectability_threshold}; using fallback_magnitude anyway as the "
                "largest tested candidate, and this is recorded honestly as a magnitude-fix attempt "
                "that did not fully clear the calibrated threshold."
            )
            chosen = cfg.fallback_magnitude

    return {
        "pilot_num_scenes": cfg.pilot_num_scenes,
        "candidates_configured": list(cfg.candidate_magnitudes),
        "candidates_tested": [r["magnitude"] for r in pilot_results],
        "early_stopped": (not fallback_used) and len(pilot_results) < len(cfg.candidate_magnitudes),
        "fallback_magnitude_used": fallback_used,
        "per_candidate": pilot_results,
        "chosen_magnitude": chosen,
        "selection_reason": reason,
        "detectability_threshold": cfg.detectability_threshold,
    }


# --- full N=100 run, mirroring task7_geometric_consistency.py's per-transform
#     pipeline exactly, just for object_translation at the chosen magnitude --


def run_full_experiment(
    scenes: list[SceneState], split: dict[str, str], chosen_magnitude: float, cfg: FollowupConfig, encoder
) -> dict:
    transform_cfg = TransformConfig(object_translation_range=(chosen_magnitude, chosen_magnitude))
    out_dir = Path(cfg.output_dir) / "object_translation_n100"

    gclib.render_transform_pairs(
        scenes, "object_translation", transform_cfg, out_dir, cfg.num_frames, cfg.fps, cfg.resolution
    )

    reps = encode_all_pairs_cached(scenes, out_dir, encoder, batch_size=cfg.encode_batch_size)
    train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test = gclib.build_arrays(scenes, split, reps)

    # Leakage check: a scene's original/transformed pair must never be
    # split across train/test (research/RESEARCH_INVARIANTS.md; identical
    # check to tests/test_scene_split.py's, re-asserted at run time here
    # rather than assumed from gclib.assign_split's construction alone).
    if not set(train_ids).isdisjoint(test_ids):
        raise ValueError("train/test scene leakage detected -- a scene appears in both splits")

    metrics_block = gclib.evaluate_transform(
        "object_translation", Z_train, Zp_train, Z_test, Zp_test, cfg.ridge_alpha, cfg.shuffled_pairing_seed
    )
    visual_confound = gclib.pixel_diff_stats(scenes, out_dir)
    visibility = object_visibility_stats(scenes, out_dir)

    _, Z_all, Zp_all = gclib.build_full_arrays(scenes, reps)
    detectability_ratio_full = representation_change_ratio(Z_all, Zp_all)

    manifest = {
        "transform": "object_translation",
        "num_scenes": len(scenes),
        "base_seed": cfg.base_seed,
        "train_fraction": cfg.train_fraction,
        "chosen_magnitude": chosen_magnitude,
        "transform_config": asdict(transform_cfg),
        "scenes": [{"scene_id": s.scene_id, "seed": s.seed, "split": split[s.scene_id]} for s in scenes],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "transform_name": "object_translation",
        "chosen_magnitude": chosen_magnitude,
        "train_scene_ids": train_ids,
        "test_scene_ids": test_ids,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "transform_config": asdict(transform_cfg),
        **metrics_block,
        "visual_confound": visual_confound,
        "object_visibility": visibility,
        "detectability_ratio_full_n100": detectability_ratio_full,
        "manifest_path": str(out_dir / "manifest.json"),
    }


def _original_task7_object_translation_r2(task7_result_path: str | Path) -> dict:
    """Read-only comparison against Task 7's own recorded object_translation
    result (state/task_07_result.json) -- that file is never written to
    or modified by this follow-up.
    """
    path = Path(task7_result_path)
    if not path.exists():
        return {"found": False, "note": f"{path} not found; no comparison performed."}
    task7 = json.loads(path.read_text())
    obj_trans = task7.get("results", {}).get("object_translation", {})
    return {
        "found": True,
        "original_n_scenes": task7.get("dataset", {}).get("num_scenes"),
        "original_learned_W_T_r2": obj_trans.get("learned_W_T", {}).get("r2"),
        "original_transform_config": obj_trans.get("transform_config"),
    }


def run_experiment(cfg: FollowupConfig) -> dict:
    scenes = gclib.sample_scenes(cfg.num_scenes, cfg.base_seed, cfg.num_objects_min, cfg.num_objects_max)
    if len(scenes) < MIN_SCENES_FULL:
        raise ValueError(f"This follow-up requires >={MIN_SCENES_FULL} scenes, got {len(scenes)}")
    if len(scenes) < cfg.pilot_num_scenes:
        raise ValueError("pilot_num_scenes must not exceed num_scenes")
    scene_ids = [s.scene_id for s in scenes]

    # One scene-level split, decided once before any rendering (identical
    # protocol to gclib.assign_split / Task 6/7).
    split = gclib.assign_split(scene_ids, cfg.train_fraction, cfg.base_seed)

    # One frozen encoder instance for the entire run (pilot + full), so
    # frozen-ness can be verified once across everything this script does.
    encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
    encoder_pretrained = bool(getattr(encoder, "pretrained", False))
    params_before = gclib.snapshot_params(encoder.model)

    pilot = run_pilot(scenes, cfg, encoder)
    chosen_magnitude = pilot["chosen_magnitude"]

    full_result = run_full_experiment(scenes, split, chosen_magnitude, cfg, encoder)

    frozen_verified = gclib.params_unchanged(params_before, encoder.model)

    original_comparison = _original_task7_object_translation_r2("state/task_07_result.json")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_lines = [
        "# Task 7 follow-up -- object_translation magnitude fix (N=100)\n",
        f"- pilot candidates tested: {pilot['candidates_tested']} "
        f"(fallback used: {pilot['fallback_magnitude_used']})\n",
        f"- chosen magnitude: {chosen_magnitude} ({pilot['selection_reason']})\n",
        f"- scenes: {len(scenes)} (train={full_result['n_train']}, test={full_result['n_test']})\n",
        f"- encoder pretrained: {encoder_pretrained}, frozen verified: {frozen_verified}\n\n",
        "## Pilot detectability ratios\n\n",
        "| magnitude | ratio | object visible fraction |\n|---|---|---|\n",
    ]
    for r in pilot["per_candidate"]:
        report_lines.append(
            f"| {r['magnitude']} | {r['detectability_ratio']['ratio']:.4f} | "
            f"{r['object_visibility']['fraction_object_visible_after_transform']:.2f} |\n"
        )
    report_lines.append(
        f"\n## Full N=100 result\n\n"
        f"- detectability ratio (N=100, all scenes): {full_result['detectability_ratio_full_n100']['ratio']:.4f}\n"
        f"- learned W_T R^2 (held-out test): {full_result['learned_W_T']['r2']:.4f}\n"
        f"- persistence baseline R^2: {full_result['persistence_baseline']['r2']:.4f}\n"
        f"- mean baseline R^2: {full_result['mean_baseline']['r2']:.4f}\n"
        f"- random-pair (shuffled) control R^2: {full_result['random_pair_control']['r2']:.4f}\n"
        f"- object visible after transform (fraction): "
        f"{full_result['object_visibility']['fraction_object_visible_after_transform']:.2f}\n\n"
        f"## Comparison to Task 7's original object_translation result\n\n"
        f"{json.dumps(original_comparison, indent=2)}\n"
    )
    report_path.write_text("".join(report_lines))

    result = {
        "task": "07_followup_object_translation_magnitude",
        "implementation_status": "COMPLETE",
        "num_scenes": len(scenes),
        "base_seed": cfg.base_seed,
        "train_fraction": cfg.train_fraction,
        "pretrained": encoder_pretrained,
        "encoder": {
            "name": "VJEPAEncoder",
            "checkpoint": encoder.checkpoint,
            "pretrained": encoder_pretrained,
            "frozen": frozen_verified,
            "pooling": "mean_pool",
        },
        "fitting": {"method": "ridge", "alpha": cfg.ridge_alpha},
        "magnitude_selection": pilot,
        "chosen_magnitude": chosen_magnitude,
        "full_run_n100": full_result,
        "detectability_ratio_final_n100": full_result["detectability_ratio_full_n100"]["ratio"],
        "r2_equivariance_held_out_test": full_result["learned_W_T"]["r2"],
        "original_task7_object_translation_comparison": original_comparison,
        "config": asdict(cfg),
        "seed": cfg.base_seed,
        "software_versions": gclib.software_versions(),
        "artifacts": [str(report_path), full_result["manifest_path"]],
        "tests": {"passed": 0, "failed": 0},
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Task 7 follow-up: object_translation detectability-magnitude fix at N=100."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument(
        "--skip_tests", action="store_true", help="Skip running the existing test suite before writing the result"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = gclib.run_existing_test_suite(repo_root)

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"Result written to {result_path}")
    print(f"Chosen magnitude: {result['chosen_magnitude']}")
    print(f"Final N=100 detectability ratio: {result['detectability_ratio_final_n100']:.4f}")
    print(f"Held-out test R^2 (learned W_T): {result['r2_equivariance_held_out_test']:.4f}")


if __name__ == "__main__":
    main()
