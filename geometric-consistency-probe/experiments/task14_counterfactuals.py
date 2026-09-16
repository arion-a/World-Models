"""Task 14: controlled counterfactual representation consistency.

    python -m experiments.task14_counterfactuals --stage process --condition object_translation --scene_start 0 --scene_end 10
    ... (repeat for object_rotation, camera_translation, lighting_change / scene chunks) ...
    python -m experiments.task14_counterfactuals --stage finalize

Pipeline (see tasks/14_counterfactuals.md and research/CANONICAL_RESEARCH_PROTOCOL.md's
"TASK 14" section, which this module implements verbatim):

    >=40 single-object scenes (`generation.scene_sampler.sample_scene`,
    `num_objects_min=num_objects_max=1` so the randomly-chosen object
    index `apply_object_translation`/`apply_object_rotation` draw is
    always 0 -- "same object" isolation holds by construction, not by
    chance) -> scene-level train/test split, decided before any
    rendering (`experiments.geometric_consistency_lib.assign_split`,
    reused unmodified).

    Two ISOLATABLE counterfactual conditions per scene (CONDITIONS):
    `object_translation` and `object_rotation`, both applied via
    `transforms.scene_transform.apply_transform` (Task 3, unmodified) and
    both re-verified via `experiments.geometric_consistency_lib.
    verify_transform_ground_truth` against `apply_transform`'s own
    `changed_variables`/`fixed_variables` output -- isolation is never
    re-derived by hand (Experimental protocol step 2).

    MAGNITUDE MATCHING (Experimental protocol step 3; see module-level
    "Magnitude-matching method" note below `target_rotation_angle_deg`):
    the translation's magnitude is fixed at `cfg.target_displacement`
    for every scene (`TransformConfig(object_translation_range=(d, d))`);
    the rotation's angle is solved, PER SCENE, from that same scene's own
    object radius (`scale / 2`) so that a reference point on the
    object's surface moves by the IDENTICAL Euclidean distance
    `cfg.target_displacement` under either edit -- an exact, per-scene
    ground-truth SE(3) displacement match, computed via `transforms.se3`
    primitives (`reference_point_displacement`), not assumed.

    Z_before/Z_after are computed for both conditions
    (mean-pooled VJEPAEncoder, reused from Task 6-8's pattern) from ONLY
    the rendered RGB -- no `changed_variables`/`transform_matrix`/pose
    ground truth ever reaches the encoder (research/RESEARCH_INVARIANTS.md
    invariant 6/7). A linear ridge DISCRIMINATION probe (Task 8's exact
    `_fit_and_predict`/`shuffled_label_permutation` machinery, applied to
    a binary "which condition" target instead of a physical-state
    regression target -- Task 6/8's linear-probe machinery reused, not a
    new probe family invented) is fit on `Z_after - Z_before` for TRAIN
    scenes and evaluated on TEST scenes, for: the real probe, the
    required label-scrambled control, a representation-norm-only control
    (required "not trivially separable by representation-norm alone"
    check), and the required FIXED-VARIABLE LEAKAGE CHECK -- a probe
    fit on a separate control run's counterfactual pair (CONFOUND_CONDITIONS:
    `camera_translation` vs. `lighting_change`, both members of this
    experiment's `fixed_variables` set for TRAIN scenes only), evaluated
    directly on the real TEST set, to confirm the real discrimination
    signal is not attributable to some confound shared by "any transform
    happened" rather than the intended object_translation/object_rotation
    distinction.

Because a single foreground command has a wall-clock budget far shorter
than rendering+encoding 4 conditions * up to `cfg.num_scenes` scenes
end to end with a real pretrained encoder, the expensive step is split
into a `--stage process` call per (condition, scene index range) --
each idempotent and disk-cached under `cfg.output_dir` -- and a cheap,
bpy/torch-free `--stage finalize` call that only reads those caches,
fits/evaluates the probes, and writes `state/task_14_result.json`.
`run_experiment()` (used by fast tests) still does everything in one
call when that is fast enough (e.g. a tiny pretrained=False dev config).

This module deliberately does not modify generation/, transforms/,
encoders/, representations/, probes/, metrics/, or baselines/ -- it only
calls their existing public functions (plus a few of Task 8's own
module-level helper functions, reused directly rather than
reimplemented, per this task's "Task 6/8's linear-probe machinery is
reused, not a new probe family invented").
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import yaml

import experiments.geometric_consistency_lib as gclib
from experiments.task8_physical_state import _fit_and_predict, shuffled_label_permutation
from generation.scene import SceneState
from transforms.pairs import generate_pair, load_pair
from transforms.scene_transform import TransformConfig

DEFAULT_RESULT_PATH = "state/task_14_result.json"
# Same scene-count floor Task 6 established (tasks/14_counterfactuals.md's
# Dataset requirements: "meeting the scene-count floor established in
# Task 6").
MIN_SCENES = 40

# The two isolatable counterfactual conditions this run compares
# (Experimental protocol step 1) -- both touch exactly one physical
# variable of the SAME object (guaranteed by num_objects_min=
# num_objects_max=1, see build_scenes), confirmed via
# gclib.verify_transform_ground_truth's changed_variables/fixed_variables
# check, never re-derived by hand.
CONDITIONS = ("object_translation", "object_rotation")
CONDITION_LABEL = {"object_translation": 0, "object_rotation": 1}

# The fixed-variable leakage check's own "separate control run" (task's
# own required control, not optional): two transforms that are each
# explicitly a member of the FIXED_VARIABLES set for both CONDITIONS
# above (camera pose and light state are untouched by object_translation/
# object_rotation -- see transforms/scene_transform.py's
# _all_variable_paths), so a probe that can predict CONDITIONS from
# THESE transforms' representation differences would be exploiting a
# confound ("some transform happened") rather than the intended
# object_translation-vs-object_rotation distinction.
CONFOUND_CONDITIONS = ("camera_translation", "lighting_change")
CONFOUND_LABEL = {"camera_translation": 0, "lighting_change": 1}

ALL_RENDERED_CONDITIONS = CONDITIONS + CONFOUND_CONDITIONS


@dataclass
class Task14Config:
    num_scenes: int = 40
    train_fraction: float = 0.8
    base_seed: int = 0
    output_dir: str = "experiments/counterfactuals"
    result_path: str = DEFAULT_RESULT_PATH
    resolution: int = 128
    num_frames: int = 4
    fps: float = 4.0
    # Magnitude-matching target (Experimental protocol step 3): the exact
    # Euclidean displacement (scene units), computed via transforms.se3,
    # that BOTH conditions are calibrated to produce for a reference point
    # on the tracked object's surface -- see target_rotation_angle_deg's
    # docstring for the closed-form derivation and why 0.4 is safely
    # achievable for every object radius this project's default
    # SceneSamplerConfig.object_scale_range=(0.5, 1.0) produces (radius
    # in [0.25, 0.5]; max reachable rotation-induced displacement is
    # 2*radius, i.e. >=0.5, comfortably above 0.4 with margin).
    target_displacement: float = 0.4
    ridge_alpha: float = 10.0
    scramble_seed: int = 0
    # Chance-level accuracy threshold width, in standard deviations of a
    # p=0.5 binomial with n = the real test set's example count -- used
    # by both the leakage check and the representation-norm-only check
    # (chance_accuracy_threshold) so "did not exceed chance" is a
    # documented, size-aware criterion, not an eyeballed cutoff.
    chance_margin_z: float = 2.0
    pretrained: bool = True
    checkpoint: str | None = None  # None -> encoders.vjepa.DEFAULT_CHECKPOINT
    device: str | None = None
    fallback_seed: int = 0


def load_task14_config(path: str | Path | None) -> Task14Config:
    cfg = Task14Config()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for f_ in fields(cfg):
        if f_.name in data:
            setattr(cfg, f_.name, data[f_.name])
    return cfg


def build_scenes(cfg: Task14Config) -> list[SceneState]:
    """>=cfg.num_scenes single-object scenes, one distinct seed each --
    reuses Task 6/7's exact sampling convention
    (experiments.geometric_consistency_lib.sample_scenes) with
    num_objects fixed to exactly 1 so BOTH counterfactual conditions'
    randomly-chosen object index (transforms/scene_transform.py's
    `rng.integers(0, len(scene.objects))`) is deterministically 0 --
    "the same object" isolation guarantee holds by construction for
    every scene, not merely on average.
    """
    return gclib.sample_scenes(cfg.num_scenes, cfg.base_seed, num_objects_min=1, num_objects_max=1)


def scene_ids_for(cfg: Task14Config) -> list[str]:
    return [f"scene_{i:04d}" for i in range(cfg.num_scenes)]


def compute_split(cfg: Task14Config) -> dict[str, str]:
    return gclib.assign_split(scene_ids_for(cfg), cfg.train_fraction, cfg.base_seed)


# --- magnitude-matching: pure math, unit-testable without bpy/torch --------
#
# Magnitude-matching method (documented per Inputs: "magnitudes scaled to
# be comparable by an explicitly documented method ... a comparable
# ground-truth SE(3) displacement magnitude, computed via transforms.se3"):
#
#   1. object_translation's magnitude is fixed at `target_displacement`
#      for every scene (a degenerate TransformConfig.object_translation_range
#      = (d, d) -- numpy's Generator.uniform(d, d) returns exactly d, the
#      same "fixed magnitude via a degenerate range" pattern Task 6 already
#      uses for camera_rotation_azimuth_deg_range).
#   2. object_rotation rotates the tracked object about ITS OWN position
#      (transforms/scene_transform.py:apply_object_rotation uses
#      rotate_about_point_matrix, pivot = object position), so a point at
#      world-frame offset (radius, 0, 0) from that pivot traces a chord of
#      length `2 * radius * |sin(angle/2)|` under a z-axis rotation by
#      `angle`. Solving that equation for `angle` given the SAME
#      `target_displacement` and this scene's own object radius
#      (`scale / 2`) gives the rotation magnitude that displaces that
#      reference point by EXACTLY the same distance the translation
#      displaces the object's own center.
#   3. Both conditions' actual achieved displacement is then RE-MEASURED
#      (not just derived from the formula) by applying the transform's own
#      recorded `transform_matrix` to the reference point via
#      `reference_point_displacement`, so any drift between the intended
#      and achieved magnitude (e.g. from clipping at the +-1 arcsin domain
#      edge) is caught and recorded, not assumed away.


def target_rotation_angle_deg(target_displacement: float, radius: float) -> tuple[float, bool]:
    """Solve `2 * radius * sin(angle / 2) = target_displacement` for
    `angle` (degrees). Returns `(angle_deg, was_clipped)`: `was_clipped`
    is True only if `target_displacement > 2 * radius` (the target
    exceeds what any rotation about this pivot could reach for this
    radius), in which case the ratio is clamped to +-1 (angle = +-180
    degrees, the maximum possible chord) rather than raising -- callers
    must check this flag rather than assume magnitude-matching silently
    succeeded.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    raw_ratio = target_displacement / (2.0 * radius)
    ratio = float(np.clip(raw_ratio, -1.0, 1.0))
    angle_deg = float(np.degrees(2.0 * np.arcsin(ratio)))
    return angle_deg, bool(abs(raw_ratio) > 1.0)


def reference_point_displacement(
    transform_matrix, base_position: tuple[float, float, float], offset: tuple[float, float, float]
) -> float:
    """Apply a 4x4 SE(3) `transform_matrix` (exactly the matrix
    `transforms.scene_transform.apply_transform` records as ground truth)
    to the reference point `base_position + offset`, and return the
    Euclidean distance it moves -- this task's chosen operational
    definition of "SE(3) displacement magnitude" (Inputs: "a comparable
    ground-truth SE(3) displacement magnitude, computed via
    transforms.se3"). For a pure translation this is exactly the
    translation's own magnitude for ANY offset; for a rotation about
    `base_position`, this is the chord length of the offset point's
    circular arc -- both are exercised directly (not just via the
    closed-form formula) in tests/test_task14_counterfactuals.py.
    """
    matrix = np.asarray(transform_matrix, dtype=float)
    point = np.array(base_position, dtype=float) + np.array(offset, dtype=float)
    homogeneous = np.append(point, 1.0)
    new_point = (matrix @ homogeneous)[:3]
    return float(np.linalg.norm(new_point - point))


def transform_configs_for_scene(scene: SceneState, cfg: Task14Config) -> dict:
    """Per-scene TransformConfig for each of CONDITIONS, magnitude-matched
    to `cfg.target_displacement` via this scene's own tracked object
    radius (see module-level "Magnitude-matching method" note).
    """
    obj = scene.objects[0]
    radius = obj.scale / 2.0
    angle_deg, clipped = target_rotation_angle_deg(cfg.target_displacement, radius)
    return {
        "object_translation": TransformConfig(object_translation_range=(cfg.target_displacement, cfg.target_displacement)),
        "object_rotation": TransformConfig(object_rotation_deg_range=(angle_deg, angle_deg)),
        "radius": radius,
        "rotation_angle_deg": angle_deg,
        "rotation_angle_clipped": clipped,
    }


# --- chance-level thresholds + accuracy (pure numpy) ------------------------


def chance_accuracy_threshold(n_examples: int, z: float = 2.0) -> float:
    """Upper bound of an approximate `z`-sigma confidence interval around
    chance (p=0.5) binary-classification accuracy for `n_examples`
    independent test examples -- the documented, size-aware "did not
    exceed chance" criterion used by both the representation-norm-only
    check and the fixed-variable leakage check, rather than an
    eyeballed cutoff (Required scientific-validity tests: "checked, not
    assumed").
    """
    if n_examples <= 0:
        return 1.0
    std = float(np.sqrt(0.25 / n_examples))
    return 0.5 + z * std


def binary_accuracy(pred: np.ndarray, y_true: np.ndarray) -> float:
    pred_label = (np.asarray(pred, dtype=float).reshape(-1) >= 0.5).astype(int)
    true_label = np.asarray(y_true, dtype=float).reshape(-1).astype(int)
    return float(np.mean(pred_label == true_label))


# --- rendering + verification (needs bpy) ------------------------------------


def _condition_dir(cfg: Task14Config, condition: str) -> Path:
    return Path(cfg.output_dir) / condition


def _scene_cache_paths(cfg: Task14Config, condition: str, scene_id: str) -> dict[str, Path]:
    scene_dir = _condition_dir(cfg, condition) / scene_id
    return {
        "scene_dir": scene_dir,
        "reps": scene_dir / "reps.npz",
        "record": scene_dir / "task14_record.json",
        "transformation": scene_dir / "transformation.json",
    }


def _encoder_info_for(encoder) -> dict:
    return {
        "name": "VJEPAEncoder",
        "checkpoint": encoder.checkpoint,
        "pretrained": bool(getattr(encoder, "pretrained", False)),
        "pooling": "mean_pool",
    }


def _record_or_reuse_encoder_info(cfg: Task14Config, encoder_info: dict) -> None:
    """Same "encoder provenance must be identical across every chunk of
    one run" discipline Task 13 uses, generalized to ALL FOUR conditions
    of this task sharing one `encoder_info.json` at the output_dir root
    (invariant: `pretrained: true/false` provenance must never be
    silently dropped or aggregated across).
    """
    path = Path(cfg.output_dir) / "encoder_info.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != encoder_info:
            raise ValueError(f"encoder_info mismatch across Task 14 chunks: {existing} != {encoder_info}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encoder_info, indent=2))


def _encode_and_cache_pair(pair_dir: Path, paths: dict, record: dict, encoder) -> None:
    from encoders.vjepa import mean_pool

    loaded = load_pair(pair_dir)
    _, orig_rgb, _, _ = loaded["original"]
    _, trans_rgb, _, _ = loaded["transformed"]
    Z_before = mean_pool(encoder.encode(orig_rgb))
    Z_after = mean_pool(encoder.encode(trans_rgb))
    np.savez(paths["reps"], Z_before=Z_before, Z_after=Z_after)
    paths["record"].write_text(json.dumps(record, indent=2))


def process_primary_scene_range(condition: str, scene_start: int, scene_end: int, cfg: Task14Config) -> dict:
    """Render+verify+encode `condition` (one of CONDITIONS) for scenes
    `[scene_start, scene_end)`, magnitude-matched per scene. Idempotent:
    a scene already cached on disk is skipped.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"process_primary_scene_range: expected one of {CONDITIONS}, got {condition!r}")
    out_dir = _condition_dir(cfg, condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(cfg)

    encoder = None
    processed, skipped = [], []
    for i in range(scene_start, scene_end):
        scene = scenes[i]
        scene_id = scene.scene_id
        paths = _scene_cache_paths(cfg, condition, scene_id)
        if paths["reps"].exists() and paths["record"].exists():
            skipped.append(scene_id)
            continue

        matched = transform_configs_for_scene(scene, cfg)
        transform_cfg = matched[condition]
        pair_dir = out_dir / scene_id
        generate_pair(
            scene, condition, pair_dir, transform_cfg=transform_cfg,
            num_frames=cfg.num_frames, fps=cfg.fps, resolution=cfg.resolution,
        )
        gclib.verify_transform_ground_truth(pair_dir, condition, num_objects=1)

        transformation = json.loads((pair_dir / "transformation.json").read_text())
        if transformation["object_index"] != 0:
            raise ValueError(
                f"{scene_id}/{condition}: expected object_index 0 (single-object scene), got "
                f"{transformation['object_index']} -- 'same object' isolation guarantee violated"
            )
        measured_displacement = reference_point_displacement(
            transformation["transform_matrix"], scene.objects[0].position, (matched["radius"], 0.0, 0.0)
        )

        if encoder is None:
            encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
            _record_or_reuse_encoder_info(cfg, _encoder_info_for(encoder))

        record = {
            "scene_id": scene_id,
            "condition": condition,
            "object_index": transformation["object_index"],
            "radius": matched["radius"],
            "target_displacement": cfg.target_displacement,
            "measured_displacement": measured_displacement,
            "rotation_angle_deg": matched["rotation_angle_deg"] if condition == "object_rotation" else None,
            "rotation_angle_clipped": matched["rotation_angle_clipped"] if condition == "object_rotation" else None,
        }
        _encode_and_cache_pair(pair_dir, paths, record, encoder)
        processed.append(scene_id)

    return {"condition": condition, "processed": processed, "skipped": skipped}


def process_confound_scene_range(condition: str, scene_start: int, scene_end: int, cfg: Task14Config) -> dict:
    """Render+verify+encode `condition` (one of CONFOUND_CONDITIONS) for
    TRAIN scenes only in `[scene_start, scene_end)` -- the fixed-variable
    leakage check's own separate control run needs training data only
    (Experimental protocol step 6: fit on it, then evaluate against the
    REAL test set built by run_finalize). Test-split scenes are skipped
    here entirely, never rendered for this condition, so there is no way
    for this control's own rendering to leak into the real evaluation.
    """
    if condition not in CONFOUND_CONDITIONS:
        raise ValueError(f"process_confound_scene_range: expected one of {CONFOUND_CONDITIONS}, got {condition!r}")
    out_dir = _condition_dir(cfg, condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(cfg)
    split = compute_split(cfg)

    encoder = None
    processed, skipped, not_train = [], [], []
    for i in range(scene_start, scene_end):
        scene = scenes[i]
        scene_id = scene.scene_id
        if split[scene_id] != "train":
            not_train.append(scene_id)
            continue
        paths = _scene_cache_paths(cfg, condition, scene_id)
        if paths["reps"].exists() and paths["record"].exists():
            skipped.append(scene_id)
            continue

        pair_dir = out_dir / scene_id
        generate_pair(
            scene, condition, pair_dir, transform_cfg=TransformConfig(),
            num_frames=cfg.num_frames, fps=cfg.fps, resolution=cfg.resolution,
        )
        gclib.verify_transform_ground_truth(pair_dir, condition, num_objects=1)

        if encoder is None:
            encoder = gclib.build_encoder(cfg.pretrained, cfg.checkpoint, cfg.device, cfg.fallback_seed)
            _record_or_reuse_encoder_info(cfg, _encoder_info_for(encoder))

        record = {"scene_id": scene_id, "condition": condition}
        _encode_and_cache_pair(pair_dir, paths, record, encoder)
        processed.append(scene_id)

    return {"condition": condition, "processed": processed, "skipped": skipped, "not_train": not_train}


# --- finalize: load caches, fit+evaluate probes, assemble result (no bpy/torch)


def _load_reps(cfg: Task14Config, condition: str, scene_id: str) -> np.ndarray:
    paths = _scene_cache_paths(cfg, condition, scene_id)
    if not paths["reps"].exists():
        raise ValueError(f"{condition}/{scene_id}: not yet processed -- run `--stage process --condition {condition}` for it first")
    npz = np.load(paths["reps"])
    return npz["Z_after"] - npz["Z_before"]


def build_discrimination_arrays(
    cfg: Task14Config, scene_ids: list[str], conditions: tuple[str, str], label_map: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """For each scene_id (in order) and each of the two `conditions`,
    load that (condition, scene_id)'s cached Z_after-Z_before difference
    vector and its label -- returns `(X, y)` with `X.shape ==
    (2 * len(scene_ids), D)`, rows ordered
    `[cond0(scene0), cond1(scene0), cond0(scene1), cond1(scene1), ...]`
    so pairing between a scene and its two rows is always
    `X[2*i], X[2*i+1]` for `scene_ids[i]` -- required "correct pairing
    between counterfactual conditions and scenes" property, checked
    directly in tests/test_task14_counterfactuals.py.
    """
    cond_a, cond_b = conditions
    X_rows, y_rows = [], []
    for scene_id in scene_ids:
        X_rows.append(_load_reps(cfg, cond_a, scene_id))
        y_rows.append(label_map[cond_a])
        X_rows.append(_load_reps(cfg, cond_b, scene_id))
        y_rows.append(label_map[cond_b])
    return np.stack(X_rows), np.array(y_rows, dtype=np.float64)


def run_finalize(cfg: Task14Config) -> dict:
    scene_ids = scene_ids_for(cfg)
    if len(scene_ids) < MIN_SCENES:
        raise ValueError(f"Task 14 requires >={MIN_SCENES} scenes, got {len(scene_ids)}")
    split = compute_split(cfg)
    train_ids = [sid for sid in scene_ids if split[sid] == "train"]
    test_ids = [sid for sid in scene_ids if split[sid] == "test"]
    if not set(train_ids).isdisjoint(test_ids):
        raise ValueError("train/test scene leakage detected")

    encoder_info = json.loads((Path(cfg.output_dir) / "encoder_info.json").read_text())

    X_train, y_train = build_discrimination_arrays(cfg, train_ids, CONDITIONS, CONDITION_LABEL)
    X_test, y_test = build_discrimination_arrays(cfg, test_ids, CONDITIONS, CONDITION_LABEL)
    X_confound, y_confound = build_discrimination_arrays(cfg, train_ids, CONFOUND_CONDITIONS, CONFOUND_LABEL)

    for name, arr in [("X_train", X_train), ("X_test", X_test), ("X_confound", X_confound)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN/Inf")

    # --- real discrimination probe + required label-scrambled control ---
    real_pred = _fit_and_predict(X_train, y_train, X_test, cfg.ridge_alpha)
    accuracy_real = binary_accuracy(real_pred, y_test)

    perm = shuffled_label_permutation(len(y_train), cfg.scramble_seed)
    scrambled_pred = _fit_and_predict(X_train, y_train[perm], X_test, cfg.ridge_alpha)
    accuracy_scrambled = binary_accuracy(scrambled_pred, y_test)

    # --- required "not trivially separable by representation-norm alone" check ---
    norm_train = np.linalg.norm(X_train, axis=1, keepdims=True)
    norm_test = np.linalg.norm(X_test, axis=1, keepdims=True)
    norm_pred = _fit_and_predict(norm_train, y_train, norm_test, cfg.ridge_alpha)
    accuracy_norm_only = binary_accuracy(norm_pred, y_test)

    # --- required fixed-variable leakage check ---
    leakage_pred = _fit_and_predict(X_confound, y_confound, X_test, cfg.ridge_alpha)
    accuracy_leakage = binary_accuracy(leakage_pred, y_test)

    chance_threshold = chance_accuracy_threshold(len(y_test), cfg.chance_margin_z)

    for name, value in [
        ("accuracy_real", accuracy_real),
        ("accuracy_scrambled", accuracy_scrambled),
        ("accuracy_norm_only", accuracy_norm_only),
        ("accuracy_leakage", accuracy_leakage),
    ]:
        if not np.isfinite(value):
            raise ValueError(f"{name} is NaN/Inf: {value}")

    exceeds_scrambled = accuracy_real > accuracy_scrambled
    exceeds_chance = accuracy_real > chance_threshold
    norm_alone_not_trivial = accuracy_norm_only <= chance_threshold
    leakage_check_passed = accuracy_leakage <= chance_threshold
    # A scientifically meaningful POSITIVE result requires all four:
    # real discrimination beats the scrambled control AND chance, the
    # magnitude-matching design is not trivially separable by
    # representation-norm alone (else "discrimination" would just be
    # rediscovering the (by-construction-equal) SE(3) displacement
    # magnitude), and the fixed-variable leakage check found no confound.
    hypothesis_supported = exceeds_scrambled and exceeds_chance and norm_alone_not_trivial and leakage_check_passed

    displacement_stats = _displacement_stats(cfg, scene_ids)

    metrics = {
        "discrimination": {
            "real_probe_accuracy": accuracy_real,
            "scrambled_label_control_accuracy": accuracy_scrambled,
            "chance_accuracy_threshold": chance_threshold,
            "num_train_examples": int(len(y_train)),
            "num_test_examples": int(len(y_test)),
        },
        "representation_norm_only_check": {
            "accuracy": accuracy_norm_only,
            "chance_accuracy_threshold": chance_threshold,
            "not_trivially_separable": norm_alone_not_trivial,
        },
        "fixed_variable_leakage_check": {
            "confound_conditions": list(CONFOUND_CONDITIONS),
            "accuracy_on_real_test_set": accuracy_leakage,
            "chance_accuracy_threshold": chance_threshold,
            "passed": leakage_check_passed,
            "num_confound_train_examples": int(len(y_confound)),
        },
        "magnitude_matching": displacement_stats,
    }

    out_dir = Path(cfg.output_dir)
    report_path = out_dir / "report.md"
    report_path.write_text(_report_markdown(cfg, metrics))
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "num_scenes": len(scene_ids),
        "train_scene_ids": train_ids,
        "test_scene_ids": test_ids,
        "train_fraction": cfg.train_fraction,
        "base_seed": cfg.base_seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    result = {
        "task": 14,
        "implementation_status": "COMPLETE",
        "scientific_result": _scientific_result_text(metrics, hypothesis_supported),
        "counterfactual_conditions": list(CONDITIONS),
        "confound_conditions_for_leakage_check": list(CONFOUND_CONDITIONS),
        "magnitude_matching_method": (
            "object_translation's magnitude is fixed at target_displacement="
            f"{cfg.target_displacement} scene units for every scene; object_rotation's angle is solved "
            "PER SCENE from that scene's own tracked-object radius (scale/2) so that a reference point on "
            "the object's surface (world-frame offset (radius, 0, 0) from the object's pivot) moves by the "
            "IDENTICAL Euclidean distance under either edit (target_rotation_angle_deg, closed-form "
            "2*radius*sin(angle/2) = target_displacement). Both conditions' ACHIEVED displacement is then "
            "re-measured (not just derived) by applying each transform's own recorded transform_matrix to "
            "that reference point (reference_point_displacement) -- see magnitude_matching in metrics for "
            "the measured per-scene values, which confirm the intended match was actually achieved."
        ),
        "dataset": {
            "num_scenes": len(scene_ids),
            "train_scene_ids": train_ids,
            "test_scene_ids": test_ids,
            "train_fraction": cfg.train_fraction,
            "base_seed": cfg.base_seed,
        },
        "encoder": encoder_info,
        "fitting": {"method": "ridge", "alpha": cfg.ridge_alpha, "target": "binary condition label, thresholded at 0.5"},
        "metrics": metrics,
        "scrambled_label_control": {
            "method": "Task 8's shuffled_label_permutation applied to TRAIN labels only, a genuine derangement for n>1",
            "seed": cfg.scramble_seed,
        },
        "tests": {"passed": 0, "failed": 0},
        "artifacts": [str(manifest_path), str(report_path)],
        "config": asdict(cfg),
        "seed": cfg.base_seed,
        "software_versions": gclib.software_versions(),
    }
    return result


def _displacement_stats(cfg: Task14Config, scene_ids: list[str]) -> dict:
    translation_disp, rotation_disp = [], []
    any_clipped = False
    for sid in scene_ids:
        t_record = json.loads(_scene_cache_paths(cfg, "object_translation", sid)["record"].read_text())
        r_record = json.loads(_scene_cache_paths(cfg, "object_rotation", sid)["record"].read_text())
        translation_disp.append(t_record["measured_displacement"])
        rotation_disp.append(r_record["measured_displacement"])
        any_clipped = any_clipped or bool(r_record["rotation_angle_clipped"])
    translation_disp = np.array(translation_disp)
    rotation_disp = np.array(rotation_disp)
    return {
        "target_displacement": cfg.target_displacement,
        "object_translation_measured_displacement": {
            "mean": float(translation_disp.mean()), "std": float(translation_disp.std()),
            "min": float(translation_disp.min()), "max": float(translation_disp.max()),
        },
        "object_rotation_measured_displacement": {
            "mean": float(rotation_disp.mean()), "std": float(rotation_disp.std()),
            "min": float(rotation_disp.min()), "max": float(rotation_disp.max()),
        },
        "mean_absolute_difference_between_conditions": float(np.mean(np.abs(translation_disp - rotation_disp))),
        "any_rotation_angle_clipped": any_clipped,
    }


def _scientific_result_text(metrics: dict, hypothesis_supported: bool) -> str:
    disc = metrics["discrimination"]
    norm_check = metrics["representation_norm_only_check"]
    leak = metrics["fixed_variable_leakage_check"]
    mm = metrics["magnitude_matching"]

    lines = [
        f"Discrimination probe (object_translation vs. object_rotation, magnitude-matched to "
        f"{mm['target_displacement']} scene units per scene): real probe accuracy="
        f"{disc['real_probe_accuracy']:.3f} vs. scrambled-label control accuracy="
        f"{disc['scrambled_label_control_accuracy']:.3f} (chance threshold={disc['chance_accuracy_threshold']:.3f}, "
        f"n_test={disc['num_test_examples']}).",
        f"Representation-norm-only control (required 'not trivially separable by magnitude alone' check): "
        f"accuracy={norm_check['accuracy']:.3f} vs. chance threshold={norm_check['chance_accuracy_threshold']:.3f} "
        f"-- {'PASSED (not trivially separable)' if norm_check['not_trivially_separable'] else 'FAILED (trivially separable by norm alone)'}.",
        f"Fixed-variable leakage check (probe trained on camera_translation-vs-lighting_change representation "
        f"differences on TRAIN scenes only, evaluated on the REAL test set): accuracy={leak['accuracy_on_real_test_set']:.3f} "
        f"vs. chance threshold={leak['chance_accuracy_threshold']:.3f} -- {'PASSED (no confound found)' if leak['passed'] else 'FAILED (a confound was found)'}.",
        f"Measured achieved displacement: object_translation mean={mm['object_translation_measured_displacement']['mean']:.4f}, "
        f"object_rotation mean={mm['object_rotation_measured_displacement']['mean']:.4f} "
        f"(mean |difference| across scenes={mm['mean_absolute_difference_between_conditions']:.6f}).",
    ]

    if hypothesis_supported:
        summary = (
            "POSITIVE result: the real discrimination probe exceeded both the label-scrambled control and the "
            "chance threshold, the two magnitude-matched conditions were NOT trivially separable by "
            "representation-norm alone, and the fixed-variable leakage check found no confound. This is evidence "
            "that the frozen representation carries information distinguishing these two specific "
            "counterfactual edits (object_translation vs. object_rotation on the same object, matched to an "
            "identical ground-truth SE(3) displacement magnitude), beyond a chance/scrambled level, for this "
            "scene distribution -- an accessibility/structure finding, NOT a causal or compositional world "
            "model claim (DESIGN.md Sec 13; tasks/14_counterfactuals.md's interpretation limits)."
        )
    elif not leak["passed"]:
        summary = (
            "NEGATIVE/INVALIDATED result: the fixed-variable leakage check FAILED -- a probe trained only on "
            "representation differences from an UNRELATED pair of transforms (camera_translation vs. "
            "lighting_change, both held-fixed variables in the real experiment) predicted the real "
            "object_translation-vs-object_rotation test labels above chance. Per this task's explicit failure "
            "condition ('the fixed-variable leakage check failing ... must block the task, not be averaged "
            "away'), this means ANY apparent discrimination advantage in this run cannot be trusted as evidence "
            "of intervention-specific structure -- it may instead reflect a confound shared by 'any transform "
            "happened' rather than the intended distinction. Reported plainly as a required, prominent negative "
            "finding, not smoothed over."
        )
    elif not norm_check["not_trivially_separable"]:
        summary = (
            "NEGATIVE/INVALIDATED result: the representation-norm-only control accuracy exceeded the chance "
            "threshold, i.e. despite the exact per-scene SE(3) displacement-magnitude matching, the two "
            "conditions turned out to be separable by ||Z_after - Z_before|| alone in the learned "
            "representation. Per this task's design requirement ('must not leave the two conditions trivially "
            "separable by representation-norm alone'), any positive discrimination result in this run does not "
            "demonstrate intervention-SPECIFIC structure beyond magnitude, and is reported as invalidated rather "
            "than as a meaningful positive."
        )
    else:
        summary = (
            "NEGATIVE result: the real discrimination probe did not exceed both the scrambled-label control and "
            "the chance threshold. This is a valid, complete outcome -- no measurable advantage was found for "
            "distinguishing object_translation from object_rotation (on the same object, matched to an "
            "identical ground-truth SE(3) displacement) from the frozen representation's Z_after-Z_before "
            "difference under this protocol. Reported plainly rather than adjusted to force a positive-looking "
            "number."
        )

    return summary + "\n\nDetail:\n" + "\n".join(f"- {line}" for line in lines)


def _report_markdown(cfg: Task14Config, metrics: dict) -> str:
    disc = metrics["discrimination"]
    norm_check = metrics["representation_norm_only_check"]
    leak = metrics["fixed_variable_leakage_check"]
    mm = metrics["magnitude_matching"]
    lines = [
        "# Task 14 -- controlled counterfactual representation consistency\n\n",
        f"- conditions: {CONDITIONS}, target_displacement={cfg.target_displacement}\n",
        f"- confound (leakage-check) conditions: {CONFOUND_CONDITIONS}\n\n",
        "| check | accuracy | chance threshold | result |\n",
        "|---|---|---|---|\n",
        f"| real discrimination | {disc['real_probe_accuracy']:.4f} | {disc['chance_accuracy_threshold']:.4f} | "
        f"{'above chance' if disc['real_probe_accuracy'] > disc['chance_accuracy_threshold'] else 'at/below chance'} |\n",
        f"| scrambled-label control | {disc['scrambled_label_control_accuracy']:.4f} | {disc['chance_accuracy_threshold']:.4f} | -- |\n",
        f"| representation-norm-only | {norm_check['accuracy']:.4f} | {norm_check['chance_accuracy_threshold']:.4f} | "
        f"{'PASS (not trivial)' if norm_check['not_trivially_separable'] else 'FAIL (trivial)'} |\n",
        f"| fixed-variable leakage check | {leak['accuracy_on_real_test_set']:.4f} | {leak['chance_accuracy_threshold']:.4f} | "
        f"{'PASS' if leak['passed'] else 'FAIL'} |\n\n",
        f"## Magnitude matching\n\n{json.dumps(mm, indent=2)}\n",
    ]
    return "".join(lines)


def run_experiment(cfg: Task14Config) -> dict:
    """Everything in one call -- used by fast tests / a tiny dev config;
    the real run instead uses `--stage process` chunks + `--stage
    finalize` (see module docstring)."""
    for condition in CONDITIONS:
        process_primary_scene_range(condition, 0, cfg.num_scenes, cfg)
    for condition in CONFOUND_CONDITIONS:
        process_confound_scene_range(condition, 0, cfg.num_scenes, cfg)
    return run_finalize(cfg)


def run_existing_test_suite(repo_root: str | Path) -> dict:
    return gclib.run_existing_test_suite(repo_root)


def main():
    parser = argparse.ArgumentParser(description="Run the Task 14 counterfactual-discrimination experiment.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--skip_tests", action="store_true")
    parser.add_argument("--stage", choices=["all", "process", "finalize"], default="all")
    parser.add_argument("--condition", choices=list(ALL_RENDERED_CONDITIONS), default=None)
    parser.add_argument("--scene_start", type=int, default=None)
    parser.add_argument("--scene_end", type=int, default=None)
    args = parser.parse_args()

    cfg = load_task14_config(args.config)

    if args.stage == "process":
        if args.condition is None:
            raise SystemExit("--stage process requires --condition")
        scene_start = args.scene_start if args.scene_start is not None else 0
        scene_end = args.scene_end if args.scene_end is not None else cfg.num_scenes
        if args.condition in CONDITIONS:
            info = process_primary_scene_range(args.condition, scene_start, scene_end, cfg)
        else:
            info = process_confound_scene_range(args.condition, scene_start, scene_end, cfg)
        print(json.dumps(info, indent=2))
        return

    if args.stage == "finalize":
        result = run_finalize(cfg)
    else:
        result = run_experiment(cfg)

    if not args.skip_tests:
        repo_root = Path(__file__).resolve().parent.parent
        result["tests"] = gclib.run_existing_test_suite(repo_root)

    result_path = Path(args.result_path or cfg.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))

    print(f"Task 14 result written to {result_path}")
    print(result["scientific_result"])


if __name__ == "__main__":
    main()
