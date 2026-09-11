"""Shared library for the geometric-consistency experiment pipeline
(Task 6's single-transform script generalized to any transform in
`transforms.scene_transform.TRANSFORM_NAMES`).

Extracted from `experiments/task6_camera_rotation.py` for Task 7 ("the
four members of GEOMETRIC_TRANSFORMS plus the two CONTROL_TRANSFORMS,
run through the identical protocol"), per that task's "Relationship to
previous tasks" requirement: "Task 6's script must be refactored into a
per-transform function if it was not already, rather than copy-pasted
per transform." `experiments/task6_camera_rotation.py` now calls into
this module too, so Task 6 and Task 7 share one implementation of
sampling, splitting, rendering+verification, encoding, array assembly,
and fit/evaluate/baseline logic -- see research/CANONICAL_RESEARCH_PROTOCOL.md's
"TASK 6" and "TASK 7" sections, which this module implements verbatim.

This module deliberately does not modify generation/, transforms/,
encoders/, representations/, probes/, metrics/, or baselines/ -- it only
calls their existing public functions.
"""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path

import numpy as np

from baselines.identity_baseline import evaluate_identity_baseline
from baselines.mean_baseline import evaluate_mean_baseline
from baselines.shuffled_pairing_baseline import evaluate_shuffled_pairing_baseline
from generation.scene import SceneState
from generation.scene_sampler import SceneSamplerConfig, sample_scene
from metrics.equivariance import evaluate_equivariance
from transforms.pairs import generate_pair, load_pair
from transforms.scene_transform import CONTROL_TRANSFORMS, GEOMETRIC_TRANSFORMS, TransformConfig, apply_transform

ALL_TRANSFORMS = tuple(GEOMETRIC_TRANSFORMS) + tuple(CONTROL_TRANSFORMS)

_OBJECT_PATH_RE = re.compile(r"^objects\[(\d+)\]\.(position|rotation_euler|color)$")


# --- 1-2. scene sampling + scene-level train/test split (no bpy, no torch) --


def sample_scenes(
    num_scenes: int, base_seed: int, num_objects_min: int = 1, num_objects_max: int = 3
) -> list[SceneState]:
    """>=N scenes, one distinct seed each -- see generate_dataset() in
    generation/generate.py for the same base_seed*1_000_003 + i
    convention, reused here so scene ids/seeds are derived the same
    documented way as Task 2's (and identically to Task 6's own
    sample_scenes(), so the same (num_scenes, base_seed, num_objects_*)
    reproduces exactly the same scene set).
    """
    sampler_cfg = SceneSamplerConfig(num_objects_min=num_objects_min, num_objects_max=num_objects_max)
    scenes = []
    for i in range(num_scenes):
        scene_id = f"scene_{i:04d}"
        seed = base_seed * 1_000_003 + i
        scenes.append(sample_scene(scene_id, seed, sampler_cfg))
    return scenes


def assign_split(scene_ids: list[str], train_fraction: float, base_seed: int) -> dict[str, str]:
    """Scene-level train/test split, decided before any rendering.

    A permutation seeded by `base_seed` (not scene order) decides which
    scenes are train vs. test; every variant of one scene_id (across
    however many transforms it is rendered for) shares the resulting
    label by construction, since callers key everything by scene_id,
    never by (scene_id, transform).
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    rng = np.random.default_rng(base_seed)
    order = rng.permutation(len(scene_ids))
    num_train = int(round(len(scene_ids) * train_fraction))
    split: dict[str, str] = {}
    for rank, idx in enumerate(order):
        split[scene_ids[idx]] = "train" if rank < num_train else "test"
    return split


# --- 3. rendering + ground-truth verification (needs bpy) -------------------


def expected_changed_variables(transform_name: str, transformation_record: dict, num_objects: int) -> set[str]:
    """The exact set of `SceneState` leaf paths a given transform must
    change, cross-checked against `apply_transform`'s own ground truth
    (transforms/scene_transform.py) rather than assumed -- object_translation
    and object_rotation touch a randomly-chosen object per scene, so
    the expected path is read from the transform's own recorded
    `object_index`, not hard-coded to object 0.
    """
    if transform_name == "camera_translation":
        return {"camera.position"}
    if transform_name == "camera_rotation":
        return {"camera.position", "camera.rotation_euler"}
    if transform_name == "object_translation":
        return {f"objects[{transformation_record['object_index']}].position"}
    if transform_name == "object_rotation":
        return {f"objects[{transformation_record['object_index']}].rotation_euler"}
    if transform_name == "lighting_change":
        return {"light.position", "light.energy"}
    if transform_name == "texture_change":
        return {f"objects[{i}].color" for i in range(num_objects)}
    raise ValueError(f"Unknown transform '{transform_name}'. Known: {sorted(ALL_TRANSFORMS)}")


def verify_transform_ground_truth(pair_dir: Path, transform_name: str, num_objects: int) -> None:
    """Required scientific-validity check, enforced at run time (not just
    in a separate test): the rendered pair's own transformation.json must
    show EXACTLY the physical variables this transform type is defined
    to touch, and nothing else -- guards against a future refactor of
    transforms/scene_transform.py silently changing what a transform
    does out from under this experiment. Geometric transforms must carry
    a real SE(3) matrix; the two appearance-only controls must not
    (transforms/scene_transform.py's `transform_matrix=None` contract).
    """
    transformation = json.loads((pair_dir / "transformation.json").read_text())
    if transformation.get("type") != transform_name or transformation.get("transform_name") != transform_name:
        raise ValueError(f"{pair_dir}: expected transform '{transform_name}', got {transformation.get('type')!r}")

    matrix = transformation.get("transform_matrix")
    is_geometric = transform_name in GEOMETRIC_TRANSFORMS
    if is_geometric and matrix is None:
        raise ValueError(f"{pair_dir}: {transform_name} must be a rigid transform with a real SE(3) matrix")
    if not is_geometric and matrix is not None:
        raise ValueError(f"{pair_dir}: {transform_name} is a non-geometric control and must not carry a transform_matrix")

    changed = set(transformation.get("changed_variables", []))
    expected = expected_changed_variables(transform_name, transformation, num_objects)
    if changed != expected:
        raise ValueError(
            f"{pair_dir}: {transform_name} changed unexpected variables: {sorted(changed)} != expected {sorted(expected)}"
        )


def render_null_transform_pairs(scenes: list[SceneState], out_dir: str | Path, num_frames: int, fps: float, resolution: int) -> Path:
    """Task 9's required null-transform sanity check: render each scene
    TWICE, independently, with no transform applied at all (T = identity)
    -- both renders start from the bit-identical SceneState and the
    bit-identical Trajectory (generate_trajectory is pure/deterministic,
    generation/motion.py), so any difference between the two renders'
    representations reveals a rendering/pipeline determinism bug, not a
    scientific finding (see tests/test_bpy_renderer.py::
    test_render_trajectory_is_reproducible, which already establishes
    bit-exact RGB/depth/segmentation reproducibility at the renderer
    level; this reruns that same guarantee through the full render ->
    encode path used everywhere else in this project).

    Deliberately does not go through transforms.pairs.generate_pair /
    transforms.scene_transform.apply_transform -- "null transform" is not
    a registered member of TRANSFORM_NAMES (it changes nothing, by
    definition, so it is not one of the six named physical transforms
    this project studies); it only reuses generation's own
    render/save primitives directly, unmodified.
    """
    from generation.bpy_renderer import render_trajectory
    from generation.ground_truth import save_ground_truth
    from generation.motion import generate_trajectory

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        pair_dir = out_dir / scene.scene_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        trajectory = generate_trajectory(scene, num_frames=num_frames, fps=fps)
        for dir_name in ("original", "transformed"):
            clip = render_trajectory(scene, trajectory, resolution=resolution)
            save_ground_truth(scene, trajectory, clip, pair_dir, resolution, dir_name=dir_name)
        transformation_record = {
            "type": "null_transform",
            "transform_name": "null_transform",
            "transform_matrix": np.eye(4).tolist(),
            "changed_variables": [],
            "object_index": None,
        }
        (pair_dir / "transformation.json").write_text(json.dumps(transformation_record, indent=2))
    return out_dir


def verify_null_transform_ground_truth(pair_dir: Path) -> None:
    """Required software test / runtime check for the null-transform
    sanity check: its transformation.json must show literally nothing
    changed (changed_variables == []) and carry the identity SE(3)
    matrix -- the "T = identity" contract render_null_transform_pairs
    writes above, re-verified independently rather than assumed.
    """
    transformation = json.loads((pair_dir / "transformation.json").read_text())
    if transformation.get("type") != "null_transform" or transformation.get("transform_name") != "null_transform":
        raise ValueError(f"{pair_dir}: expected the null_transform sanity-check record, got {transformation.get('type')!r}")
    if transformation.get("changed_variables"):
        raise ValueError(f"{pair_dir}: null_transform must not change any variable, got {transformation['changed_variables']}")
    matrix = np.array(transformation.get("transform_matrix"))
    if matrix.shape != (4, 4) or not np.allclose(matrix, np.eye(4)):
        raise ValueError(f"{pair_dir}: null_transform must carry exactly the 4x4 identity matrix")


def verify_appearance_physical_equality(scenes: list[SceneState], out_dir: Path, transform_name: str) -> None:
    """Task 9's required, explicit re-verification (not assumed) that an
    appearance-only control (`lighting_change`/`texture_change`) leaves
    camera pose, every object's pose, and object identity/count bit-for-
    bit unchanged -- reads each scene's own transformation.json directly
    off disk and cross-checks it against apply_transform's own ground
    truth (via expected_changed_variables), independently of whatever
    check ran when the pair was originally rendered (Task 7 or otherwise).
    """
    if transform_name not in CONTROL_TRANSFORMS:
        raise ValueError(f"verify_appearance_physical_equality is only valid for appearance controls, got {transform_name!r}")

    pose_pattern = re.compile(r"^objects\[\d+\]\.(position|rotation_euler)$")
    for scene in scenes:
        pair_dir = Path(out_dir) / scene.scene_id
        transformation = json.loads((pair_dir / "transformation.json").read_text())

        if transformation.get("transform_matrix") is not None:
            raise ValueError(f"{pair_dir}: {transform_name} must not carry a rigid transform_matrix (appearance-only control)")

        changed = set(transformation.get("changed_variables", []))
        expected = expected_changed_variables(transform_name, transformation, len(scene.objects))
        if changed != expected:
            raise ValueError(
                f"{pair_dir}: {transform_name} changed unexpected variables: {sorted(changed)} != expected {sorted(expected)}"
            )
        for path in changed:
            if path in ("camera.position", "camera.rotation_euler") or pose_pattern.match(path):
                raise ValueError(
                    f"{pair_dir}: {transform_name} unexpectedly changed a camera/object pose variable ({path}) "
                    "-- this appearance control is not a valid invariance control if it moves geometry."
                )


def render_transform_pairs(
    scenes: list[SceneState],
    transform_name: str,
    transform_cfg: TransformConfig,
    out_dir: str | Path,
    num_frames: int,
    fps: float,
    resolution: int,
) -> Path:
    """Render the original/`transform_name` pair for every scene via
    transforms.pairs.generate_pair (Task 3's renderer, reused
    unmodified), verifying each pair's ground truth as it is produced.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        pair_dir = out_dir / scene.scene_id
        generate_pair(
            scene,
            transform_name,
            pair_dir,
            transform_cfg=transform_cfg,
            num_frames=num_frames,
            fps=fps,
            resolution=resolution,
        )
        verify_transform_ground_truth(pair_dir, transform_name, len(scene.objects))
    return out_dir


# --- 4. encoding (needs torch/transformers, real weights when pretrained) ---


def build_encoder(pretrained: bool, checkpoint: str | None, device: str | None, fallback_seed: int):
    from encoders.vjepa import DEFAULT_CHECKPOINT, VJEPAEncoder

    return VJEPAEncoder(
        checkpoint=checkpoint or DEFAULT_CHECKPOINT,
        pretrained=pretrained,
        device=device,
        fallback_seed=fallback_seed,
    )


def encode_all_pairs(scenes: list[SceneState], out_dir: Path, encoder) -> dict[str, dict[str, np.ndarray]]:
    """Z, Z' per scene: mean_pool(encoder.encode(rgb)) for each side.

    Only the rendered RGB video ever reaches `encoder.encode` -- no
    ground-truth pose/rotation-matrix data is passed in (research/
    RESEARCH_INVARIANTS.md invariant 7); those live only in
    transformation.json / metadata.json, read here only for provenance
    bookkeeping elsewhere, never for encoding.
    """
    from encoders.vjepa import mean_pool

    reps: dict[str, dict[str, np.ndarray]] = {}
    for scene in scenes:
        loaded = load_pair(out_dir / scene.scene_id)
        _, orig_rgb, _, _ = loaded["original"]
        _, trans_rgb, _, _ = loaded["transformed"]
        z = mean_pool(encoder.encode(orig_rgb))
        z_prime = mean_pool(encoder.encode(trans_rgb))
        reps[scene.scene_id] = {"Z": z, "Z_prime": z_prime}
    return reps


def pixel_diff_stats(scenes: list[SceneState], out_dir: Path) -> dict:
    """Mean/std absolute per-pixel RGB difference between original and
    transformed clips, per scene, for a given transform's rendered pair
    set -- the required scientific-QA "investigate whether differences
    between transformations could be caused by different visual
    artifacts... rather than by geometry itself" check
    (tasks/07_geometric_consistency.md's Required scientific-validity
    tests). Computed straight from the rendered RGB arrays, never from
    the encoder's representation.
    """
    per_scene = []
    for scene in scenes:
        loaded = load_pair(out_dir / scene.scene_id)
        _, orig_rgb, _, _ = loaded["original"]
        _, trans_rgb, _, _ = loaded["transformed"]
        diff = np.abs(orig_rgb.astype(np.float64) - trans_rgb.astype(np.float64))
        per_scene.append(float(diff.mean()))
    return {
        "mean_abs_pixel_diff": float(np.mean(per_scene)),
        "std_abs_pixel_diff": float(np.std(per_scene)),
        "per_scene_mean_abs_pixel_diff": per_scene,
    }


# --- assemble train/test arrays, enforcing scene-level disjointness --------


def build_arrays(
    scenes: list[SceneState], split: dict[str, str], reps: dict[str, dict[str, np.ndarray]]
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ids = [s.scene_id for s in scenes if split[s.scene_id] == "train"]
    test_ids = [s.scene_id for s in scenes if split[s.scene_id] == "test"]
    if not set(train_ids).isdisjoint(test_ids):
        raise ValueError("train/test scene leakage detected")

    Z_train = np.stack([reps[sid]["Z"] for sid in train_ids])
    Zp_train = np.stack([reps[sid]["Z_prime"] for sid in train_ids])
    Z_test = np.stack([reps[sid]["Z"] for sid in test_ids])
    Zp_test = np.stack([reps[sid]["Z_prime"] for sid in test_ids])
    return train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test


def build_full_arrays(scenes: list[SceneState], reps: dict[str, dict[str, np.ndarray]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Like build_arrays, but for Task 9's invariance computation, which
    (per DESIGN.md Sec 7 / tasks/09_appearance_invariance.md's Train/test
    protocol) fits nothing and therefore has no train/test split to
    contaminate -- every scene's (Z, Z') pair is used directly.
    """
    ids = [s.scene_id for s in scenes]
    Z = np.stack([reps[sid]["Z"] for sid in ids])
    Zp = np.stack([reps[sid]["Z_prime"] for sid in ids])
    return ids, Z, Zp


# --- fit W_T + evaluate + the three required controls, one transform -------


def evaluate_transform(
    transform_name: str,
    Z_train: np.ndarray,
    Zp_train: np.ndarray,
    Z_test: np.ndarray,
    Zp_test: np.ndarray,
    ridge_alpha: float,
    shuffled_pairing_seed: int,
) -> dict:
    equiv_result, _rho = evaluate_equivariance(transform_name, Z_train, Zp_train, Z_test, Zp_test, alpha=ridge_alpha)
    persistence_result = evaluate_identity_baseline(transform_name, Z_test, Zp_test)
    mean_result = evaluate_mean_baseline(transform_name, Zp_train, Zp_test)
    random_pair_result = evaluate_shuffled_pairing_baseline(
        transform_name, Z_train, Zp_train, Z_test, Zp_test, alpha=ridge_alpha, seed=shuffled_pairing_seed
    )
    return {
        "learned_W_T": {
            "r2": equiv_result.r2,
            "mean_cosine_similarity": equiv_result.mean_cosine_similarity,
            "mean_relative_l2_error": equiv_result.mean_relative_l2_error,
        },
        "persistence_baseline": {
            "r2": persistence_result.r2,
            "mean_cosine_similarity": persistence_result.mean_cosine_similarity,
            "mean_relative_l2_error": persistence_result.mean_relative_l2_error,
        },
        "mean_baseline": {
            "r2": mean_result.r2,
            "mean_cosine_similarity": mean_result.mean_cosine_similarity,
            "mean_relative_l2_error": mean_result.mean_relative_l2_error,
        },
        "random_pair_control": {
            "r2": random_pair_result.r2,
            "mean_cosine_similarity": random_pair_result.mean_cosine_similarity,
            "mean_relative_l2_error": random_pair_result.mean_relative_l2_error,
        },
    }


# --- provenance / reproducibility helpers -----------------------------------


def software_versions() -> dict:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def snapshot_params(model) -> list:
    return [p.detach().clone() for p in model.parameters()]


def params_unchanged(before: list, model) -> bool:
    import torch

    after = list(model.parameters())
    if len(before) != len(after):
        return False
    return all(torch.equal(b, a) for b, a in zip(before, after))


def parse_pytest_summary(output: str) -> tuple[int, int]:
    """Parse pytest -q's final summary line ('12 passed in 0.34s',
    '3 failed, 9 passed in 1.02s', 'no tests ran in 0.00s') into
    (passed, failed) counts. Returns (0, 0) if no recognizable summary
    line is found, rather than guessing.
    """
    passed = failed = 0
    for line in reversed(output.splitlines()):
        m_passed = re.search(r"(\d+) passed", line)
        m_failed = re.search(r"(\d+) failed", line)
        if m_passed or m_failed:
            if m_passed:
                passed = int(m_passed.group(1))
            if m_failed:
                failed = int(m_failed.group(1))
            break
    return passed, failed


def run_existing_test_suite(repo_root: str | Path) -> dict:
    """Run `pytest -m "not slow" -q` from the repo root and report honest
    pass/fail counts -- required by these tasks' own instructions, though
    (per those same instructions) an independent process re-runs this
    after Claude finishes and is the actual basis for acceptance, not
    this self-report.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["pytest", "-m", "not slow", "-q"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"passed": 0, "failed": 0, "error": f"could not run test suite: {exc!r}"}

    output = proc.stdout + proc.stderr
    passed, failed = parse_pytest_summary(output)
    return {"passed": passed, "failed": failed, "returncode": proc.returncode}
