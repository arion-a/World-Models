"""Task 7B protocol freeze (contract Sec. 4): a content-hashed,
immutable protocol.json + protocol.md written BEFORE any test-scene
rendering/encoding. Any change after a sealed test evaluation requires a
new protocol ID and a new, clearly labelled follow-on study -- enforced
here by refusing to overwrite an existing protocol.json whose content
differs from what would be (re)generated, once a sealed test result
exists alongside it (see freeze_protocol's guard).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from experiments.task7b_latent_transformation_discovery import data
from experiments.task7b_latent_transformation_discovery.models import LOW_RANK_RANKS, MLP_HIDDEN_SIZES

PROTOCOL_NOTES = [
    (
        "Task 7B is a new, independent discovery study -- it does not reuse Task 7's 40-scene "
        "population or its state/task_07_result.json in any fitting or evaluation step. Its own "
        "master population (see master_population_size) is sampled fresh via the same deterministic "
        "sampler (geometric_consistency_lib.sample_scenes) Task 6/7 use, with Task 7B's own base_seed, "
        "so its scene set is disjoint in general from Task 7's (both start numbering at scene_0000, "
        "but are independent draws from the sampler and are never cross-referenced)."
    ),
    (
        "camera_rotation's azimuth is pinned to a single fixed elevation of 0.0 degrees for the "
        "entire Task 7B study (transforms.scene_transform.TransformConfig.fixed_elevation_deg), making "
        "it a pure single-axis (world +Z) yaw -- verified, not assumed, to satisfy an exact one-parameter "
        "rotation-group composition law in tests/test_task7b_composition_law.py before this protocol "
        "relies on it for Stage 3's structure tests."
    ),
    (
        "N_train=1024 (the full training pool) was included in the schedule per an explicit, "
        "session-compute-budget decision made WITH the user before any rendering began -- not silently "
        "dropped. See compute_budget_note below."
    ),
]


def encoder_config() -> dict:
    """Identical encoder/preprocessing settings to Task 7 (same checkpoint,
    frozen, mean-pooled) -- an explicit, recorded choice for consistency
    with the audited pipeline, not an unexamined default."""
    return {
        "encoder_name": "VJEPAEncoder",
        "checkpoint": "facebook/vjepa2-vitl-fpc64-256",
        "pretrained": True,
        "frozen": True,
        "pooling": "mean_pool",
        "declared_dimension_D": 1024,
        "resolution": 128,
        "num_frames": 4,
        "fps": 4.0,
        "fallback_seed": 0,
    }


def transformation_config() -> dict:
    return {
        "family": data.TRANSFORM_NAME,
        "coordinate_convention": "world +Z axis yaw (azimuth), fixed elevation=0.0 deg -- see generation/COORDINATE_SYSTEM.md for the base world-frame convention this azimuth is defined within",
        "sign_convention": "positive azimuth_deg = positive rotation about world +Z per transforms.se3.rotation_about_axis's right-hand-rule convention",
        "magnitudes_deg": list(data.MAGNITUDES_DEG),
        "primary_magnitude_deg": data.PRIMARY_MAGNITUDE_DEG,
        "interpolation_reserved_magnitude_deg": data.INTERPOLATION_RESERVED_MAGNITUDE_DEG,
        "stage2_direct_fit_magnitudes_deg": list(data.STAGE2_DIRECT_FIT_MAGNITUDES_DEG),
        "composition_law": "T_theta2(T_theta1(S)) == T_(theta1+theta2)(S); verified numerically in tests/test_task7b_composition_law.py before use",
        "valid_one_parameter_group": True,
    }


def model_grid() -> dict:
    return {
        "M0_persistence": {"parameters": 0},
        "M1_scalar": {"parameters": 1},
        "M2_diagonal": {"parameters": encoder_config()["declared_dimension_D"]},
        "M3_low_rank_residual": {
            "ranks": list(LOW_RANK_RANKS),
            # string keys deliberately -- JSON has no integer dict keys, and
            # round-tripping int keys through json.dumps/loads changes their
            # sort order under sort_keys=True (numeric vs. lexicographic),
            # which would silently change this protocol's content hash on
            # every reload. Using strings from the start makes the hash
            # stable across write/read cycles, not just within one process.
            "parameters_by_rank": {str(r): 2 * encoder_config()["declared_dimension_D"] * r for r in LOW_RANK_RANKS},
            "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "M4_full_linear": {
            "parameters": encoder_config()["declared_dimension_D"] ** 2,
            "alpha_grid": [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0],
        },
        "M5_affine": {
            "parameters": encoder_config()["declared_dimension_D"] ** 2 + encoder_config()["declared_dimension_D"],
            "alpha_grid": [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0],
        },
        "M6_residual_mlp": {
            "hidden_sizes": list(MLP_HIDDEN_SIZES),
            "parameters_by_h": {
                str(h): 2 * encoder_config()["declared_dimension_D"] * h + encoder_config()["declared_dimension_D"] + h
                for h in MLP_HIDDEN_SIZES
            },
            "note": (
                "Parameter count is for the equation Zhat'=Z+B tanh(AZ+a)+b as implemented "
                "(2Dh+D+h) -- see models.py's module docstring for a documented discrepancy "
                "with the contract's own worked example, which this protocol does not silently adopt."
            ),
            "optimizer": "Adam",
            "early_stopping": "validation MSE, patience=100 epochs, max_epochs=3000",
        },
        "controls": ["control_train_mean", "control_persistence", "control_shuffled_pair"],
    }


def selection_and_stopping_rules() -> dict:
    return {
        "selection_rule": "At each N, choose the candidate/hyperparameter combo with highest mean VALIDATION R^2. Ties within 0.01 -> lower parameter count wins, then earlier position in the M0-M6 hierarchy.",
        "stopping_rule": "Adequately powered if the final two scheduled N's mean validation R^2 differ by <0.02 AND their 95% seed-bootstrap CIs overlap. Otherwise the full schedule is run and the curve is labeled NOT_STABILIZED.",
        "sealed_test_rule": "The sealed test command evaluates ONLY the single already-selected (model, hyperparameter) configuration plus all pre-specified controls, exactly once, on the fixed test split. A rerun is permitted only for a documented software failure discovered BEFORE result inspection.",
        "acceptance_criteria": [
            "All QA gates (software + scientific/leakage) pass.",
            "Every feasible scheduled N_train x seed combination is represented, including failures.",
            "Selection record proves validation-only choice and the complexity tie-break rule.",
            "Sealed test artifact includes all controls, CIs, and a contamination check.",
            "Every mathematically-valid Stage 3 structure test has a result or a NOT_APPLICABLE reason.",
        ],
    }


def compute_budget_note() -> str:
    return (
        "N_train=1024 requires a 1463-scene master population (70% train pool = 1024, per "
        "compute_population_size()). Measured throughput this session: ~5s/scene render+encode "
        "at 4 frames/128px. Stage 1's full population was rendered as a single ~2 hour background "
        "job -- an explicit choice confirmed with the user (not a silent scope cut) given the "
        "alternative of capping the schedule at N_train=512. Stage 2/3 (the 7 additional magnitudes' "
        "structure tests) use a much smaller, separately fixed scene subset (not the full 1024-scene "
        "pool) since algebraic-property verification does not need learning-curve-scale N -- see "
        "STRUCTURE_TEST_SUBSET_SIZE in run_structure_tests.py."
    )


def build_protocol(seed: int = data.BASE_SEED) -> dict:
    total_population = data.compute_population_size()
    val_size = round(total_population * data.VAL_FRACTION)
    train_pool_size = data.TRAIN_POOL_SIZE
    scenes = data.build_master_population(total_population, base_seed=seed)
    scene_ids = [s.scene_id for s in scenes]
    split = data.three_way_split(scene_ids, train_pool_size=train_pool_size, val_size=val_size, seed=seed)

    protocol = {
        "task": "7B",
        "title": "Discovery of Latent Geometric Transformation Structure",
        "encoder": encoder_config(),
        "transformation": transformation_config(),
        "master_population_size": split.total_population,
        "base_seed": seed,
        "split": {
            "train_pool_ids": list(split.train_pool_ids),
            "val_ids": list(split.val_ids),
            "test_ids": list(split.test_ids),
            "train_pool_size": len(split.train_pool_ids),
            "val_size": len(split.val_ids),
            "test_size": len(split.test_ids),
            "min_test_scenes_required": data.MIN_TEST_SCENES,
        },
        "n_train_schedule": list(data.N_TRAIN_SCHEDULE),
        "seeds_per_n": data.SEEDS_PER_N,
        "model_grid": model_grid(),
        "metrics": ["r2_multi_output (undefined on zero variance)", "mse", "relative_l2_error", "cosine_similarity", "95pct_bootstrap_ci_over_seeds"],
        "rules": selection_and_stopping_rules(),
        "compute_budget_note": compute_budget_note(),
        "protocol_notes": PROTOCOL_NOTES,
        "immutable_historical_paths": [
            "state/task_07_result.json",
            "experiments/geometric_consistency/",
            "experiments/task7_forensic_audit/",
            "experiments/task7_diagnostic/",
        ],
    }
    return protocol


def content_hash(protocol: dict) -> str:
    canonical = json.dumps(protocol, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def render_protocol_markdown(protocol: dict, protocol_hash: str) -> str:
    lines = [
        f"# Task 7B Protocol (frozen, hash `{protocol_hash[:16]}`)",
        "",
        f"**Master population:** {protocol['master_population_size']} scenes (base_seed={protocol['base_seed']})",
        f"**Split:** train_pool={protocol['split']['train_pool_size']}, val={protocol['split']['val_size']}, test={protocol['split']['test_size']}",
        f"**N_train schedule:** {protocol['n_train_schedule']} ({protocol['seeds_per_n']} seeds each)",
        f"**Transform:** {protocol['transformation']['family']}, magnitudes {protocol['transformation']['magnitudes_deg']} deg, primary={protocol['transformation']['primary_magnitude_deg']} deg",
        "",
        "## Selection & stopping rules",
        "",
        f"- Selection: {protocol['rules']['selection_rule']}",
        f"- Stopping: {protocol['rules']['stopping_rule']}",
        f"- Sealed test: {protocol['rules']['sealed_test_rule']}",
        "",
        "## Compute budget",
        "",
        protocol["compute_budget_note"],
        "",
        "## Protocol notes",
        "",
        *[f"- {note}" for note in protocol["protocol_notes"]],
        "",
        "## Immutable historical paths (never modified by Task 7B)",
        "",
        *[f"- `{p}`" for p in protocol["immutable_historical_paths"]],
        "",
    ]
    return "\n".join(lines)


def freeze_protocol(out_dir: Path, seed: int = data.BASE_SEED) -> tuple[dict, str]:
    """Writes protocol.json + protocol.md + a separate split manifest, and
    refuses to silently overwrite a DIFFERENT protocol once a sealed test
    result already exists next to it (contract: 'any change after a test
    evaluation requires a new protocol ID')."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(seed=seed)
    protocol_hash = content_hash(protocol)

    existing_protocol_path = out_dir / "protocol.json"
    sealed_test_path = out_dir / "sealed_test_result.json"
    if existing_protocol_path.exists() and sealed_test_path.exists():
        existing = json.loads(existing_protocol_path.read_text())
        existing.pop("protocol_hash", None)  # was not present in the dict content_hash() was originally computed over
        if content_hash(existing) != protocol_hash:
            raise RuntimeError(
                "A sealed test result already exists for a DIFFERENT protocol. "
                "Per contract Sec. 4, changing the protocol after a test evaluation "
                "requires a new protocol ID and a new, separately labelled follow-on study."
            )

    protocol["protocol_hash"] = protocol_hash
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out_dir / "protocol.md").write_text(render_protocol_markdown(protocol, protocol_hash))
    (out_dir / "split_manifest.json").write_text(json.dumps(protocol["split"], indent=2))
    return protocol, protocol_hash
