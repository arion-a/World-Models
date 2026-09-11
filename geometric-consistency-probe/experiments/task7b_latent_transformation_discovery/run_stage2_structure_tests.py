"""Task 7B Stage 2/3: multi-magnitude sweep + structure tests (contract
Sec. 8-9). Requires run_learning_curve.py to have already produced
model_selection_record.json (the SAME selected functional form is reused
for every magnitude here, per contract Sec. 8 -- Stage 2 never re-runs
model selection).

Compute-budget scoping (recorded here, not hidden): fits F_theta using a
FIXED 256-scene subset of the train pool (STRUCTURE_TEST_TRAIN_SIZE),
not the full N_train selected in Stage 1 -- algebraic-property
verification does not need learning-curve-scale N, and this keeps
Stage 2/3 cost bounded regardless of how large Stage 1's stabilization
point turned out to be (contract's own compute_budget_note in
protocol.py already documents this choice). Structure-test evaluation
itself uses the FULL fixed test split (all test_ids), since that is
comparatively cheap and is the actual "held-out scenes" the contract
requires.

New renders needed, all TRANSFORMED-side only (the original side is
magnitude-independent and already cached from Stage 1):
  - TRAIN (256 scenes) x 6 new direct-fit magnitudes {-60,-30,-20,-10,10,60}
    (excludes the primary +30, already available, and 0/+20 which need no
    new fitting -- see below)
  - TEST (up to 220 scenes) x 7 magnitudes (the 6 above + the
    interpolation-reserved +20, needed ONLY as ground truth, never fit)

theta=0 needs NO new rendering: T_0 is the identity transform by
definition, so Z'(0) == Z exactly -- the identity test below fits the
selected form on (Z_train, Z_train) pairs directly, using data already
cached from Stage 1.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.task7b_latent_transformation_discovery import data, learning_curve, state, structure_tests as st
from transforms.scene_transform import TransformConfig

OUT_DIR = Path(__file__).resolve().parent
RENDER_DIR = OUT_DIR / "rendered" / "camera_rotation_stage2"
Z_CACHE_STAGE2_PATH = OUT_DIR / "z_cache_stage2.npz"
STATE_PATH = Path(__file__).resolve().parent.parent.parent / "state" / "task_07b_result.json"

STRUCTURE_TEST_TRAIN_SIZE = 256
# If the selected form is M6 (residual MLP), its early stopping needs a
# genuine held-out validation slice -- carved out of the 256-scene
# structure-test subset itself (rather than rendering more scenes at
# every new magnitude just for this) so early stopping is never
# evaluated on the same rows it is trained on, for any selected form.
STRUCTURE_TEST_INTERNAL_VAL_SIZE = 56
NEW_DIRECT_FIT_MAGNITUDES = (-60.0, -30.0, -20.0, -10.0, 10.0, 60.0)  # excludes primary +30 (already have it)
GROUND_TRUTH_ONLY_MAGNITUDES = (data.INTERPOLATION_RESERVED_MAGNITUDE_DEG,)  # +20 -- never fit, ground truth only

# (theta1, theta2) pairs whose sum is ALSO among the fitted magnitudes
# (primary +30 included as a "fitted" magnitude for this purpose).
COMPOSITION_PAIRS = [(-10.0, -10.0), (-30.0, 60.0), (-20.0, -10.0), (-10.0, -20.0), (30.0, 30.0)]
INVERSE_PAIRS = [(10.0, -10.0), (30.0, -30.0), (60.0, -60.0)]
INTERPOLATION_NEIGHBORS = (10.0, 30.0)  # bracket the reserved 20.0


def _load_z_cache(path: Path) -> tuple[dict, dict]:
    Z, Zp = {}, {}
    if path.exists():
        npz = np.load(path)
        for key in npz.files:
            if key.startswith("Z__"):
                Z[key[len("Z__"):]] = npz[key]
            elif key.startswith("Zp__"):
                Zp[key[len("Zp__"):]] = npz[key]
    return Z, Zp


def _save_z_cache(path: Path, Zp_by_magnitude: dict[float, dict]) -> None:
    payload = {}
    for magnitude, zp_dict in Zp_by_magnitude.items():
        for sid, vec in zp_dict.items():
            payload[f"Zp__{magnitude}__{sid}"] = vec
    np.savez(path, **payload)


def render_and_encode_magnitude(magnitude: float, scene_ids: list, scenes_by_id: dict, encoder, enc_cfg: dict) -> dict:
    """Renders (if not cached) and encodes the TRANSFORMED side only, for
    one magnitude, over the given scenes. Returns {scene_id: Zp_vector}."""
    from encoders.vjepa import mean_pool

    from experiments.task7b_latent_transformation_discovery.fast_render import generate_pair_fast

    cfg = TransformConfig(fixed_azimuth_deg=magnitude, fixed_elevation_deg=data.FIXED_ELEVATION_DEG)
    out = {}
    mag_dir = RENDER_DIR / f"mag_{magnitude:g}"
    for sid in scene_ids:
        pair_dir = mag_dir / sid
        if not (pair_dir / "transformed" / "rgb.npy").exists():
            # Transformed side only: the original is magnitude-independent
            # and already in z_cache_primary.npz from Stage 1 -- rendering
            # it again here per magnitude would double this stage's cost
            # for output that's discarded (render_and_encode_magnitude
            # below only ever reads the transformed side).
            generate_pair_fast(
                scenes_by_id[sid], data.TRANSFORM_NAME, pair_dir,
                transform_cfg=cfg, num_frames=enc_cfg["num_frames"], fps=enc_cfg["fps"], resolution=enc_cfg["resolution"],
                sides=("transformed",),
            )
        trans_rgb = np.load(pair_dir / "transformed" / "rgb.npy")
        out[sid] = mean_pool(encoder.encode(trans_rgb))
    return out


def main():
    import experiments.geometric_consistency_lib as gclib

    protocol = json.loads((OUT_DIR / "protocol.json").read_text())
    selection = json.loads((OUT_DIR / "model_selection_record.json").read_text())
    model_key, hyperparams = selection["selected_model_key"], selection["selected_hyperparams"]
    print(f"Reusing selected form from Stage 1: {model_key} {hyperparams}")

    split = protocol["split"]
    train_pool_ids = tuple(split["train_pool_ids"])
    test_ids = split["test_ids"]
    train_ids_256 = list(data.nested_train_subset(train_pool_ids, STRUCTURE_TEST_TRAIN_SIZE, seed=0))
    # Internal train/val split carved out of the 256-scene structure-test
    # subset itself (never rendering extra scenes just for this): the last
    # STRUCTURE_TEST_INTERNAL_VAL_SIZE scenes are held out as a genuine
    # validation set so that, if the selected form is M6, its early
    # stopping is never evaluated against rows it was trained on.
    train_ids_fit = train_ids_256[: STRUCTURE_TEST_TRAIN_SIZE - STRUCTURE_TEST_INTERNAL_VAL_SIZE]
    val_ids_internal = train_ids_256[STRUCTURE_TEST_TRAIN_SIZE - STRUCTURE_TEST_INTERNAL_VAL_SIZE :]

    enc_cfg = protocol["encoder"]
    scenes = data.build_master_population(protocol["master_population_size"], base_seed=protocol["base_seed"])
    scenes_by_id = {s.scene_id: s for s in scenes}
    encoder = gclib.build_encoder(pretrained=True, checkpoint=None, device=None, fallback_seed=enc_cfg["fallback_seed"])

    Z_primary, Zp_primary = _load_z_cache(OUT_DIR / "z_cache_primary.npz")  # Z is magnitude-independent

    all_magnitudes_to_render = list(NEW_DIRECT_FIT_MAGNITUDES) + list(GROUND_TRUTH_ONLY_MAGNITUDES)
    t0 = time.time()
    Zp_by_magnitude: dict[float, dict] = {30.0: Zp_primary}  # primary already available
    for magnitude in all_magnitudes_to_render:
        print(f"Rendering/encoding magnitude={magnitude} deg for {len(train_ids_256)} train + {len(test_ids)} test scenes...")
        scene_ids_needed = train_ids_256 + list(test_ids) if magnitude != data.INTERPOLATION_RESERVED_MAGNITUDE_DEG else list(test_ids)
        Zp_by_magnitude[magnitude] = render_and_encode_magnitude(magnitude, scene_ids_needed, scenes_by_id, encoder, enc_cfg)
        _save_z_cache(Z_CACHE_STAGE2_PATH, Zp_by_magnitude)
        print(f"  done, elapsed={time.time() - t0:.0f}s")

    # --- fit F_theta for every directly-fit magnitude (primary reused) ---
    # Fit on train_ids_fit only; val_ids_internal is a genuine held-out
    # slice used for early stopping (relevant only to M6 -- M0-M5 ignore
    # the val arguments entirely, per learning_curve._fit_candidate).
    fitted_by_magnitude = {}
    Z_train_fit = np.stack([Z_primary[sid] for sid in train_ids_fit])
    Z_val_internal = np.stack([Z_primary[sid] for sid in val_ids_internal])
    for magnitude in (30.0,) + NEW_DIRECT_FIT_MAGNITUDES:
        Zp_train_fit = np.stack([Zp_by_magnitude[magnitude][sid] for sid in train_ids_fit])
        Zp_val_internal = np.stack([Zp_by_magnitude[magnitude][sid] for sid in val_ids_internal])
        fitted_by_magnitude[magnitude] = learning_curve._fit_candidate(
            model_key, hyperparams, Z_train_fit, Zp_train_fit, Z_val_internal, Zp_val_internal, seed=0,
        )
    # identity: fit on (Z_train, Z_train) directly -- no new rendering (theta=0 is the identity transform)
    fitted_by_magnitude[0.0] = learning_curve._fit_candidate(
        model_key, hyperparams, Z_train_fit, Z_train_fit, Z_val_internal, Z_val_internal, seed=0,
    )

    D = Z_train_fit.shape[1]
    W_by_magnitude = {m: st.operator_matrix(f, D) for m, f in fitted_by_magnitude.items()}
    Z_test = np.stack([Z_primary[sid] for sid in test_ids])

    results = {
        "selected_model_key": model_key,
        "selected_hyperparams": hyperparams,
        "structure_test_train_size": STRUCTURE_TEST_TRAIN_SIZE,
        "structure_test_internal_val_size": STRUCTURE_TEST_INTERNAL_VAL_SIZE,
        "structure_test_fit_size": len(train_ids_fit),
    }

    # composition
    composition_results = []
    for theta1, theta2 in COMPOSITION_PAIRS:
        theta_sum = theta1 + theta2
        if theta1 not in fitted_by_magnitude or theta2 not in fitted_by_magnitude or theta_sum not in fitted_by_magnitude:
            composition_results.append({"theta1": theta1, "theta2": theta2, **st.not_applicable("one of the required operators was not fit")})
            continue
        res = st.composition_test(
            fitted_by_magnitude[theta1].predict, fitted_by_magnitude[theta2].predict, fitted_by_magnitude[theta_sum].predict,
            Z_test, W1=W_by_magnitude[theta1], W2=W_by_magnitude[theta2], W_sum=W_by_magnitude[theta_sum],
        )
        composition_results.append({"theta1": theta1, "theta2": theta2, "theta_sum": theta_sum, **res})
    results["composition"] = composition_results

    # inverse
    inverse_results = []
    for theta, neg_theta in INVERSE_PAIRS:
        res = st.inverse_test(fitted_by_magnitude[theta].predict, fitted_by_magnitude[neg_theta].predict, Z_test, W=W_by_magnitude[theta], W_neg=W_by_magnitude[neg_theta])
        inverse_results.append({"theta": theta, **res})
    results["inverse"] = inverse_results

    # identity
    results["identity"] = st.identity_test(fitted_by_magnitude[0.0].predict, Z_test, W_zero=W_by_magnitude[0.0])

    # interpolation: reserve +20, bracket with +10/+30, compare against the TRUE rendered +20 target
    theta_a, theta_b = INTERPOLATION_NEIGHBORS
    theta_interior = data.INTERPOLATION_RESERVED_MAGNITUDE_DEG
    lam = (theta_interior - theta_a) / (theta_b - theta_a)
    Zp_true_interior = np.stack([Zp_by_magnitude[theta_interior][sid] for sid in test_ids])
    results["interpolation"] = st.interpolation_test(fitted_by_magnitude[theta_a].predict, fitted_by_magnitude[theta_b].predict, lam, Z_test, Zp_true_interior)

    (OUT_DIR / "structure_test_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({k: v for k, v in results.items() if k not in ("selected_model_key", "selected_hyperparams")}, indent=2, default=str)[:4000])

    s = state.load(STATE_PATH)
    if s.status == state.TEST_EVALUATED:
        s.transition(state.STRUCTURE_TESTED)
        state.save(s, STATE_PATH)
    print(f"\nTotal Stage 2/3 wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
