"""Task 7B orchestration: DATA_VALIDATED -> LEARNING_CURVE_COMPLETE ->
MODEL_SELECTED -> TEST_EVALUATED, in that order, each gated on the
previous stage's artifacts actually existing (contract Sec. 14's
advancement rules) -- never skips a stage silently.

    python -m experiments.task7b_latent_transformation_discovery.run_learning_curve

Requires Stage 1 data collection (run_stage1_data.py) to have already
completed (protocol.json, z_cache_primary.npz present).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.task7b_latent_transformation_discovery import data, learning_curve, qa_gates, sealed_test, state

OUT_DIR = Path(__file__).resolve().parent
STATE_PATH = Path(__file__).resolve().parent.parent.parent / "state" / "task_07b_result.json"


def _require(path: Path, stage_name: str):
    if not path.exists():
        raise RuntimeError(f"{stage_name}: required artifact {path} does not exist -- run the prior stage first.")


def main():
    s = state.load(STATE_PATH)

    protocol_path = OUT_DIR / "protocol.json"
    z_cache_path = OUT_DIR / "z_cache_primary.npz"
    _require(protocol_path, "PROTOCOL_FROZEN")
    protocol = json.loads(protocol_path.read_text())
    if s.status == state.NOT_STARTED:
        s.transition(state.PROTOCOL_FROZEN, protocol_hash=protocol["protocol_hash"])
        state.save(s, STATE_PATH)

    # --- DATA_VALIDATED ---------------------------------------------------
    _require(z_cache_path, "DATA_VALIDATED")
    print("Running data-validation QA gates...")
    qa_result = qa_gates.run_all(OUT_DIR)
    print(qa_result.render())
    (OUT_DIR / "data_validation_qa_report.md").write_text(qa_result.render())
    (OUT_DIR / "data_validation_qa_report.json").write_text(json.dumps({"passed": qa_result.passed, "layers": qa_result.layers}, indent=2))
    if not qa_result.passed:
        s.transition(state.FAILED_SOFTWARE, reason="data validation QA failed")
        state.save(s, STATE_PATH)
        raise SystemExit("Data validation QA FAILED -- see data_validation_qa_report.md. Stopping (FAILED_SOFTWARE).")
    if s.status == state.PROTOCOL_FROZEN:
        s.transition(state.DATA_VALIDATED)
        state.save(s, STATE_PATH)

    # --- LEARNING_CURVE_COMPLETE -------------------------------------------
    npz = np.load(z_cache_path)
    Z = {k[len("Z__"):]: npz[k] for k in npz.files if k.startswith("Z__")}
    Zp = {k[len("Zp__"):]: npz[k] for k in npz.files if k.startswith("Zp__")}

    split = protocol["split"]
    train_pool_ids = tuple(split["train_pool_ids"])
    val_ids = split["val_ids"]
    test_ids = split["test_ids"]

    all_points = []
    aggregated_by_n = []
    t_start = time.time()
    n_schedule = protocol["n_train_schedule"]
    seeds_per_n = protocol["seeds_per_n"]
    for n_train in n_schedule:
        # contract: >=5 seeds at every point, including the full pool --
        # at n_train == max(n_schedule) (the full training pool), every
        # seed's nested_train_subset returns the SAME 1024 scenes (just
        # differently ordered, which doesn't affect order-invariant fits
        # M0-M5); the seed varies M6's random weight initialization
        # instead, which is exactly "model/training seeds" for a full
        # pool per contract Sec. 5 -- not a special case to branch on here.
        n_seeds = seeds_per_n
        points_this_n = []
        for seed in range(n_seeds):
            train_ids = data.nested_train_subset(train_pool_ids, n_train, seed=seed)
            print(f"Fitting all candidates: N_train={n_train} seed={seed} ({len(train_ids)} scenes)...")
            point = learning_curve.run_point(n_train, seed, list(train_ids), val_ids, Z, Zp)
            points_this_n.append(point)
            all_points.append(point)
            key, entry = learning_curve.select_best_candidate(point)
            print(f"  best this point: {entry['model_key']} {entry['hyperparams']} val_r2={entry['val_metrics']['r2']}")
        agg = learning_curve.aggregate_over_seeds(points_this_n)
        aggregated_by_n.append(agg)
        print(f"N_train={n_train}: mean val R2={agg['mean']:.4f} CI=[{agg['low']:.4f},{agg['high']:.4f}] elapsed={time.time() - t_start:.0f}s")

        (OUT_DIR / "learning_curve_all_points.json").write_text(json.dumps(all_points, indent=2, default=str))
        (OUT_DIR / "learning_curve_aggregated.json").write_text(json.dumps(aggregated_by_n, indent=2))

        if learning_curve.check_stabilization(aggregated_by_n):
            print(f"Stabilization rule satisfied at N_train={n_train}.")
            break

    stabilized = learning_curve.check_stabilization(aggregated_by_n)
    ran_full_schedule = len(aggregated_by_n) == len(n_schedule)
    if not stabilized and not ran_full_schedule:
        raise RuntimeError("Loop exited before stabilizing or exhausting the schedule -- this should not happen.")

    curve_status = "STABILIZED" if stabilized else "NOT_STABILIZED"
    print(f"\nLearning curve status: {curve_status}")
    if s.status == state.DATA_VALIDATED:
        s.transition(state.LEARNING_CURVE_COMPLETE, curve_status=curve_status, n_points=len(all_points))
        state.save(s, STATE_PATH)

    # --- MODEL_SELECTED -----------------------------------------------------
    # Select using the LAST scheduled point that ran (largest N reached),
    # per contract Sec. 7: "largest scheduled point that satisfies the
    # stopping rule; if none stabilizes, use the largest point and mark
    # the result scale-limited."
    final_n = aggregated_by_n[-1]["n_train"]
    final_points = [p for p in all_points if p["n_train"] == final_n]
    # tie-break across seeds at the final N: pick the seed whose best
    # candidate has the highest val R^2 (a single configuration must be
    # selected, contract Sec. 7 "Select exactly one configuration")
    best_per_seed = [(p, *learning_curve.select_best_candidate(p)) for p in final_points]
    best_per_seed.sort(key=lambda t: t[2]["val_metrics"]["r2"], reverse=True)
    winning_point, winning_key, winning_entry = best_per_seed[0]

    selection_record = {
        "final_n_train": final_n,
        "curve_status": curve_status,
        "selected_model_key": winning_entry["model_key"],
        "selected_hyperparams": winning_entry["hyperparams"],
        "selected_n_params": winning_entry["n_params"],
        "selected_seed": winning_point["seed"],
        "selection_rule": protocol["rules"]["selection_rule"],
        "all_candidates_at_final_n_across_seeds": [
            {"seed": p["seed"], "best_key": k, "best_entry": e} for p, k, e in best_per_seed
        ],
        "rejected_candidates_summary": {
            k: v["val_metrics"]["r2"] for k, v in winning_point["candidates"].items() if v["status"] == "ok" and isinstance(v["val_metrics"]["r2"], (int, float))
        },
    }
    (OUT_DIR / "model_selection_record.json").write_text(json.dumps(selection_record, indent=2))
    print(f"\nSelected: {winning_entry['model_key']} {winning_entry['hyperparams']} (val R2={winning_entry['val_metrics']['r2']:.4f}) at N_train={final_n}, seed={winning_point['seed']}")

    if s.status == state.LEARNING_CURVE_COMPLETE:
        s.transition(state.MODEL_SELECTED, selected_model_key=winning_entry["model_key"])
        state.save(s, STATE_PATH)

    # --- TEST_EVALUATED (sealed, once) --------------------------------------
    train_ids_final = data.nested_train_subset(train_pool_ids, final_n, seed=winning_point["seed"])
    result = sealed_test.run_sealed_test(
        winning_entry["model_key"], winning_entry["hyperparams"], list(train_ids_final), test_ids, Z, Zp, seed=winning_point["seed"],
    )
    print(f"\nSealed test: r2={result['selected_config_test_metrics']['r2']}, "
          f"diff_vs_best_control={result['difference_vs_best_control_r2']}")

    if s.status == state.MODEL_SELECTED:
        s.transition(state.TEST_EVALUATED, sealed_test_r2=result["selected_config_test_metrics"]["r2"])
        state.save(s, STATE_PATH)

    print(f"\nTotal learning-curve + selection + sealed test wall time: {time.time() - t_start:.0f}s")
    print("Next: run run_stage2_structure_tests.py for Stage 2/3 (magnitude sweep + composition/inverse/identity/interpolation).")


if __name__ == "__main__":
    main()
