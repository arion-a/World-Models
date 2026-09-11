"""Forensic audit: independently re-derive Task 7's Z/Z' representations
from the real, already-rendered rgb.npy files on disk (no re-rendering),
re-run the EXACT SAME evaluate_transform/evaluate_equivariance/baseline
code Task 7 used, and diff the recomputed numbers against the recorded
state/task_07_result.json. This is a from-scratch reproduction, not a
re-read of the original numbers -- if it matches, that is strong evidence
against a pairing/indexing/caching bug and confirms encoder determinism
in situ (same code, same weights, same real input files, independently
invoked). It also dumps per-scene pairing records and raw Z arrays for
the representation-statistics and regression-conditioning sections of
the audit, and prints pairing-integrity + video-integrity fields for a
random sample of scenes.

Does NOT touch or modify state/task_07_result.json or experiments/
geometric_consistency/ -- read-only against those.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

import experiments.geometric_consistency_lib as gclib
from transforms.pairs import load_pair

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
RENDER_DIR = REPO_ROOT / "experiments" / "geometric_consistency"
RESULT_PATH = REPO_ROOT / "state" / "task_07_result.json"


def main():
    t_start = time.time()
    recorded = json.loads(RESULT_PATH.read_text())
    cfg = recorded["config"]
    ridge_alpha = cfg["ridge_alpha"]
    shuffled_pairing_seed = cfg["shuffled_pairing_seed"]

    train_ids = recorded["dataset"]["train_scene_ids"]
    test_ids = recorded["dataset"]["test_scene_ids"]
    all_scene_ids = sorted(set(train_ids) | set(test_ids))
    print(f"train={len(train_ids)} test={len(test_ids)} total={len(all_scene_ids)}")

    # --- 1. PAIRING INTEGRITY: print >=20 randomly sampled pairing records,
    # verified straight from disk paths + transformation.json, before any
    # encoding happens. -----------------------------------------------------
    rng = random.Random(12345)
    transforms = list(recorded["transforms_evaluated"])
    pairing_records = []
    sample_specs = [(rng.choice(transforms), rng.choice(all_scene_ids)) for _ in range(24)]
    for transform_name, scene_id in sample_specs:
        pair_dir = RENDER_DIR / transform_name / scene_id
        orig_meta = json.loads((pair_dir / "original" / "metadata.json").read_text())
        trans_meta = json.loads((pair_dir / "transformed" / "metadata.json").read_text())
        transformation = json.loads((pair_dir / "transformation.json").read_text())
        record = {
            "transform": transform_name,
            "scene_id": scene_id,
            "original_scene_id_in_metadata": orig_meta["scene_id"],
            "transformed_scene_id_in_metadata": trans_meta["scene_id"],
            "original_seed": orig_meta["seed"],
            "transformed_seed": trans_meta["seed"],
            "original_rgb_path": str(pair_dir / "original" / "rgb.npy"),
            "transformed_rgb_path": str(pair_dir / "transformed" / "rgb.npy"),
            "transformation_type": transformation["type"],
            "transformation_matches_dir": transformation["type"] == transform_name,
            "scene_id_matches_dir_name": (orig_meta["scene_id"] == scene_id and trans_meta["scene_id"] == scene_id),
            "seeds_match_each_other": orig_meta["seed"] == trans_meta["seed"],
            "changed_variables": transformation.get("changed_variables"),
            "transform_matrix_present": transformation.get("transform_matrix") is not None,
        }
        pairing_records.append(record)
    print(f"\n=== PAIRING INTEGRITY: {len(pairing_records)} sampled records ===")
    for r in pairing_records:
        print(json.dumps(r, indent=None))
    all_ok = all(r["transformation_matches_dir"] and r["scene_id_matches_dir_name"] and r["seeds_match_each_other"] for r in pairing_records)
    print(f"\nALL {len(pairing_records)} SAMPLED PAIRS INTERNALLY CONSISTENT: {all_ok}")
    (OUT_DIR / "pairing_integrity_sample.json").write_text(json.dumps(pairing_records, indent=2))

    # --- 3. VIDEO PAIR INTEGRITY across ALL scenes/transforms (cheap, from
    # metadata + rgb.npy header, no encoding). -------------------------------
    video_integrity_issues = []
    video_checks = 0
    resolutions, fpses, frame_counts, dtypes, shapes = set(), set(), set(), set(), set()
    for transform_name in transforms:
        for scene_id in all_scene_ids:
            pair_dir = RENDER_DIR / transform_name / scene_id
            for side in ("original", "transformed"):
                meta = json.loads((pair_dir / side / "metadata.json").read_text())
                rgb = np.load(pair_dir / side / "rgb.npy", mmap_mode="r")
                resolutions.add(meta["resolution"])
                fpses.add(meta["fps"])
                frame_counts.add(meta["num_frames"])
                dtypes.add(str(rgb.dtype))
                shapes.add(rgb.shape)
                video_checks += 1
                if rgb.shape != (meta["num_frames"], meta["resolution"], meta["resolution"], 3):
                    video_integrity_issues.append(f"{pair_dir}/{side}: rgb.shape {rgb.shape} != metadata-implied shape")
                if str(rgb.dtype) != "uint8":
                    video_integrity_issues.append(f"{pair_dir}/{side}: dtype {rgb.dtype} != uint8")
    print(f"\n=== VIDEO PAIR INTEGRITY: checked {video_checks} clips ===")
    print(f"distinct resolutions={resolutions} fps={fpses} frame_counts={frame_counts} dtypes={dtypes} shapes={shapes}")
    print(f"issues found: {len(video_integrity_issues)}")
    for issue in video_integrity_issues[:20]:
        print(" -", issue)

    # --- 2. TRANSFORMATION INTEGRITY: for a sample, confirm camera/object
    # pose actually changed by exactly the recorded transform, and that
    # UNRELATED state (per fixed_variables) is bit-identical original vs
    # transformed. -----------------------------------------------------------
    transform_integrity_records = []
    for transform_name, scene_id in sample_specs:
        pair_dir = RENDER_DIR / transform_name / scene_id
        orig_meta = json.loads((pair_dir / "original" / "metadata.json").read_text())
        trans_meta = json.loads((pair_dir / "transformed" / "metadata.json").read_text())
        transformation = json.loads((pair_dir / "transformation.json").read_text())
        fixed_vars = transformation.get("fixed_variables", [])
        changed_vars = transformation.get("changed_variables", [])

        def get_path(meta, path):
            # path like "camera.position" or "objects[0].position"
            import re
            m = re.match(r"^objects\[(\d+)\]\.(\w+)$", path)
            if m:
                idx, field = int(m.group(1)), m.group(2)
                if field in ("position", "rotation_euler"):
                    return meta["objects"][idx]["per_frame_pose"][0][field]
                return meta["objects"][idx][field]
            if path in ("camera.position", "camera.rotation_euler"):
                field = path.split(".")[1]
                return meta["camera"]["per_frame_pose"][0][field]
            if path == "light.position":
                return meta["light"]["position"]
            if path == "light.energy":
                return meta["light"]["energy"]
            return None

        def values_equal(ov, tv):
            if isinstance(ov, str) or isinstance(tv, str):
                return ov == tv
            return bool(np.allclose(np.asarray(ov, dtype=float), np.asarray(tv, dtype=float), atol=1e-9))

        fixed_ok = []
        for path in fixed_vars:
            ov, tv = get_path(orig_meta, path), get_path(trans_meta, path)
            if ov is not None and tv is not None:
                fixed_ok.append((path, values_equal(ov, tv)))
        changed_ok = []
        for path in changed_vars:
            ov, tv = get_path(orig_meta, path), get_path(trans_meta, path)
            if ov is not None and tv is not None:
                changed_ok.append((path, not values_equal(ov, tv)))

        transform_integrity_records.append({
            "transform": transform_name,
            "scene_id": scene_id,
            "all_fixed_vars_unchanged": all(ok for _, ok in fixed_ok),
            "num_fixed_checked": len(fixed_ok),
            "all_changed_vars_actually_differ": all(ok for _, ok in changed_ok),
            "num_changed_checked": len(changed_ok),
            "fixed_failures": [p for p, ok in fixed_ok if not ok],
            "changed_failures": [p for p, ok in changed_ok if not ok],
        })
    print(f"\n=== TRANSFORMATION INTEGRITY: {len(transform_integrity_records)} sampled records ===")
    for r in transform_integrity_records:
        print(json.dumps(r))
    (OUT_DIR / "transform_integrity_sample.json").write_text(json.dumps(transform_integrity_records, indent=2))

    # --- Re-derive Z/Z' from scratch for every transform via the REAL
    # encoder + REAL evaluate_transform code path, and diff against the
    # recorded result. This is the reproducibility + pairing + regression
    # + representation-statistics backbone. -----------------------------------
    print("\n=== Building real encoder (pretrained VJEPAEncoder) ===")
    encoder = gclib.build_encoder(pretrained=True, checkpoint=None, device=None, fallback_seed=0)
    print(f"encoder.pretrained={encoder.pretrained} checkpoint={encoder.checkpoint}")

    # dedupe: verify "original" rendering is byte-identical across every
    # transform folder for the same scene_id (it should be -- same scene,
    # same seed, same renderer call, independently invoked per transform)
    # before relying on that to only encode each scene's original once.
    dedupe_mismatches = []
    reference_original_rgb = {}
    for scene_id in all_scene_ids:
        ref = None
        for transform_name in transforms:
            rgb = np.load(RENDER_DIR / transform_name / scene_id / "original" / "rgb.npy")
            if ref is None:
                ref = rgb
                reference_original_rgb[scene_id] = rgb
            elif not np.array_equal(ref, rgb):
                dedupe_mismatches.append((scene_id, transform_name))
    print(f"\noriginal-render identical across all 6 transform folders for every scene: {len(dedupe_mismatches) == 0}")
    if dedupe_mismatches:
        print("MISMATCHES:", dedupe_mismatches[:20])
    (OUT_DIR / "original_render_cross_transform_identity.json").write_text(
        json.dumps({"all_identical": len(dedupe_mismatches) == 0, "mismatches": dedupe_mismatches}, indent=2)
    )

    print("\nEncoding all unique original scenes once (dedup)...")
    t0 = time.time()
    Z_by_scene = {}
    for i, scene_id in enumerate(all_scene_ids):
        from encoders.vjepa import mean_pool
        Z_by_scene[scene_id] = mean_pool(encoder.encode(reference_original_rgb[scene_id]))
        if (i + 1) % 10 == 0:
            print(f"  original {i + 1}/{len(all_scene_ids)}  elapsed={time.time() - t0:.0f}s")
    print(f"originals done in {time.time() - t0:.0f}s")

    from encoders.vjepa import mean_pool

    recomputed_results = {}
    all_z_for_stats = {"Z": [], "Zp_by_transform": {}}
    for transform_name in transforms:
        print(f"\nEncoding transformed side for '{transform_name}'...")
        t0 = time.time()
        Zp_by_scene = {}
        for scene_id in all_scene_ids:
            rgb = np.load(RENDER_DIR / transform_name / scene_id / "transformed" / "rgb.npy")
            Zp_by_scene[scene_id] = mean_pool(encoder.encode(rgb))
        print(f"  done in {time.time() - t0:.0f}s")

        Z_train = np.stack([Z_by_scene[sid] for sid in train_ids])
        Zp_train = np.stack([Zp_by_scene[sid] for sid in train_ids])
        Z_test = np.stack([Z_by_scene[sid] for sid in test_ids])
        Zp_test = np.stack([Zp_by_scene[sid] for sid in test_ids])

        metrics_block = gclib.evaluate_transform(transform_name, Z_train, Zp_train, Z_test, Zp_test, ridge_alpha, shuffled_pairing_seed)
        recomputed_results[transform_name] = metrics_block
        all_z_for_stats["Zp_by_transform"][transform_name] = {sid: Zp_by_scene[sid].tolist() for sid in all_scene_ids}

    all_z_for_stats["Z"] = {sid: Z_by_scene[sid].tolist() for sid in all_scene_ids}
    np.savez_compressed(
        OUT_DIR / "recomputed_representations.npz",
        **{f"Z__{sid}": Z_by_scene[sid] for sid in all_scene_ids},
        **{f"Zp__{t}__{sid}": all_z_for_stats["Zp_by_transform"][t][sid] for t in transforms for sid in all_scene_ids},
    )

    # --- diff recomputed vs recorded -----------------------------------------
    diffs = {}
    for transform_name in transforms:
        recorded_block = recorded["results"][transform_name]
        recomputed_block = recomputed_results[transform_name]
        diff = {}
        for method in ("learned_W_T", "persistence_baseline", "mean_baseline", "random_pair_control"):
            for metric in ("r2", "mean_cosine_similarity", "mean_relative_l2_error"):
                rec_v = recorded_block[method][metric]
                new_v = recomputed_block[method][metric]
                diff[f"{method}.{metric}"] = {"recorded": rec_v, "recomputed": new_v, "abs_diff": abs(rec_v - new_v)}
        diffs[transform_name] = diff

    max_abs_diff = max(d["abs_diff"] for t in diffs.values() for d in t.values())
    print(f"\n=== REPRODUCTION DIFF: max abs diff across all transforms/methods/metrics = {max_abs_diff:.3e} ===")
    for transform_name, diff in diffs.items():
        worst = max(diff.items(), key=lambda kv: kv[1]["abs_diff"])
        print(f"  {transform_name}: worst={worst[0]} recorded={worst[1]['recorded']:.6f} recomputed={worst[1]['recomputed']:.6f} diff={worst[1]['abs_diff']:.2e}")

    (OUT_DIR / "reproduction_diff.json").write_text(json.dumps({"max_abs_diff": max_abs_diff, "per_transform": diffs}, indent=2))
    (OUT_DIR / "recomputed_results.json").write_text(json.dumps(recomputed_results, indent=2))

    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
