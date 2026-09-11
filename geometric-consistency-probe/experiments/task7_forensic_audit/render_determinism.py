"""Section 4: render the exact same scene twice (independent bpy renders,
same scene_id/seed/transform) and compare pixel arrays numerically.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import experiments.geometric_consistency_lib as gclib
from transforms.pairs import generate_pair
from transforms.scene_transform import TransformConfig

OUT_DIR = Path(__file__).resolve().parent
TMP_DIR = OUT_DIR / "_render_determinism_tmp"


def main():
    scenes = gclib.sample_scenes(2, base_seed=0)
    scene = scenes[0]

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    results = {}
    for transform_name in ["camera_rotation", "lighting_change"]:
        pair_dir_a = TMP_DIR / f"{transform_name}_a"
        pair_dir_b = TMP_DIR / f"{transform_name}_b"
        generate_pair(scene, transform_name, pair_dir_a, transform_cfg=TransformConfig(), num_frames=4, fps=4.0, resolution=128)
        generate_pair(scene, transform_name, pair_dir_b, transform_cfg=TransformConfig(), num_frames=4, fps=4.0, resolution=128)

        entry = {}
        for side in ("original", "transformed"):
            rgb_a = np.load(pair_dir_a / side / "rgb.npy")
            rgb_b = np.load(pair_dir_b / side / "rgb.npy")
            identical = np.array_equal(rgb_a, rgb_b)
            max_abs_diff = int(np.abs(rgb_a.astype(np.int16) - rgb_b.astype(np.int16)).max())
            mean_abs_diff = float(np.abs(rgb_a.astype(np.float64) - rgb_b.astype(np.float64)).mean())
            entry[side] = {"bit_identical": identical, "max_abs_pixel_diff": max_abs_diff, "mean_abs_pixel_diff": mean_abs_diff}

            trans_a = json.loads((pair_dir_a / "transformation.json").read_text())
            trans_b = json.loads((pair_dir_b / "transformation.json").read_text())
            entry["transformation_json_identical"] = trans_a == trans_b

            # also check whether the 4 frames WITHIN one render are
            # identical to each other (static-clip / zero-motion clip ->
            # they should be, unless the renderer has per-frame stochastic
            # noise not tied to a fixed seed).
            within_clip_frames_identical = all(np.array_equal(rgb_a[0], rgb_a[i]) for i in range(1, rgb_a.shape[0]))
            entry[f"{side}_within_clip_frames_identical"] = within_clip_frames_identical

        results[transform_name] = entry

    print(json.dumps(results, indent=2))
    (OUT_DIR / "render_determinism.json").write_text(json.dumps(results, indent=2))
    shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    main()
