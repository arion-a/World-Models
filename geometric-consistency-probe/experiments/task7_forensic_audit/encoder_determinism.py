"""Section 5: encode the exact same video twice with the real pretrained
encoder and measure ||Z1-Z2||/||Z1|| and cosine similarity. Also confirms
model.eval() / no_grad / requires_grad=False (by inspection, printed
here) and reports the actual token-sequence shape produced for this
project's actual (4-frame, 128x128) clips vs. the checkpoint's native
training configuration (frames_per_clip=64), to speak to whether the
encoder is being run far outside its training distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parent.parent


def main():
    import experiments.geometric_consistency_lib as gclib
    from encoders.vjepa import mean_pool

    encoder = gclib.build_encoder(pretrained=True, checkpoint=None, device=None, fallback_seed=0)

    checks = {
        "model.training (should be False -- eval mode)": bool(encoder.model.training),
        "any_param_requires_grad (should be False -- frozen)": any(p.requires_grad for p in encoder.model.parameters()),
        "checkpoint_config": {
            "frames_per_clip": encoder.model.config.frames_per_clip,
            "tubelet_size": encoder.model.config.tubelet_size,
            "crop_size": encoder.model.config.crop_size,
            "patch_size": encoder.model.config.patch_size,
            "hidden_size": encoder.model.config.hidden_size,
        },
    }

    rgb = np.load(REPO_ROOT / "experiments/geometric_consistency/camera_rotation/scene_0000/original/rgb.npy")
    checks["actual_input_clip_shape"] = list(rgb.shape)

    z1 = encoder.encode(rgb)
    z2 = encoder.encode(rgb)
    identical = np.array_equal(z1, z2)
    diff_norm = float(np.linalg.norm(z1 - z2))
    z1_norm = float(np.linalg.norm(z1))
    rel_diff = diff_norm / z1_norm if z1_norm > 0 else float("nan")
    cos = float(np.dot(z1.flatten(), z2.flatten()) / (np.linalg.norm(z1.flatten()) * np.linalg.norm(z2.flatten())))

    checks["encode_determinism"] = {
        "tokens_shape": list(z1.shape),
        "bit_identical_across_two_calls": identical,
        "relative_l2_diff": rel_diff,
        "cosine_similarity": cos,
    }

    # native temporal tubelets vs. what this project actually feeds
    native_temporal_tubelets = encoder.model.config.frames_per_clip // encoder.model.config.tubelet_size
    actual_temporal_tubelets = rgb.shape[0] // encoder.model.config.tubelet_size
    checks["temporal_context_analysis"] = {
        "native_temporal_tubelets_from_checkpoint_training_config": native_temporal_tubelets,
        "actual_temporal_tubelets_fed_by_this_project": actual_temporal_tubelets,
        "fraction_of_native_temporal_context_used": actual_temporal_tubelets / native_temporal_tubelets,
        "note": (
            "This checkpoint's config.json declares frames_per_clip=64 (checkpoint name itself "
            "is 'vjepa2-vitl-fpc64-256' -- fpc64 = frames per clip 64). This project renders "
            "4-frame clips (see configs/experiments/task7_geometric_consistency.yaml's num_frames=4), "
            "i.e. 2 of the 32 temporal tubelets (tubelet_size=2) the model was pretrained with -- "
            "6.25% of its native temporal context. VJEPA2's attention uses RoPE-based relative "
            "position ids computed from the ACTUAL input token count (transformers/models/vjepa2/"
            "modeling_vjepa2.py's VJEPA2RopeAttention.get_position_ids), not a fixed-size learned "
            "positional-embedding buffer, so this does not crash or silently misindex -- but it is "
            "still a large distribution shift in temporal context length relative to pretraining, "
            "which is a plausible contributor to degraded representation quality independent of any "
            "code bug."
        ),
    }

    # also encode a video that differs meaningfully, to sanity check the
    # encoder is not producing a near-constant output for ANY input
    # (which would look like 'representation collapse' independent of
    # pairing).
    rgb2 = np.load(REPO_ROOT / "experiments/geometric_consistency/camera_rotation/scene_0001/original/rgb.npy")
    z3 = encoder.encode(rgb2)
    cross_scene_cos = float(np.dot(z1.flatten(), z3.flatten()) / (np.linalg.norm(z1.flatten()) * np.linalg.norm(z3.flatten())))
    checks["cross_scene_sanity"] = {
        "cosine_similarity_scene0000_vs_scene0001": cross_scene_cos,
        "note": "should be clearly lower than the same-video repeat-encode cosine similarity above if the encoder is not collapsed.",
    }

    print(json.dumps(checks, indent=2))
    (OUT_DIR / "encoder_determinism.json").write_text(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
