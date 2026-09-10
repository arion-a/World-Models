"""The V0 end-to-end experiment.

    Scene -> video -> transformed video (known transform T)
    -> frozen encoder -> Z, Z'
    -> learn linear rho(T) on TRAIN scenes
    -> evaluate geometric consistency on TEST scenes
    -> also: identity baseline, shuffled-pairing baseline, raw invariance,
             and one physical-state accessibility probe (camera azimuth)

Runs every transform in `cfg.experiment.transforms`, not just the
flagship one, because the project explicitly wants a like-for-like
comparison between the geometric transforms (which should show real
equivariance) and the appearance-only controls (which should not).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from baselines.identity_baseline import evaluate_identity_baseline
from baselines.shuffled_pairing_baseline import evaluate_shuffled_pairing_baseline
from configs.config import Config, apply_override, load_config
from generation.generate_dataset import generate_dataset
from generation.state_features import camera_azimuth_deg
from metrics.equivariance import evaluate_equivariance
from metrics.invariance import evaluate_invariance
from probes.linear_state_probe import LinearStateProbe
from representations.dataset import build_state_probe_split, build_transform_split, load_manifest
from representations.extract import build_encoder, extract_representations


def run_experiment(cfg: Config, regenerate: bool = False, reextract: bool = False) -> dict:
    out_dir = Path(cfg.dataset.output_dir)
    manifest_path = out_dir / "manifest.json"
    if regenerate or not manifest_path.exists():
        print(f"Generating dataset at {out_dir} ...")
        generate_dataset(cfg, cfg.experiment.transforms)
    manifest = load_manifest(out_dir)

    encoder = build_encoder(cfg)
    encoder_is_pretrained = getattr(encoder, "pretrained", True)
    cache = extract_representations(cfg, encoder, overwrite=reextract)

    results = {
        "encoder": cfg.encoder.name,
        "encoder_pretrained": encoder_is_pretrained,
        "num_scenes": cfg.dataset.num_scenes,
        "transforms": {},
    }

    for transform_name in cfg.experiment.transforms:
        split = build_transform_split(manifest, cache, transform_name)

        equiv_result, rho = evaluate_equivariance(
            transform_name,
            split.Z_train,
            split.Z_prime_train,
            split.Z_test,
            split.Z_prime_test,
            alpha=cfg.probe.ridge_alpha,
        )
        identity_result = evaluate_identity_baseline(transform_name, split.Z_test, split.Z_prime_test)
        shuffled_result = evaluate_shuffled_pairing_baseline(
            transform_name,
            split.Z_train,
            split.Z_prime_train,
            split.Z_test,
            split.Z_prime_test,
            alpha=cfg.probe.ridge_alpha,
        )
        invariance_result = evaluate_invariance(transform_name, split.Z_test, split.Z_prime_test)

        results["transforms"][transform_name] = {
            "equivariance": asdict(equiv_result),
            "identity_baseline": asdict(identity_result),
            "shuffled_pairing_baseline": asdict(shuffled_result),
            "raw_invariance": asdict(invariance_result),
        }
        print(
            f"{transform_name:20s} learned R^2={equiv_result.r2:+.3f} "
            f"identity R^2={identity_result.r2:+.3f} "
            f"shuffled R^2={shuffled_result.r2:+.3f} "
            f"raw cos-sim={invariance_result.mean_cosine_similarity:+.3f}"
        )

    # Component C: physical-state accessibility probe (camera azimuth, degrees).
    Z_train, y_train, Z_test, y_test = build_state_probe_split(manifest, cache, out_dir, camera_azimuth_deg)
    probe = LinearStateProbe.fit(Z_train, y_train, alpha=cfg.probe.ridge_alpha)
    state_probe_r2 = probe.score(Z_test, y_test)
    results["state_probe_camera_azimuth_r2"] = state_probe_r2
    print(f"physical-state probe (camera azimuth) held-out R^2 = {state_probe_r2:+.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the V0 geometric consistency experiment.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--regenerate", action="store_true", help="Force re-generation of the dataset")
    parser.add_argument("--reextract", action="store_true", help="Force re-extraction of representations")
    parser.add_argument("--report", type=str, default=None, help="Path to write the JSON report")
    args = parser.parse_args()

    default_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    cfg = load_config(default_path)
    if args.config:
        cfg = apply_override(cfg, args.config)

    results = run_experiment(cfg, regenerate=args.regenerate, reextract=args.reextract)

    report_path = Path(args.report) if args.report else Path(cfg.dataset.output_dir) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
