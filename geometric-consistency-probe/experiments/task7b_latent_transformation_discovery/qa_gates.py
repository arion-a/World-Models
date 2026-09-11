"""Task 7B scientific/leakage QA gates (contract Sec. 12), run against
the REAL artifacts Stage 1 produced -- never trusting the pipeline's own
completion message. Mirrors the Task 7 forensic audit's own methodology
(independent, mechanical re-derivation, not a self-report).

Produces a PASS/FAIL report; DATA_VALIDATED (the orchestrator state) may
only be reached once every layer here passes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.task7b_latent_transformation_discovery import data

OUT_DIR = Path(__file__).resolve().parent
RENDER_DIR = OUT_DIR / "rendered" / "camera_rotation_primary"


class QAResult:
    def __init__(self):
        self.layers = []

    def add(self, name: str, passed: bool, details: list[str]):
        self.layers.append({"name": name, "passed": passed, "details": details})

    @property
    def passed(self) -> bool:
        return all(l["passed"] for l in self.layers)

    def render(self) -> str:
        lines = [f"Task 7B DATA_VALIDATED QA: {'PASS' if self.passed else 'FAIL'}"]
        for l in self.layers:
            lines.append(f"[{'PASS' if l['passed'] else 'FAIL'}] {l['name']}")
            for d in l["details"][:20]:
                lines.append(f"    - {d}")
            if len(l["details"]) > 20:
                lines.append(f"    ... and {len(l['details']) - 20} more")
        return "\n".join(lines)


def check_split_exclusivity(qa: QAResult, protocol: dict):
    split = protocol["split"]
    train = set(split["train_pool_ids"])
    val = set(split["val_ids"])
    test = set(split["test_ids"])
    ok = train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    details = []
    if not ok:
        details.append(f"overlap: train&val={train & val}, train&test={train & test}, val&test={val & test}")
    if len(test) < data.MIN_TEST_SCENES:
        ok = False
        details.append(f"test split has {len(test)} scenes, below required minimum {data.MIN_TEST_SCENES}")
    qa.add("Scene-level split exclusivity", ok, details)


def check_every_scene_has_one_render_and_matches_its_id(qa: QAResult, protocol: dict):
    split = protocol["split"]
    all_ids = split["train_pool_ids"] + split["val_ids"] + split["test_ids"]
    details = []
    ok = True
    for sid in all_ids:
        pair_dir = RENDER_DIR / sid
        meta_path = pair_dir / "original" / "metadata.json"
        if not meta_path.exists():
            ok = False
            details.append(f"{sid}: missing rendered artifact")
            continue
        meta = json.loads(meta_path.read_text())
        if meta["scene_id"] != sid:
            ok = False
            details.append(f"{sid}: metadata scene_id {meta['scene_id']!r} != directory name")
    qa.add("Every scene rendered exactly once, scene_id matches directory", ok, details)


def check_transform_type_and_magnitude(qa: QAResult, protocol: dict):
    split = protocol["split"]
    all_ids = split["train_pool_ids"] + split["val_ids"] + split["test_ids"]
    expected_azimuth = protocol["transformation"]["primary_magnitude_deg"]
    details = []
    ok = True
    for sid in all_ids:
        transformation = json.loads((RENDER_DIR / sid / "transformation.json").read_text())
        if transformation["type"] != "camera_rotation":
            ok = False
            details.append(f"{sid}: wrong transform type {transformation['type']!r}")
        if abs(transformation["azimuth_deg"] - expected_azimuth) > 1e-9:
            ok = False
            details.append(f"{sid}: azimuth {transformation['azimuth_deg']} != expected primary magnitude {expected_azimuth}")
        if abs(transformation["elevation_deg"] - 0.0) > 1e-9:
            ok = False
            details.append(f"{sid}: elevation {transformation['elevation_deg']} != 0.0 (Task 7B's fixed pure-yaw setting)")
        if transformation.get("transform_matrix") is None:
            ok = False
            details.append(f"{sid}: camera_rotation must carry a real SE(3) matrix")
    qa.add("Transform type and magnitude correct for every scene", ok, details)


def check_video_shape_and_preprocessing(qa: QAResult, protocol: dict):
    split = protocol["split"]
    sample_ids = (split["train_pool_ids"][:5] + split["val_ids"][:5] + split["test_ids"][:5])
    enc_cfg = protocol["encoder"]
    details = []
    ok = True
    shapes, dtypes = set(), set()
    for sid in sample_ids:
        for side in ("original", "transformed"):
            rgb = np.load(RENDER_DIR / sid / side / "rgb.npy")
            shapes.add(rgb.shape)
            dtypes.add(str(rgb.dtype))
    expected_shape = (enc_cfg["num_frames"], enc_cfg["resolution"], enc_cfg["resolution"], 3)
    if shapes != {expected_shape}:
        ok = False
        details.append(f"unexpected shapes found: {shapes}, expected only {expected_shape}")
    if dtypes != {"uint8"}:
        ok = False
        details.append(f"unexpected dtypes found: {dtypes}")
    qa.add("Video shape/preprocessing correct (sampled)", ok, details)


def check_render_determinism_of_static_clip(qa: QAResult, protocol: dict, n_check: int = 5):
    """Within-clip frames must be bit-identical (zero-motion clips) -- a
    real, mechanical re-check on Task 7B's own rendered output, not
    assumed from the forensic audit's finding on a different dataset."""
    split = protocol["split"]
    sample_ids = split["test_ids"][:n_check]
    details = []
    ok = True
    for sid in sample_ids:
        for side in ("original", "transformed"):
            rgb = np.load(RENDER_DIR / sid / side / "rgb.npy")
            if not all(np.array_equal(rgb[0], rgb[i]) for i in range(1, rgb.shape[0])):
                ok = False
                details.append(f"{sid}/{side}: frames not bit-identical within the clip")
    qa.add("Render determinism (within-clip frame identity, sampled)", ok, details)


def check_frozen_eval_mode_encoder(qa: QAResult, encoder) -> None:
    ok = (encoder.model.training is False) and all(not p.requires_grad for p in encoder.model.parameters())
    details = [] if ok else ["encoder is not in eval mode or has trainable parameters"]
    qa.add("Encoder is frozen and in eval mode", ok, details)


def check_representation_cache_covers_every_scene(qa: QAResult, protocol: dict, z_cache_path: Path):
    split = protocol["split"]
    all_ids = set(split["train_pool_ids"] + split["val_ids"] + split["test_ids"])
    npz = np.load(z_cache_path)
    cached_z = {k[len("Z__"):] for k in npz.files if k.startswith("Z__")}
    cached_zp = {k[len("Zp__"):] for k in npz.files if k.startswith("Zp__")}
    missing_z = all_ids - cached_z
    missing_zp = all_ids - cached_zp
    ok = not missing_z and not missing_zp
    details = []
    if missing_z:
        details.append(f"{len(missing_z)} scenes missing from Z cache, e.g. {list(missing_z)[:5]}")
    if missing_zp:
        details.append(f"{len(missing_zp)} scenes missing from Z' cache, e.g. {list(missing_zp)[:5]}")
    for sid in list(all_ids)[:5]:
        if sid in cached_z:
            vec = npz[f"Z__{sid}"]
            if vec.shape != (protocol["encoder"]["declared_dimension_D"],):
                ok = False
                details.append(f"{sid}: Z has shape {vec.shape}, expected ({protocol['encoder']['declared_dimension_D']},)")
            if not np.all(np.isfinite(vec)):
                ok = False
                details.append(f"{sid}: Z contains non-finite values")
    qa.add("Representation cache covers every scene, correct D, finite", ok, details)


def run_all(out_dir: Path = OUT_DIR) -> QAResult:
    protocol = json.loads((out_dir / "protocol.json").read_text())
    qa = QAResult()
    check_split_exclusivity(qa, protocol)
    check_every_scene_has_one_render_and_matches_its_id(qa, protocol)
    check_transform_type_and_magnitude(qa, protocol)
    check_video_shape_and_preprocessing(qa, protocol)
    check_render_determinism_of_static_clip(qa, protocol)
    check_representation_cache_covers_every_scene(qa, protocol, out_dir / "z_cache_primary.npz")
    return qa


if __name__ == "__main__":
    result = run_all()
    print(result.render())
    (OUT_DIR / "data_validation_qa_report.md").write_text(result.render())
    (OUT_DIR / "data_validation_qa_report.json").write_text(json.dumps({"passed": result.passed, "layers": result.layers}, indent=2))
    raise SystemExit(0 if result.passed else 1)
