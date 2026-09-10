"""Assemble (Z, Z') arrays for a transform type, respecting the scene-level
train/test split recorded in the dataset manifest.

This is the one place that reads `manifest["scenes"][i]["split"]`, so the
"never split a scene's original/transformed pair across train and test"
guarantee lives in a single, unit-tested function
(tests/test_scene_split.py) rather than being re-implemented per experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generation.scene import SceneState


@dataclass
class TransformSplit:
    transform_name: str
    Z_train: np.ndarray
    Z_prime_train: np.ndarray
    Z_test: np.ndarray
    Z_prime_test: np.ndarray
    train_scene_ids: list[str]
    test_scene_ids: list[str]


def load_manifest(output_dir: str | Path) -> dict:
    return json.loads((Path(output_dir) / "manifest.json").read_text())


def build_transform_split(manifest: dict, cache: dict[str, np.ndarray], transform_name: str) -> TransformSplit:
    train_scene_ids, test_scene_ids = [], []
    Z_train, Zp_train, Z_test, Zp_test = [], [], [], []

    for scene in manifest["scenes"]:
        scene_id = scene["scene_id"]
        z = cache[f"{scene_id}__original"]
        z_prime = cache[f"{scene_id}__{transform_name}"]
        if scene["split"] == "train":
            Z_train.append(z)
            Zp_train.append(z_prime)
            train_scene_ids.append(scene_id)
        else:
            Z_test.append(z)
            Zp_test.append(z_prime)
            test_scene_ids.append(scene_id)

    assert set(train_scene_ids).isdisjoint(test_scene_ids), "train/test scene leakage detected"

    def _stack(items: list[np.ndarray]) -> np.ndarray:
        if not items:
            return np.empty((0,), dtype=np.float32)
        return np.stack(items)

    return TransformSplit(
        transform_name=transform_name,
        Z_train=_stack(Z_train),
        Z_prime_train=_stack(Zp_train),
        Z_test=_stack(Z_test),
        Z_prime_test=_stack(Zp_test),
        train_scene_ids=train_scene_ids,
        test_scene_ids=test_scene_ids,
    )


def load_scene_state(output_dir: str | Path, scene_id: str, variant: str = "original") -> SceneState:
    path = Path(output_dir) / "states" / scene_id / f"{variant}.json"
    return SceneState.from_json(path.read_text())


def build_state_probe_split(
    manifest: dict,
    cache: dict[str, np.ndarray],
    output_dir: str | Path,
    feature_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (Z_train, y_train, Z_test, y_test) for a linear state probe.

    `feature_fn(SceneState) -> float | np.ndarray` computes the physical
    quantity we're testing accessibility of (e.g. camera azimuth).
    """
    Z_train, y_train, Z_test, y_test = [], [], [], []
    for scene in manifest["scenes"]:
        scene_id = scene["scene_id"]
        state = load_scene_state(output_dir, scene_id, "original")
        z = cache[f"{scene_id}__original"]
        y = feature_fn(state)
        if scene["split"] == "train":
            Z_train.append(z)
            y_train.append(y)
        else:
            Z_test.append(z)
            y_test.append(y)
    return (
        np.stack(Z_train),
        np.atleast_2d(np.array(y_train, dtype=np.float32).reshape(len(y_train), -1)),
        np.stack(Z_test),
        np.atleast_2d(np.array(y_test, dtype=np.float32).reshape(len(y_test), -1)),
    )
