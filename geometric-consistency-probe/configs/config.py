"""Minimal, dependency-light config loading.

We deliberately do not pull in Hydra/OmegaConf: for a project this size a
single YAML file loaded into nested dataclasses is enough, is trivial to
read top-to-bottom, and keeps the dependency list short.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    num_scenes: int = 20
    train_fraction: float = 0.8
    resolution: int = 128
    num_frames: int = 8
    base_seed: int = 0
    output_dir: str = "data/v0"
    num_objects_min: int = 1
    num_objects_max: int = 3


@dataclass
class EncoderConfig:
    name: str = "vjepa2"  # "vjepa2" or "pixel_baseline"
    checkpoint: str = "facebook/vjepa2-vitb-fpc64-256"
    pretrained: bool = True
    device: str = "cpu"
    grid_size: int = 8  # only used by pixel_baseline


@dataclass
class ProbeConfig:
    ridge_alpha: float = 10.0


@dataclass
class ExperimentConfig:
    transforms: tuple = (
        "camera_translation",
        "camera_rotation",
        "object_translation",
        "object_rotation",
        "lighting_change",
        "texture_change",
    )
    flagship_transform: str = "camera_rotation"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def _merge_into_dataclass(instance, data: dict):
    for f in fields(instance):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            if isinstance(value, list):
                value = tuple(value)
            setattr(instance, f.name, value)
    return instance


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _merge_into_dataclass(cfg, data)


def apply_override(cfg: Config, path: str | Path) -> Config:
    """Merge a YAML file's fields onto an already-loaded Config, in place."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _merge_into_dataclass(cfg, data)


# --- Task 2: scene generation config -----------------------------------
#
# Deliberately a separate top-level dataclass from Config above, not a
# new field on it: this config has nothing to do with encoders, probes,
# or experiment transforms (Task 2 explicitly does not touch any of
# that), so nesting it inside the V0 experiment Config would only
# suggest a coupling that does not exist.


@dataclass
class ObjectMotionConfig:
    enabled: bool = True
    # Magnitude ranges; direction (linear) and axis (angular, when not
    # fixed to world z) are sampled per object -- see generation/generate.py.
    linear_speed_range: tuple[float, float] = (0.0, 0.4)  # scene-units/second
    angular_speed_deg_range: tuple[float, float] = (0.0, 30.0)  # degrees/second, about world z


@dataclass
class CameraMotionConfig:
    mode: str = "static"  # "static" | "orbit"
    orbit_deg_per_sec_range: tuple[float, float] = (5.0, 20.0)


@dataclass
class GenerationConfig:
    num_scenes: int = 10
    base_seed: int = 0
    output_dir: str = "data/generation_v0"
    resolution: int = 128
    num_frames: int = 8
    fps: float = 12.0
    num_objects_min: int = 1
    num_objects_max: int = 3
    depth_near: float = 0.05
    depth_far: float = 30.0
    max_instances: int = 255
    object_motion: ObjectMotionConfig = field(default_factory=ObjectMotionConfig)
    camera_motion: CameraMotionConfig = field(default_factory=CameraMotionConfig)


def load_generation_config(path: str | Path | None) -> GenerationConfig:
    cfg = GenerationConfig()
    if path is None:
        return cfg
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _merge_into_dataclass(cfg, data)
