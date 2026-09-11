"""Task 7B data layer: deterministic master scene population, three-way
scene-level split (contract Sec. 5), and nested per-seed training
subsets for the learning-curve schedule.

N always counts unique source SCENES, never clips/frames/tokens -- every
function here operates on scene_id strings, and a scene's original and
every transformed variant always stay together (this module produces
splits; transforms/rendering apply per scene_id afterward, in
run_pipeline.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import experiments.geometric_consistency_lib as gclib
from generation.scene import SceneState

TRANSFORM_NAME = "camera_rotation"
FIXED_ELEVATION_DEG = 0.0
# The contract's Theta set (Sec. 8); 0.0 is identity validation, never
# used to tune the model. The PRIMARY magnitude (used for the Stage 1
# learning curve / model hierarchy comparison) is +30 deg -- chosen
# because it is a member of this same set (so no separate render pass is
# needed for it in Stage 2) and matches Task 6's original historical
# camera_rotation magnitude for continuity, though Task 7B's own result
# does not depend on reproducing Task 6/7's numbers (see PROTOCOL_NOTES
# in protocol.py).
MAGNITUDES_DEG = (-60.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 60.0)
PRIMARY_MAGNITUDE_DEG = 30.0
# Reserved for interpolation (contract Sec. 9) -- excluded from Stage 2's
# direct-fit magnitude set, predeclared here (before any fitting) rather
# than chosen post hoc.
INTERPOLATION_RESERVED_MAGNITUDE_DEG = 20.0
STAGE2_DIRECT_FIT_MAGNITUDES_DEG = tuple(
    m for m in MAGNITUDES_DEG if m not in (0.0, INTERPOLATION_RESERVED_MAGNITUDE_DEG)
)

N_TRAIN_SCHEDULE = (64, 128, 256, 512, 1024)
SEEDS_PER_N = 5
TRAIN_POOL_SIZE = max(N_TRAIN_SCHEDULE)  # 1024 -- the full training pool IS the N=1024 point
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
MIN_TEST_SCENES = 64

BASE_SEED = 0
NUM_OBJECTS_MIN = 1
NUM_OBJECTS_MAX = 3


@dataclass(frozen=True)
class SplitPlan:
    total_population: int
    train_pool_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def compute_population_size(train_pool_size: int = TRAIN_POOL_SIZE) -> int:
    """Smallest total population such that a 70/15/15 (by scene-ID count)
    split gives at least `train_pool_size` training-pool scenes, rounding
    by scene ID and fixed once, before any rendering."""
    import math

    total = math.ceil(train_pool_size / TRAIN_FRACTION)
    return total


def build_master_population(total_population: int, base_seed: int = BASE_SEED) -> list[SceneState]:
    """Task 7B's own master scene list -- generated via the SAME
    deterministic sampler Task 6/7 use (geometric_consistency_lib.
    sample_scenes), so scene_id/seed derivation is not invented, but this
    is Task 7B's OWN population (never Task 7's 40 scenes reused), per
    the contract's "newly sampled deterministic master population."""
    return gclib.sample_scenes(total_population, base_seed, NUM_OBJECTS_MIN, NUM_OBJECTS_MAX)


def three_way_split(scene_ids: list[str], train_pool_size: int, val_size: int, seed: int = BASE_SEED) -> SplitPlan:
    """Deterministic, seed-permutation-based three-way split by scene ID.
    Every scene lands in exactly one of train-pool/val/test; all variants
    of one scene (however many magnitudes it is later rendered for) share
    its split label by construction, since callers always key by
    scene_id, never by (scene_id, magnitude)."""
    n = len(scene_ids)
    test_size = n - train_pool_size - val_size
    if test_size < MIN_TEST_SCENES:
        raise ValueError(f"test split would have {test_size} scenes, below the required minimum {MIN_TEST_SCENES}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    train_pool = tuple(scene_ids[i] for i in order[:train_pool_size])
    val = tuple(scene_ids[i] for i in order[train_pool_size : train_pool_size + val_size])
    test = tuple(scene_ids[i] for i in order[train_pool_size + val_size :])
    assert len(train_pool) + len(val) + len(test) == n
    assert set(train_pool).isdisjoint(val) and set(train_pool).isdisjoint(test) and set(val).isdisjoint(test)
    return SplitPlan(total_population=n, train_pool_ids=train_pool, val_ids=val, test_ids=test)


def nested_train_subset(train_pool_ids: tuple[str, ...], n_train: int, seed: int) -> tuple[str, ...]:
    """A deterministic, seeded permutation of the train pool, truncated to
    `n_train` -- guarantees nesting: nested_train_subset(pool, 64, s) is a
    PREFIX of nested_train_subset(pool, 128, s) for the same seed s,
    because both come from the identical permutation of `pool`."""
    if n_train > len(train_pool_ids):
        raise ValueError(f"n_train={n_train} exceeds train pool size {len(train_pool_ids)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_pool_ids))
    return tuple(train_pool_ids[i] for i in order[:n_train])
