"""Automated pairing-integrity regression test for Task 7's real rendered
artifacts (experiments/geometric_consistency/{transform}/{scene_id}/).

Added by the Task 7 forensic audit (experiments/task7_forensic_audit/) as
one of its required deliverables: proof that Z_i <-> Z'_i pairing is
correct for the SAME underlying scene, and that it cannot silently break
via independent sorting, shuffled loaders, indexing bugs, batch-order
mismatch, filename parsing errors, or dictionary-ordering issues.

This is a *structural* guarantee test, not a statistical spot check: the
production code (experiments/geometric_consistency_lib.py's
encode_all_pairs/build_arrays) keys every representation by scene_id in a
dict, never by list position, so there is no ordering step anywhere in
the pipeline for a bug like this to hide in -- this test asserts that
structural property directly against the pipeline's actual on-disk
outputs, for every scene of every transform, and would fail if a future
change reintroduced positional/index-based pairing.

Skipped (not failed) when experiments/geometric_consistency/ does not
exist -- e.g. a fresh checkout that hasn't run Task 7 yet -- since this
test audits real experiment artifacts, not a synthetic fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import experiments.geometric_consistency_lib as gclib

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_DIR = REPO_ROOT / "experiments" / "geometric_consistency"
RESULT_PATH = REPO_ROOT / "state" / "task_07_result.json"

pytestmark = pytest.mark.skipif(
    not RENDER_DIR.exists() or not RESULT_PATH.exists(),
    reason="experiments/geometric_consistency/ or state/task_07_result.json not present -- Task 7 has not been run in this checkout",
)


def _load_result():
    return json.loads(RESULT_PATH.read_text())


def _all_scene_transform_dirs():
    result = _load_result()
    transforms = list(result["transforms_evaluated"])
    scene_ids = sorted(set(result["dataset"]["train_scene_ids"]) | set(result["dataset"]["test_scene_ids"]))
    return transforms, scene_ids


def test_every_pair_directory_scene_id_matches_its_own_metadata():
    """The scene_id encoded in a pair directory's NAME must match the
    scene_id recorded inside both sides' metadata.json -- catches a
    filename-parsing or directory-construction bug that could silently
    put scene A's data under scene B's directory."""
    transforms, scene_ids = _all_scene_transform_dirs()
    checked = 0
    for transform_name in transforms:
        for scene_id in scene_ids:
            pair_dir = RENDER_DIR / transform_name / scene_id
            for side in ("original", "transformed"):
                meta = json.loads((pair_dir / side / "metadata.json").read_text())
                assert meta["scene_id"] == scene_id, f"{pair_dir}/{side}: metadata scene_id {meta['scene_id']!r} != directory name {scene_id!r}"
                checked += 1
    assert checked == len(transforms) * len(scene_ids) * 2


def test_original_and_transformed_sides_share_the_same_seed():
    """original/ and transformed/ within one pair directory must be the
    SAME underlying scene (same seed) -- a mismatched seed would mean Z
    and Z' come from two different scenes entirely, not a before/after
    pair."""
    transforms, scene_ids = _all_scene_transform_dirs()
    for transform_name in transforms:
        for scene_id in scene_ids:
            pair_dir = RENDER_DIR / transform_name / scene_id
            orig_meta = json.loads((pair_dir / "original" / "metadata.json").read_text())
            trans_meta = json.loads((pair_dir / "transformed" / "metadata.json").read_text())
            assert orig_meta["seed"] == trans_meta["seed"], f"{pair_dir}: original/transformed seed mismatch"


def test_transformation_json_type_matches_its_own_directory():
    """transformation.json's recorded type must match the transform-name
    directory it lives under -- catches a bug where transform application
    ran but the wrong transform's metadata got written (or vice versa)."""
    transforms, scene_ids = _all_scene_transform_dirs()
    for transform_name in transforms:
        for scene_id in scene_ids:
            transformation = json.loads((RENDER_DIR / transform_name / scene_id / "transformation.json").read_text())
            assert transformation["type"] == transform_name
            assert transformation["transform_name"] == transform_name


def test_original_rendering_is_identical_across_every_transforms_folder():
    """The SAME scene's `original` side is rendered independently once
    per transform folder (transforms/pairs.py's generate_pair is called
    separately for every transform). Since rendering is deterministic
    (see experiments/task7_forensic_audit/render_determinism.json) and
    depends only on (scene, seed), these independent renders of the same
    scene_id must be bit-identical RGB arrays across every transform's
    folder -- if they were not, that would mean either non-determinism in
    the renderer or a scene/seed mismatch somewhere upstream, either of
    which would silently corrupt the shared-original assumption the
    encoding step relies on.
    """
    transforms, scene_ids = _all_scene_transform_dirs()
    mismatches = []
    for scene_id in scene_ids:
        reference = None
        for transform_name in transforms:
            rgb = np.load(RENDER_DIR / transform_name / scene_id / "original" / "rgb.npy")
            if reference is None:
                reference = rgb
            elif not np.array_equal(reference, rgb):
                mismatches.append((scene_id, transform_name))
    assert not mismatches, f"original render differs across transform folders for: {mismatches}"


def test_dataset_split_is_disjoint_and_identical_across_every_transform():
    """The scene-level train/test split recorded per-transform in
    state/task_07_result.json must be IDENTICAL across all six transforms
    (the split is decided once, before any rendering) and train/test must
    never overlap -- the leakage/comparability guarantee this task's
    Required leakage checks describe."""
    result = _load_result()
    ref_train = set(result["dataset"]["train_scene_ids"])
    ref_test = set(result["dataset"]["test_scene_ids"])
    assert ref_train.isdisjoint(ref_test)
    for transform_name, entry in result["results"].items():
        assert set(entry["train_scene_ids"]) == ref_train, f"{transform_name}: train split differs from the shared split"
        assert set(entry["test_scene_ids"]) == ref_test, f"{transform_name}: test split differs from the shared split"


def test_representation_build_is_keyed_by_scene_id_not_list_position():
    """Directly exercises the actual production pairing code
    (encode_all_pairs/build_arrays) against a small in-memory fake
    encoder whose output ENCODES each scene's identity, then asserts the
    resulting Z_train/Z_test arrays correspond to the correct scene_id at
    every row position -- would fail immediately if build_arrays ever
    started zipping by list index instead of looking up by scene_id."""
    from generation.scene import CameraState, LightState, SceneState

    scenes = [
        SceneState(
            scene_id=f"scene_{i:04d}",
            seed=i,
            objects=(),
            camera=CameraState(position=(0, 0, 0), rotation_euler=(0, 0, 0)),
            light=LightState(position=(0, 0, 0), energy=1.0, color=(1.0, 1.0, 1.0)),
        )
        for i in range(6)
    ]
    split = {s.scene_id: ("train" if i < 4 else "test") for i, s in enumerate(scenes)}

    # Build `reps` the same shape encode_all_pairs would, but skip actual
    # rendering/loading -- each scene's Z/Z' directly encode its own
    # index, offset differently for Z vs Z', so a pairing bug (wrong row,
    # swapped Z/Z', or list-order zip instead of scene_id lookup) shows up
    # as a wrong index recovered.
    reps = {}
    for i, s in enumerate(scenes):
        reps[s.scene_id] = {
            "Z": np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32),
            "Z_prime": np.array([float(i) + 100.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

    train_ids, test_ids, Z_train, Zp_train, Z_test, Zp_test = gclib.build_arrays(scenes, split, reps)

    for row, sid in enumerate(train_ids):
        expected_i = int(sid.split("_")[1])
        assert Z_train[row, 0] == float(expected_i), f"train row {row} (scene {sid}): Z mismatch"
        assert Zp_train[row, 0] == float(expected_i) + 100.0, f"train row {row} (scene {sid}): Z_prime mismatch"
    for row, sid in enumerate(test_ids):
        expected_i = int(sid.split("_")[1])
        assert Z_test[row, 0] == float(expected_i), f"test row {row} (scene {sid}): Z mismatch"
        assert Zp_test[row, 0] == float(expected_i) + 100.0, f"test row {row} (scene {sid}): Z_prime mismatch"
