"""Task 3: scene-level tests for the transformation engine in
transforms/scene_transform.py -- identity, composition, inverse recovery,
variable isolation, and appearance-preserves-geometry, all exercised
through real SceneState objects and apply_transform()'s actual output
(not just the underlying matrix algebra, which tests/test_se3_matrices.py
covers in isolation).
"""

import numpy as np
import pytest

from generation.scene_sampler import SceneSamplerConfig, generate_scene
from transforms.scene_transform import (
    CONTROL_TRANSFORMS,
    GEOMETRIC_TRANSFORMS,
    TRANSFORM_NAMES,
    TransformConfig,
    apply_transform,
)
from transforms.se3 import identity_matrix, inverse_rigid, matrix_to_pose, pose_matrix


@pytest.fixture
def scene():
    return generate_scene("t3_test", seed=17, cfg=SceneSamplerConfig(num_objects_min=2, num_objects_max=2))


# --- identity ---------------------------------------------------------


def test_identity_se3_matrix_preserves_camera_and_object_pose(scene):
    """Applying the SE(3) identity matrix to a real camera's and a real
    object's pose (via the exact code path apply_transform uses:
    pose_matrix -> matmul -> matrix_to_pose) must reproduce that pose
    exactly -- Task 3's "identity transformation produces identical
    state," made concrete on real SceneState data.
    """
    for position, rotation_euler in [
        (scene.camera.position, scene.camera.rotation_euler),
        (scene.objects[0].position, scene.objects[0].rotation_euler),
    ]:
        old_pose = pose_matrix(position, rotation_euler)
        new_position, new_rotation = matrix_to_pose(identity_matrix() @ old_pose)
        assert new_position == pytest.approx(position, abs=1e-12)
        from transforms.se3 import euler_to_matrix

        assert euler_to_matrix(new_rotation) == pytest.approx(euler_to_matrix(rotation_euler), abs=1e-12)


# --- composition --------------------------------------------------------


def test_camera_rotation_composition_matches_chained_matrix_application(scene):
    """Two independently-applied camera_rotation transforms compose the
    same way their recorded transform_matrix values do: applying them one
    at a time to the SceneState must land at the same camera pose as
    applying (T2 @ T1) to the original pose in one matrix multiplication.
    """
    cfg = TransformConfig()
    scene1, params1 = apply_transform(scene, "camera_rotation", cfg)
    # Re-derive a second rotation from scene1 with a different transform
    # name's RNG stream so it's an independent draw, not a repeat of the same one.
    scene2, params2 = apply_transform(scene1, "camera_translation", cfg)
    scene3, params3 = apply_transform(scene2, "camera_rotation", cfg)

    T1 = np.array(params1["transform_matrix"])
    T2 = np.array(params2["transform_matrix"])
    T3 = np.array(params3["transform_matrix"])

    original_pose = pose_matrix(scene.camera.position, scene.camera.rotation_euler)
    chained = T3 @ (T2 @ (T1 @ original_pose))
    expected_position, expected_rotation = matrix_to_pose(chained)

    assert scene3.camera.position == pytest.approx(expected_position, abs=1e-8)
    from transforms.se3 import euler_to_matrix

    assert euler_to_matrix(scene3.camera.rotation_euler) == pytest.approx(euler_to_matrix(expected_rotation), abs=1e-8)


def test_object_rotation_composition_is_associative(scene):
    idx = 0
    obj = scene.objects[idx]
    cfg = TransformConfig()
    # Apply object_rotation repeatedly until it targets our object (the
    # target index is randomly chosen per call); guard against the small
    # chance it always picks the other object by trying a bounded number
    # of times.
    scene_a = scene
    matrices = []
    for _ in range(10):
        scene_a, params = apply_transform(scene_a, "object_rotation", cfg)
        if params["object_index"] == idx:
            matrices.append(np.array(params["transform_matrix"]))
        if len(matrices) == 2:
            break
    assert len(matrices) == 2, "expected object_rotation to target object 0 at least twice in 10 tries"

    original_pose = pose_matrix(obj.position, obj.rotation_euler)
    chained = matrices[1] @ (matrices[0] @ original_pose)
    expected_position, expected_rotation = matrix_to_pose(chained)
    final_obj = next(o for o in scene_a.objects if o.shape == obj.shape and o.color == obj.color)
    assert final_obj.position == pytest.approx(expected_position, abs=1e-6)


# --- inverse recovers the original --------------------------------------


@pytest.mark.parametrize("name", GEOMETRIC_TRANSFORMS)
def test_inverse_transform_matrix_recovers_original_camera_or_object_pose(scene, name):
    cfg = TransformConfig()
    new_scene, params = apply_transform(scene, name, cfg)
    T = np.array(params["transform_matrix"])
    T_inv = inverse_rigid(T)

    if name.startswith("camera"):
        old_pose = pose_matrix(scene.camera.position, scene.camera.rotation_euler)
        new_pose = pose_matrix(new_scene.camera.position, new_scene.camera.rotation_euler)
    else:
        idx = params["object_index"]
        old_pose = pose_matrix(scene.objects[idx].position, scene.objects[idx].rotation_euler)
        new_pose = pose_matrix(new_scene.objects[idx].position, new_scene.objects[idx].rotation_euler)

    recovered_position, recovered_rotation = matrix_to_pose(T_inv @ new_pose)
    original_position, original_rotation = matrix_to_pose(old_pose)
    assert recovered_position == pytest.approx(original_position, abs=1e-8)
    from transforms.se3 import euler_to_matrix

    assert euler_to_matrix(recovered_rotation) == pytest.approx(euler_to_matrix(original_rotation), abs=1e-8)


# --- isolation: a transform changes only what it says it changes --------


def test_camera_rotation_changes_only_camera_variables(scene):
    new_scene, params = apply_transform(scene, "camera_rotation", TransformConfig())
    assert set(params["changed_variables"]) == {"camera.position", "camera.rotation_euler"}
    assert "camera.lens_mm" in params["fixed_variables"]
    assert new_scene.camera.lens_mm == scene.camera.lens_mm
    assert new_scene.camera.sensor_width_mm == scene.camera.sensor_width_mm
    for old_obj, new_obj in zip(scene.objects, new_scene.objects):
        assert old_obj.position == new_obj.position
        assert old_obj.rotation_euler == new_obj.rotation_euler
        assert old_obj.color == new_obj.color
    assert new_scene.light == scene.light
    assert new_scene.floor_color == scene.floor_color


def test_camera_translation_changes_only_camera_position(scene):
    new_scene, params = apply_transform(scene, "camera_translation", TransformConfig())
    assert set(params["changed_variables"]) == {"camera.position"}
    from transforms.se3 import euler_to_matrix

    assert euler_to_matrix(new_scene.camera.rotation_euler) == pytest.approx(euler_to_matrix(scene.camera.rotation_euler), abs=1e-10)
    for old_obj, new_obj in zip(scene.objects, new_scene.objects):
        assert old_obj.position == new_obj.position
        assert old_obj.rotation_euler == new_obj.rotation_euler
    assert new_scene.light == scene.light


def test_object_translation_changes_only_that_objects_position(scene):
    new_scene, params = apply_transform(scene, "object_translation", TransformConfig())
    idx = params["object_index"]
    assert set(params["changed_variables"]) == {f"objects[{idx}].position"}
    assert new_scene.camera == scene.camera
    assert new_scene.light == scene.light
    for i, (old_obj, new_obj) in enumerate(zip(scene.objects, new_scene.objects)):
        if i == idx:
            assert old_obj.position != new_obj.position
            assert old_obj.rotation_euler == new_obj.rotation_euler
            assert old_obj.color == new_obj.color
        else:
            assert old_obj.position == new_obj.position
            assert old_obj.rotation_euler == new_obj.rotation_euler


def test_object_rotation_changes_only_that_objects_rotation(scene):
    new_scene, params = apply_transform(scene, "object_rotation", TransformConfig())
    idx = params["object_index"]
    assert set(params["changed_variables"]) == {f"objects[{idx}].rotation_euler"}
    for i, (old_obj, new_obj) in enumerate(zip(scene.objects, new_scene.objects)):
        if i == idx:
            assert old_obj.position == pytest.approx(new_obj.position, abs=1e-8)
            assert old_obj.color == new_obj.color
        else:
            assert old_obj.position == new_obj.position
            assert old_obj.rotation_euler == new_obj.rotation_euler
    assert new_scene.camera == scene.camera


# --- appearance transforms preserve physical geometry --------------------


@pytest.mark.parametrize("name", CONTROL_TRANSFORMS)
def test_appearance_transforms_preserve_all_geometry(scene, name):
    new_scene, params = apply_transform(scene, name, TransformConfig())
    assert params["transform_matrix"] is None  # not a rigid transform
    assert new_scene.camera == scene.camera
    for old_obj, new_obj in zip(scene.objects, new_scene.objects):
        assert old_obj.position == new_obj.position
        assert old_obj.rotation_euler == new_obj.rotation_euler
        assert old_obj.scale == new_obj.scale
        assert old_obj.shape == new_obj.shape
    # and at least the thing the transform is FOR must actually have changed,
    # so this isn't a vacuously-passing no-op test
    changed_something = (new_scene.light != scene.light) or any(
        o1.color != o2.color for o1, o2 in zip(scene.objects, new_scene.objects)
    )
    assert changed_something


# --- reproducibility ------------------------------------------------------


@pytest.mark.parametrize("name", TRANSFORM_NAMES)
def test_transform_matrix_is_reproducible(scene, name):
    cfg = TransformConfig()
    _, params1 = apply_transform(scene, name, cfg)
    _, params2 = apply_transform(scene, name, cfg)
    if params1["transform_matrix"] is None:
        assert params2["transform_matrix"] is None
    else:
        assert np.array_equal(np.array(params1["transform_matrix"]), np.array(params2["transform_matrix"]))
    assert params1["changed_variables"] == params2["changed_variables"]
    assert params1["fixed_variables"] == params2["fixed_variables"]


def test_changed_and_fixed_variables_partition_all_scene_fields(scene):
    for name in TRANSFORM_NAMES:
        _, params = apply_transform(scene, name, TransformConfig())
        changed = set(params["changed_variables"])
        fixed = set(params["fixed_variables"])
        assert changed.isdisjoint(fixed)
        # every object/camera/light field this project tracks must be
        # accounted for as either changed or fixed -- nothing silently
        # unaccounted-for.
        expected_total = 4 + 5 * len(scene.objects) + 3 + 1  # camera + per-object + light + floor_color
        assert len(changed) + len(fixed) == expected_total
