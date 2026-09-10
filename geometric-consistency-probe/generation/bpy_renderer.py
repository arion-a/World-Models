"""The renderer R: SceneState -> video frames, implemented with Blender/Cycles (bpy).

Design note (see DESIGN.md "Renderer choice" for the full rationale):
we render with Blender's Cycles engine via the official `bpy` PyPI
package -- the same rendering engine Kubric itself wraps -- rather than
through the `kubric` PyPI package. Kubric's own package pulls in a large
Apache-Beam/TensorFlow dataset-pipeline stack meant for building huge
datasets (e.g. ShapeNet-scale), which is unnecessary complexity for V0's
~100 controlled primitive-object scenes. Nothing here reimplements
rendering: every pixel is produced by Blender/Cycles. The module is
structured so it can be swapped for the full Kubric pipeline later
(e.g. to add ShapeNet/GSO assets or rigid-body physics) without touching
any downstream module -- everything past this file only depends on
`render_scene` returning a `(T, H, W, 3)` uint8 array.

Because our synthetic scenes have no intrinsic camera motion within a
single physical state S (the transformations we study, T, map one static
scene to another static scene: S -> S'), `render_scene` renders exactly
one frame and repeats it `num_frames` times to build a fixed-length clip.
This is a deliberate V0 simplification, not a rendering limitation --
see DESIGN.md "Static-clip videos" for the tradeoffs and how temporal
dynamics (component D/E in the project plan) will be introduced later.
"""

from __future__ import annotations

import math

import numpy as np

from generation.scene import SceneState

_SHAPE_ADDERS = {
    "cube": "primitive_cube_add",
    "cone": "primitive_cone_add",
    "cylinder": "primitive_cylinder_add",
    "monkey": "primitive_monkey_add",
    "sphere": "primitive_uv_sphere_add",
}


def _build_blender_scene(state: SceneState, resolution: int):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)

    # Floor
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    floor = bpy.context.object
    floor_mat = bpy.data.materials.new("FloorMaterial")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*state.floor_color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(floor_mat)

    # Objects
    for i, obj_state in enumerate(state.objects):
        adder = getattr(bpy.ops.mesh, _SHAPE_ADDERS[obj_state.shape])
        # Each Blender primitive operator has a different size-parameter
        # name/shape (verified via bpy.ops.mesh.primitive_*_add.__doc__),
        # so each branch passes exactly the kwargs that operator accepts.
        if obj_state.shape == "cube":
            adder(size=1.0, location=obj_state.position)
        elif obj_state.shape == "cone":
            adder(radius1=0.6, radius2=0.0, depth=1.2, location=obj_state.position)
        elif obj_state.shape == "cylinder":
            adder(radius=0.6, depth=1.2, location=obj_state.position)
        elif obj_state.shape == "sphere":
            adder(radius=0.6, location=obj_state.position)
        else:  # monkey
            adder(size=1.0, location=obj_state.position)
        blender_obj = bpy.context.object
        blender_obj.rotation_euler = obj_state.rotation_euler
        blender_obj.scale = (obj_state.scale, obj_state.scale, obj_state.scale)
        mat = bpy.data.materials.new(f"ObjMaterial{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*obj_state.color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        blender_obj.data.materials.append(mat)

    # Light
    bpy.ops.object.light_add(type="SUN", location=state.light.position)
    light = bpy.context.object
    light.data.energy = state.light.energy
    light.data.color = state.light.color
    direction = -np.array(state.light.position)
    direction = direction / np.linalg.norm(direction)
    light.rotation_euler = _direction_to_euler(direction)

    # Camera
    bpy.ops.object.camera_add(location=state.camera.position, rotation=state.camera.rotation_euler)
    camera = bpy.context.object
    camera.data.lens = state.camera.lens_mm
    scene.camera = camera

    return scene


def _direction_to_euler(direction: np.ndarray) -> tuple[float, float, float]:
    """Euler angles pointing a light's local -Z axis along `direction`."""
    forward = -direction
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    rot = np.stack([right, up, forward], axis=1)
    sy = -rot[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    ry = math.asin(sy)
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(rot[2, 1], rot[2, 2])
        rz = math.atan2(rot[1, 0], rot[0, 0])
    else:
        rx = math.atan2(-rot[1, 2], rot[1, 1])
        rz = 0.0
    return (rx, ry, rz)


def render_frame(state: SceneState, resolution: int = 128) -> np.ndarray:
    """Render one RGB frame of `state`. Returns a (resolution, resolution, 3) uint8 array."""
    import bpy

    scene = _build_blender_scene(state, resolution)
    with _temp_render_path() as filepath:
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        image = _load_png_rgb(filepath)
    return image


def render_scene(state: SceneState, num_frames: int = 8, resolution: int = 128) -> np.ndarray:
    """Render the video V = R(S) for a static scene: (num_frames, resolution, resolution, 3) uint8.

    See module docstring: the scene is static, so this renders one frame
    and repeats it. Frame count is a downstream-encoder input-shape
    parameter, not a claim about scene dynamics.
    """
    frame = render_frame(state, resolution=resolution)
    return np.repeat(frame[None], num_frames, axis=0)


def _load_png_rgb(filepath: str) -> np.ndarray:
    import imageio.v3 as iio

    img = iio.imread(filepath)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


class _temp_render_path:
    def __enter__(self):
        import tempfile

        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self._tmp.close()
        return self._tmp.name

    def __exit__(self, *exc):
        import os

        try:
            os.remove(self._tmp.name)
        except OSError:
            pass
