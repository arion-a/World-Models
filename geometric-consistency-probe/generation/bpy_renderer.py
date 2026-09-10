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
from dataclasses import dataclass

import numpy as np

from generation.motion import Trajectory
from generation.scene import SceneState

_SHAPE_ADDERS = {
    "cube": "primitive_cube_add",
    "cone": "primitive_cone_add",
    "cylinder": "primitive_cylinder_add",
    "monkey": "primitive_monkey_add",
    "sphere": "primitive_uv_sphere_add",
}


def _build_blender_scene(state: SceneState, resolution: int, enable_passes: bool = False):
    """Build the Blender scene for `state`.

    Returns `(scene, handles)`, where `handles` is a dict of the created
    bpy objects (`"camera"`, `"light"`, `"floor"`, and
    `"objects": {instance_id: bpy_object}`) -- Task 2's per-frame
    trajectory rendering needs these to move objects/camera between
    renders without rebuilding the whole scene each frame; V0's
    single-frame `render_frame` just discards `handles`.

    `enable_passes=True` additionally tags every object with a
    `pass_index` equal to its `instance_id` (0 is the floor's implicit
    background index -- Blender objects default to `pass_index = 0`, so
    the floor is never explicitly tagged) and turns on the Z and Object
    Index view-layer passes, which `render_trajectory` below reads back
    as depth and instance segmentation. See
    generation/COORDINATE_SYSTEM.md for exactly what these passes mean.
    """
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

    if enable_passes:
        vl = bpy.context.view_layer
        vl.use_pass_z = True
        vl.use_pass_object_index = True

    # Floor (implicit instance/segmentation index 0: Blender objects
    # default to pass_index = 0, so it is never explicitly tagged).
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    floor = bpy.context.object
    floor_mat = bpy.data.materials.new("FloorMaterial")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*state.floor_color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(floor_mat)

    # Objects
    object_handles: dict[int, "bpy.types.Object"] = {}
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
        if enable_passes:
            blender_obj.pass_index = obj_state.instance_id
        mat = bpy.data.materials.new(f"ObjMaterial{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*obj_state.color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        blender_obj.data.materials.append(mat)
        object_handles[obj_state.instance_id] = blender_obj

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
    camera.data.sensor_width = state.camera.sensor_width_mm
    scene.camera = camera

    handles = {"camera": camera, "light": light, "floor": floor, "objects": object_handles}
    return scene, handles


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

    scene, _handles = _build_blender_scene(state, resolution)
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


DEFAULT_DEPTH_NEAR = 0.05
DEFAULT_DEPTH_FAR = 30.0
#: Number of distinct instance ids the segmentation channel can encode
#: (0 = background/floor, 1..N = objects). 255 comfortably covers this
#: project's "1-3 objects" scenes with headroom; see
#: generation/COORDINATE_SYSTEM.md for the exact pixel-value encoding.
DEFAULT_MAX_INSTANCES = 255


@dataclass
class ClipGroundTruth:
    """Everything `render_trajectory` produces for one clip.

    `depth` is camera-space Z (perpendicular distance from the camera's
    image plane), NOT Euclidean camera-to-point ray distance -- verified
    empirically against Blender's actual Z-pass output (a flat plane
    viewed obliquely renders with *uniform* depth, which only planar
    depth predicts) and documented in generation/COORDINATE_SYSTEM.md.
    `segmentation` gives each pixel the `instance_id` of the object
    rendered there (0 = floor/background), decoded from the 16-bit PNG
    encoding also documented there.
    """

    rgb: np.ndarray  # (T, H, W, 3) uint8
    depth: np.ndarray  # (T, H, W) float32, scene units
    segmentation: np.ndarray  # (T, H, W) uint8, instance ids
    depth_near: float
    depth_far: float
    max_instances: int


def _setup_ground_truth_compositor(scene, depth_near: float, depth_far: float, max_instances: int, out_dir: str):
    """Wire up a compositor graph that writes normalized depth and
    segmentation passes to 16-bit grayscale PNGs alongside the normal RGB
    render. See DESIGN.md/COORDINATE_SYSTEM.md for the encoding this
    implements, and IMPLEMENTATION_NOTES.md for how the exact Blender
    compositor API used here (`compositing_node_group`,
    `file_output_items`, `format.media_type`) was verified empirically --
    it differs substantially from older Blender versions' compositor API
    and is not obviously documented.
    """
    import bpy

    group = bpy.data.node_groups.new("GroundTruthCompositing", "CompositorNodeTree")
    scene.compositing_node_group = group
    render_layers = group.nodes.new("CompositorNodeRLayers")

    file_output = group.nodes.new("CompositorNodeOutputFile")
    file_output.directory = out_dir
    file_output.format.media_type = "IMAGE"
    file_output.format.file_format = "PNG"

    depth_range = group.nodes.new("ShaderNodeMapRange")
    depth_range.clamp = True
    depth_range.inputs["From Min"].default_value = depth_near
    depth_range.inputs["From Max"].default_value = depth_far
    depth_range.inputs["To Min"].default_value = 0.0
    depth_range.inputs["To Max"].default_value = 1.0
    group.links.new(render_layers.outputs["Depth"], depth_range.inputs["Value"])

    depth_item = file_output.file_output_items.new("FLOAT", "depth")
    depth_item.override_node_format = True
    depth_item.format.media_type = "IMAGE"
    depth_item.format.file_format = "PNG"
    depth_item.format.color_depth = "16"
    depth_item.format.color_mode = "BW"
    depth_item.save_as_render = False  # raw values, not color-managed
    group.links.new(depth_range.outputs["Result"], file_output.inputs["depth"])

    seg_range = group.nodes.new("ShaderNodeMapRange")
    seg_range.clamp = True
    seg_range.inputs["From Min"].default_value = 0.0
    seg_range.inputs["From Max"].default_value = float(max_instances)
    seg_range.inputs["To Min"].default_value = 0.0
    seg_range.inputs["To Max"].default_value = 1.0
    group.links.new(render_layers.outputs["Object Index"], seg_range.inputs["Value"])

    seg_item = file_output.file_output_items.new("FLOAT", "seg")
    seg_item.override_node_format = True
    seg_item.format.media_type = "IMAGE"
    seg_item.format.file_format = "PNG"
    seg_item.format.color_depth = "16"
    seg_item.format.color_mode = "BW"
    seg_item.save_as_render = False
    group.links.new(seg_range.outputs["Result"], file_output.inputs["seg"])

    return file_output


def render_trajectory(
    state: SceneState,
    trajectory: Trajectory,
    resolution: int = 128,
    depth_near: float = DEFAULT_DEPTH_NEAR,
    depth_far: float = DEFAULT_DEPTH_FAR,
    max_instances: int = DEFAULT_MAX_INSTANCES,
) -> ClipGroundTruth:
    """Render RGB + depth + instance segmentation for every frame of `trajectory`.

    Unlike V0's `render_scene` (one frame repeated), this renders the
    scene once per trajectory frame with the camera and every object
    moved to that frame's pose (`trajectory.frames[f].camera` /
    `.objects`), and is the `render_scene()` Task 2 refers to (exposed
    under that name from generation/generate.py; kept as
    `render_trajectory` here to avoid colliding with V0's existing
    `render_scene` in this module).
    """
    import bpy
    import imageio.v3 as iio

    scene, handles = _build_blender_scene(state, resolution, enable_passes=True)

    with _temp_render_dir() as tmp_dir:
        file_output = _setup_ground_truth_compositor(scene, depth_near, depth_far, max_instances, tmp_dir)

        rgb_frames, depth_frames, seg_frames = [], [], []
        for frame in trajectory.frames:
            handles["camera"].location = frame.camera.position
            handles["camera"].rotation_euler = frame.camera.rotation_euler
            for obj_pose in frame.objects:
                obj = handles["objects"][obj_pose.instance_id]
                obj.location = obj_pose.position
                obj.rotation_euler = obj_pose.rotation_euler

            prefix = f"frame_{frame.frame_index:04d}_"
            file_output.file_name = prefix
            rgb_path = f"{tmp_dir}/{prefix}rgb.png"
            scene.render.filepath = rgb_path
            bpy.ops.render.render(write_still=True)

            rgb_frames.append(_load_png_rgb(rgb_path))

            depth_png = iio.imread(f"{tmp_dir}/{prefix}depth.png").astype(np.float64)
            depth_normalized = depth_png / 65535.0
            depth_frames.append((depth_near + depth_normalized * (depth_far - depth_near)).astype(np.float32))

            seg_png = iio.imread(f"{tmp_dir}/{prefix}seg.png").astype(np.float64)
            seg_normalized = seg_png / 65535.0
            seg_frames.append(np.round(seg_normalized * max_instances).astype(np.uint8))

    return ClipGroundTruth(
        rgb=np.stack(rgb_frames),
        depth=np.stack(depth_frames),
        segmentation=np.stack(seg_frames),
        depth_near=depth_near,
        depth_far=depth_far,
        max_instances=max_instances,
    )


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


class _temp_render_dir:
    def __enter__(self):
        import tempfile

        self._tmp = tempfile.mkdtemp()
        return self._tmp

    def __exit__(self, *exc):
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)
