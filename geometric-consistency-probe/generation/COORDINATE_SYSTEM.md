# Coordinate system and ground-truth encoding (Task 2)

This is the precise reference for every geometric quantity Task 2's
generator produces. Read this before writing anything that consumes
`generation/generate.py`'s output, and before reading depth or
segmentation arrays as if they were in some other, more familiar
convention (OpenCV, COLMAP, PyTorch3D, ... all differ from at least one
choice made here).

## World frame

Right-handed, **Z-up** -- Blender's native convention, *not* the
Y-up convention common in graphics/game engines, and *not* any
particular robotics convention. Concretely: `+X` and `+Y` span the
ground plane, `+Z` is "up" (away from the floor). The floor plane is the
`z = 0` plane. All positions in this project's `SceneState` and
`Trajectory` are world-frame coordinates in this system, in Blender's
scene units (treated as meters for descriptive purposes only -- nothing
in the pipeline depends on a specific physical unit).

## Rotation representation

Every orientation (`rotation_euler` on a camera, an object, a light) is
an **XYZ-order intrinsic Euler angle triple, in radians** -- Blender's
default `rotation_mode = 'XYZ'`. `transforms/se3.py:euler_to_matrix` /
`matrix_to_euler` are the exact, tested (see `tests/test_transforms.py`)
conversions to/from a 3x3 rotation matrix, and are the only functions in
this repo that should be trusted to get the axis order and handedness
right -- do not re-derive Euler conversions ad hoc elsewhere.

## Camera frame -- Blender convention, and how it differs from OpenCV

**The camera looks down its own local `-Z` axis, with local `+Y` as
"up" and local `+X` as "right."** This is Blender's convention. It is
**not** the OpenCV/COMPUTER-VISION convention, which looks down `+Z`
with `+Y` *down* and `+X` right. If some later component (a probe, a
different renderer, a comparison to a real-video pipeline) expects
OpenCV-style camera-space coordinates, convert with the fixed change of
basis

```
R_cv = R_blender @ diag(1, -1, -1)
```

(negate the local Y and Z axes; X is already right in both
conventions) -- applied to the camera's 3x3 rotation matrix (from
`euler_to_matrix(camera.rotation_euler)`), not to world-frame points.
`generation/scene.py:camera_forward_vector` returns the world-space
viewing direction directly (already accounting for the `-Z`-forward
convention), and is the reference implementation to check any new code
against.

## Camera intrinsics

`generation/scene.py:camera_intrinsics(camera, resolution_x,
resolution_y)` returns the standard pinhole `fx, fy, cx, cy` and 3x3
matrix `K`, derived from Blender's own perspective-camera model
(`lens_mm`, `sensor_width_mm`) for a **square render**
(`resolution_x == resolution_y`, always true in this project's configs):

```
fx = fy = resolution * lens_mm / sensor_width_mm
cx = cy = resolution / 2
```

This is Blender's sensor-fit-`AUTO` behavior (sensor width maps to the
larger image dimension) specialized to the square-image case, where
there is no larger/smaller dimension to disambiguate. A non-square
render would need the sensor-fit axis handled explicitly;
`camera_intrinsics` raises `NotImplementedError` rather than silently
computing the wrong axis if `resolution_x != resolution_y`.

## Depth: camera-space Z, not ray distance

**`depth.npy` stores the perpendicular distance from the camera's image
plane to each point (camera-space Z), not the Euclidean straight-line
distance from the camera to that point.** These two quantities are
equal for a point exactly on the camera's optical axis and diverge
elsewhere (a point at the edge of the frame, viewed at an oblique angle,
has a *larger* Euclidean distance than its camera-space Z). This
matches the convention most depth-estimation datasets (e.g. KITTI,
NYU-Depth) use, so it is probably what a probe expects by default --
but it is exactly the kind of assumption that is easy to get backwards,
so it was verified empirically rather than assumed: a large flat plane
rendered face-on to the camera (camera looking straight down at a
`z = 0` ground plane) produces a **spatially uniform** raw depth value
across the whole frame, including pixels far from the image center;
Euclidean ray distance to an off-center point on that same plane would
be strictly larger than the on-axis distance, so uniformity across the
frame is only consistent with camera-space-Z (planar) depth. See
`tests/test_bpy_renderer.py::test_depth_pass_is_planar_not_euclidean`
for the automated version of this check, and
`IMPLEMENTATION_NOTES.md` for how the underlying Blender compositor API
that reads this pass out was reverse-engineered.

**Encoding on disk (already decoded when you load `depth.npy`):**
Blender's raw Z-pass is written through a `Map Range` compositor node
that linearly clamps and rescales `[depth_near, depth_far]` (per-scene,
recorded in `metadata.json["depth_encoding"]`, default
`[0.05, 30.0]`) to `[0, 1]`, saved as a 16-bit grayscale PNG, then
decoded back to scene-unit float32 by `generation/ground_truth.py`
before it ever reaches `depth.npy` -- so if you're reading `depth.npy`
directly, you already have real scene-unit floats and don't need this
paragraph; it matters only if you are reading the intermediate PNGs
Blender's compositor produces internally. **Any true depth beyond
`depth_far` is clamped to exactly `depth_far`** -- e.g. background/sky
pixels, or a floor pixel near the horizon at a shallow viewing angle --
so `depth == depth_far` should be read as "at least this far," not "at
exactly this distance."

## Segmentation: instance id as a direct label map

**`segmentation.npy` stores each pixel's `instance_id` directly** as a
`uint8` label map: `0` is reserved for background/floor (Blender objects
default to `pass_index = 0`, and the floor is never given another
value), and `1..N` identify the `N` objects in the scene, matching each
object's `ObjectState.instance_id` / `metadata.json["objects"][i]
["instance_id"]`. Internally this is produced from Blender's "Object
Index" render pass, which is only defined per-object (not per-face or
per-material), so `instance_id` is exactly Task 2's notion of "object
identity": stable across every frame of one scene's trajectory.

## Frame timing

`Trajectory.fps` and `Trajectory.frames[i].timestamp = i / fps` are
exact by construction (`generation/motion.py`); there is no variable
frame rate or dropped-frame concern anywhere in this pipeline, since
every frame is rendered from an analytically-computed pose, never
captured in real time.

## What is deliberately *not* covered here

This document describes Task 2's static-camera-intrinsics,
constant-velocity-trajectory generator. It does not describe:

- The V0 experiment pipeline's transforms (`transforms/scene_transform.py`,
  `S' = T(S)`), which predates this task's formal scope and will be
  reconciled with it in a later task (see DESIGN.md §5 for the `T`
  operator this project's research question is ultimately about).
- Any renderer other than Blender/Cycles, or any coordinate convention
  a different renderer might use.
