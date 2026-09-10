# Implementation notes (V0 code)

> This document records engineering decisions made while building the V0
> *code* (renderer choice, encoder plumbing, a shape bug that was found
> and fixed, etc.). It is a record of "why the code looks the way it
> does," not the research specification. **The research specification —
> what physical state, transformations, equivariance, invariance, probes,
> and evaluation protocol this project uses, and what conclusions they
> can and cannot support — lives in `DESIGN.md`.** Read that first;
> this file is a supplement for anyone modifying the V0 implementation.

## Scope of V0

V0 implements exactly the pipeline in the project brief:

```
Scene S --renderer R--> video V --encoder E--> Z
S' = T(S) --renderer R--> video V' --encoder E--> Z'
learn rho(T): Z' ≈ rho(T) Z    (ridge regression, fit on train scenes)
evaluate rho(T) on held-out scenes
```

for six transform types (camera translation/rotation, object
translation/rotation, lighting change, texture change), plus one
physical-state probe (component C: decoding camera azimuth from Z).
Deliberately out of scope for V0 (see "Deferred to later versions"
below): temporal consistency (D), object persistence (E), counterfactual
dynamics (F), multiple encoders, non-primitive object assets, physics
simulation, and any geometric-algebra machinery.

## Renderer choice: Blender/Cycles via `bpy`, not the `kubric` package

The project brief says "use existing tools such as Kubric wherever
practical" and "do not implement a new renderer." We satisfy both by
rendering exclusively through **Blender's Cycles engine**, the same
renderer Kubric itself wraps, via the official `bpy` PyPI package
(`pip install bpy`, no Docker required -- confirmed to render headlessly
with `scene.render.engine = 'CYCLES'`, `scene.cycles.device = 'CPU'`,
no X server needed).

We do **not** depend on the `kubric` PyPI package directly. Its
`METADATA` lists `apache-beam[gcp]`, `tensorflow`, and
`tensorflow-datasets` as hard requirements -- a dataset-pipeline stack
sized for generating ShapeNet/GSO-scale datasets on a cluster. Pulling
that in for ~100 scenes made of four mesh primitives would violate
"start extremely small" far more than it would honor "use existing
tools." Every pixel in this repo is still produced by Blender/Cycles;
nothing here reimplements rendering.

**Upgrade path:** `generation/bpy_renderer.py` is the only module that
touches `bpy`. Swapping in the full Kubric pipeline later (e.g. for
ShapeNet/GSO assets or PyBullet rigid-body physics) means rewriting this
one file to build its scene through `kubric.Scene`/`kubric.Simulator`
instead of raw `bpy.ops`, without touching `generation/scene.py`,
`transforms/`, `encoders/`, `probes/`, or `metrics/` -- they only depend
on `SceneState` in and a `(T, H, W, 3)` uint8 array out.

## Static-clip videos

The transforms we study in V0 map one static scene to another static
scene (`S -> S' = T(S)`), not a scene evolving continuously in time.
Accordingly, `generation/bpy_renderer.render_scene` renders **one**
frame and repeats it `num_frames` times to build a fixed-length clip,
rather than rendering `num_frames` distinct frames of an unmoving scene
(which would be pixel-identical anyway, just far more expensive to
render). This is a faithful implementation of the block diagram in the
project brief (two separate `S -> V -> Z` pipelines being compared), not
a shortcut around it.

**Caveat, not swept under the rug:** a repeated static frame is
out-of-distribution input for a video model pretrained on natural
motion. This is a real question for interpretation once real V-JEPA 2
weights are in use (see the "What this run does and does not show"
section of `reports/v0_smoke_test_untrained_encoder.md`), and is the
natural on-ramp to component D (temporal consistency) in a later
version: render actual short trajectories (e.g. a smooth camera pan, an
object sliding) and check whether the encoder's response to real motion
differs systematically from its response to these synthetic before/after
pairs.

## Encoder: honesty about pretrained weights

`encoders/vjepa2.py` wraps `transformers.VJEPA2Model`
(`transformers>=4.52`), with the input/output contract verified by
reading the installed library's source
(`transformers/models/vjepa2/{modeling,configuration,video_processing}_vjepa2.py`)
rather than guessed:

- Input `pixel_values_videos`: shape `(batch, num_frames, channels, H, W)`
  -- confirmed from `VJEPA2Embeddings.forward`, which reads `shape[1]`
  as the frame axis before permuting to `(B, C, T, H, W)` for the Conv3d
  patch embedding.
- `model.get_vision_features(pixel_values_videos)` returns the encoder's
  `last_hidden_state`, shape `(batch, num_patches, hidden_size)`, with no
  built-in pooling -- we mean-pool over patch tokens ourselves to get one
  vector per clip (the simplest pooling choice, per the "prefer simple
  linear probes ... before anything fancier" principle).
- Reference preprocessing (`VJEPA2VideoProcessor`) resizes so the
  shortest edge is `crop_size * 256/224`, center-crops to `crop_size`,
  and normalizes with ImageNet mean/std. We reimplement this by hand
  (`encoders/vjepa2.py:_preprocess`) instead of calling
  `AutoVideoProcessor.from_pretrained(...)`, because that call also
  requires Hugging Face Hub access.

`VJEPA2Model.from_pretrained(checkpoint)` downloads weights from
Hugging Face Hub. **This repo was first built in a sandbox whose
network policy blocks `huggingface.co` outright** (confirmed via the
proxy's own diagnostic endpoint reporting a policy-level `403`, not a
transient failure). When the download fails for any reason, the encoder
prints a loud `RuntimeWarning`, logs it, and falls back to the identical
architecture with **random weights**, so the pipeline can still be
exercised end-to-end. Every report this repo produces states explicitly
whether it used real or randomly-initialized weights
(`results["encoder_pretrained"]` in `experiments/run.py`'s output) --
see `reports/v0_smoke_test_untrained_encoder.md` for what that run does
and does not show. On a machine with Hub access, nothing needs to
change: `encoder.pretrained: true` (the default) will load the real
checkpoint.

The ViT-B/16 checkpoint id, `facebook/vjepa2-vitb-fpc64-256`, follows
the naming pattern visible in the installed library
(`facebook/vjepa2-vitl-fpc16-256-ssv2` appears in
`modeling_vjepa2.py`'s docstring) and Hugging Face Hub's V-JEPA 2
collection; it was not independently verified to exist on the Hub from
this sandbox, since that check itself needs the blocked network access.
If the exact id differs, only `configs/*.yaml`'s `encoder.checkpoint`
needs to change.

## Transform definitions and why each is designed the way it is

All six transforms live in `transforms/scene_transform.py`, are
deterministic given `(scene.seed, transform_name)` (via a
`zlib.crc32`-derived RNG seed -- **not** Python's built-in `hash()`,
which is randomized per-process for strings and would silently break
reproducibility across runs), and return both the new `SceneState` and a
dict of ground-truth parameters.

- **camera_translation**: moves the camera position by a random-direction
  vector, keeping orientation fixed. This isolates the translation
  subgroup of SE(3) (no reorientation), matching the transform's name
  literally.
- **camera_rotation**: orbits the camera around the scene origin
  (azimuth + elevation delta) and re-derives orientation so it keeps
  looking at the origin. This is "camera rotation" in the sense the
  project brief's flagship experiment means it -- a viewpoint change --
  as opposed to spinning the camera in place about its own optical axis,
  which would barely change the rendered image for a roughly-centered
  scene.
- **object_translation** / **object_rotation**: pick one object in the
  scene (uniformly, via the transform's own RNG stream) and move/rotate
  it. Rotation is about the world z-axis; this is why the shape pool
  (`generation/scene.py:SHAPES`) excludes the sphere by default -- a
  z-rotated sphere renders identically to the original, which would
  make `object_rotation` untestable for that object.
- **lighting_change** / **texture_change**: the two non-geometric
  controls. They are implemented to touch *only* light energy/position
  or object color -- `tests/test_transforms.py::test_control_transforms_preserve_geometry`
  asserts camera and object positions/rotations are bit-for-bit
  unchanged, which is what makes them a valid control for "should not
  substantially alter physical state."

## Metrics: what each one means and why more than one is needed

`metrics/equivariance.py` (component A) reports three numbers together
because they are not redundant: held-out R^2 (variance explained, scale-
and direction-sensitive), mean cosine similarity (direction only), and
mean relative L2 error (magnitude-sensitive, unnormalized by target
variance). A representation could score well on one and poorly on
another -- e.g. right direction, wrong scale -- and collapsing to a
single number would hide that.

`metrics/invariance.py` (component B) is deliberately *not* the same
computation as equivariance with `rho(T) = Identity` (that comparison
lives in `baselines/identity_baseline.py`) -- invariance is "how much did
the raw representation change", asked with no fitted map at all, and is
the natural thing to check for the appearance-only control transforms
(we want it high there) as a counterpart to equivariance (which we want
to be *predictable*, not necessarily *absent*, for the geometric
transforms).

`probes/linear_state_probe.py` (component C) reports a single held-out
R^2 for decoding one scalar physical quantity from Z with a linear map.
A high score means that quantity is linearly *accessible*; per the
project's explicit instruction, this must never be read as "the
representation understands" that quantity -- see "Interpretation
guardrails" below.

### A subtle shape bug worth documenting

`metrics/common.py` originally used `np.atleast_2d` to normalize inputs
before computing row-wise metrics. For a 1-D array of length `N`,
`np.atleast_2d` returns shape `(1, N)`, not `(N, 1)`. This is invisible
for representation vectors (`D`-dimensional, `D > 1`, so inputs are
already 2-D), but silently wrong for scalar physical-state targets
(`D = 1`): a `(N,)` array of predictions minus an `(N, 1)` array of
targets broadcasts to an `(N, N)` matrix instead of raising a shape
error, producing a nonsensical (wildly negative) R^2. This was caught
by `tests/test_probes.py::test_linear_state_probe_recovers_linear_target`
failing on synthetic data with a *known* perfect linear relationship
(R^2 should have been ~1.0, was instead -59). The fix
(`metrics/common.py:_as_samples_by_features`) reshapes a 1-D array of
length N to `(N, 1)` instead, matching the "rows are scenes" convention
used everywhere else. `tests/test_metrics.py::test_r_squared_handles_1d_vs_column_vector_shape_mismatch`
is a regression test for this specific failure mode. The lesson --
silent shape broadcasting is a real risk in this kind of numpy-heavy
code, and any metric that might see both `(N,)` and `(N, 1)` inputs
needs an explicit, tested normalization step rather than a generic
"make it 2-D" call.

## Baselines

Every experiment reports three things against the same test split, not
just the learned `rho(T)`:

- **Identity** (`baselines/identity_baseline.py`): predict `Z' = Z`. If
  this already scores well, the representation simply doesn't move much
  under the transform -- an invariance finding, not evidence that a
  learned map captured something.
- **Shuffled-pairing** (`baselines/shuffled_pairing_baseline.py`): fit
  the same ridge regression on deliberately mismatched `(Z_i, Z'_j)`
  pairs, `i != j`. If this scores nearly as well as the correctly-paired
  fit, the apparent equivariance is coming from dataset-level structure
  or regularization, not from genuine scene-by-scene correspondence.
- **Pixel-statistics encoder** (`encoders/pixel_baseline.py`): a
  non-learned encoder (mean-pooled low-resolution RGB grid) that can be
  swapped in for the whole pipeline via `encoder.name: pixel_baseline`
  in a config, to check whether a result is specific to the learned
  encoder or would show up for trivial pixel correlation too.

## Interpretation guardrails

Per the project brief, this repo must not claim a representation
"understands geometry" from probe performance alone. Concretely:

- A high equivariance R^2 says the *change* in Z under T is *linearly
  predictable* from Z. It does not by itself rule out the predictor
  exploiting a confound (see the shuffled-pairing baseline) or say
  anything about whether that predictability would hold under a
  different scene distribution.
- A high state-probe R^2 says a quantity is *linearly decodable*. It
  does not mean the representation was built to represent that quantity,
  or that decoding remains possible outside the training distribution.
- Every one of these numbers must be reported next to its baselines, not
  in isolation, and every report must state which encoder weights
  (pretrained vs. randomly initialized) were used to produce it.

## Task 2: the Blender 5.0 compositor API is not the one documented online

Task 2 needs depth and instance-segmentation ground truth, produced via
Blender's compositor (a `Depth` and an `Object Index` render pass,
routed to file outputs). Most Blender-scripting references online
describe the pre-4.x compositor API (`scene.use_nodes`,
`scene.node_tree`, a `CompositorNodeComposite` node, `file_slots`,
`CompositorNodeMapRange`, `file_output.base_path`). **None of that
exists in the installed Blender 5.0/bpy 5.0.1** -- it was redesigned
into a generic node-group system shared with geometry nodes. The
working API, verified empirically (`generation/bpy_renderer.py:
_setup_ground_truth_compositor`) rather than assumed from
documentation:

- `scene.node_tree` is gone. Create the compositor as
  `bpy.data.node_groups.new(name, 'CompositorNodeTree')` and assign it
  with `scene.compositing_node_group = group`.
- There is no `CompositorNodeComposite` node anymore, and none is
  needed: the normal RGB image still saves via `scene.render.filepath`
  regardless of what the compositor graph does, as long as nothing
  reroutes `Render Layers -> Image` away from it.
- `CompositorNodeMapRange` no longer exists in the compositor node set;
  `ShaderNodeMapRange` works fine inside a `CompositorNodeTree` (the
  node-group unification lets shader nodes be used here) and is what
  this project uses to normalize the raw `Depth`/`Object Index` passes
  into `[0, 1]` before writing 16-bit PNGs (writing raw un-normalized
  float values directly to a `PNG` file slot silently clips/saturates
  them at 1.0 -- this cost real debugging time before the fix).
- The Object Index pass output socket on the Render Layers node is
  named `"Object Index"`, not the older `"IndexOB"`.
- `CompositorNodeOutputFile` no longer has `base_path`/`file_slots`;
  use `.directory`, `.file_name` (a filename *prefix*, not a
  frame-numbered template -- set it explicitly before each render call
  to control per-frame filenames yourself), and
  `.file_output_items.new(socket_type, name)` where `socket_type` is a
  bare enum string (`'FLOAT'`, not `'NodeSocketFloat'`).
- Each `file_output_items` entry needs `override_node_format = True`
  and, critically, **`item.format.media_type = 'IMAGE'`** set *before*
  `item.format.file_format = 'PNG'` -- the node (and each item) defaults
  to `media_type = 'MULTI_LAYER_IMAGE'`, under which `file_format` only
  accepts `'OPEN_EXR_MULTILAYER'` and silently writing everything into
  one combined `.exr` instead of separate per-item PNGs is the failure
  mode if this is missed. `save_as_render = False` on each item is also
  required to get raw, non-color-managed float values for depth/
  segmentation instead of a Filmic/view-transformed image.

## Task 2: depth-pass semantics were verified empirically, not assumed

Blender's `Depth` render pass could plausibly mean either the Euclidean
distance from the camera to each point, or the camera-space Z
(perpendicular distance to the image plane) -- these differ for any
point off the camera's optical axis, and getting this backwards would
silently corrupt every depth-based probe built on top of it. It was
resolved by rendering a large flat plane face-on to a camera looking
straight down: camera-space Z predicts a spatially *uniform* depth value
across the whole frame (including off-center pixels viewed at an oblique
angle), while Euclidean ray distance would increase toward the frame
edges. The measured result was exactly uniform (`generation/scene.py`'s
camera at height 5 above the plane -> depth 5.0 at both the center and
the corners of the frame, to float precision) -- confirming camera-space
Z. See `generation/COORDINATE_SYSTEM.md` for the documented conclusion
and `tests/test_bpy_renderer.py::test_depth_pass_is_planar_not_euclidean`
for the automated version of this exact check.

## Task 2: additive SceneState changes, kept backward compatible with V0

`ObjectState.instance_id` and `CameraState.sensor_width_mm` were added
as new fields with defaults (`0` and `32.0` respectively) rather than
by renaming or restructuring anything, specifically so V0's existing
transform/experiment pipeline (`transforms/scene_transform.py`,
`experiments/run.py`, `representations/`) keeps working unchanged --
Task 2 explicitly does not touch transformation experiments, so nothing
there should have needed to change to support it, and after this
addition the full V0 pipeline and test suite were re-run to confirm
that. `generation/scene_sampler.sample_scene` now assigns each object a
1-indexed `instance_id`, and `generate_scene` is an alias for
`sample_scene` (Task 2's spec names it `generate_scene()`; V0 code
already imports it as `sample_scene`, so both names exist rather than
picking one and breaking the other). The look-at-orientation math that
`sample_scene`, `transforms/scene_transform.apply_camera_rotation`, and
`generation/motion.py`'s camera-orbit trajectory all need was
consolidated into `transforms/se3.look_at_euler` during this task
(previously duplicated in the first two places) rather than written a
third time.

## Task 3: a unified SE(3) engine replaces four separate ad hoc computations

V0's `transforms/scene_transform.py` computed each geometric transform's
new position/orientation directly (e.g. `camera_rotation` called
`orbit_position` then separately reconstructed orientation with
`look_at_euler`) and never produced an explicit transformation matrix.
Task 3 asks for "the exact ground-truth transformation matrix where
appropriate" and "SE(3) conventions consistently," so every geometric
transform (`transforms/scene_transform.py`) was rewritten around one
shared computation: construct a 4x4 matrix `T`, then
`new_pose = T @ old_pose` (`transforms/se3.py`'s `pose_matrix` /
`matrix_to_pose`), decomposed back to `(position, rotation_euler)`. The
matrix returned in each transform's metadata (`transform_matrix`) is
always exactly the matrix that produced the result -- there is no
second, separately-derived computation that could silently drift out of
sync with it.

This unification revealed that all four geometric transforms are one of
three primitive SE(3) operations:

- **camera_translation** / **object_translation**: `translation_matrix(delta)`.
- **camera_rotation**: a pure rotation about the *world origin*
  (`rotation_only_matrix`) -- see `orbit_rotation_matrix` below.
- **object_rotation**: a rotation about the *object's own current
  position* (`rotate_about_point_matrix`, the
  `Translate(pivot) @ Rotate(R) @ Translate(-pivot)` sandwich), which
  is what "rotate an object in place" means concretely.

### `orbit_rotation_matrix`: deriving one rotation matrix for an azimuth+elevation orbit

`camera_rotation` moves the camera by an azimuth delta and an elevation
delta on a sphere -- two numbers, not obviously "a rotation matrix" on
their own, since elevation is defined relative to the *current*
azimuth. By Euler's rotation theorem, though, any composition of
rotations about axes through a common point (here, the world origin) is
itself some single rotation about that point, so one must exist; it was
derived as `R_elevation(tangent axis at the NEW azimuth) @
R_azimuth(world Z)`, with the tangent axis
`(sin(az_new), -cos(az_new), 0)` (a horizontal direction perpendicular
to the new radial direction). The sign of that axis, and the whole
formula, was verified numerically against the pre-existing (and
already-tested) `orbit_position` function over 500 random trials before
being trusted -- not derived and assumed correct. A second, independent
check confirmed that applying this *same* matrix to the camera's full
orientation (not just its position) exactly reproduces
`look_at_euler(new_position, origin)`, i.e. one matrix correctly moves
the whole rigid pose, which is what makes it usable as `T` for the
whole camera, not just its position (`tests/test_se3_matrices.py`). This
only holds for `pivot = origin` and away from the +-90 degree elevation
poles, where `orbit_position`'s own clamping (to avoid a singularity)
makes the two diverge -- not a concern for this project's configured
elevation ranges, but a real, documented limit of the function.

### A second Euler-angle-uniqueness gotcha, this time in the transforms themselves

Early versions of `apply_camera_translation`/`apply_object_translation`/
`apply_object_rotation` round-tripped *every* pose component (including
the one that is mathematically guaranteed unchanged -- rotation for a
pure translation, position for a rotation about the object's own
pivot) through `matrix_to_pose`. `tests/test_scene_transform_matrices.py`
caught this: a translated object's `rotation_euler` field changed even
though nothing rotated, because `matrix_to_euler` can legitimately
return a *different* Euler triple for the exact same (bit-identical)
rotation matrix (the same non-uniqueness noted in
`tests/test_transforms.py::test_euler_matrix_roundtrip`, now caught one
level up, at the transform level rather than the raw-math level). The
fix: when a component is known analytically to be unchanged, the code
now keeps the literal original tuple for it instead of deriving it from
the matrix -- which also makes `fixed_variables` true at the field
level (byte-identical), not just "physically-equivalent-up-to-Euler-
representation."

### World-to-camera vs. camera-to-world

Per Task 3's explicit instruction not to assume these are
interchangeable: `generation/scene.py:camera_to_world_matrix(camera)`
is the camera's pose in world coordinates (`pose_matrix` applied to its
position/rotation); `world_to_camera_matrix(camera)` is its *exact*
rigid inverse (`transforms/se3.py:inverse_rigid`, `[R^T, -R^T @ t]`, not
the common wrong shortcut `[R^T, -t]`).
`tests/test_se3_matrices.py::test_inverse_rigid_is_not_naive_negate_translation`
is a regression test specifically against that wrong shortcut, and
`test_world_to_camera_and_camera_to_world_are_inverses_but_not_equal`
checks both that they invert each other and that they are not the same
matrix.

### Pairs render full ground truth on both sides, not one RGB frame

`transforms/pairs.py`'s `pair_dir/{original,transformed}/` each contain
a full Task 2 `save_ground_truth` output (RGB + depth + segmentation +
metadata), produced via a trivial static (zero-motion) `Trajectory`,
rather than V0's original single-RGB-frame-per-variant approach. This
reuses Task 2's already-tested renderer and gives depth/segmentation
ground truth for transformation pairs "for free" -- e.g.
`tests/test_pairs.py::test_geometric_transform_changes_depth_but_appearance_transform_need_not`
checks that a camera move changes the depth map while a lighting change
does not, a cross-check that would not be possible with RGB alone.

### Backward compatibility re-verified, not just asserted

`transforms/scene_transform.py`'s public surface
(`TRANSFORM_NAMES`/`GEOMETRIC_TRANSFORMS`/`CONTROL_TRANSFORMS`/
`TransformConfig`/`apply_transform`) is unchanged, and V0's full
experiment pipeline (`experiments/run.py`) was re-run after this rewrite
on the same seeds it was run on for the V0 smoke-test report; the
reported R^2/cosine-similarity numbers came back bit-for-bit identical,
confirming the new matrix-based computation is not just "similar to"
but exactly equivalent to what V0 computed directly, for every case V0
exercises.

## Task 4: V-JEPA 2.1, and a real reproducibility gap V0 had

### Upstream model, verified rather than assumed

Task 4 asks for "the official V-JEPA 2.1 implementation," which is a
real, distinct release from V-JEPA 2 (not a typo) -- confirmed by
reading the `facebookresearch/vjepa2` GitHub repo directly rather than
assuming "2.1" meant the same thing as "2." Its ViT-B/16 checkpoint:
80M parameters, **384** resolution (V-JEPA 2's was 256), checkpoint file
`vjepa2_1_vitb_dist_vitG_384.pt` (distilled from a ViT-G teacher, per
the checkpoint's own filename), config `configs/train_2_1/vitb16`,
downloaded upstream from
`https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt`.
Architecturally, V-JEPA 2.1's changes (dense predictive loss, deep
self-supervision, distillation) are to the *training recipe*, not the
model class, so it loads through the same `transformers.VJEPA2Model`
already verified for V-JEPA 2 in `encoders/vjepa2.py` (V0) -- only the
resolution and (for the untrained fallback) the architecture-size
constants needed to change, not the loading code's structure.

As of this writing, no checkpoint was found under the official
`facebook/` Hugging Face Hub namespace specifically for V-JEPA 2.1 (a
hosting request for it, covering the ViT-B/L/g/G variants, was open on
the upstream GitHub issue tracker as of March 2026). A third-party
conversion exists at `apiantonio/vjepa2.1-vit-base-384`, whose own model
card claims bit-exact agreement with Meta's reference implementation;
`encoders/vjepa.py:DEFAULT_CHECKPOINT` points there, explicitly labeled
as an unofficial community conversion rather than a `facebook/` release,
with `checkpoint=` fully overridable once an official repo exists. This
sandbox cannot independently verify any of this by downloading it --
both `huggingface.co` and `dl.fbaipublicfiles.com` are network-policy-
blocked here, the same constraint documented for V-JEPA 2 in this file's
earlier "Encoder: honesty about pretrained weights" section, now true of
V-JEPA 2.1 too regardless of which host the real weights end up on.

### A real reproducibility gap in V0, found by writing Task 4's smoke test

Task 4 requires "the same video produces consistent representations."
V0's `encoders/vjepa2.py` falls back to a randomly-initialized model
when pretrained weights can't be loaded (this sandbox's permanent
situation), but never seeded that initialization -- so two separate
Python processes hitting the fallback path would silently get two
*different* random encoders, and "consistent representations" would
only hold within a single already-constructed encoder instance, not
across runs (exactly the kind of gap Task 1's DESIGN.md warns a
selectivity/control-task mindset should catch). `encoders/vjepa.py`
fixes this: `torch.manual_seed(fallback_seed)` (default 0) runs
immediately before constructing the untrained model, and
`tests/test_vjepa_encoder.py::test_same_video_different_instances_same_seed_is_bit_identical`
constructs two *independent* `VJEPAEncoder` instances and checks their
output is bit-identical -- verified directly (not just asserted) by
running the equivalent check as two separate OS processes and diffing
the saved `.npy` arrays before writing the test. A companion test
(`test_different_fallback_seed_gives_different_untrained_weights`)
checks the seed is actually doing something, so the determinism test
above isn't trivially true for an unrelated reason (e.g. an
accidentally-always-zero-initialized model).

### Representation extraction is deliberately not pooled

`VJEPAEncoder.encode()` returns the encoder's native token sequence
(`(num_tokens, hidden_size)`), not a single pooled vector like V0's
`FrozenEncoder.encode_video()`. This is a deliberate difference, not an
oversight: Task 4's own pipeline diagram is "video -> preprocessing ->
V-JEPA 2.1 -> representation/tokens -> saved tensor," and Task 4
explicitly defers evaluation ("do not implement evaluation metrics
yet"). Pooling is an evaluation-time decision (mean pooling is what V0
happened to choose for its own linear-probe pipeline, per DESIGN.md
§4); baking it into extraction now would decide that question on a
later task's behalf and throw away information (which tokens moved)
that a later, more spatially-aware probe might want. `mean_pool()` is
provided as a convenience, not applied automatically -- see
`encoders/base.py`'s `VideoEncoder` docstring and
`encoders/REPRESENTATION_FORMAT.md` for the full reasoning and the
resulting tensor format.

### Two encoder interfaces, on purpose

`encoders/base.py` now defines both `FrozenEncoder` (V0's
`encode_video -> pooled vector`, used by `representations/`/
`experiments/run.py`) and `VideoEncoder` (Task 4's `encode -> native
tokens`). They are not unified into one hierarchy: doing so now would
mean picking, on Task 4's behalf, whether pooling lives inside or
outside the encoder -- exactly the question the previous section says
Task 4 defers. `encoders/vjepa2.py` (V0, pooled, still used by the V0
experiment pipeline) and `encoders/vjepa.py` (Task 4, unpooled, new)
therefore coexist rather than one replacing the other, following the
same pattern as Task 2/3's `generate_scene`/`sample_scene` and
`render_trajectory`/`render_scene` coexistence: each formal task adds
its own well-specified module without an unrequested migration of the
pipeline that predates it.

### Task 4 -- real pretrained weights

The environment's network-egress policy was changed (by the user, in
Claude Code on the web's environment settings) to allow `huggingface.co`,
which this project had no ability to change from inside a session.
`dl.fbaipublicfiles.com` remains blocked, but is no longer needed (see
below). Verifying real pretrained-weight loading, with live Hub access,
surfaced two problems with what had only ever been *assumed* while the
sandbox was network-blocked:

1. **`facebook/vjepa2-vitb-fpc64-256` (V0's `encoders/vjepa2.py` default)
   does not exist on the Hub.** `huggingface_hub.HfApi().model_info(...)`
   returns `RepositoryNotFoundError` for it. Listing `facebook`'s actual
   V-JEPA 2 models (`api.list_models(author="facebook", search="vjepa")`)
   shows only ViT-L/H/G sizes (`vjepa2-vitl-fpc64-256`,
   `vjepa2-vith-fpc64-256`, `vjepa2-vitg-fpc64-256`/`-384`, plus a few
   `-ssv2`/`-diving48` fine-tunes) -- **no ViT-B**. This checkpoint name
   had been in the codebase, unverified, since before this sandbox had
   any Hub access at all.
2. **"V-JEPA 2.1" has no official `facebook/`-namespaced checkpoint on
   the Hub, at any size.** Searching the Hub for `vjepa2.1` / `vjepa
   2.1` / `v-jepa2.1` returns only individual/community namespaces
   (`apiantonio`, `davevanveen`, `Dev-Jahn`, `dgrauet`, `RAntonello`),
   none under `facebook`. The one this project's `encoders/vjepa.py`
   pointed `DEFAULT_CHECKPOINT` at,
   `apiantonio/vjepa2.1-vit-base-384`, has a `config.json` declaring
   `"model_type": "vjepa21"` and an `auto_map` pointing at that repo's
   own `modeling_vjepa21.py`/`configuration_vjepa21.py` -- i.e. loading
   it requires `AutoModel.from_pretrained(..., trust_remote_code=True)`,
   which executes that individual's Python code, not
   `transformers.VJEPA2Model` as the old module docstring claimed. There
   is also no way to confirm from the Hub alone that its weights are a
   faithful conversion of anything Meta actually released.

Both were genuinely unverifiable claims while `huggingface.co` was
blocked -- this is exactly the situation DESIGN.md's "honesty about
pretrained weights" principle exists for, and why every report and
`metadata.json` already carried an explicit `pretrained` field rather
than assuming success.

Given the choice, put to the user directly rather than resolved
silently (`AskUserQuestion`, per the standing QA protocol's "no silent
hypothesis/protocol changes" rule): switch to
**`facebook/vjepa2-vitl-fpc64-256`** -- official Meta weights, loads
through `transformers`' built-in `VJEPA2Model`/`VJEPA2Config` with no
`trust_remote_code`, `config.json` verified live
(`"architectures": ["VJEPA2Model"]`, `"model_type": "vjepa2"`,
`hidden_size=1024`, `num_hidden_layers=24`, `num_attention_heads=16`,
`crop_size=256`). This is a ViT-L, not ViT-B -- larger and slower on
CPU -- but is the smallest official, code-verified, no-remote-code
V-JEPA 2 checkpoint that actually exists.

Changes made, all mechanical (checkpoint identifier + matching config
constants, everywhere the old ViT-B numbers appeared): `encoders/vjepa.py`
(`DEFAULT_CHECKPOINT`, `DEFAULT_CROP_SIZE=256`, `VITB16_CONFIG_KWARGS`
renamed `VITL16_CONFIG_KWARGS` with real ViT-L values, `output_dim` is
now `1024`), `encoders/vjepa2.py` (V0, same checkpoint/config fix, so V0
doesn't keep a now-known-wrong checkpoint name), `configs/config.py`,
`configs/default.yaml`, `configs/experiments/camera_rotation_v0.yaml`
(checkpoint string updated), `tests/test_vjepa_encoder.py`
(`output_dim == 1024`), `encoders/REPRESENTATION_FORMAT.md` (hidden_size,
example `metadata.json`), `README.md`, `DESIGN.md` (an explicit
amendment note, not a silent rewrite of the original ViT-B/16 text). The
research design itself -- transforms, probes, train/test split,
baselines -- is unaffected; only the encoder's size changed.

New evidence this actually works, not just that the code compiles:
`tests/test_vjepa_encoder.py::test_pretrained_checkpoint_loads_real_weights`
(marked `@pytest.mark.slow`, needs live Hub access) constructs
`VJEPAEncoder(pretrained=True)` for real, asserts `encoder.pretrained is
True` (i.e. the except-and-fall-back branch was NOT taken),
`encoder.checkpoint == DEFAULT_CHECKPOINT`, `output_dim == 1024`, and
runs a full `encode()` forward pass on a synthetic video, checking the
output shape and that it's finite. Run directly: real weights
downloaded and loaded in ~4 minutes on this machine's network/CPU, the
forward pass produced a finite `(64, 1024)` representation, and the full
fast suite (102 tests, `pytest -m "not slow"`) still passes with the new
checkpoint's config wired through.

## Deferred to later versions

Explicitly out of scope until V0's core loop is validated with real
pretrained weights (per "do not expand the project until this
experiment works end-to-end"):

- **Temporal consistency (D) / object persistence (E) / counterfactual
  dynamics (F)**: need clips with actual motion, not repeated static
  frames -- see "Static-clip videos" above.
- **Additional encoders**: `encoders/base.py`'s `FrozenEncoder` interface
  is the only thing a new encoder needs to implement; V-JEPA 2 and the
  pixel baseline are the first two.
- **Non-primitive assets, physics, larger scenes**: this is exactly the
  point at which swapping in the full Kubric pipeline (see "Renderer
  choice" above) would start paying for its dependency weight.
- **Neural (non-linear) probes**: only after linear probes are
  established as a baseline, per "prefer simple linear probes ... before
  using neural probes."
- **Clifford/geometric algebra**: explicitly excluded from V0 by the
  project brief; nothing here should be read as a step toward it.
