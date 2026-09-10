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
