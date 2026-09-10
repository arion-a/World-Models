# DESIGN.md — Research specification for the Geometric Consistency Probe

**Status: research definition, written before (and binding on) the
implementation.** This document specifies *what* the project measures
and *why* those measurements mean what we claim they mean. It is the
protocol later code must follow. It is not a description of the current
V0 code — for that, see `IMPLEMENTATION_NOTES.md`, which records
engineering decisions made while building the pipeline this spec
describes, and is subordinate to this document wherever the two disagree.

**Amendment (post-Task-4, user-approved):** every mention of "V-JEPA 2
ViT-B/16" below is superseded by **V-JEPA 2 ViT-L/16**
(`facebook/vjepa2-vitl-fpc64-256`). Once real Hugging Face Hub access was
available, neither of this project's originally planned checkpoints
turned out to exist under an official namespace: `facebook/vjepa2-vitb-fpc64-256`
is not hosted at all, and "V-JEPA 2.1" has no official
`facebook/`-namespaced checkpoint (only unverified third-party
conversions requiring `trust_remote_code=True`). ViT-L/16 is the
smallest official, code-verified V-JEPA 2 checkpoint on the Hub. This
was surfaced to and chosen by the user rather than changed silently --
see `encoders/vjepa.py`'s module docstring and `IMPLEMENTATION_NOTES.md`'s
"Task 4 -- real pretrained weights" section for the full account. The
research question, transforms, probes, and everything else in this
document are unaffected by this substitution -- only the encoder's size
changed, not its family or the experimental design.

## 0. Background: what we're building on

This project sits inside two lines of existing work, and borrows its
vocabulary and its caution from both.

**V-JEPA / V-JEPA 2.** V-JEPA 2 (Assran, Bardes, Garrido, et al., Meta,
*"V-JEPA 2: Self-Supervised Video Models Enable Understanding,
Prediction and Planning,"* arXiv:2506.09985, June 2025) is a
**joint-embedding predictive architecture**: a Vision Transformer
encoder (with 3D rotary position embeddings) maps a video, divided into
spatiotemporal "tubelets" (e.g. 2 frames × 16×16 pixels), to a sequence
of latent tokens. Unlike a generative video model, it is trained to
predict *masked tokens in this latent space*, not raw pixels — a
"visual mask-denoising objective" applied self-supervised over more
than a million hours of internet video, explicitly avoiding pixel-level
reconstruction. Its stated motivation is exactly our research question,
approached from the model-building side: build a representation that
captures the "predictable" structure of the physical world (object
permanence, extrinsic physics, "intuitive physics") well enough that a
lightweight action-conditioned predictor on top of it supports zero-shot
robot planning. That motivation is a hypothesis, not an established fact
about the representation — which is what this project probes.

**Representation-level probing and equivariant representation
learning.** A representation is a linear probe's input, evaluated by how
well a simple classifier or regressor recovers a target property from
it — a methodology with well-documented failure modes: Hewitt & Liang
(2019) showed a probe can achieve high accuracy by memorizing the
probing dataset rather than reading information out of the
representation, and proposed *control tasks* (fit the same probe
architecture on deliberately randomized targets) to measure *selectivity*
— the gap between real and control performance. Belinkov's survey,
*"Probing Classifiers: Promises, Shortcomings, and Advances"*
(Computational Linguistics 48(1), 2022), further finds that **linear
probes tend to have higher selectivity than non-linear ones**, i.e. are
less prone to this failure mode — direct support for this project's
"prefer linear probes" principle on methodological, not just
simplicity, grounds. Separately, the *equivariant representation
learning* literature gives a precise target for what "the representation
encodes a transformation" should mean: an encoder `f` is equivariant to
a group `G` acting on inputs via `φ(g)` if there exists a representation
`ρ(g)` of that same group acting on the embedding space such that
`f(φ(g)(x)) ≈ ρ(g) f(x)`, with `ρ` a group homomorphism
(`ρ(g∘h) = ρ(g)ρ(h)`) and — critically — **independent of the specific
input `x`** ("steerability"). Section 6 spells out exactly how our
operational test in this project relates to, and falls short of, that
strict definition, and why that gap matters. Prior synthetic benchmarks
built for this kind of question exist (e.g. 3DIEBench, built for
disentangling invariant/equivariant image representations under
3D-controllable rotations) but target static images and a fixed encoder
family, not a pretrained *video* model under a battery of *scene-level*
transformations with ground-truth 3D state — which is the gap this
project occupies. Contemporaneous work (*"What, Where, and How: Probing
Spatiotemporal Representations in Video Foundation Models,"* which
probes V-JEPA 2 and VideoMAE-v2 for camera-motion understanding via
layer-wise linear probes) confirms the question is live and being asked
by others too, from a classification-probe angle; this project's
distinguishing move is using *known, applied, ground-truth* scene
transformations and asking about *predictable change* (equivariance),
not just post-hoc classification accuracy. *"Probing the 3D Awareness of
Visual Foundation Models"* (dense multi-layer probes for depth/surface
normals) is the closest precedent for this project's depth probe
(§10) and motivates a future move from single pooled vectors to
per-token probing (§4).

**Basic SE(3) transformations.** The rigid-body transformations in this
project act within `SE(3)`, the group of position + orientation changes
in 3D space: a translation (`R^3`, commutative, no orientation change)
and a rotation (`SO(3)`, the group of 3×3 orthogonal matrices with
determinant 1, non-commutative). A full rigid pose is `(R, t)` with
`R ∈ SO(3)`, `t ∈ R^3`, composing as `(R2, t2)∘(R1, t1) = (R2 R1, R2 t1 + t2)`.
Every "geometric" transform below is an explicit, named element of this
group (or a constrained subset of it), applied to one part of the scene
at a time.

**Synthetic 3D evaluation.** Kubric (Greff et al., CVPR 2022) is the
reference design point: physics-simulated (PyBullet) multi-object scenes
rendered in Blender, chosen specifically because synthetic generation
gives exact, free ground truth (segmentation, depth, optical flow, and
here, full physical state) that is either unavailable or noisy for real
video — the same reason this project is synthetic-only. See
`IMPLEMENTATION_NOTES.md` for why V0's renderer uses `bpy` directly
rather than the `kubric` package itself.

## 1. Research question

> Do self-supervised video representations encode the geometry of the
> underlying physical world, or do they primarily encode appearance and
> temporal correlations?

This project does not answer this in the abstract, for "video
representations" in general. It answers a narrower, checkable version of
it: for one specific frozen encoder (V-JEPA 2 ViT-B/16), under one
specific pooling scheme, on one specific synthetic scene distribution,
is a known 3D transformation's effect on that encoder's output
*predictable* (§6), and is physical state *linearly accessible* from it
(§10)? Every later section exists to make that narrower question
precise enough to have an unambiguous answer, and §13 is explicit about
which broader claims that answer can and cannot support.

## 2. Physical state `S`

`S` is the complete, exact, generative description of one scene at a
single instant — the ground truth a synthetic renderer has and a
real-video pipeline would not. We partition it into a **geometric**
subspace (pose information: what a rigid transformation acts on) and an
**appearance** subspace (nuisance parameters a geometric transformation
must leave untouched, and an appearance transformation acts on
exclusively):

```
S = (C, {O_1, ..., O_N}, Light)

C  (camera, geometric+intrinsic):
     position      p_cam  ∈ R^3
     orientation   R_cam  ∈ SO(3)
     intrinsics    K      (focal length / field of view; fixed across
                           this project's experiments unless stated)

O_i (object i, geometric+identity+appearance):
     position      p_i    ∈ R^3
     orientation   R_i    ∈ SO(3)
     velocity      v_i    ∈ R^3 ∪ {undefined}   (see note below)
     identity      id_i           persistent instance id across a clip
     category      shape_i        {cube, cone, cylinder, monkey, ...}
     appearance    color_i, material_i   (geometric transforms hold these fixed)

Light (appearance):
     position/direction, energy, color
```

**Velocity is defined but not always observable.** `v_i` is only
well-defined relative to a time window in which `O_i` moves. A single
static instant (or, as in V0, a clip built by repeating one rendered
frame — see `IMPLEMENTATION_NOTES.md`, "Static-clip videos") has no
such window, so `v_i` is `undefined` there by construction, not zero.
The velocity probe (§10) only becomes meaningful once clips with real
multi-frame motion exist — explicitly future work, not claimed by V0.

**Depth is not part of `S`; it is a derived observable.** Given `S` and
a camera, per-pixel depth is a deterministic function of scene geometry
that a synthetic renderer can export directly (`D(S, C)`), the same way
Kubric exports depth/segmentation "for free." It is listed as a probe
target (§10) precisely because it is cheap, exact ground truth here and
expensive or noisy to obtain for real video — not because it is an
independent generative parameter of the scene.

**Object identity** (`id_i`) is a persistent label, not a geometric
quantity, and is what makes "move *object 3*" a well-defined operation
(§8) and what a future object-persistence probe (component E, deferred)
would test recoverability of across a transformed pair.

## 3. Observation: `V = R(S)`

`R` is the renderer: a (for our purposes deterministic, given a fixed
random seed for any stochastic sampling inside the renderer, e.g. Monte
Carlo path tracing noise at a fixed sample count) function from physical
state to a video tensor, `V ∈ {0,...,255}^{T×H×W×3}`. `R` is fixed
throughout this project (Blender/Cycles — see `IMPLEMENTATION_NOTES.md`)
and is never learned or modified; it is the one part of the pipeline
whose input/output relationship is known exactly by construction, which
is what licenses treating `S` as ground truth for everything downstream.

## 4. Representation: `Z = E(V)`

`E` is the frozen, pretrained encoder (V-JEPA 2 ViT-B/16; see §0).
"Frozen" means no parameter of `E` is ever updated by this project —
every probe and every `ρ(T)` fit downstream is a separate, disposable
linear model trained on top of `E`'s fixed output. `E` in general
produces a sequence of spatiotemporal tokens, not a single vector; this
project's primary representation `Z` is the **mean pooling of that
token sequence into one vector per clip**, chosen because it is the
simplest well-defined reduction (matching "prefer simple ... before
anything fancier") and because it makes every downstream probe an
ordinary vector regression. This is a real, stated limitation: mean
pooling can only reveal geometric information that survives *averaging
over space and time*, and cannot localize *where* in the token sequence
any geometric signal lives. Per-token/patch probing (in the style of
"Probing the 3D Awareness of Visual Foundation Models," §0) is the
natural refinement once the pooled-vector question has been asked and
answered, and is explicitly future work, not part of this
specification's initial claims.

## 5. Transformation: `S' = T(S)`, `Z' = E(R(S'))`

`T` is a named, parameterized operator on physical state space —
applied to `S` directly, in the exact representation of §2, never to
pixels or to `Z`. This ordering is the entire reason to use synthetic
data: `T` is known exactly (we chose its parameters), so any relationship
we later find between `Z` and `Z'` can be checked against a
ground-truth transformation rather than inferred from an unknown or
noisy one. `S'` is rendered and encoded exactly as `S` was
(`V' = R(S')`, `Z' = E(V')`), through the same fixed `R` and `E` — the
only thing that differs between the two pipelines is the input state.

## 6. Geometric equivariance: what `Z' ≈ ρ(T)Z` means, precisely

**Strict definition** (from the equivariant representation learning
literature, §0): `E` is equivariant to a transformation group `𝒯` acting
on physical states if there is a group homomorphism `ρ: 𝒯 → (maps on Z)`
— i.e. `ρ(T2 ∘ T1) = ρ(T2) ρ(T1)` — such that for **every** scene `S`,

```
E(R(T(S))) ≈ ρ(T) E(R(S))
```

with `ρ(T)` depending only on `T`, never on `S` ("steerability" —
the same map works no matter which scene it's applied to).

**What this project actually tests, and why it is weaker.** For each
*transform type* (e.g. "camera_rotation"), the applied `T` varies
continuously in its parameters (a random azimuth/elevation delta drawn
per scene — see §8), and we fit a *single* linear `ρ` by regression
across many `(Z, Z')` pairs spanning that whole parameter range, on a
training set of scenes, then evaluate its fit on held-out scenes. This
is an **aggregate, parameter-marginalized** approximation to `ρ`, not a
per-transform-element representation, and we do not test the
homomorphism property (`ρ(T2)ρ(T1) ≈ ρ(T2∘T1)`) at all in this version.
A high score under this protocol is evidence that *the net effect of
this transform family on `Z`, averaged over the sampled parameter
range, is well-approximated by one linear map* — a real, useful, but
strictly weaker claim than strict group equivariance. **A stronger
version of this test — fitting `ρ` as an explicit function of the
transform's continuous parameter (e.g. `ρ(θ)` for rotation angle `θ`)
and checking the homomorphism property directly on composed
transformations — is a natural next step, explicitly out of scope for
this specification's initial protocol.**

Practically, `ρ(T)` is estimated by ridge regression (`Z' ≈ W_T Z + b`,
fit on train-split scenes, `W_T` unconstrained — not restricted to be
orthogonal), because it is the simplest model that can express a linear
map; the literature note in §0 (Procrustes-constrained `ρ` when a
rotation/orthogonal structure is assumed) is a natural follow-up if
early results suggest that constraint is appropriate, not a starting
assumption.

We report three complementary numbers for every fit, evaluated **only
on held-out test scenes** (§11): held-out `R^2` (variance of `Z'`
explained beyond predicting its mean — scale- and direction-sensitive),
mean cosine similarity between predicted and actual `Z'` (direction
only), and mean relative L2 error (magnitude-sensitive). These are not
redundant: a map can get the direction right and the scale wrong, or
vice versa, and collapsing to one number would hide that.

## 7. Geometric invariance for appearance-only transformations

For a transformation `T_a` drawn from the **appearance** subspace of
§2 (lighting, texture/material — see §9), the geometric subspace of `S`
is held bit-for-bit identical (`T_a` acts only on `Light`/`color_i`/
`material_i`, never on `p_cam, R_cam, p_i, R_i`). The corresponding
claim we test is **invariance**, not equivariance: we expect the raw
representation to change little,

```
E(R(T_a(S))) ≈ E(R(S))    i.e.    Z' ≈ Z
```

measured directly (mean cosine similarity, mean relative L2 error
between `Z` and `Z'`, **no map fit at all**) — deliberately a different
computation from "equivariance with `ρ(T_a) = Identity`" (that specific
comparison is the *identity baseline*, §12), because invariance asks
"how much is there to explain in the first place," while equivariance
(with any baseline) asks "how well can the change be predicted." A
transform can be highly invariant (little raw change) while its residual
change is nonetheless *unpredictable* by any of our baselines — that
combination is itself informative and is exactly why both numbers are
reported for every transform, not just the geometric ones.

## 8. The first four geometric transformations

All four act on exactly one part of `S`'s geometric subspace, holding
everything else (including all appearance parameters) fixed, and are
each a named element of `SE(3)` or one of its subgroups (§0):

1. **Camera translation.** `p_cam → p_cam + Δp`, `Δp ∈ R^3` a random
   direction and magnitude; `R_cam` unchanged. This isolates the pure
   translation subgroup of `SE(3)` acting on the camera — no
   reorientation, matching the transform's name literally.

2. **Camera rotation.** The camera *orbits* a fixed scene reference
   point (the world origin): `p_cam` moves along a sphere by a random
   azimuth/elevation delta, and `R_cam` is re-derived so the camera
   keeps looking at that point. This is a **viewpoint change** — a
   combined position+orientation update, constrained by the "always
   look at the reference point" condition — rather than a rotation of
   the camera about its own optical center with position fixed. The
   latter alternative is also a valid element of `SO(3)` acting on
   `R_cam` alone, but for a roughly centered scene it barely changes the
   rendered image (the scene stays in frame, largely unchanged in scale
   and layout), making it a weak test case; the orbit formalization is
   adopted as the primary definition of "camera rotation" for exactly
   this reason, and is explicitly named as a design choice here so a
   later version can add the in-place variant as a second, harder case
   rather than silently redefining this one.

3. **Object translation.** One object `O_i` (chosen by index) has
   `p_i → p_i + Δp`; camera and all other objects, and `O_i`'s
   orientation, are unchanged.

4. **Object rotation.** One object `O_i` has `R_i → ΔR · R_i`,
   `ΔR ∈ SO(3)` a rotation about a fixed axis (world `z`) by a random
   angle; position and everything else unchanged. This transform is only
   meaningful for objects without the corresponding rotational
   symmetry — a sphere rotated about any axis through its center renders
   identically to the original, so the object category pool used for
   this transform must exclude rotationally-symmetric shapes (see
   `IMPLEMENTATION_NOTES.md` for the concrete shape list).

## 9. Invariance controls: lighting and texture/appearance

Both act **exclusively** on the appearance subspace of §2 and must leave
`p_cam, R_cam` and every `p_i, R_i` bit-for-bit unchanged — this is the
falsifiable condition that makes them valid controls, not just "small"
transforms:

- **Lighting change.** Modifies `Light` (position/direction, energy,
  color) only.
- **Texture/appearance change.** Modifies `color_i`/`material_i` for one
  or more objects only.

Per §7, the expectation is high raw invariance (`Z' ≈ Z`) and,
correspondingly, a low bar for any learned `ρ` to clear — if a learned
map scores much higher than the raw-invariance number here, that is a
sign the map is fitting something other than "predicting a real change,"
since by construction there is very little geometric change to predict.

## 10. Physical-state probes (component C: accessibility)

For each physical quantity below, a **linear ridge probe** is fit from
`Z` (of the *original*, untransformed scene, unless noted) to that
quantity, trained on train-split scenes and scored by held-out `R^2` on
test-split scenes (§11). A high score means the quantity is **linearly
accessible** from `Z` under this scene distribution — see §13 for what
this does and does not imply.

1. **3D position** — an object's `p_i ∈ R^3` (or a scalar projection of
   it, e.g. one coordinate or distance from origin, as a simpler first
   case).
2. **Depth** — a scene-level depth statistic (e.g. the rendered depth
   map's mean, or the depth at the image center) computed from `S` and
   the camera as in §2, standing in for the per-pixel depth-probing
   approach of "Probing the 3D Awareness of Visual Foundation Models"
   (§0) at the single global-vector granularity this specification's
   pooling (§4) supports.
3. **Camera pose** — `p_cam` (or a derived scalar such as azimuth,
   elevation, or distance from origin) and/or `R_cam`.
4. **Velocity** — `v_i` for an object, **only once `v_i` is defined**
   (§2): this probe is specified here for completeness but is not
   claimed to produce a meaningful result until clips with real
   multi-frame motion exist.

Every probe here is linear by the same "prefer linear probes" principle
as §6, reinforced by the Belinkov/Hewitt & Liang selectivity finding in
§0: a linear probe that succeeds is harder to explain away as
memorization than a non-linear one that succeeds, which matters more
here than probe expressiveness does, precisely because a failed linear
probe is *not* strong evidence the quantity is absent (§13).

## 11. Train/test protocol and leakage prevention

**Unit of splitting is the scene, not the render.** A "scene" is one
sampled base physical state `S` (drawn from a fixed generative
distribution — object count/shape/position ranges, camera pose range,
lighting range — via a single seed) together with **every** transformed
variant `T(S)` derived from it for every transform type in the
experiment. The train/test assignment is made **once per scene, before
any transform is applied or any rendering happens**, and every variant
of that scene — the original and all of `T_1(S), T_2(S), ...` — inherits
that single assignment. Concretely: it is never valid for `S`'s original
render to be a training example while `camera_rotation(S)`'s render is a
test example, or for two different transforms of the same base scene to
land on opposite sides of the split. This is what "strict scene-level
train/test separation" and "avoid leakage between transformed versions
of the same scene" mean operationally, and it must be checked in code,
not just intended — the current implementation enforces and unit-tests
this (see `IMPLEMENTATION_NOTES.md`).

**Why this matters here specifically.** Because a transform's parameters
(§8) are drawn per-scene from a fixed range, a model that has *seen this
exact object/camera configuration before* (even under a different
transform draw) could fit `Z → Z'` by partially recognizing the scene
rather than by capturing the transform's general effect — the synthetic
equivalent of an evaluation-set leak. Splitting at the scene level, not
the render level, is the only way to rule that out.

**A second, stronger protocol for later work.** The protocol above tests
generalization to *novel scenes drawn from the same generative
distribution and the same transform-parameter range* — an
in-distribution held-out test. A strictly stronger claim would require
an **out-of-distribution** split: e.g. training `ρ(T)` on small
transform magnitudes and evaluating on larger ones, or training on one
subset of object categories and evaluating `ρ(T)` or a state probe on
categories never seen during fitting. This project's initial protocol
does not attempt this, and any result from it should be read as "holds
within the tested distribution," not "holds in general" — this
distinction is carried through explicitly in §13.

## 12. Baseline experiments

Every equivariance/invariance result (§6, §7) and every state-probe
result (§10) is reported **alongside**, never instead of, all of the
following, evaluated identically on the same held-out split:

1. **Identity baseline** (`ρ(T) = Identity`, i.e. predict `Z' = Z`). If
   this alone scores well, the representation simply does not move much
   under `T` — an invariance finding, not evidence a *learned* map
   captured anything.
2. **Shuffled-pairing baseline.** Fit the same regression on
   deliberately mismatched `(Z_i, Z'_j)` pairs, `i ≠ j`, drawn from the
   training split. This is this project's version of a Hewitt & Liang
   control task (§0): if a map fit on *wrong* correspondences scores
   nearly as well as the correctly-paired fit, the apparent
   predictability is coming from dataset-level structure (e.g.
   regularization shrinking everything toward a shared mean), not from
   genuine scene-by-scene correspondence.
3. **Non-learned pixel-statistics encoder.** Swap the entire encoder `E`
   for a trivial, non-learned featurization (e.g. a pooled low-resolution
   RGB grid) and re-run the identical protocol. If this trivial
   "encoder" achieves comparable scores, the result is about the scene
   distribution and metric, not about anything specific to a learned
   video representation.
4. **Matched-architecture, randomly-initialized encoder.** Re-run with
   the same encoder architecture (V-JEPA 2 ViT-B/16) but random,
   untrained weights instead of the pretrained checkpoint. This isolates
   the *contribution of self-supervised pretraining specifically* from
   whatever structure the architecture provides for free (e.g. a ViT's
   patch-grid inductive bias might already yield some translation-related
   structure with no learning at all). Comparing pretrained vs. this
   baseline, under otherwise identical conditions, is the single most
   important comparison this project can make, and no equivariance or
   accessibility claim about "what pretraining buys you" is
   interpretable without it.

## 13. What this framework can and cannot support

**Can support**, for the specific encoder, pooling scheme, scene
distribution, and transform-parameter range actually tested:

- Whether a transform's effect on `Z` is *linearly predictable* from
  `Z` alone (§6), better than the identity/shuffled-pairing/pixel/
  untrained-encoder baselines (§12).
- Whether appearance-only transforms leave `Z` comparatively unchanged
  relative to geometric ones (§7).
- Whether a given physical quantity is *linearly accessible* from `Z`
  (§10), again relative to the same baselines.
- **Relative, comparative** statements: transform X is more/less
  predictable than transform Y under this protocol; the pretrained
  encoder is more/less predictable or accessible than its untrained
  counterpart or the pixel baseline, under matched conditions.

**Cannot support:**

- Any causal or mechanistic claim about what the encoder "understands,"
  "represents," or "knows" — accessibility and predictability are
  operational, statistical properties of a fitted linear model, not
  claims about internal computation.
- Generalization beyond what was actually tested: a different encoder,
  a different pooling scheme, a different scene distribution, object
  set, or transform-magnitude range, or a non-linear probe, may give a
  different answer, and this framework's results should not be
  extrapolated to any of those without re-running it there. The
  weaker-vs-stronger equivariance distinction in §6 and the
  in-distribution-only split in §11 are both instances of this general
  caution, not one-off exceptions to it.
- **Absence from a negative result.** Per Hewitt & Liang / Belinkov
  (§0), a low score from a *linear* probe or `ρ` fit is evidence the
  quantity is not *linearly* accessible/predictable under this setup —
  it is not evidence the quantity is absent from `Z` in some
  non-linearly-decodable form. Only a validated non-linear probe,
  compared against its own appropriate baselines, could speak to that,
  and this specification's initial protocol deliberately does not
  include one (per "prefer simple linear probes ... before using neural
  probes").
- Any claim about real (non-synthetic) video, temporal dynamics beyond
  a single before/after pair (components D/E/F, explicitly deferred —
  see `IMPLEMENTATION_NOTES.md`), or transformations not among the six
  specified here.

## Sources

- Assran, Bardes, Garrido, et al., ["V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"](https://arxiv.org/abs/2506.09985), arXiv:2506.09985, 2025; [Meta AI blog announcement](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/).
- Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks," EMNLP 2019 (control tasks / selectivity).
- Belinkov, ["Probing Classifiers: Promises, Shortcomings, and Advances"](https://aclanthology.org/2022.cl-1.7.pdf), Computational Linguistics 48(1), 2022.
- Equivariant representation learning / `ρ(g)` definition and steerability: see e.g. ["Equivariant Representation Learning via Class-Pose Decomposition"](https://arxiv.org/pdf/2207.03116) and the broader group-equivariant representation learning literature.
- 3DIEBench, from ["Self-supervised learning of Split Invariant Equivariant representations"](https://arxiv.org/pdf/2302.10283) — synthetic 3D-controllable benchmark for invariant/equivariant image representation probing.
- ["What, Where, and How: Probing Spatiotemporal Representations in Video Foundation Models"](https://arxiv.org/html/2609.01551) — contemporaneous linear-probing study of V-JEPA 2 and VideoMAE-v2.
- ["Probing the 3D Awareness of Visual Foundation Models"](https://arxiv.org/html/2404.08636v1) — dense multi-layer depth/surface-normal probing, precedent for §4/§10.
- Greff et al., "Kubric: A Scalable Dataset Generator," CVPR 2022 — synthetic controllable-scene precedent (see `IMPLEMENTATION_NOTES.md` for the renderer decision this motivates).
