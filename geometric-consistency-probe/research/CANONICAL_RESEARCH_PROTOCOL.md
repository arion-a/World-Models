# CANONICAL_RESEARCH_PROTOCOL.md — The authoritative Task 6–18 research contract

**Status: canonical research contract, binding on all future implementation
work.** This document is not ordinary documentation. It is the single
source of truth Claude Code and the autonomous orchestrator
(`orchestrator/`) must trace every Task 6–18 implementation decision,
QA judgment, and final-report claim back to. Where a future task prompt,
`tasks/{NN}_*.md` file, or orchestrator behavior appears to conflict with
this document, this document wins, and the conflict must be resolved
explicitly and recorded (never silently) — see Global Invariant 27/§
"Task Dependency Rules" below.

Subordinate to, and must never contradict: `DESIGN.md` (the Task 1
research specification governing Tasks 1–5, whose definitions of
equivariance, invariance, physical state, and the train/test protocol
this document inherits and extends rather than re-derives). Consistent
with, and extends: `research/RESEARCH_PLAN.md` (the task-pipeline
tracking document) and `research/RESEARCH_INVARIANTS.md` (the 18
invariants currently wired into `orchestrator/qa.py`'s automated checks).
This document's 30 Global Scientific Invariants (below) are the fuller,
canonical statement of that same discipline; `research/
RESEARCH_INVARIANTS.md` should be read as the orchestrator's current
*mechanically-enforced subset* of what is stated here in full, not as a
competing or superseding list — reconciling the two documents (updating
`orchestrator/qa.py` to check against this canonical numbering) is
future orchestrator-engineering work, out of scope for this document
itself, and is called out explicitly in this document's own creation
report rather than resolved silently.

**This document does not implement, run, or report any experiment.**
Tasks 6–18 remain unimplemented as of this document's creation. Nothing
below is a claimed result.

---

## 0. Repository grounding — what this protocol is built on

Every mathematical object, module name, and function referenced below
was verified against the actual repository, not assumed:

- **Central pipeline**: `S --renderer R--> V --frozen encoder E--> Z`,
  `S' = T(S) --R--> V' --E--> Z'`, testing `Z' ≈ ρ(T) Z` — DESIGN.md §5–6,
  unchanged here.
- **Physical state `S`**: `generation.scene.SceneState` /
  `ObjectState` / `CameraState` / `LightState` dataclasses.
  `ObjectState.instance_id` (int, unique within a scene, `0` reserved for
  background) is the persistent per-object identity Task 13 depends on.
- **Scene sampling**: `generation.scene_sampler.sample_scene` /
  `generation.generate.generate_scene`, seeded per `(base_seed, scene
  index)` — `generation/generate_dataset.py`'s `_train_test_split`
  is the existing scene-level split reference implementation Tasks 6+
  must reuse the *pattern* of (assign split once, before any transform
  or render, and propagate it to every variant of that scene).
- **Motion / temporal**: `generation.motion.ObjectMotion`,
  `CameraMotion` (`mode: "static"|"orbit"`), `generate_trajectory()` —
  pure math, no bpy dependency; `generation.bpy_renderer.render_trajectory`
  is the real multi-frame renderer with RGB+depth+segmentation ground
  truth (`ClipGroundTruth`), used by Task 12+.
- **Transforms**: `transforms.se3` (pure SE(3) matrix math: rigid
  inverse, composition, `orbit_rotation_matrix`, `rotate_about_point_matrix`).
  `transforms.scene_transform.apply_transform(scene, transform_name,
  TransformConfig)` returns the transformed `SceneState`, an explicit 4x4
  `transform_matrix` (or `null` for the two appearance-only controls),
  and `changed_variables`/`fixed_variables` — the exact machine-checkable
  isolation guarantee Task 14's counterfactual design depends on.
  `TransformConfig`'s fields (e.g.
  `camera_rotation_azimuth_deg_range: (10.0, 35.0)`) currently sample a
  **range**; a fixed-magnitude experiment (Task 6) must set that
  magnitude explicitly rather than accept the sampled default, and this
  must be done by exercising the existing config surface, not by adding
  a second, parallel way to specify the angle.
  `GEOMETRIC_TRANSFORMS = (camera_translation, camera_rotation,
  object_translation, object_rotation)`;
  `CONTROL_TRANSFORMS = (lighting_change, texture_change)`.
  `transforms.pairs.generate_pair(scene, transform_name, out_dir, ...)`
  renders a matched original/transformed pair (or, with
  `object_motions`/`camera_motion`, a matched pair of *videos*), reusing
  the Task 2 renderer on both sides.
- **Encoder**: `encoders.vjepa.VJEPAEncoder` (`checkpoint =
  "facebook/vjepa2-vitl-fpc64-256"`, loaded through
  `transformers.VJEPA2Model` directly, no `trust_remote_code`). Frozen:
  `requires_grad=False` on every parameter, `eval()` mode, no optimizer
  anywhere. `encode(video) -> (num_tokens, 1024)` float32, **not pooled**
  — `encoders.vjepa.mean_pool(representation) -> (1024,)` is the provided
  first reduction, applied by the caller, not inside `encode()`.
  `encoders.extract.extract_and_save`/`load_representation` persist
  representations with full provenance (`checkpoint`, `pretrained`,
  `preprocessing_config`, `seed`-independent determinism via a seeded
  untrained-fallback path). See `IMPLEMENTATION_NOTES.md`'s "Task 4 —
  real pretrained weights" section for why ViT-L/16 (not the originally
  planned ViT-B/16 or "V-JEPA 2.1") is the actual encoder identity from
  here on.
- **The linear map `ρ(T)`**: implemented as
  `probes.linear_rep_transform.LinearRepTransform` — ridge regression
  (`sklearn.linear_model.Ridge`), `.fit(Z, Z_prime, alpha) -> W, b`,
  `.predict(Z) -> Z @ W.T + b`. This document's `W_T` (matching the task
  brief's notation) **is** DESIGN.md's `ρ(T)`, concretely `W_T = W`,
  `b_T = b` from this class. `alpha` (ridge strength) is a
  reproducibility-recorded hyperparameter, never tuned by looking at
  test-split performance (Global Invariant 11).
- **Metrics**: `metrics.common.{cosine_similarity, relative_l2_error,
  r_squared, _as_samples_by_features}` (the last one exists specifically
  because `np.atleast_2d` silently mis-shapes 1D arrays — always route
  scalar-per-scene targets through it, never call `np.atleast_2d`
  directly — see `IMPLEMENTATION_NOTES.md`'s documented incident).
  `metrics.equivariance.evaluate_equivariance(transform_name, Z_train,
  Z_prime_train, Z_test, Z_prime_test, alpha) -> (EquivarianceResult,
  LinearRepTransform)` fits `W_T` on train and reports held-out R²,
  mean cosine similarity, mean relative L2 error on test — this is the
  canonical implementation of Task 6/7's central test. `metrics.
  invariance.evaluate_invariance(transform_name, Z, Z_prime) ->
  InvarianceResult` computes the same three quantities directly between
  `Z` and `Z'` with **no map fit** — the canonical implementation of
  Task 9's appearance-invariance test and DESIGN.md §7's distinction
  between "how much changed" (invariance) and "was the change
  predictable" (equivariance).
- **Baselines**: DESIGN.md §12 specifies exactly four, all evaluated on
  the identical held-out split as the learned map:
  (1) **Identity baseline** — `baselines.identity_baseline.
  evaluate_identity_baseline` (`Z_hat' = Z`, this document's
  "persistence" baseline); (2) **Shuffled-pairing baseline** —
  `baselines.shuffled_pairing_baseline.evaluate_shuffled_pairing_baseline`
  (fits `W_T` on deliberately mismatched train pairs — this document's
  "random-pair control"); (3) **Non-learned pixel-statistics encoder** —
  `encoders.pixel_baseline` (swap `E` entirely for a trivial
  non-learned featurization); (4) **Matched-architecture, randomly-
  initialized encoder** — `VJEPAEncoder(pretrained=False)` (isolates
  what pretraining specifically buys, vs. architecture alone). A "mean
  transformed representation" baseline (`Z_hat' = mean(Z'_train)`) is
  named explicitly in the Task 6 brief and does **not** yet exist as a
  module — Task 6 must add it (e.g. `baselines/mean_baseline.py`,
  matching `identity_baseline.py`'s structure) as a fifth baseline
  alongside DESIGN.md's canonical four, not as a replacement for any of
  them.
- **Physical-state probes**: `generation.state_features.
  {camera_azimuth_deg, camera_elevation_deg, camera_distance,
  primary_object_position_xy}` (DESIGN.md §10's camera-pose and
  3D-position probe targets, already implemented). `velocity` (DESIGN.md
  §10 item 4) is explicitly **not yet meaningful** — `v_i` is undefined
  until clips have real multi-frame motion, which only exists from Task
  12 onward via `generation.motion`. `probes.linear_state_probe.
  LinearStateProbe` and `representations.dataset.build_state_probe_split`
  are the existing linear-probe-fitting and scene-split-respecting
  assembly code Task 8 must reuse.
- **Train/test split enforcement**: `representations.dataset.
  build_transform_split`/`load_manifest` (`tests/test_scene_split.py`
  is the existing regression test for "never split a scene's
  original/transformed pair across train and test" — the pattern, not
  necessarily the exact manifest schema, that Tasks 6+ must replicate
  for whatever new data layout they introduce).
- **Configuration**: `configs.config` — dataclass-based, YAML-loaded
  (`load_config`/`apply_override`/`load_generation_config`). Tasks 6+
  should extend this pattern (a new dataclass per new config surface)
  rather than introduce a second config mechanism.
- **Orchestrator**: `orchestrator/qa.py`'s seven layers (A. Software
  Correctness, B. Task Acceptance, C. Scientific Validity, D. Research
  Alignment, E. Data Leakage, F. Reproducibility, G. Regression) are
  the mechanical enforcement of a subset of this document's invariants.
  `state/task_{NN}_result.json` is each task's required machine-readable
  result. A passing task is git-checkpointed as `task-{NN}-pass`.
  Claude's own `"implementation_status": "COMPLETE"` self-report is one
  field among many QA checks, never sufficient by itself (Global
  Invariant 30 below; `research/RESEARCH_INVARIANTS.md` invariant 18).

---

## 1. Central research question

**"Does a World Model Actually Understand 3D?"**

Operationally: when a learned video/world-model representation is
exposed to controlled visual observations generated from known 3D
physical states, does its latent representation encode geometric and
physical structure in a way that is predictable, generalizable,
temporally coherent, and robust to superficial appearance changes?

This project distinguishes between the following, in increasing order
of strength, and treats only the first six as directly, operationally
testable:

1. Information being **present** in a representation.
2. Information being **linearly accessible**.
3. Geometric transformations being **consistently represented**
   (predictable, generalizing across scenes).
4. **Temporal dynamics** being represented.
5. **Object identity/state persisting** under occlusion.
6. **Counterfactual interventions** producing structured representation
   changes.
7. The model **"understands 3D."**

Claim 7 must never be made merely from successful experiments on 1–6.
Every task below states explicitly what a positive result does and does
not establish, and no synthesis step (Task 16's report, or any
intermediate task) may collapse 1–6 into 7.

## 2. Core mathematical framework

```
S → R → V → E → Z
```

- `S` = known physical scene state (`generation.scene.SceneState`)
- `R` = renderer (`generation.bpy_renderer`, Blender/Cycles)
- `V` = rendered video
- `E` = frozen learned video representation encoder
  (`encoders.vjepa.VJEPAEncoder`)
- `Z` = learned representation (`encoders.vjepa.VJEPAEncoder.encode`,
  pooled via `mean_pool` where a single vector per clip is needed)

For a controlled physical transformation `T`:

```
S' = T(S)          (transforms.scene_transform.apply_transform)
V  = R(S),  V' = R(S')
Z  = E(V),  Z' = E(V')
```

The central geometric consistency formulation:

```
Z' ≈ W_T Z + b_T            (W_T, b_T = probes.linear_rep_transform.LinearRepTransform.fit(Z_train, Z'_train))
```

`W_T` (equivalently DESIGN.md's `ρ(T)`) must be learned **only** from
training scenes and evaluated **only** on completely unseen test
scenes. The initial experiment (Task 6) is explicitly **not** "can we
predict the transformation parameter from `Z`" (that would be a
physical-state-accessibility question, Task 8's domain) — it is "does
the known physical transformation induce a predictable,
scene-generalizable transformation *of the representation itself*."

Per DESIGN.md §6, this is a real but **weaker** claim than strict group
equivariance: `T`'s parameters vary continuously per scene, a single
linear `W_T` is fit as an aggregate, parameter-marginalized
approximation across the whole sampled parameter range, and the
homomorphism property (`ρ(T2)ρ(T1) ≈ ρ(T2∘T1)`) is not tested. Every
task inheriting this formulation inherits this caveat too.

## 3. Global Scientific Invariants

These 30 invariants are binding on every task from 6 through 18. They
are reproduced here in full — not summarized — because every task
section below refers back to them by number rather than restating them.

1. The primary object of study is the learned representation Z.
2. The physical scene state S is known because the world is controlled.
3. Physical transformations T must be explicitly defined and known.
4. Rendering must preserve the intended physical intervention.
5. The primary representation encoder is frozen.
6. No ground-truth 3D coordinates may be secretly provided to the encoder.
7. Train/test separation must be performed at the SCENE level.
8. Every transformed or modified version of a scene must remain in the same split as its source scene.
9. No test scene may contribute to fitting W_T.
10. No test scene may contribute to fitting probes.
11. No test scene may be used to tune experimental hyperparameters in a way that contaminates the evaluation.
12. Linear probes are the initial test for physical-state accessibility.
13. Nonlinear probes must not replace linear probes in Task 8.
14. Every important claim requires an actual experiment.
15. Every experiment requires a plausible negative outcome.
16. Negative scientific results are valid results.
17. Software failure and scientific negative results must be distinguished.
18. Controls must be implemented rather than merely discussed.
19. Baselines must be evaluated under the same protocol.
20. Metrics must be defined before interpreting results.
21. Seeds must be recorded.
22. Configuration must be recorded.
23. Experimental provenance must be recorded.
24. Results must be machine-readable.
25. No fabricated experimental results.
26. No cherry-picking.
27. No changing the methodology simply to obtain positive results.
28. No complexity merely for visual impressiveness.
29. Every task must state interpretation limits.
30. Task 18 must be allowed to challenge conclusions produced by Tasks 6–17.

**Repository-level enforcement mapping** (how these are, or will be,
mechanically checked, not just stated):

- Invariants 5, 6 → frozen-encoder checks already exercised by
  `tests/test_vjepa_encoder.py` (`test_all_parameters_have_requires_grad_false`,
  `test_no_optimizer_exists_anywhere_on_the_encoder`,
  `test_encode_does_not_change_any_parameter_value`); `orchestrator/qa.py`
  layer D/research-alignment checks for a frozen-encoder declaration in
  each task's result JSON.
- Invariants 7, 8, 9, 10 → `orchestrator/qa.py` layer E (DATA LEAKAGE):
  independently re-checks that every `train_scene_ids`/`test_scene_ids`
  pair declared anywhere in a task's result JSON is disjoint. The
  underlying pattern is `representations.dataset.build_transform_split`
  / `tests/test_scene_split.py`.
- Invariants 21, 22, 23, 24 → `orchestrator/qa.py` layer F
  (REPRODUCIBILITY): a task's result JSON must declare `seed` and
  `config`, non-null.
- Invariant 25 (no fabricated results) → `orchestrator/qa.py` layer A's
  stub-marker scan and all-metrics-are-zero heuristic are a first line
  of defense; Task 18's adversarial audit is the deep check.
- Invariant 17 (software failure vs. negative result) → `orchestrator/
  qa.py`'s layers never inspect a metric's *value* (only its presence,
  finiteness, and baseline comparison) — see §5 below for the full
  semantics.
- Invariant 30 → `orchestrator/qa.py`'s entire design (never trusts
  Claude's own `"implementation_status"` field alone); `research/
  RESEARCH_INVARIANTS.md` invariant 18 states the same rule for the
  orchestrator specifically.

## 4. Task dependency chain

```
Tasks 1–5 (complete)
     ↓
Task 6  — first camera-rotation geometric consistency experiment
     ↓
Task 7  — does the phenomenon generalize across geometric transformations
     ↓
Task 8  — are physical variables linearly accessible (related but distinct question)
     ↓
Task 9  — sensitivity to superficial appearance factors; controls against appearance-based explanations
     ↓
Task 10 — fair baselines, established as reusable infrastructure
     ↓
Task 11 — do effects survive scale
     ↓
Task 12 — extend into temporal dynamics
     ↓
Task 13 — persistence through occlusion
     ↓
Task 14 — controlled counterfactual interventions
     ↓
Task 15 — unify the collection into one evaluation framework
     ↓
Task 16 — derive conclusions from actual evidence
     ↓
Task 17 — make the complete work reproducible and open
     ↓
Task 18 — attempt to falsify the resulting conclusions
```

Each later task must **inherit and preserve** the relevant methodology
of earlier tasks (scene-level splitting, the four DESIGN.md baselines,
frozen-encoder discipline, linear-probes-first, seeded reproducibility)
unless an explicit scientific reason for changing it is documented in
that task's own result JSON (Global Invariant 27; `research/
RESEARCH_INVARIANTS.md` invariant 15's "protocol changes must be
explicit and recorded"). **Do not reorder these tasks without a
documented scientific reason.**

## 5. Task completion semantics

Every task's outcome must be classified into exactly one of:

- **SOFTWARE FAILURE** — the implementation is broken: it crashes, a
  required file is missing, tensors have wrong shape/dtype, NaN/Inf
  appears, a scene-level split leaks, the encoder was accidentally
  fine-tuned, ground-truth coordinates reached the encoder, or a
  required baseline/control was silently skipped. **This must be
  fixed** — the orchestrator's fix loop (`orchestrator/run.py`,
  `MAX_FIX_ATTEMPTS`) exists specifically for this category.
- **VALID NEGATIVE SCIENTIFIC RESULT** — the implementation is correct
  and the measured effect is weak, absent, or opposite to the
  hypothesis (e.g. `W_T` performs no better than the persistence
  baseline). **This is an acceptable, complete task outcome** and must
  never trigger a retry on its own (Global Invariants 16, 17).
- **INCONCLUSIVE SCIENTIFIC RESULT** — the implementation is correct
  but the experiment's design cannot distinguish a real effect from
  noise/confound at the scale or configuration actually run (e.g. too
  few scenes, a metric with no discriminating power at the observed
  scale). This must be stated as inconclusive, not silently rounded up
  to positive or down to negative, and not treated as grounds for an
  automatic retry unless the inconclusiveness itself stems from a
  methodological defect (in which case it is reclassified as a software
  failure).
- **VALID POSITIVE SCIENTIFIC RESULT** — the implementation is correct
  and the measured effect exceeds all applicable baselines by a
  documented margin, under the exact protocol specified.

**Worked example, given verbatim in the task brief and preserved here:**
if Task 6 runs correctly and `W_T` performs poorly against the
baselines, this is *not* automatically a Task 6 failure — it may be an
important negative scientific result, and the task should be recorded
as PASSED with that result honestly stated. However, if Task 6
accidentally places transformed versions of the same scene in both
train and test, that *is* a methodological (software) failure, and the
orchestrator must retry/fix it. **The orchestrator must never retry
merely because the scientific result is negative.**

## 6. Orchestrator compatibility

Each task section below is written so the orchestrator can extract it
as an independent execution specification: every task states its own
Objective (Purpose + Scientific question), Requirements
(Inputs/Outputs/Experimental protocol/Dataset requirements/Train-test
protocol/Controls/Baselines/Metrics/Required artifacts), Tests
(Required software tests/scientific-validity tests/research-alignment
checks/leakage checks/reproducibility requirements), Failure conditions,
and Acceptance criteria. The orchestrator's actual loop
(`orchestrator/run.py`) is: read the current task → provide its
specification to Claude Code → let Claude implement it → independently
run QA (`orchestrator/qa.py`) → repair if necessary (up to
`MAX_FIX_ATTEMPTS`) → checkpoint successful work (`task-{NN}-pass` git
commit) → advance to the next task. **Claude Code's own declaration of
PASS must never be treated as sufficient** — this is Global Invariant 30
and `research/RESEARCH_INVARIANTS.md` invariant 18, restated here as
binding on every task below without exception.

---

# TASK 6 — First camera-rotation geometric consistency experiment

### Purpose

Establish the first controlled test of whether a known physical
geometric transformation produces a predictable transformation in
latent representation space.

### Scientific question

If the camera undergoes a known rotation, does the frozen V-JEPA
representation change according to a predictable transformation that
generalizes across unseen scenes?

### Why this task exists

This is the first experiment directly testing the central geometric
consistency hypothesis (§2) with the real Task 4/5 encoder
(`encoders.vjepa.VJEPAEncoder`, real pretrained ViT-L/16 weights) and
the real Task 2/3 scene/transform pipeline, rather than the exploratory
V0 pipeline that predates the formal task numbering. Everything later
(Tasks 7–18) either generalizes this protocol (Task 7), asks a related
but distinct question about it (Task 8), controls for confounds in it
(Task 9), formalizes its baselines (Task 10), stress-tests it at scale
(Task 11), or extends it into new regimes (temporal — Task 12,
occlusion — Task 13, counterfactual — Task 14). Getting Task 6's
protocol right is a precondition for all of them.

### Relationship to previous tasks

Inherits directly, without modification: `generation.scene_sampler
.sample_scene`/`generation.generate.generate_scene` (Task 2 scene
sampling), `transforms.scene_transform.apply_transform` and
`transforms.pairs.generate_pair` (Task 3's transform + matched-pair
rendering), `encoders.vjepa.VJEPAEncoder`/`encoders.extract` (Task 4/5's
frozen encoder). Does **not** modify these modules except for a
minimal, justified bug fix (recorded per Global Invariant 27 /
`protected_files_justification` in the result JSON).

### Hypothesis

A fixed-magnitude camera rotation applied to a controlled synthetic
scene induces a change in the frozen encoder's pooled representation
that is well-approximated by a single linear map `W_T`, fit on a set of
training scenes, and that this map generalizes to unseen test scenes
better than the persistence baseline, the mean-transformed-
representation baseline, and the shuffled-pairing (random-pair) control.

### Mathematical formulation

```
S' = camera_rotation(S)        (transforms.scene_transform.apply_transform)
V  = R(S),   V' = R(S')        (transforms.pairs.generate_pair, reusing generation's renderer)
Z  = E(V),   Z' = E(V')        (encoders.vjepa.VJEPAEncoder.encode, then mean_pool)

Z' ≈ W_T Z + b_T                (probes.linear_rep_transform.LinearRepTransform.fit, TRAIN scenes only)
```

Evaluated via `metrics.equivariance.evaluate_equivariance` on
`(Z_test, Z'_test)`, per DESIGN.md §6's weaker, aggregate,
parameter-marginalized notion of equivariance (not strict group
equivariance; the homomorphism property is not tested).

### Inputs

- A set of at least 40 sampled `SceneState`s (`generation.scene_sampler
  .sample_scene`, one seed per scene), split into train/test at the
  **scene level**, before any rendering.
- A fixed camera-rotation magnitude (e.g. 30 degrees azimuth), set
  explicitly via `TransformConfig`'s existing
  `camera_rotation_azimuth_deg_range` field (e.g. `(30.0, 30.0)`) rather
  than left at its sampled `(10.0, 35.0)` default or specified through a
  new, parallel mechanism — the exact convention (azimuth vs. elevation,
  sign, units) must be inspected from `transforms/scene_transform.py`
  and `transforms/se3.py`, never invented.
- `encoders.vjepa.VJEPAEncoder(pretrained=True)` — real pretrained
  weights; if unreachable at run time, the encoder's own documented
  fallback applies and `pretrained: false` **must** be recorded
  plainly, never hidden (Global Invariant 25).

### Outputs

- Rendered original/transformed clip pairs for every scene (reusing
  `transforms.pairs.generate_pair`'s output layout — no new renderer
  output convention).
- `state/task_06_result.json` (schema below).
- A runnable script/module under `experiments/` (e.g.
  `experiments/task6_camera_rotation.py`) with a `main()` callable via
  `python -m ...` and a `--config` flag.

### Experimental protocol

1. Sample ≥40 scenes with distinct seeds.
2. Assign train/test split at the scene level, before rendering
   (Global Invariants 7, 8).
3. For every scene, render the original and the fixed-magnitude
   `camera_rotation`-transformed clip via `transforms.pairs.
   generate_pair`.
4. Encode both clips with `VJEPAEncoder`, reduce to a single vector per
   clip with `mean_pool` (pooling choice stated explicitly, not an
   unstated default — ridge regression over the raw unpooled
   `(num_tokens, hidden_size)` tensor is dimensionally unreasonable at
   this scene count).
5. Fit `W_T` on `(Z_train, Z'_train)` only.
6. Evaluate `W_T`, and every baseline (below), on `(Z_test, Z'_test)`.

### Dataset requirements

≥40 scenes, each with a distinct seed, each rendered on both the
original and `camera_rotation`-transformed side. An 80/20 (or similar,
explicitly recorded) train/test fraction.

### Train/test protocol

Scene-level, assigned once before any rendering. Every variant of
`scene_001` (original, rotated) belongs to the same split as every
other variant of `scene_001`. It is forbidden to place different
variants of the same underlying scene into train and test (Global
Invariants 7, 8; DESIGN.md §11).

### Controls

1. **Persistence** (`Z_hat' = Z`) — `baselines.identity_baseline.
   evaluate_identity_baseline`.
2. **Mean transformed representation** (`Z_hat' = mean(Z'_train)`) —
   does not yet exist; add `baselines/mean_baseline.py`, matching
   `identity_baseline.py`'s structure (a dataclass result + one
   function), using `metrics.common`'s existing primitives.
3. **Random-pair control**, which must destroy the true scene
   correspondence — `baselines.shuffled_pairing_baseline.
   evaluate_shuffled_pairing_baseline`.

### Baselines

The three controls above are this task's required baseline set. Task
10 will later formalize DESIGN.md §12's full four-baseline set
(identity, shuffled-pairing, pixel-statistics encoder,
randomly-initialized encoder) as reusable infrastructure; Task 6 is not
required to run the pixel-statistics or randomly-initialized-encoder
baselines itself, but must not report results in a way that would
conflict with their later addition (e.g. must not claim "beats every
reasonable baseline" — only "beats persistence, mean, and random-pair").

### Metrics

Held-out R² (`metrics.common.r_squared`), mean cosine similarity, mean
relative L2 error — via `metrics.equivariance.evaluate_equivariance`,
computed identically for the learned `W_T` and all three controls.
Metric definitions are fixed by this document and by
`metrics.equivariance`/`metrics.common` before any result is inspected;
the metric must not be chosen or changed after inspecting results to
produce a favorable conclusion (Global Invariant 20, 27).

### Required artifacts

- `experiments/camera_rotation/` (or the task's own output directory,
  consistent with `transforms.pairs.generate_pair`'s layout) containing
  the rendered scene data.
- `state/task_06_result.json`, containing at minimum: `task`,
  `implementation_status`, `scientific_result`, `transform` (=
  `"camera_rotation"`), `transform_params` (the exact fixed magnitude
  and any other `TransformConfig` fields used), `encoder` (`name`,
  `checkpoint`, `pretrained`, `frozen`, `pooling`), `dataset`
  (`num_scenes`, `train_scene_ids`, `test_scene_ids`, `train_fraction`,
  `base_seed`), `fitting` (`method`, `alpha`), `metrics` (`learned_W_T`,
  `persistence_baseline`, `mean_baseline`, `random_pair_control`, each
  with `r2`/`mean_cosine_similarity`/`mean_relative_l2_error`), `tests`
  (`passed`, `failed`), `artifacts`, `config`, `seed`,
  `software_versions`.

### Required software tests

- Transformation runs; rendering runs; encoding runs; `W_T` fitting
  runs; prediction runs; metrics run; result files are generated.
- No NaN/Inf anywhere in the result.
- Expected tensor dimensions at every stage (pooled `Z`/`Z'` are
  `(1024,)`; stacked train/test arrays are `(N, 1024)`).
- Unit test(s) for the new `baselines/mean_baseline.py`.
- Full existing suite (`pytest -m "not slow"`) still passes — no
  regression in Tasks 1–5.

### Required scientific-validity tests

- The transformation is actually applied (the rendered transformed
  video's ground truth — `changed_variables`/`fixed_variables`/
  `transform_matrix` from `apply_transform` — reflects exactly the
  intended camera rotation and nothing else).
- The transformed video genuinely corresponds to the transformed scene,
  and `Z`/`Z'` genuinely correspond to `V`/`V'` (no accidental
  mismatch/misindexing between scene, render, and representation).
- `W_T` was fit on train scenes only; test scenes were never seen during
  fitting.
- Transformed variants never cross the train/test split.
- The encoder is frozen throughout (parameters unchanged before/after
  encoding, per `tests/test_vjepa_encoder.py`'s existing pattern).
- No ground-truth coordinates (rotation matrices, camera pose) enter
  the encoder as input — only as labels/config for constructing `T` and
  as provenance metadata.

### Required research-alignment checks

Consistent with `research/RESEARCH_INVARIANTS.md` and this document's
§3: frozen encoder declared and verified, scene-level split declared,
baselines present, seed/config recorded, no fabricated/stub metrics
(`orchestrator/qa.py` layers A/C/D/F).

### Required leakage checks

- `train_scene_ids` and `test_scene_ids` are disjoint sets.
- Every `test_scene_ids` entry's representation was excluded from
  `LinearRepTransform.fit`'s training data.
- No test-split scene contributed to the mean-baseline's computed mean.

### Reproducibility requirements

Fixed seed produces a bit-reproducible (or documented-tolerance)
result across independent runs; `base_seed`, the fixed rotation
magnitude, ridge `alpha`, encoder checkpoint/`pretrained` flag, and
software versions (`transformers`, `torch`, `python`) are all recorded
in `state/task_06_result.json`.

### Failure conditions

Any of: missing/invalid result JSON; NaN/Inf; train/test scene overlap;
a baseline silently skipped; encoder parameters changed by the run;
ground-truth coordinates fed to the encoder; scene count below 40;
`alpha` or any hyperparameter chosen by inspecting test metrics;
unjustified modification of Task 1–5 modules. Any of these is a
SOFTWARE FAILURE (§5) — the orchestrator must retry/fix, not accept.

### Acceptance criteria

1. ≥40 scenes generated and rendered through both original and
   `camera_rotation`-transformed sides.
2. Real `VJEPAEncoder` (pretrained per the Inputs section) used for
   both sides of every scene.
3. `W_T` fit on train only, evaluated on test only.
4. All three required controls computed on the identical test split.
5. `state/task_06_result.json` present and matches the schema above.
6. No NaN/Inf in any reported metric.
7. Full existing test suite still passes.
8. Scene-level split integrity holds (leakage checks above).

### Interpretation rules

Allowed: "The representation exhibits measurable predictive consistency
under the tested camera rotation, exceeding [baseline] by [margin]."
Allowed: "No measurable predictive consistency was found under the
tested protocol." **Not allowed, under any result:** "V-JEPA understands
3D," or any claim beyond DESIGN.md §13's "can support" list.

### Explicit prohibited shortcuts

- Feeding ground-truth camera pose/rotation matrices into the encoder
  as input.
- Fitting `W_T` or choosing its ridge `alpha` by looking at test
  metrics.
- Reducing scene count to make ridge regression trivially "work."
- Silently swapping `pretrained=False` to dodge slow/networked runs
  without recording it.
- Modifying `transforms/`, `generation/`, or `encoders/` to make this
  experiment's numbers look better (a genuine bug fix is allowed, must
  be minimal and justified in `protected_files_justification`).

### What a positive result means

Camera-rotation's effect on this frozen encoder's pooled representation
is linearly predictable and generalizes to unseen scenes, under this
scene distribution, this pooling scheme, and this rotation magnitude —
nothing more.

### What a negative result means

Under this protocol, no linear map recovers camera-rotation's effect on
`Z` better than trivial controls — a real, complete, reportable finding
(Global Invariants 16, 17), not a task failure.

### What this task does NOT establish

Generalization to other rotation magnitudes or axes (Task 7's
question); that physical camera state is *itself* decodable from `Z`
(Task 8's question, related but distinct — DESIGN.md §13); robustness
to appearance confounds (Task 9); behavior at scale (Task 11); anything
about temporal dynamics, occlusion, or counterfactual structure (Tasks
12–14); and, under any outcome, any claim that the model "understands"
3D, has an internal camera model, or performs geometric reasoning.

---

# TASK 7 — Multiple geometric transformations

### Purpose

Determine whether the phenomenon tested in Task 6 generalizes beyond a
single camera rotation.

### Scientific question

Does the representation exhibit predictable, scene-generalizable
responses to multiple distinct physical geometric transformations?

### Why this task exists

Task 6 establishes the protocol on one transform. A single positive or
negative result on `camera_rotation` alone cannot support any claim
about "geometric transformations" as a class — it could be an artifact
specific to rotation's particular visual signature. Task 7 exists to
turn Task 6's anecdote into a comparison across the full set of
transforms this project defines as geometric, using the appearance-only
controls as an explicit contrast class.

### Relationship to previous tasks

Directly reuses Task 6's scene generation, splitting, encoding, fitting,
and metric code as a library — Task 6's script must be refactored into
a per-transform function if it was not already, rather than
copy-pasted per transform. Inherits Task 6's baseline set unchanged.

### Hypothesis

The four members of `transforms.scene_transform.GEOMETRIC_TRANSFORMS`
(`camera_rotation`, `camera_translation`, `object_rotation`,
`object_translation`) each induce a linearly predictable,
scene-generalizing change in `Z`, to varying degrees; the two members of
`CONTROL_TRANSFORMS` (`lighting_change`, `texture_change`) — which are
not rigid transforms (`transform_matrix` is `null` for them) — show
markedly weaker equivariance under the identical protocol.

### Mathematical formulation

For each `T` in `GEOMETRIC_TRANSFORMS ∪ CONTROL_TRANSFORMS`:
`S' = T(S)`, `V=R(S)`, `V'=R(S')`, `Z=E(V)`, `Z'=E(V')`,
`Z' ≈ W_T Z + b_T` fit on train, evaluated on test — identical
formulation to Task 6, run once per transform, on the **same scene set
and same train/test split across all six transforms** (so results are
comparable transform-to-transform, not confounded by different scene
samples).

### Inputs

The same scene set used in Task 6 (or a fresh, equally-sized set,
explicitly recorded either way), rendered through all four
`GEOMETRIC_TRANSFORMS` plus both `CONTROL_TRANSFORMS`, using transform
magnitudes drawn from `TransformConfig`'s existing ranges unless a
specific fixed magnitude is scientifically motivated and recorded.

### Outputs

`state/task_07_result.json` with a `results: {transform_name: {...}}`
map (one entry per transform, each shaped like Task 6's top-level
metrics block), plus a comparison table/report.

### Experimental protocol

1. Reuse Task 6's scene generation and the identical train/test split
   for all six transforms in one run.
2. Render every transform's original/transformed pair for every scene
   (reusing `transforms.pairs.generate_pair`).
3. Encode, pool, fit, and evaluate exactly as in Task 6, independently
   per transform — each transformation must be independently evaluated;
   transformations must not be pooled into one result in a way that
   hides transformation-specific failures.
4. Run Task 6's three baselines for every one of the six transforms.

### Dataset requirements

Same scene count/seed discipline as Task 6 (≥40 scenes), all six
transforms rendered for every scene.

### Train/test protocol

Identical scene-level split as Task 6, reused verbatim across all six
transforms in the same run (a scene's split assignment does not vary by
transform).

### Controls

The same three as Task 6 (persistence, mean, random-pair), applied
per-transform.

### Baselines

Same as Task 6's; DESIGN.md's full four-baseline set remains Task 10's
responsibility to formalize, not Task 7's to add.

### Metrics

R², mean cosine similarity, mean relative L2 error, per transform, for
the learned `W_T` and all three baselines — a 6×4-metric comparison
table at minimum, using the same metric implementations as Task 6
(no transform-specific metric substitution).

### Required artifacts

`state/task_07_result.json` (per-transform breakdown as above); a
comparison table/report (markdown or the result JSON itself).

### Required software tests

- Every transformation executes; the renderer correctly reflects the
  transformation (checked against `apply_transform`'s own
  `changed_variables`/`fixed_variables`/`transform_matrix` ground
  truth); representations generated correctly; `W_T` fits; evaluation
  executes; outputs generated for all six transforms.
- No NaN/Inf anywhere.
- Regression: Task 6 remains valid (its own tests, and ideally its own
  recorded numbers, still reproduce under the shared library code).

### Required scientific-validity tests

- Expected physical properties actually change under each transform,
  and unintended properties remain fixed where intended (cross-checked
  against `changed_variables`/`fixed_variables`).
- Scene-level split holds across all six transforms simultaneously.
- Transformed variants stay with their source scene.
- No test-set fitting; frozen encoder; no ground-truth leakage into the
  encoder.
- Investigate, explicitly, whether differences between transformations
  could be caused by different visual artifacts (e.g. one transform
  producing a larger average pixel-level change than another) rather
  than by geometry itself — this is a required scientific-QA step, not
  optional discussion.

### Required research-alignment checks

Same as Task 6, applied per transform; additionally, confirm the
identical scene/split is used for all six (a dedicated check comparing
`train_scene_ids`/`test_scene_ids` across every transform's entry).

### Required leakage checks

Same as Task 6, checked independently per transform, plus: the shared
split's `train_scene_ids`/`test_scene_ids` are identical across every
transform's entry in the result JSON.

### Reproducibility requirements

Same as Task 6; additionally record which transform magnitudes
(fixed or ranged) were used per transform.

### Failure conditions

Same categories as Task 6, plus: using six different train/test splits
(confounding transform effect with sample effect); pooling
transform-level results in a way that obscures a transform-specific
leakage or failure; silently overwriting Task 6's own recorded result
when reusing its scene set (a numeric difference from re-sampling must
be recorded as an explicit, intentional protocol note per Global
Invariant 27, not silently accepted).

### Acceptance criteria

1. All four geometric transforms and both appearance controls run
   through the identical protocol on the identical scene split.
2. All three baselines present for every transform.
3. No NaN/Inf anywhere in the six-transform results.
4. Task 6's own recorded result is not silently contradicted without an
   explicit, recorded protocol note.

### Interpretation rules

A ranking across transforms is evidence about *this frozen encoder,
this scene distribution, this transform-magnitude range* — not a
general claim about geometric transformations in video representations.
If appearance controls score similarly to geometric transforms, that is
a meaningful negative finding (weak evidence `Z` is not distinguishing
geometry from any change), to be reported plainly, not fixed until it
goes away.

### Explicit prohibited shortcuts

Cherry-picking which transforms to report based on which look best; using
different train/test splits per transform; adjusting a transform's
magnitude after seeing its score to improve it.

### What a positive result means

For the transforms that show it: this frozen encoder's representation
responds to that transform type in a way that is linearly predictable
and generalizes across scenes, under the tested magnitude range.

### What a negative result means

For the transforms that do not show it: no evidence, under this
protocol, that this transform type's effect is linearly predictable in
`Z` — a valid, complete, reportable finding, potentially different
per transform.

### What this task does NOT establish

Any claim about transforms not tested; universal geometric
understanding from success on one or a few transforms; whether the
underlying physical parameters (not just the representation's response)
are themselves accessible (Task 8); robustness to appearance confounds
beyond the two controls tested here (Task 9 goes deeper); behavior at
scale (Task 11).

---

# TASK 8 — Physical-state accessibility

### Purpose

Test whether physically meaningful variables are linearly accessible
from `Z`.

### Scientific question

Even if representation transformations are not perfectly equivariant
(Tasks 6–7), does the latent representation contain linearly
recoverable information about physical scene state?

### Why this task exists

Equivariance (Tasks 6–7) asks whether a transform's *effect* on `Z` is
predictable. Accessibility asks a different, complementary question:
whether the *absolute* physical state of the (untransformed) scene can
be read out of `Z` at all. DESIGN.md §13 is explicit that these are
related but distinct claims, and a project answering "does a world
model understand 3D" cannot skip either — a representation could be
highly equivariant under a transform while the underlying physical
quantity itself remains linearly inaccessible, or vice versa, and that
combination is itself scientifically informative.

### Relationship to previous tasks

Reuses the scene/encoding pipeline from Tasks 6–7 and, ideally, the
same scenes, so results are comparable across tasks. Reuses
`representations.dataset.build_state_probe_split`'s pattern for
assembling `(Z_train, y_train, Z_test, y_test)`, and
`probes.linear_state_probe.LinearStateProbe` if it already covers the
needed target — inspected and extended, not reimplemented, before
adding anything new. Reuses `generation.state_features
.{camera_azimuth_deg, camera_elevation_deg, camera_distance,
primary_object_position_xy}` as probe targets (DESIGN.md §10);
`velocity` remains explicitly out of scope until Task 12 defines `v_i`
via real multi-frame motion.

### Hypothesis

At least camera azimuth (continuity with the V0 pipeline's existing
`camera_azimuth_deg` probe) and one additional physical-state variable
are linearly decodable from `Z` (of the original, untransformed scene)
above chance/mean-baseline level, under this scene distribution.

### Mathematical formulation

For target `y` (a physical-state scalar or vector):

```
y ≈ W_probe Z + b_probe     (linear ridge probe, fit on TRAIN scenes only)
```

Evaluated by held-out R² (regression) on test-split scenes.

### Inputs

`Z` for the original (untransformed) render of each scene in the
train/test split; ground-truth physical-state labels computed directly
from each scene's `SceneState` via `generation.state_features` feature
functions (`feature_fn(SceneState) -> float | np.ndarray`).

### Outputs

`state/task_08_result.json`: which state variables were probed,
per-variable R² for the real probe / shuffled-label control / mean
baseline, train/test scene IDs, seed, config.

### Experimental protocol

1. Define/select at least two `feature_fn`s, one of which is
   `camera_azimuth_deg` for continuity with V0.
2. For a rotation-based variable (azimuth), explicitly decide and
   record whether raw degrees or `(sin, cos)` components are probed —
   an angle must not be probed linearly across its wrap-around
   discontinuity without addressing it.
3. Assemble `(Z_train, y_train, Z_test, y_test)` via the scene-level
   split (reusing `representations.dataset.build_state_probe_split`'s
   pattern).
4. Fit a **linear** ridge probe per variable on train only; evaluate
   held-out R² on test.
5. Run the shuffled-label control (fit the identical probe on
   scene-shuffled `(Z_train, y_train_shuffled)`) and the mean-prediction
   baseline (`y_hat = mean(y_train)`).

### Dataset requirements

Reuses Tasks 6–7's scene set where practical; if a fresh set is
sampled, it must follow the same scene-level split discipline and scene
count floor.

### Train/test protocol

Scene-level split (Global Invariants 7–10), ideally identical to Tasks
6–7's for cross-task comparability.

### Controls

- **Shuffled-label control**: fit the same probe on scene-shuffled
  labels — if this scores nearly as well as the real fit, the
  probe/metric setup itself is suspect. The shuffled-label control's
  permutation must not be re-derivable to the identity by construction
  (guard against a trivial/identity permutation for small `N`, mirroring
  the existing guard in `baselines/shuffled_pairing_baseline.py`).
- **Mean-prediction baseline** (`y_hat = mean(y_train)`).

### Baselines

The shuffled-label control and mean-prediction baseline above are this
task's required baseline set (a different baseline family than Tasks
6–7/10's representation-transform baselines, since the target here is a
scalar physical quantity, not a representation vector).

### Metrics

R² (primary); MAE and RMSE reported for interpretability; angular error
for any orientation-valued quantity probed via `(sin, cos)` decoding.

### Required artifacts

`state/task_08_result.json` per the Outputs section above.

### Required software tests

- Correct labels (feature functions verified against known `SceneState`
  fixtures); correct dimensions; correct metric implementation; no
  NaN/Inf; reproducibility.
- Unit test that the shuffled-label control uses a genuine permutation
  (not an identity permutation for small N).
- Regression (existing suite).

### Required scientific-validity tests

- **Linear probes only** — nonlinear probes must not replace linear
  probes in this task (Global Invariant 13); reaching for a nonlinear
  probe to explain away a null linear result is prohibited without
  first exhausting the linear-probe design space (regularization
  strength, pooling choice) and recording why linear was insufficient.
- Correct alignment between representation and label (the label
  computed from exactly the same `SceneState` whose render produced the
  probed `Z`).
- Verify that high probe performance cannot be trivially explained by
  scene identity, camera identity, a near-constant object position
  across the sampled scenes, leakage, or duplicate scenes — an explicit
  scientific-QA step, not assumed away.
- Pose/angle convention documented (which axis, which sign, degrees vs.
  radians, raw vs. `(sin, cos)`).
- No ground-truth state fed into the encoder (invariant 6) — state is
  the probe *target*, never encoder input.

### Required research-alignment checks

Consistent with Global Invariants 12–13 (linear probes first, no
nonlinear substitution) and §3's general mapping.

### Required leakage checks

Scene-level train/test disjointness (Global Invariants 7–10); the
shuffled-label control's permutation must not accidentally leak the
real correspondence back in.

### Reproducibility requirements

Seed, ridge `alpha`, feature-function choice, and angle-encoding
convention all recorded in `state/task_08_result.json`.

### Failure conditions

Any of: a nonlinear probe substituted for the required linear one; an
angle probed naively across its discontinuity with no stated
convention; train/test overlap; a missing shuffled-label or mean
baseline; feature-function/label misalignment with the probed `Z`.

### Acceptance criteria

1. At least two physical-state variables probed, one of them
   `camera_azimuth_deg`.
2. Real probe, shuffled-label control, and mean baseline all reported
   per variable.
3. Train/test split disjoint and consistent with Tasks 6–7's scenes
   where practical.
4. No NaN/Inf.

### Interpretation rules

Allowed: "Variable X is linearly accessible from Z, under this scene
distribution." **Not allowed**: "The model explicitly represents X," or
"the model understands X" — accessibility is an operational, statistical
property of a fitted linear model (DESIGN.md §13), not a claim about
internal computation.

### Explicit prohibited shortcuts

Feeding ground-truth `SceneState` fields into the encoder; selecting
which state variables to report based on which probe performs best
after the fact; replacing a null linear result with a nonlinear probe
instead of reporting the null result.

### What a positive result means

The named variable is linearly recoverable from `Z` for this scene
distribution and pooling scheme — an accessibility finding, not a
representational or causal one.

### What a negative result means

Per DESIGN.md §13 (echoing Hewitt & Liang / Belinkov, §0): a low score
from a linear probe is evidence the quantity is not *linearly*
accessible under this setup — it is **not** evidence the quantity is
absent from `Z` in some non-linearly-decodable form. A null result here
alongside a positive equivariance result in Task 6/7 (or vice versa) is
itself a valid, informative combination and must be reported as such,
not forced to agree.

### What this task does NOT establish

That the encoder "represents" or "knows" the variable in any stronger
sense than linear accessibility; anything about non-linear
accessibility (explicitly out of scope, per Global Invariant 13); any
causal role of the variable in the encoder's computation.

---

# TASK 9 — Appearance invariance controls

### Purpose

Test whether representation variation is specifically related to
physical state or whether superficial appearance changes can produce
comparable effects.

### Scientific question

When physical state is held constant, how sensitive is `Z` to changes
in lighting, texture, material, or color?

### Why this task exists

Tasks 6–7 already include `lighting_change`/`texture_change` as
within-task appearance controls, but only as a side comparison to the
geometric transforms. Task 9 exists to make that comparison the primary
object of a dedicated experiment, deepen it with a fixed sanity-check
(the null-transform case), and rule out that Tasks 6–7's apparent
geometric sensitivity is actually explainable by lighting/texture/
material/rendering-pipeline artifacts rather than genuine geometric
change.

### Relationship to previous tasks

Directly reuses Task 7's `(Z, Z')` pairs for the appearance controls
and, for comparison, the geometric transforms — no need to re-render if
Task 7's outputs are available and unmodified. Uses `metrics.invariance
.evaluate_invariance` (DESIGN.md §7's canonical implementation:
`Z' ≈ Z` measured directly, with **no map fit**, deliberately distinct
from "equivariance with `ρ(T)=Identity`," which is the identity
baseline in Tasks 6/7/10).

### Hypothesis

`Z` stays comparatively invariant (high cosine similarity, low relative
L2 movement) under `lighting_change`/`texture_change`, in contrast to
its behavior under the four geometric transforms.

### Mathematical formulation

For `T_a` in `CONTROL_TRANSFORMS`:

```
E(R(T_a(S))) ≈ E(R(S))     i.e.    Z' ≈ Z
```

measured directly via `metrics.invariance.evaluate_invariance(transform
_name, Z, Z_prime)` — no `W_T` fit at all, per DESIGN.md §7.

### Inputs

`(Z, Z')` pairs for `lighting_change` and `texture_change` (reused from
Task 7, or freshly rendered under the identical protocol); a
"null-transform" pair per scene (re-render the identical `SceneState`
with no transform applied at all, `T = identity`).

### Outputs

`state/task_09_result.json`: invariance metrics for `lighting_change`,
`texture_change`, the null-transform sanity check, and (for comparison)
the four geometric transforms' invariance scores.

### Experimental protocol

1. For each `CONTROL_TRANSFORMS` entry, compute invariance metrics
   between `Z` (original) and `Z'` (transformed).
2. Compute the null-transform sanity check: re-render the same
   `SceneState` with no transform, compute invariance between the two
   renders' representations. If this does not score near-perfect
   invariance, that indicates a pipeline determinism bug (e.g.
   nondeterministic rendering), which is a software-correctness
   finding, not a scientific one, and must block this task until fixed.
3. Compute invariance metrics for the four geometric transforms (reused
   from Task 7's `(Z, Z')` pairs) for direct comparison.

### Dataset requirements

Same scene set as Task 7 where reused; otherwise a fresh, equally-sized
set under the identical scene-level split discipline.

### Train/test protocol

No fitting occurs in this task (invariance is measured directly, not
via a fit map), so there is no train/test split to contaminate for the
invariance computation itself; the underlying scene set nonetheless
retains its scene-level split labeling for consistency with Tasks 6–7
and for any control that does involve fitting (see below).

### Controls

**Null-transform sanity check** (identity re-render) — required, not
optional; it is the upper-bound/sanity-check control this task adds
beyond Tasks 6–7's use of the same transforms.

### Baselines

Geometric transforms' invariance scores (reused from Task 7) serve as
the contrastive comparison baseline for this task specifically — the
appearance controls are expected to be markedly *more* invariant.

### Metrics

Mean cosine similarity and mean relative L2 error between `Z` and `Z'`
(via `metrics.invariance.evaluate_invariance`) — the same primitives as
Tasks 6–7, applied without a fitted map.

### Required artifacts

`state/task_09_result.json` per the Outputs section.

### Required software tests

- Physical-state equality verified for every appearance intervention
  (not assumed): geometry unchanged, object pose unchanged, camera pose
  unchanged, object identity unchanged — checked against
  `changed_variables`/`fixed_variables` from `apply_transform`, which
  for `lighting_change`/`texture_change` must show only appearance
  fields as changed.
- Correct representation/video pairing; no NaN/Inf; reproducibility.
- The null-transform sanity check is itself asserted in a test (e.g.
  invariance score above a documented near-1.0 threshold), not just
  reported informally.
- Regression (existing suite, plus `tests/test_metrics.py`'s existing
  invariance coverage if present).

### Required scientific-validity tests

- Verify the intended physical equality rather than assuming it (every
  appearance intervention explicitly recorded and checked, per above).
- Investigate, explicitly, whether apparent geometric sensitivity in
  Tasks 6–7 could actually be caused by lighting, texture, material,
  color, or rendering-pipeline artifacts rather than genuine geometric
  change — a required scientific-QA step for this task.
- Scene-level split respected even though no fitting occurs here (for
  cross-task consistency and any future extension of this task that
  does fit something).

### Required research-alignment checks

Consistent with DESIGN.md §7's explicit distinction between invariance
(no map fit) and equivariance-with-identity-baseline (the identity
baseline in Tasks 6/7/10) — this task's result JSON must not conflate
the two.

### Required leakage checks

Same scene-level split discipline as prior tasks, for the underlying
scene set.

### Reproducibility requirements

Seed and config recorded; the null-transform sanity check's threshold
is fixed in advance, not tuned after seeing appearance-control results.

### Failure conditions

Any of: an appearance intervention that inadvertently changes geometry,
pose, or object identity (per `changed_variables`); the null-transform
sanity check failing its threshold (pipeline determinism bug); tuning
the invariance threshold after seeing results to make appearance
controls look more invariant than they are.

### Acceptance criteria

1. Both appearance controls and the null-transform sanity check
   evaluated with `metrics.invariance`.
2. Geometric transforms' invariance scores included for direct
   comparison.
3. Null-transform sanity check passes its documented threshold, or the
   task is marked blocked with the anomaly explained (a pipeline bug,
   not a valid negative result).

### Interpretation rules

Invariance under appearance change is evidence the representation is
not merely reacting to *any* pixel-level change. It does **not** by
itself prove the representation encodes geometry (that inference needs
Tasks 6–8's positive results too, taken together, still bounded by
their own limits). Do not claim complete appearance invariance — only
claim what was tested (lighting, texture; not every possible appearance
factor).

### Explicit prohibited shortcuts

Assuming physical-state equality for an appearance intervention instead
of verifying it; tuning the invariance metric's threshold after seeing
the appearance-control results.

### What a positive result means

Appearance-only change moves `Z` markedly less than the tested
geometric transforms, under this scene distribution — supporting
(not proving) that `Z`'s sensitivity in Tasks 6–7 tracks geometry
rather than arbitrary visual change.

### What a negative result means

Appearance changes move `Z` almost as much as geometric ones — a valid,
meaningful negative finding that directly undercuts confidence in
Tasks 6–7's geometric interpretation, and must be reported prominently,
not minimized.

### What this task does NOT establish

Invariance to appearance factors not tested (e.g. novel textures/
materials never sampled); any claim about *which* geometric property
drives Tasks 6–7's results (only that appearance alone does not fully
explain it, if the result is positive).

---

# TASK 10 — Baseline framework

### Purpose

Determine whether observed representation behavior is meaningful
relative to reasonable alternative representations and trivial
controls, and consolidate this project's baseline logic into one
shared, tested framework.

### Scientific question

Does the primary representation demonstrate behavior that cannot be
explained by simple or alternative baselines?

### Why this task exists

Tasks 6–9 each called baseline logic individually (persistence, mean,
shuffled-pairing) and DESIGN.md §12 additionally specifies two
baselines — the non-learned pixel-statistics encoder and the
matched-architecture randomly-initialized encoder — that have not yet
been wired into any Task 6–9 experiment. Task 10 exists both to
implement the full DESIGN.md §12 baseline set and to consolidate it
into one reusable, fairly-applied framework so Tasks 11+ invoke one
function instead of re-wiring baseline calls by hand each time, making
"mandatory baselines" (Global Invariant 18–19) enforced in code, not
just convention.

### Relationship to previous tasks

Formalizes and extends `baselines/identity_baseline.py`,
`baselines/shuffled_pairing_baseline.py`, and Task 6's new
`baselines/mean_baseline.py`. Adds the two DESIGN.md §12 baselines not
yet implemented: the non-learned pixel-statistics encoder
(`encoders/pixel_baseline.py`, already scaffolded per `README.md`'s
"pixel_baseline" mention — inspected and reused, not reimplemented, if
it already exists) and the matched-architecture randomly-initialized
encoder (`VJEPAEncoder(pretrained=False)`, already supported by the
existing encoder interface). Refactors Tasks 6–9's experiment code to
call the new consolidated entry point.

### Hypothesis

None new — this is an infrastructure-consolidation and
baseline-completion task, not a new experiment. Its scientific content
is whatever Tasks 6–9's *re-run* baseline comparisons reveal once the
full DESIGN.md §12 set is applied uniformly.

### Mathematical formulation

No new formulation; formalizes DESIGN.md §12's four baselines
(identity/persistence, shuffled-pairing/random-pair, pixel-statistics
encoder, randomly-initialized encoder) plus Task 6's mean baseline, as
one uniformly-applied set.

### Inputs

Whatever `(Z_train, Z'_train, Z_test, Z'_test)` (or `(Z_train, y_train,
Z_test, y_test)` for Task 8-style probes) a calling task already
assembles.

### Outputs

A single entry point, e.g. `baselines.run_all_baselines(Z_train,
Z_prime_train, Z_test, Z_prime_test, transform_name, alpha, seed) ->
dict[str, BaselineResultLike]`, running the full baseline set and
returning results uniformly; `state/task_10_result.json`.

### Experimental protocol

1. Implement (or confirm-and-wire, if partially present) all five
   baselines: persistence, mean, shuffled-pairing, pixel-statistics
   encoder, randomly-initialized encoder.
2. Every baseline must receive the identical scenes, split, labels,
   transformations, metrics, and evaluation protocol as the primary
   representation's result — no baseline may receive privileged
   ground-truth information the primary representation does not also
   have access to (e.g. the pixel-statistics baseline gets rendered
   pixels, not ground-truth `SceneState` coordinates).
3. Refactor Tasks 6–9's experiment code to call `run_all_baselines`
   instead of the three separate calls each currently makes — a real
   refactor, not a parallel implementation left to bit-rot.
4. Verify numerical parity: after refactoring, Tasks 6–9's previously
   recorded persistence/mean/shuffled-pairing numbers must reproduce
   (bit-for-bit or within documented floating-point tolerance) when
   re-run with the same seed/data through the new consolidated call.
5. Re-run Tasks 6–9's flagship comparisons with the two newly-added
   baselines (pixel-statistics, randomly-initialized encoder) and
   report the expanded comparison.

### Dataset requirements

Reuses Tasks 6–9's existing datasets; no new scene generation required
unless a baseline's fair-comparison requirement cannot otherwise be met
(e.g. the pixel-statistics baseline needs the same rendered frames,
already available).

### Train/test protocol

Unchanged from whichever task's data is being re-evaluated; this task
must not alter any train/test split, only the baseline-computation
pathway.

### Controls

The baseline set itself *is* this task's set of controls, per Global
Invariant 18 ("controls must be implemented rather than merely
discussed").

### Baselines

The full five: persistence, mean, shuffled-pairing/random-pair,
non-learned pixel-statistics encoder, matched-architecture
randomly-initialized encoder. A missing baseline in any downstream
task's use of this framework is an explicit failure, not a silent skip.

### Metrics

Whatever metric the calling task specifies (R²/cosine/relative-L2 for
equivariance-style tasks; R²/MAE/RMSE for probe-style tasks like Task
8) — `run_all_baselines` must not hardcode one metric family.

### Required artifacts

`baselines/run_all_baselines` (or equivalently named) function,
implemented and used by Tasks 6–9's code; `state/task_10_result.json`.

### Required software tests

- `run_all_baselines` returns exactly the five baselines with the same
  field names/types as calling them individually.
- A regression test that re-running Task 6 (or a small synthetic
  stand-in with the same seed/inputs) through the new consolidated call
  reproduces its previously recorded metrics.
- Every baseline executes without error on real data.
- No NaN/Inf; reproducibility.

### Required scientific-validity tests

- Identical scene split, labels, and metrics confirmed across every
  baseline and the primary representation for a given comparison.
- No baseline receives privileged ground-truth information (an explicit
  check, e.g. the pixel-statistics baseline's inputs are traced back to
  rendered pixels only).
- Ask, explicitly, whether a trivial representation (pixel-statistics)
  can obtain performance comparable to the primary encoder — a required
  scientific-QA step, and if so, this must be reported prominently, not
  buried.

### Required research-alignment checks

Global Invariants 18–19 (controls implemented, baselines evaluated
under the identical protocol) become mechanically checkable once
`run_all_baselines` exists — future orchestrator QA for Tasks 11+
should call it directly rather than re-deriving baseline logic.

### Required leakage checks

None new (no new experiments run beyond what Tasks 6–9 already
established) — a leakage regression introduced by this refactor is
itself a failure to catch immediately.

### Reproducibility requirements

Same seed/config discipline as the task being re-evaluated; the
refactor itself must not require new hyperparameters beyond what was
already recorded.

### Failure conditions

Any of: a baseline's math silently changed while "consolidating" it
(invalidating Tasks 6–9's already-recorded results); the old per-task
baseline-calling code left in place unused alongside the new function;
a baseline receiving privileged ground-truth information; a missing
baseline silently skipped instead of raising.

### Acceptance criteria

1. `run_all_baselines` (or equivalent) exists, tested, and used by
   Tasks 6–9's experiment code.
2. Numerical parity with pre-refactor baseline numbers (documented
   tolerance).
3. The two newly-added DESIGN.md §12 baselines (pixel-statistics,
   randomly-initialized encoder) are wired in and reported for at least
   Task 6/7's flagship comparison.
4. Full existing suite passes, including Tasks 6–9's own tests.

### Interpretation rules

Do not call the primary representation superior to a baseline unless
the comparison is fair (identical protocol) and the margin is reported
with its metric explicitly, not asserted qualitatively.

### Explicit prohibited shortcuts

Changing any baseline's math while consolidating it; leaving duplicate
baseline-calling code paths; giving any baseline privileged information;
silently skipping a baseline instead of failing loudly.

### What a positive result means

The primary representation outperforms the full DESIGN.md §12 baseline
set (not just persistence/mean/random-pair) under a fair, identical
protocol — the single most important comparison this project can make,
per DESIGN.md §12 item 4's note on isolating pretraining's contribution.

### What a negative result means

A baseline (especially the pixel-statistics or randomly-initialized
encoder) performs comparably to the primary representation — direct
evidence that Tasks 6–9's apparent findings are not specific to
pretrained V-JEPA representations, and must be reported as a major,
not minor, finding.

### What this task does NOT establish

Any new scientific claim about geometric consistency itself — only
whether previously reported findings survive a fair, complete baseline
comparison.

---

# TASK 11 — Scale experiment

### Purpose

Determine whether observed effects survive increasing dataset size.

### Scientific question

Are the observed representation properties robust to dataset scale, or
are they artifacts of a small controlled dataset?

### Why this task exists

Every prior task (6–10) has operated on scene counts in the tens. A
result that only appears (or only fails to appear) at that scale could
be a small-sample artifact in either direction. Task 11 exists to test
this directly, using the flagship transform as a tractable
representative case rather than re-running the entire Task 7 matrix at
every scale (which would be computationally prohibitive given
rendering is this project's bottleneck).

### Relationship to previous tasks

Reuses Task 10's consolidated experiment/baseline framework; targets
the flagship transform (`camera_rotation`, per `configs.config
.ExperimentConfig.flagship_transform`) as established by the existing
V0 config convention.

### Hypothesis

Where appropriate: the flagship transform's equivariance margin over
baselines (Task 6/7's finding) persists, strengthens, weakens, or
saturates as scene count grows from tens to (at minimum) approximately
100, and, if computationally feasible, approximately 1000 scenes.

### Mathematical formulation

Identical to Task 6's `Z' ≈ W_T Z + b_T` for `T = camera_rotation`, run
independently at each scale point on a scale-specific train/test split,
with multiple random seeds per scale point.

### Inputs

Scene counts on a documented ladder (at minimum ~100 scenes; ~1000 if
feasible), each with its own fresh scene-level train/test split at a
fixed fraction (e.g. always 80/20), and multiple seeds per scale point.

### Outputs

`state/task_11_result.json`: a `scale_points: [{num_scenes, seed,
metrics for learned map and all Task 10 baselines, mean, std,
per-seed results, wall_clock_seconds}, ...]` list.

### Experimental protocol

1. For each scale point on the ladder, sample a fresh scene-level
   train/test split at the fixed fraction, under multiple seeds.
2. Hold constant, wherever scientifically possible, across scale
   points: encoder, transformation, metrics, split methodology,
   preprocessing, evaluation protocol — do not silently change
   methodology between scales.
3. Run the flagship transform's equivariance protocol (Task
   6/10-consolidated) at each scale point, for every seed.
4. Report per-scale-point: sample count, seed, mean, standard
   deviation, and every per-seed result (not just the best seed).
5. Run Task 10's full baseline set at every scale point too — a
   baseline's score changing with scale (e.g. the mean baseline's R²
   naturally rising as the training-mean estimate stabilizes) is
   expected and must be shown alongside the learned map's curve, not
   omitted.

### Dataset requirements

At minimum 3 distinct scene-count points on the ladder (e.g. ~40, ~100,
~1000, budget permitting), each with multiple seeds.

### Train/test protocol

Fresh scene-level split per scale point, at a fixed train/test
fraction; scale is varied on absolute scene count on both sides, not on
the split ratio.

### Controls

Multiple seeds per scale point (a "control" against seed-specific
noise, not a baseline in the Task 6/10 sense).

### Baselines

Task 10's full consolidated baseline set, run at every scale point.

### Metrics

Same as Task 6 (R², cosine similarity, relative L2 error), aggregated
as mean ± standard deviation across seeds per scale point, with every
per-seed value also recorded (not just the aggregate).

### Required artifacts

`state/task_11_result.json` per the Outputs section; wall-clock timing
per scale point (a genuine practical constraint on this project, given
rendering is the bottleneck).

### Required software tests

- The requested scale was actually used (scene counts verified, not
  assumed); scene IDs recorded; no overlap between scale points'
  splits (each is independent, though overlap across *different* scale
  points is not itself a leakage bug — only overlap between a single
  scale point's own train and test is).
- Multiple seeds actually executed per scale point; same protocol
  applied at every scale point; seed recorded.
- No test-set fitting; correct mean/std computation; failed runs not
  silently discarded (a failed run at one scale point must be reported,
  not dropped from the average).
- Reproducibility; no NaN/Inf.

### Required scientific-validity tests

- Determine, explicitly, whether the effect persists, strengthens,
  weakens, saturates, or disappears with scale — a required conclusion
  in the result JSON, not left implicit in raw numbers.
- Report all seeds/scale points, not only the best seed or best scale.

### Required research-alignment checks

Global Invariants 21–24 (seed, config, provenance, machine-readable
results) checked per scale point, not just once for the whole task.

### Required leakage checks

Per scale point: disjoint scene sets, no test-scene influence on any
fitted parameter — same discipline as Task 6, independently re-verified
at each point.

### Reproducibility requirements

Every scale point's seed(s), scene IDs, and config recorded
independently; the train/test fraction held fixed and stated once.

### Failure conditions

Fewer than 3 scale points completed; a scale point's split not
disjoint; only the best seed/scale reported; methodology silently
changed between scale points (e.g. a different pooling scheme at a
larger scale) without an explicit, recorded justification.

### Acceptance criteria

1. At least 3 distinct scene-count points run end-to-end.
2. Learned map and all baselines reported at every point.
3. Wall-clock timing recorded per point.
4. No NaN/Inf at any point.

### Interpretation rules

A trend across 3–5 scale points on one synthetic scene distribution and
one transform is suggestive, not an asymptotic scaling law. Do not
extrapolate beyond the tested range (e.g. do not claim what would
happen at 10,000 scenes from a 40–1000 ladder).

### Explicit prohibited shortcuts

Stopping the ladder early and extrapolating a trend from fewer than 3
points; reusing the exact same train/test split across scale points by
only adding scenes to one side; reporting only the best-performing
seed or scale point.

### What a positive result means

The flagship transform's advantage over baselines persists or
strengthens with scale, under this scene distribution — evidence the
effect is not a small-sample artifact, bounded to the tested range.

### What a negative result means

The effect weakens, saturates, or disappears with scale — a valid,
important finding that directly qualifies (and may substantially
weaken confidence in) Tasks 6–7's conclusions, and must be reported
prominently.

### What this task does NOT establish

Behavior beyond the tested scale range; behavior for transforms other
than the flagship one tested here (a full Task 7 matrix at scale is
explicitly out of scope given rendering cost, unless a documented
protocol change extends it).

---

# TASK 12 — Temporal geometric consistency

### Purpose

Extend the static geometric consistency framework into temporal
dynamics.

### Scientific question

Does representation change over time in a way that corresponds to
controlled physical changes in the underlying world?

### Why this task exists

Tasks 6–11 all operate on a single discrete before/after transform pair
per scene (a static-clip or single-transform-application design,
explicitly noted in DESIGN.md §13's "cannot support" list as "any claim
about ... temporal dynamics beyond a single before/after pair"). Task
12 exists to close exactly that gap, using `generation.motion`'s
already-implemented real multi-frame trajectory rendering (orbit
camera motion, constant-velocity object motion) that Tasks 6–11
deliberately did not need.

### Relationship to previous tasks

Uses `generation.motion.generate_trajectory`
(`ObjectMotion`/`CameraMotion`, pure math, no bpy dependency) and
`generation.bpy_renderer.render_trajectory` (Task 2's real multi-frame
renderer with depth+segmentation ground truth) — genuinely new
infrastructure usage for this project's task sequence, since Tasks 6–11
used `transforms.pairs.generate_pair`'s (optionally motion-enabled, but
not required to be) matched-pair rendering. Inherits the `Z' ≈ W_T Z`
formulation from Task 6, now applied between two temporal windows of
one continuous clip instead of two separately-rendered original/
transformed clips. `v_i` (object velocity, DESIGN.md §10 item 4,
explicitly deferred until real motion existed) becomes meaningful for
the first time from this task onward, unblocking a future re-visit of
Task 8's velocity probe.

### Hypothesis

For a clip with real, continuous motion (camera orbiting, or an object
moving at constant velocity), the representation at one temporal window
relates to the representation at another temporal window through a
transform that is predictable from the known relative
camera/object displacement between the two windows, generalizing across
scenes the same way Task 6 tested for a single discrete transform.

### Mathematical formulation

For temporal windows `t1`, `t2` within one clip, with known relative
transform `T_{t1→t2}` derived from the trajectory's recorded per-frame
poses:

```
Z_{t1} = E(V[window t1]),   Z_{t2} = E(V[window t2])
Z_{t2} ≈ W_{T} Z_{t1} + b_T      (fit on TRAIN scenes only)
```

structurally identical to Task 6's formulation, with `T` now derived
from continuous motion instead of a single `apply_transform` call.

### Inputs

Clips rendered with a non-trivial `camera_motion` (mode `"orbit"`) or
`object_motions` via `generation.motion.generate_trajectory` +
`generation.bpy_renderer.render_trajectory`; at least two distinct
temporal windows per clip (e.g. frames 0–3 vs. frames 4–7, or two
overlapping windows — the exact windowing scheme decided and documented
explicitly, not left implicit); the ground-truth relative
camera/object transform between the two windows, read from the
trajectory's per-frame pose records (reused, not re-derived).

### Outputs

`state/task_12_result.json`, following Task 6's schema shape, with an
added `motion_config` field (exact `ObjectMotion`/`CameraMotion`
parameters used) and a `static_clip_control` metrics block.

### Experimental protocol

1. Render clips with real motion (orbit camera or moving object) via
   `generate_trajectory` + `render_trajectory`.
2. Encode each clip at two (or more) distinct temporal windows,
   producing `Z_{t1}`, `Z_{t2}` per scene.
3. Compute the known ground-truth relative transform between the two
   windows' poses.
4. Fit `W_T` between the two windows' representations on train scenes;
   evaluate on test scenes — identical protocol to Task 6, applied to
   this temporally-derived transform.
5. Run a **static-clip control**: repeat the identical experiment with
   `camera_motion=None`/no object motion, to confirm the temporal-
   window split alone (without real motion) does not itself produce
   spurious "consistency" (e.g. from encoder positional biases across
   the token sequence).

### Dataset requirements

At least one real-motion condition (orbit camera or moving object) and
the static-clip control, each with its own scene set meeting the
scene-count floor established in Task 6 (≥40), and — per the leakage
requirement below — kept disjoint from each other or explicitly
documented as sharing scenes.

### Train/test protocol

Standard scene-level split (Global Invariants 7–10), applied
independently to the real-motion condition and the static-clip control;
if the same scenes are reused across both (motion-enabled vs.
motion-disabled variant of the same scene), this must not leak train/
test assignment across the two conditions — documented explicitly
either way.

### Controls

**Static-clip control** (no real motion) is required, not optional,
for this task specifically — it is the only way to rule out that
temporal-windowing itself, independent of genuine motion, produces
spurious apparent consistency.

### Baselines

The same three from Task 6 (persistence, mean, random-pair), reported
for both the real-motion condition and the static-clip control.

### Metrics

Same as Task 6 (R², cosine similarity, relative L2 error) applied to
`(Z_{t1}, Z_{t2})` pairs instead of `(Z, Z')` pairs.

### Required artifacts

`state/task_12_result.json` per the Outputs section.

### Required software tests

- Frame ordering and indices correct; adjacent/chosen-window pairing
  correct; physical-transition ground truth correctly derived from the
  trajectory; representation-transition computation correct.
- Regression (existing suite, including `tests/test_motion.py`,
  `tests/test_bpy_renderer.py`).
- No NaN/Inf; reproducibility.

### Required scientific-validity tests

- A leakage test that the two temporal windows are computed from the
  same scene's single trajectory (not accidentally mixing frames
  across different scenes).
- Rule out, explicitly: simple adjacent-frame visual similarity, static
  scene identity, camera artifacts, or rendering artifacts as
  alternative explanations for any positive result — a required
  scientific-QA step for this task.
- Temporal smoothness alone (i.e. `Z_{t1}` and `Z_{t2}` simply being
  similar because adjacent frames look similar) is explicitly **not**
  sufficient evidence of physical understanding, and the static-clip
  control is the primary tool for ruling this out — a passing result
  requires the real-motion condition's predictive margin over the
  static-clip control, not just over the standard Task 6 baselines.

### Required research-alignment checks

Same frozen-encoder/no-ground-truth-input/linear-map-first discipline
as Task 6, now applied to a temporal pair instead of a transform pair.

### Required leakage checks

Standard scene-level split (invariants 4–5-equivalent, i.e. Global
Invariants 7–10), plus: the static-clip control must use a different,
disjoint scene set from the real-motion experiment, or, if the same
scenes' motion-disabled variant is used, an explicit check that this
does not leak train/test assignment across the two conditions.

### Reproducibility requirements

Seed, `motion_config` (exact `ObjectMotion`/`CameraMotion` parameters),
windowing scheme, and ridge `alpha` all recorded.

### Failure conditions

Any of: temporal windows drawn from mismatched scenes; the static-clip
control omitted; windowing scheme chosen after seeing which gives the
best score; ground-truth trajectory poses fed into the encoder as
input.

### Acceptance criteria

1. At least one real-motion condition run end-to-end with the
   two-window protocol.
2. Static-clip control run under the identical protocol.
3. All three Task 6-style baselines reported for both conditions.
4. No NaN/Inf.

### Interpretation rules

A positive result here is evidence of consistency under one motion
type (orbit / constant velocity) and one windowing scheme — it does
not establish general temporal-dynamics understanding, and does not
imply the encoder tracks arbitrary motion patterns.

### Explicit prohibited shortcuts

Using ground-truth trajectory poses as encoder input; picking temporal
windows after seeing which windowing gives the best score; omitting the
static-clip control.

### What a positive result means

The tested motion type induces a temporally-local, linearly predictable
representation transition that generalizes across scenes, beyond what
the static-clip control alone produces.

### What a negative result means

No measurable temporal consistency beyond the static-clip control, or
no advantage over Task 6-style baselines — a valid, complete finding
for this motion type and windowing scheme.

### What this task does NOT establish

Anything about motion types not tested; general temporal-dynamics
understanding; object permanence or identity persistence specifically
(Task 13's distinct question); causal or predictive-world-model claims.

---

# TASK 13 — Object persistence under occlusion

### Purpose

Test whether object-related information persists through controlled
visual occlusion and reappearance.

### Scientific question

When an object becomes visually unavailable and later reappears, does
the representation preserve information sufficient to associate the
reappearing object with the same underlying physical object?

### Why this task exists

Task 12 establishes that temporal representation transitions can be
studied at all. Task 13 asks a qualitatively different temporal
question: not "is the transition between two visible states
predictable," but "does object-level information survive a period of
zero direct visual evidence." This is a distinct, harder claim
(component E in this project's six operationally-testable claims, §1)
that neither Tasks 6–11's static-pair design nor Task 12's
continuously-visible-motion design can address.

### Relationship to previous tasks

Uses `generation.scene.ObjectState.instance_id` (the persistent,
segmentation-mask-tied per-object identity already defined in Task 2)
and `generation.bpy_renderer`'s segmentation-pass rendering (Task 2) to
both construct and **verify** occlusion, and Task 12's motion/windowing
infrastructure to construct the pre-occlusion/occluded/post-occlusion
temporal structure.

### Hypothesis

For a tracked object (by `instance_id`) that becomes occluded for a
documented sub-sequence of frames and then reappears, the
representation's post-occlusion window carries information about that
object's state (e.g. position) that is more accurately recoverable by a
linear probe than the same probe applied under a scrambled-identity
control, as a function of occlusion duration.

### Mathematical formulation

A linear probe (reusing Task 8's `y ≈ W_probe Z + b_probe` pattern) is
fit to predict the tracked object's post-occlusion physical state from
the post-occlusion representation window, compared against the
identical probe under: (a) an object-identity-scrambled control (labels
permuted across scenes), and (b) a no-occlusion control (upper bound).

### Inputs

Scenes constructed (via `generation.motion` trajectories or camera
framing) so that one tracked object (by `instance_id`) becomes occluded
— by another object or by leaving the camera frustum — for a documented
sub-sequence of frames, then reappears; the segmentation ground truth
(`ClipGroundTruth.segmentation`, Task 2) for **verifying** occlusion
actually occurred.

### Outputs

`state/task_13_result.json`: occlusion verification evidence (frame
ranges + segmentation-based confirmation), the recoverability metric
for the real, scrambled, and no-occlusion conditions.

### Experimental protocol

1. Construct at least one occlusion scenario via object motion or
   camera framing.
2. **Verify** occlusion programmatically using segmentation ground
   truth: the tracked instance's segmentation mask pixel count drops to
   zero (or below a documented threshold) for the intended frames — do
   not assume a scripted trajectory produces occlusion without checking
   the rendered ground truth.
3. Extract representations from pre-occlusion, occluded, and
   post-occlusion windows (reusing Task 12's windowing approach).
4. Fit the recoverability probe (post-occlusion representation →
   post-occlusion object state) on train scenes; evaluate on test
   scenes, for the real, scrambled-identity, and no-occlusion
   conditions.
5. Where feasible, vary occlusion duration and report recoverability as
   a function of duration.

### Dataset requirements

At least one occlusion scenario, verified via segmentation, meeting the
scene-count floor established in Task 6; ideally multiple occlusion
durations.

### Train/test protocol

Standard scene-level split (Global Invariants 7–10).

### Controls

- **Object-identity-scrambled control**: permute object identity/
  position labels across scenes (same spirit as
  `shuffled_pairing_baseline.py`) — its permutation must be a real
  derangement over scenes, not accidentally close to the identity
  permutation.
- **No-occlusion control**: identical scene setup with a trajectory
  that keeps the object visible throughout, as the upper-bound
  comparison.
- A chance/random-identity baseline is required at minimum, per the
  task brief.

### Baselines

The scrambled-identity and no-occlusion controls above serve as this
task's baseline set; Task 8's linear-probe machinery is reused, not
a new probe family invented.

### Metrics

Held-out R²/MAE/RMSE for the recoverability probe (matching Task 8's
metric family), compared across real/scrambled/no-occlusion conditions
and, where feasible, across occlusion durations.

### Required artifacts

`state/task_13_result.json` per the Outputs section.

### Required software tests

- Unit test on the occlusion-verification logic itself (given a
  synthetic segmentation array, correctly detects occlusion vs. no
  occlusion).
- Persistent IDs correctly assigned/tracked; occlusion duration
  correctly computed; trajectory correctness; reappearance genuinely
  corresponds to the same physical object (`instance_id` match).
- No NaN/Inf; reproducibility; regression (existing suite).

### Required scientific-validity tests

- Occlusion claimed only when the segmentation-based check actually
  passes (not assumed from the trajectory alone).
- Investigate, explicitly, trivial cues that could explain successful
  re-identification without genuine persistence: object color, object
  location continuity, background, or scene identity — a required
  scientific-QA step.
- No identity-label leakage (the scrambled control must not
  accidentally preserve the true correspondence).

### Required research-alignment checks

No ground-truth segmentation/instance-ID data fed into the encoder
(invariant 6) — used only for verification and probe labels.

### Required leakage checks

Standard scene-level split; the scrambled-identity control's
permutation must be a genuine derangement, not trivially close to
identity for small `N`.

### Reproducibility requirements

Seed, occlusion-construction parameters, and duration(s) tested all
recorded.

### Failure conditions

Occlusion claimed without a passing segmentation-based verification;
the scrambled or no-occlusion control omitted; identity labels leaking
into the scrambled control; ground-truth segmentation/instance IDs fed
to the encoder.

### Acceptance criteria

1. At least one occlusion scenario constructed and verified via
   ground-truth segmentation.
2. Recoverability metric computed for real, scrambled, and
   no-occlusion conditions.
3. No NaN/Inf.

### Interpretation rules

Successful re-identification/recoverability alone does not prove an
internal object model — it must be reported alongside the explicit
ruling-out of trivial cues (color, location, background, scene
identity) required above.

### Explicit prohibited shortcuts

Claiming occlusion occurred without the segmentation-based check
passing; feeding segmentation masks or instance IDs into the encoder.

### What a positive result means

Some object-level continuity signal is linearly recoverable after
occlusion, for this occlusion pattern and scene distribution, beyond
what the scrambled-identity control achieves.

### What a negative result means

No recoverability advantage over the scrambled control — a valid,
complete outcome, and evidence against object-level persistence in this
representation under this protocol.

### What this task does NOT establish

Object permanence in any general sense; that "the encoder tracks the
object" as opposed to "the encoder reconstructs a plausible
continuation from scene-level statistics" (both remain consistent with
a positive result unless further distinguished); behavior for occlusion
patterns, durations, or object types not tested.

---

# TASK 14 — Controlled counterfactual representation consistency

### Purpose

Test whether controlled interventions generate structured and
reproducible representation changes.

### Scientific question

When two matched worlds differ only in a controlled intervention, does
the representation change in a way specific to that intervention?

### Why this task exists

Tasks 6–7 test whether a *given* transform's effect is predictable.
Task 14 asks a sharper question: whether the representation
*distinguishes* between two different counterfactual interventions of
comparable magnitude, holding everything else fixed — a test more
directly probing whether `Z` carries intervention-specific structure
rather than only "some change happened." This uses Task 3's
already-established per-variable isolation guarantee
(`changed_variables`/`fixed_variables` from `apply_transform`), which
is exactly the ground truth a counterfactual-isolation claim needs to
be checkable.

### Relationship to previous tasks

Reuses Tasks 6–8's full experiment/probe/baseline infrastructure and
`transforms.scene_transform.apply_transform`'s explicit
`changed_variables`/`fixed_variables` contract (Task 3) directly, as
ground truth of isolation, rather than re-deriving it.

### Hypothesis

For two isolatable counterfactual edits already supported by
`transforms/scene_transform.py`'s per-variable isolation guarantees
(e.g. `object_translation` vs. `object_rotation` on the same object),
scaled to comparable magnitude, a probe trained to distinguish "which
variable changed" from `(Z_before, Z_after)` pairs (or their
difference) generalizes to held-out scenes, beyond a label-scrambled
control.

### Mathematical formulation

Construct `S`, `S_A = intervention_A(S)`, `S_B = intervention_B(S)`,
with `changed_variables`/`fixed_variables` confirming only the intended
variable differs between `S` and each of `S_A`/`S_B`. Fit a
discrimination probe (reusing Task 6/8's linear-probe machinery) on
`(Z, Z_A)` / `(Z, Z_B)` pairs or their differences, on train scenes;
evaluate generalization to test scenes, against a label-scrambled
control.

### Inputs

At least two counterfactual conditions per scene, constructed via
`transforms.scene_transform.apply_transform` with isolation guarantees
confirmed by its own `changed_variables`/`fixed_variables` output;
magnitudes scaled to be comparable by an explicitly documented method
(e.g. matched by a comparable pixel-level change or a comparable
ground-truth SE(3) displacement magnitude, computed via `transforms.
se3`), so the two conditions are not trivially separable by
representation-norm alone.

### Outputs

`state/task_14_result.json`: which counterfactual conditions were
compared, the magnitude-matching method, discrimination accuracy for
real vs. scrambled labels, and the fixed-variable leakage check's
result.

### Experimental protocol

1. Choose at least two isolatable counterfactual edits already
   supported by `transforms/scene_transform.py` (e.g.
   `object_translation` vs. `object_rotation` on the same object).
2. Confirm isolation via `apply_transform`'s own
   `changed_variables`/`fixed_variables` output — never re-derived by
   hand.
3. Magnitude-match the two conditions using an explicit, documented
   method.
4. Compute `Z` before/after each condition; fit a discrimination probe
   on train scenes; evaluate on test scenes.
5. Run the label-scrambled control (scene-to-condition labels permuted).
6. Run the **fixed-variable leakage check**: verify that a probe
   trained only on representations from scenes where a *held-fixed*
   variable was (independently, in a separate control run) varied does
   **not** predict the counterfactual condition — i.e. the
   discrimination signal is not coming from a confound correlated with,
   but not equal to, the intended variable.

### Dataset requirements

At least two counterfactual conditions, magnitude-matched, meeting the
scene-count floor established in Task 6.

### Train/test protocol

Standard scene-level split (Global Invariants 7–10).

### Controls

- **Label-scrambled control** (same shuffled-pairing-style pattern as
  every prior task).
- **Fixed-variable leakage check** (this task's own, specifically
  required control — not optional): confirm the discrimination signal
  is not attributable to a confound correlated with, but distinct from,
  the intended variable.

### Baselines

The label-scrambled control above; Task 6/8's linear-probe machinery is
reused, not a new probe family invented.

### Metrics

Discrimination accuracy (real vs. scrambled labels); the
magnitude-matching method's own displacement-magnitude comparison is
reported alongside, not just the resulting accuracy.

### Required artifacts

`state/task_14_result.json` per the Outputs section.

### Required software tests

- Unit test verifying the magnitude-matching computation itself (given
  two known transform params, correctly computes/compares their
  displacement magnitude).
- Correct pairing between counterfactual conditions and scenes; no
  NaN/Inf; reproducibility; regression (existing suite).

### Required scientific-validity tests

- Ask, explicitly, whether representation differences could arise from
  unintended differences rather than the intended intervention — this
  is exactly what the fixed-variable leakage check tests, and it must
  be performed, not merely discussed.
- Magnitude-matching must not leave the two conditions trivially
  separable by representation-norm alone (checked, not assumed).

### Required research-alignment checks

No ground-truth `changed_variables`/`fixed_variables` labels fed into
the encoder (invariant 6) — they are ground truth for constructing
conditions and for the leakage check only, never encoder input.

### Required leakage checks

Standard scene-level split; the fixed-variable leakage check above is
itself a required leakage check specific to this task, not optional.

### Reproducibility requirements

Seed, the two counterfactual conditions chosen, and the
magnitude-matching method all recorded.

### Failure conditions

Comparing two counterfactual edits of wildly different, unmatched
magnitude and treating easy discrimination as a meaningful positive
result; the fixed-variable leakage check failing (a real finding that
must block the task, not be averaged away); ground-truth
`changed_variables`/`fixed_variables` fed to the encoder.

### Acceptance criteria

1. At least two counterfactual conditions compared, magnitude-matched,
   with `changed_variables`/`fixed_variables` used as ground truth of
   isolation.
2. Real discrimination probe and label-scrambled control both
   reported.
3. Fixed-variable leakage check performed and reported (pass or fail).
4. No NaN/Inf.

### Interpretation rules

Do not use causal language unless the experimental design actually
supports a causal claim — a positive discrimination result shows the
representation carries *some* information distinguishing the two
tested counterfactual edits under this magnitude-matching scheme, not a
causal or compositional world model.

### Explicit prohibited shortcuts

Comparing magnitude-mismatched conditions; feeding
`changed_variables`/`fixed_variables` labels into the encoder.

### What a positive result means

The representation carries information distinguishing these two
specific counterfactual edits, under this magnitude-matching scheme,
beyond the label-scrambled control, and the fixed-variable leakage
check did not reveal a confound.

### What a negative result means

No discrimination advantage over the scrambled control, or the
fixed-variable leakage check reveals a confound — both are valid,
complete, and (in the leakage-check case) important findings requiring
prominent reporting.

### What this task does NOT establish

A causal or compositional world model; generalization to
edits/variables not tested here; anything beyond the two (or few)
specific counterfactual conditions actually compared.

---

# TASK 15 — Unified evaluation framework

### Purpose

Unify Tasks 6–14 into a coherent, reproducible evaluation framework.

### Scientific question

None new — the individual experiments should become components of a
consistent measurement framework rather than unrelated scripts.

### Why this task exists

By this point, nine distinct experiment scripts (Tasks 6–14) exist,
each with its own slightly different result-JSON shape. Task 16's
report generation and Task 18's adversarial audit both need to consume
all nine consistently; without unification, both would have to
special-case nine formats, multiplying the chance of a transcription or
interpretation error entering the final claims.

### Relationship to previous tasks

Unifies configuration, datasets, scene splits, transformations,
representation extraction, probes, baselines, metrics, controls, result
storage, and provenance across Tasks 6–14, extending `configs.config`'s
existing dataclass-based, YAML-loaded pattern rather than introducing a
new configuration mechanism. Does **not** re-run Tasks 6–14's
experiments to produce new numbers unless a genuine bug is found in the
process — this task is about the harness, not about re-deriving
results.

### Hypothesis

None — infrastructure task.

### Mathematical formulation

None new.

### Inputs

Tasks 6–14's existing result JSON files and experiment code.

### Outputs

One shared result schema (Python dataclass or JSON Schema) that is a
strict superset of every field Tasks 6–14 actually produced (fields
extended, never dropped; optional fields for task-type-specific data
like `motion_config`); one CLI entry point (e.g. `python -m
evaluation.run_task --task 6..14`) that dispatches to each task's
existing experiment code; `state/task_15_result.json`.

### Experimental protocol

1. Define the shared schema as a strict superset of Tasks 6–14's
   existing fields.
2. Build one CLI entry point that imports and calls each task's
   existing experiment logic (no logic rewrite).
3. Re-validate every existing `state/task_0{6..9}_result.json`/
   `state/task_1{0..4}_result.json` against the new shared schema;
   migrate the files in place if field names changed, recording that
   migration explicitly as a protocol change (Global Invariant 27) even
   though it is a schema/naming change, not a re-run.
4. Experiments must be runnable through configuration rather than
   manual source-code editing from this point forward.

### Dataset requirements

None new.

### Train/test protocol

Unchanged from each unified task; this task must not alter any prior
split.

### Controls

None new; every experiment's own controls are preserved unchanged
through the unification.

### Baselines

None new; every experiment's own baselines are preserved unchanged
through the unification.

### Metrics

None new; every experiment's own metrics are preserved unchanged.

### Required artifacts

`evaluation/` module (or similarly named) with the shared schema and
dispatcher; `state/task_15_result.json` confirming all nine prior
result files validate against the new schema.

### Required software tests

- Schema validation tests for all of Tasks 6–14's (migrated) result
  files.
- Tasks 6–14 invocable through the unified interface; configuration
  parsing works; invalid configuration fails loudly (not silently);
  split validation; provenance fields present; unique experiment
  identification; no hidden defaults.
- Regression tests for Tasks 6–14 (their own prior tests still pass).
- No NaN/Inf; reproducibility.

### Required scientific-validity tests

Confirm no previously recorded metric value was altered by this
consolidation (only structure/field names, if anything, and that
change is recorded) — a direct byte/value comparison between
pre-migration and post-migration result files for every unchanged
field.

### Required research-alignment checks

Every one of Global Invariants 21–24 (seed, config, provenance,
machine-readable) must now be satisfiable via the unified schema for
every task, not just some.

### Required leakage checks

None new (no new experiments run); a leakage regression introduced by
this refactor is itself an immediate failure.

### Reproducibility requirements

The migration itself is deterministic and reversible in principle
(documented mapping from old field names to new ones).

### Failure conditions

Any previously recorded metric value silently altered; a task's
experiment logic rewritten rather than wrapped; a methodological change
introduced without explicit documentation (Global Invariant 27); this
task used as a pretext to quietly redo a task whose result the operator
would prefer to be different.

### Acceptance criteria

1. Shared schema exists, is documented, and validates all of Tasks
   6–14's result files.
2. One CLI entry point can invoke each of Tasks 6–14's experiment code.
3. No prior task's recorded numbers were altered by this consolidation
   (only structure/field names, if anything, and that change is
   recorded).

### Interpretation rules

None — infrastructure only; no new scientific claim is made by this
task.

### Explicit prohibited shortcuts

Silently changing any previously recorded metric value while
"unifying" format; using this task as a pretext to quietly redo a task
whose result is undesired.

### What a positive result means

Not applicable (infrastructure task) — "success" here means the unified
framework faithfully reproduces every prior task's recorded result
under a common interface.

### What a negative result means

Not applicable in the scientific sense; a discovered discrepancy
between pre- and post-unification numbers is a software failure to
fix, not a scientific negative result.

### What this task does NOT establish

Any new scientific claim about geometric consistency, accessibility,
invariance, scale, temporal consistency, occlusion, or counterfactual
structure — all such claims remain exactly as established (or not) by
Tasks 6–14 individually.

---

# TASK 16 — Final research report

### Purpose

Produce a rigorous research report based only on actual experimental
evidence.

### Scientific question

What, precisely, can and cannot be concluded about geometric
consistency and physical-state accessibility in this frozen encoder's
representation, across everything tested in Tasks 6–15?

### Why this task exists

Nine-plus experiments (Tasks 6–14, unified in Task 15) produce numbers,
not conclusions. Task 16 exists to synthesize them into a single
document that states exactly what was found, with the same evidentiary
discipline DESIGN.md §13 already applies to Tasks 1–5 — every claim
traceable to a specific metric in a specific task result, and no claim
stronger than what §1's six operationally-testable components actually
support.

### Relationship to previous tasks

Reads (not re-derives) every result in `state/task_06_result.json`
through `state/task_15_result.json` (or their Task-15-unified
equivalents). **The report must be written FROM THE RESULTS** — it must
not be written from expected conclusions and then populated with
supporting evidence.

### Hypothesis

Not applicable — this task synthesizes, it does not test a new
hypothesis.

### Mathematical formulation

Not applicable, beyond restating §2's central formulation for context.

### Inputs

`state/task_06_result.json` through `state/task_15_result.json` (or
Task 15's unified equivalents), and any generated reports/plots from
those tasks.

### Outputs

`reports/final_research_report.md`, `state/task_16_result.json`.

### Experimental protocol

Not an experiment; a synthesis protocol:

1. Restate the research question and method summary, one paragraph per
   Task 6–15, each citing its actual result file.
2. Build a results table aggregating every task's headline metric(s)
   against its baselines.
3. Explicitly reconcile any tasks whose results conflicted (e.g. strong
   equivariance in Task 6/7 but weak accessibility in Task 8) rather
   than silently picking the more favorable one to headline.
4. Write the required structure in full: Abstract; Research question;
   Motivation; Experimental setup; Controlled 3D environment;
   Rendering; Representation extraction; Geometric transformations;
   Physical-state probes; Appearance controls; Baselines; Scale
   experiments; Temporal experiments; Occlusion experiments;
   Counterfactual experiments; Results; Negative results; Failure
   analysis; Alternative explanations; Limitations; Conclusions; Future
   work.
5. Build the claim-evidence matrix for every major claim: CLAIM →
   EXPERIMENT → METRIC → CONTROL → ACTUAL RESULT → LIMITATION.

### Dataset requirements

Not applicable.

### Train/test protocol

Not applicable — reports on splits already used by Tasks 6–15.

### Controls

Not applicable as a new element — the report cites, rather than
re-runs, each source task's own controls.

### Baselines

Every result reported must include its baseline comparison (Global
Invariant 19) — no result may be reported in this document without
also reporting its baseline comparison.

### Metrics

Not applicable beyond citing each task's own metrics verbatim.

### Required artifacts

`reports/final_research_report.md` with the full required structure
above and the claim-evidence matrix; `state/task_16_result.json`
pointing at the report path and confirming every cited task result
file was actually read, not assumed.

### Required software tests

A test that every `state/task_0{6..9}|1{0..5}_result.json` referenced
by the report actually exists, and that at least one number from each
is quoted correctly (a simple string/number match between the report
and the JSON source, to guard against transcription drift).

### Required scientific-validity tests

- No numerical claim in the report lacks a metric citation.
- No claim in the report describes a result that isn't backed by a
  metric actually computed and recorded in the cited task's result
  JSON.
- Every result reported includes its baseline comparison.

### Required research-alignment checks

- No unsupported "understanding" claims: the phrase "understands 3D"
  (or equivalent) must never appear without the specific operational
  metric immediately following it (Global Invariant 29's interpretation
  discipline, applied at the report level).
- Negative results included, not omitted to make the narrative more
  positive (Global Invariants 16, 26).

### Required leakage checks

Not applicable (no new data/splits introduced).

### Reproducibility requirements

The report's own reproducibility: every cited number must be
re-derivable by re-reading the named result file at the named field.

### Failure conditions

Any claim not directly traceable to a result file; any negative result
omitted; any use of "understands 3D" (or equivalent) without an
immediate operational metric; any fabricated value, graph, statistical
significance claim, successful-experiment claim, comparison, or missing
result; any uncertainty silently converted into confidence.

### Acceptance criteria

1. Report covers every task 6–15 with a traceable citation.
2. Explicit "supported" vs. "not supported" claims section present
   (mirroring DESIGN.md §13's structure).
3. No claim in the report lacks a metric citation.
4. Every numerical result traces to machine-readable output; plots
   correspond to actual data; claims correspond to experiments;
   limitations included; negative results included; report
   reproducibility holds.

### Interpretation rules

This report's claims are bounded by everything already stated in each
task's own "What this task DOES NOT establish" section — the report may
not claim more than the sum of its parts. If an experiment failed
technically, state that. If an experiment produced a negative
scientific result, state that. If evidence is inconclusive, state that.

### Explicit prohibited shortcuts

Writing the report from expected conclusions and retrofitting evidence;
fabricating any value, graph, comparison, or missing result; omitting a
task's negative result to improve the overall narrative; using
"understands 3D" or equivalent language without an immediate
operational metric.

### What a positive result means

Not applicable in the aggregate — the report's job is to state
precisely which of §1's six operationally-testable components were
positively, negatively, or inconclusively supported, per task, not to
render one aggregate verdict.

### What a negative result means

Not applicable in the aggregate, for the same reason.

### What this task does NOT establish

Anything beyond what Tasks 6–15 actually measured — the report is not
license to extrapolate, round up uncertainty, or claim general
understanding of 3D structure by a video representation model.

---

# TASK 17 — Open-source release

### Purpose

Make the complete research project independently reproducible and
inspectable.

### Scientific question

Not applicable — this is a release-readiness task, not an experiment.

### Why this task exists

A research project whose results cannot be independently reproduced by
someone outside this session/environment does not meet ordinary
scientific standards, and Task 18's adversarial audit is far more
credible if it (or an external party) can actually re-run the pipeline
from a clean checkout rather than trusting the environment that
produced the original results.

### Relationship to previous tasks

Operates on the entire repository as it stands after Task 16;
verifies, rather than re-derives, that `README.md`'s documented setup
(`geometric-consistency-probe/`'s `pip install -r requirements.txt`,
`pytest -m "not slow"`) actually works end-to-end on a fresh checkout.

### Hypothesis

Not applicable.

### Mathematical formulation

Not applicable.

### Inputs

The full repository as of Task 16's completion.

### Outputs

`LICENSE` (only after the user's explicit direction — do not assume
one); a verified fresh-clone setup path; a top-level "start here"
pointer (in `README.md` or a new `CONTRIBUTING.md`) sending a new
reader to `DESIGN.md` first, then `research/CANONICAL_RESEARCH_PROTOCOL.md`
(this document) and `research/RESEARCH_PLAN.md`, then the final report;
`state/task_17_result.json`.

### Experimental protocol

Not an experiment; a release-readiness protocol:

1. Add a `LICENSE` file per the user's explicit choice.
2. Verify `requirements.txt` is complete and accurately pinned given
   Task 4/5's real pretrained-checkpoint work (`facebook/
   vjepa2-vitl-fpc64-256`, `transformers>=4.52`).
3. Actually run a clean `git clone` plus the documented `Setup` steps
   in `README.md` on a fresh checkout/environment — this must be done
   for real, not re-read from the instructions.
4. Add or verify the top-level reading-order pointer.
5. Grep the current tree (not full history) for credentials, API keys,
   or machine-specific absolute paths.

### Dataset requirements

Not applicable.

### Train/test protocol

Not applicable.

### Controls

Not applicable.

### Baselines

Not applicable.

### Metrics

Not applicable.

### Required artifacts

`state/task_17_result.json` recording: license added, fresh-clone setup
verified (with the exact commands run and their exit codes), dependency
audit result, secret-scan result.

### Required software tests

An automated fresh-environment smoke test: create a new virtualenv,
install `requirements.txt`, run `pytest -m "not slow"`, and record the
outcome — done for real, not assumed from the instructions.

### Required scientific-validity tests

Not applicable in the Tasks-6-14 sense; the analogous requirement here
is that the minimal reproduction actually works: install dependencies,
run tests, execute a minimal experiment, produce machine-readable
output.

### Required research-alignment checks

The reading-order pointer must correctly route a new reader through
`DESIGN.md` → this canonical protocol / `research/RESEARCH_PLAN.md` →
the final report, preserving the same authority ordering stated in this
document's own header.

### Required leakage checks

Not applicable.

### Reproducibility requirements

The fresh-clone verification's exact commands and exit codes are
recorded, not summarized as "it worked."

### Failure conditions

The fresh-clone verification not actually run (only asserted); a
license chosen without the user's explicit direction; a secret or
machine-specific absolute path found in the current tree; a required
dependency undocumented or a documented `README.md` command that does
not actually work.

### Acceptance criteria

1. `LICENSE` present.
2. Fresh-clone setup verified end-to-end with recorded command output.
3. No secrets/credentials found in the current tree.
4. Top-level reading-order pointer present.

### Interpretation rules

Not applicable — this task makes no scientific claims.

### Explicit prohibited shortcuts

Marking the fresh-clone verification complete without actually running
it in an isolated environment; picking a license without the user's
explicit direction.

### What a positive result means

Not applicable.

### What a negative result means

Not applicable; a failed fresh-clone verification is a software failure
to fix before this task can pass, not a scientific finding.

### What this task does NOT establish

Any scientific claim; this task only establishes reproducibility of the
existing pipeline and results.

---

# TASK 18 — Adversarial independent research audit

### Purpose

Attempt to falsify the conclusions of Tasks 6–17.

### Scientific question

Could the observed results be explained by something other than the
claimed geometric/physical structure in the learned representation?

### Why this task exists

Every prior task (6–17) was implemented and QA'd by the same
Claude-Code-plus-orchestrator process that is, by construction,
motivated to produce a passing result. Task 18 exists specifically to
break that symmetry: it must **not** act as an advocate for the
project, and per Global Invariant 30, it must be allowed to challenge —
and, where warranted, overturn — any conclusion produced by Tasks
6–17, including this canonical document's own framing if evidence
warrants it.

### Relationship to previous tasks

Independently re-derives, from raw data (not from Tasks 6–17's own
reported numbers), a sample of their headline results, and
independently re-verifies every train/test split's disjointness from
raw manifest/result files rather than trusting each task's own
leakage-check report.

### Hypothesis

None in the confirmatory sense — this task's working assumption is
adversarial: each headline result is assumed challengeable until an
independent re-derivation and a deliberate falsification attempt fail
to break it.

### Mathematical formulation

Reuses Tasks 6–14's own formulations for the purpose of independent
re-computation; introduces no new formulation of its own.

### Inputs

Raw representations, manifests, and result files from Tasks 6–17
(or their Task 15-unified equivalents); the full codebase and its git
history for the audited period.

### Outputs

`RESEARCH_AUDIT.md` (also referenced in the task brief as
`reports/independent_audit.md` — both names are acceptable as long as
one canonical file is produced and referenced from `state/
task_18_result.json`); `state/task_18_result.json` summarizing pass/
fail per audited claim.

### Experimental protocol

1. Independently re-derive, from raw data, at least one headline result
   from Task 6/7 and one from Task 8 — actually re-run the metric
   computation from the saved representations/manifests, not from the
   reported JSON numbers.
2. Independently re-verify disjointness and same-split-per-scene for
   every train/test split used anywhere in Tasks 6–14, from the raw
   manifest/result files.
3. Attempt at least one deliberate "attack": substitute a scene-identity
   feature (not a physical-state feature) as a probe target and check
   whether it performs suspiciously close to the real probe — a
   leakage finding to report, not to bury, if so.
4. Re-run every audited baseline with a different random seed than the
   original task used, and confirm the qualitative conclusion (learned
   map beats baselines, or does not) is seed-stable.
5. Review every task's "Explicit prohibited shortcuts" section against
   what was actually implemented (read the code, not just each task's
   own self-report) and flag any shortcut taken anyway.
6. Audit systematically against all 20 named categories: scene identity
   leakage; low-level visual similarity; camera artifacts; rendering
   artifacts; preprocessing artifacts; memorization; dataset size
   limitations; seed dependence; weak baselines; transformation
   confounds; temporal leakage; test-set contamination; appearance
   confounds; alternative representations; metric dependence;
   implementation bugs; object identity leakage; counterfactual
   validity; report/result inconsistencies; any methodological
   deviation from the original protocol (this document).

### Dataset requirements

Whatever data Tasks 6–17 already produced; this task does not sample
new scenes except where a deliberate attack (e.g. the scene-identity
substitution) requires constructing one.

### Train/test protocol

Independently re-verified, not re-designed; if the audit finds a split
violation, that is reported as a BLOCKER finding against the affected
task, not silently corrected by this task.

### Controls

The deliberate "attacks" listed above (scene-identity substitution,
seed re-randomization) are this task's own controls against Tasks
6–17's conclusions.

### Baselines

Re-uses Tasks 6–14's own baselines for the seed-stability re-check;
introduces no new baseline family.

### Metrics

Re-uses each audited task's own metrics for independent re-derivation,
so results are directly comparable to the original.

### Required artifacts

`RESEARCH_AUDIT.md` (or `reports/independent_audit.md`): every finding
(confirmed, refuted, or newly discovered issue), with severity, and —
for anything found — either a proposed fix or an explicit statement
that a prior claim must be downgraded/retracted; `state/
task_18_result.json`.

### Required software tests

The audit's own re-derivation code must have basic tests: it must
reproduce a known, hand-computed toy example correctly before being
trusted against real project data.

### Required scientific-validity tests

For every one of the 20 audit categories, classify as **BLOCKER**,
**MAJOR**, **MINOR**, or **PASS**, with, for each finding: issue,
evidence, affected task, scientific consequence, recommended
correction, and whether the conclusion survives.

### Required research-alignment checks

This task's own conduct must satisfy Global Invariant 30 — it must not
soften or omit a finding that contradicts `reports/
final_research_report.md`'s conclusions; the report must be corrected
(with a recorded justification per Global Invariant 27) if the audit
overturns it, never the other way around.

### Required leakage checks

This entire task *is* a leakage check, applied adversarially across
Tasks 6–17; it must not perform the audit by re-reading the same result
JSON files and re-stating their conclusions — the entire point is
independent re-derivation from more raw data.

### Reproducibility requirements

The audit's own re-derivation must be re-runnable and must state its
own seed(s)/configuration, to the same standard it applies to the tasks
it audits.

### Failure conditions

An audit that only re-reads existing result files without independent
re-derivation from raw data; a finding softened or omitted because
fixing it is inconvenient; a finding downgraded in severity without
justification; the scene-identity-substitution attack or the
seed-stability re-check skipped entirely.

### Acceptance criteria

1. At least one Task 6/7 result and one Task 8 result independently
   re-derived from raw data.
2. Seed-stability re-check performed for at least the flagship
   transform's headline result.
3. Scene-identity-substitution attack attempted and reported.
4. Every finding (positive or negative) recorded in `RESEARCH_AUDIT.md`
   / `reports/independent_audit.md`, including "no issue found" for
   anything explicitly checked.

### Interpretation rules

This audit cannot prove the absence of all possible confounds — only
that the specific attacks attempted here did or did not succeed. This
limitation must be stated explicitly in the audit's own limitations
section, mirroring every other task's own interpretation-limits
discipline.

### Explicit prohibited shortcuts

Auditing by re-reading and re-stating existing result JSON conclusions
instead of independent re-derivation; softening or omitting a finding
that contradicts the final report's conclusions; downgrading a finding's
severity because fixing it is inconvenient.

### What a positive result means

("Positive" here means "the audited claim survives adversarial
scrutiny," not "a favorable scientific effect.") A claim classified
PASS survived independent re-derivation, seed re-randomization, and the
scene-identity attack, under the specific checks this task actually
ran.

### What a negative result means

A claim classified MAJOR or BLOCKER means a specific, documented
alternative explanation was found, or the original result did not
reproduce under independent re-derivation or a different seed — this
must propagate back to a correction or retraction in `reports/
final_research_report.md` (Global Invariant 27, with justification
recorded), not be left standing alongside a contradicting audit
finding.

### What this task does NOT establish

That no other confound exists beyond the ones actually audited; a
clean audit (all PASS) is evidence the specific attacks tried did not
break the conclusions, not proof the conclusions are correct in an
absolute sense.

---

## Validation performed on this document

- Confirmed every task number 6–18 is present, each with all 26
  required fields (Purpose, Scientific question, Why the task exists,
  Relationship to previous tasks, Hypothesis, Mathematical formulation,
  Inputs, Outputs, Experimental protocol, Dataset requirements,
  Train/test protocol, Controls, Baselines, Metrics, Required artifacts,
  Required software tests, Required scientific-validity tests, Required
  research-alignment checks, Required leakage checks, Reproducibility
  requirements, Failure conditions, Acceptance criteria, Interpretation
  rules, Explicit prohibited shortcuts, What a positive result means,
  What a negative result means, What the task DOES NOT establish).
- Confirmed all 30 Global Scientific Invariants are reproduced verbatim,
  numbered 1–30, matching the task brief exactly (not paraphrased).
- Confirmed the task-dependency chain (§4) lists all of Tasks 1–18 in
  order, with the one-line relationship statement for each of 6–18
  given verbatim in the task brief.
- Confirmed no experimental result, number, or outcome is asserted
  anywhere in this document — every task's content is a specification
  of what to do and how to judge it, never a claimed finding.
- Confirmed every concrete implementation detail (module names, class
  names, function signatures, config field names, existing baseline/
  metric implementations) was checked against the actual repository
  (`generation/`, `transforms/`, `encoders/`, `probes/`, `metrics/`,
  `baselines/`, `configs/`, `DESIGN.md`) rather than invented.
