# Final Research Report: Does a Frozen V-JEPA 2 Representation Encode 3D Geometric Structure?

**Task 16 of the Geometric Consistency Probe (GCP) research pipeline.** This report
synthesizes Tasks 6-15. It makes no new measurement — every number below is
read from an existing `state/task_NN_result.json` (or, for Task 7B, from
`state/task_07b_result.json` and `experiments/task7b_latent_transformation_discovery/final_report.md`,
included as directly relevant supplementary evidence to the camera-rotation
question Tasks 6/7/11 also test). Every quantitative claim below cites the
exact field it is drawn from. Numbers are quoted as they appear in each
result file's own `scientific_result` text (rounded to 3 decimals in most
source files) so they are directly string-matchable against the source.

---

## Abstract

Across ten completed tasks (6-15, plus the supplementary Task 7B), this
project tested whether a frozen, pretrained V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`,
ViT-L/16) video representation encodes 3D geometric structure in a way that
is linearly predictable, linearly accessible, temporally coherent, persistent
through occlusion, and specific to controlled counterfactual interventions.
At this project's primary scene-count regime (N=40 scenes, 32 train / 8
test), the results are **predominantly negative**: camera rotation (Task 6),
all six tested transforms including appearance-only controls (Task 7), a
5-baseline comparison (Task 10), a temporal-window transition (Task 12),
post-occlusion object-state recovery (Task 13), and counterfactual
intervention discrimination (Task 14) all failed to exceed their required
controls. Physical-state linear accessibility (Task 8) was **mixed**: 2 of 4
probed variables (camera elevation, camera distance) exceeded their
baselines; camera azimuth and object position did not. Appearance invariance
(Task 9) showed **no clean separation** between geometric transforms and
appearance-only controls — a finding that itself undercuts confidence in any
geometric interpretation of Tasks 6/7's numbers. The one clear positive
result, Task 7B (a much larger-scale, narrower-parameterized, preregistered
follow-up study restricted to camera rotation alone), found a sealed-test
R²=0.557 far above its controls (best control R²=0.157) — but this result is
explicitly **scale-limited** (the learning curve did not stabilize by
N_train=1024) and used a different transform parameterization (fixed pure
yaw only) and roughly 25x more training scenes than Tasks 6/7/11's floor.
No task in this pipeline supports, or claims, that the model "understands
3D" in any sense beyond the specific operational metrics reported here.

## Research question

*What, precisely, can and cannot be concluded about geometric consistency
and physical-state accessibility in this frozen encoder's representation,
across everything tested in Tasks 6-15?*
(`research/CANONICAL_RESEARCH_PROTOCOL.md` §"TASK 16", restating §1's central
question: "Does a World Model Actually Understand 3D?", decomposed into six
operationally-testable components — see "Supported vs. not-supported claims"
below.)

## Motivation

Nine-plus experiments (Tasks 6-14, unified by Task 15) each produce numbers,
not conclusions. This report exists to state exactly what was found, with
every claim traceable to a specific metric in a specific task result, and no
claim stronger than what §1's six operationally-testable components actually
support (`tasks/16_report.md`).

---

## Experimental setup

### Controlled 3D environment

All scenes are procedurally sampled `SceneState`/`ObjectState`/`CameraState`/`LightState`
objects (`generation/scene.py`, `generation/scene_sampler.py`), 1-3 objects
per scene, seeded per `(base_seed, scene index)`. Every task in this report
uses `base_seed=0` and either 40 scenes (Tasks 6-10, 12-14) or a documented
scale ladder up to 100 scenes (Task 11) or a fresh 1463-scene population
(Task 7B).

### Rendering

100% Blender/Cycles (`bpy`, `generation/bpy_renderer.py`), producing RGB (+
depth/segmentation ground truth where used), 128px resolution, 4 frames at
4 fps for the static-pair tasks (6-10, 13-14), and multi-frame trajectories
(`generation/motion.py`, `render_trajectory`) for Task 12's continuous
camera-orbit motion.

### Representation extraction

The frozen encoder is `encoders.vjepa.VJEPAEncoder`, checkpoint
`facebook/vjepa2-vitl-fpc64-256`, loaded through `transformers.VJEPA2Model`
directly (`encoder.frozen: true`, verified per `tests/test_vjepa_encoder.py`
in every task). Every task in this report used `pretrained: true` (real
weights) except where a task's own result explicitly records a matched-
architecture untrained variant as a *baseline* (Task 10's `random_encoder`).
The native `(num_tokens, 1024)` unpooled representation is reduced to a
single `(1024,)` vector per clip via `mean_pool` — every task in this report
operates on this mean-pooled representation; no task performed a token-level
analysis (explicitly out of scope, see Task 7B's "token-level boundary").

### Geometric transformations

`transforms.scene_transform.apply_transform` applies one of
`GEOMETRIC_TRANSFORMS = (camera_translation, camera_rotation,
object_translation, object_rotation)` or `CONTROL_TRANSFORMS =
(lighting_change, texture_change)`, returning an explicit 4x4 transform
matrix (or `null` for the two appearance-only controls) plus
`changed_variables`/`fixed_variables` ground truth. `Z' ≈ W_T Z + b_T` is fit
by ridge regression (`probes.linear_rep_transform.LinearRepTransform`, α=10.0
throughout Tasks 6-14) on **train scenes only** and evaluated on held-out
**test scenes only**.

### Physical-state probes

`generation.state_features.{camera_azimuth_deg, camera_elevation_deg,
camera_distance, primary_object_position_xy}` are probed via a linear ridge
regression (`probes.linear_state_probe`) fit on train scenes, evaluated on
held-out test scenes.

### Appearance controls

`lighting_change`/`texture_change` are non-rigid, appearance-only
interventions (`transform_matrix: null`); Task 9 measures `Z` vs. `Z'`
directly (`metrics.invariance.evaluate_invariance`, no map fit), including a
required null-transform (`T=identity`) sanity check.

### Baselines

Task 10 formalizes DESIGN.md §12's full five-baseline set: persistence
(`Z_hat'=Z`), mean-transformed-representation (`Z_hat'=mean(Z'_train)`),
shuffled-pairing (random-pair control), a non-learned pixel-statistics
encoder (`encoders/pixel_baseline.py`), and a matched-architecture
randomly-initialized encoder (`VJEPAEncoder(pretrained=False)`). All five
are evaluated under the identical protocol as the primary representation.

### Scale experiments

Task 11 reruns the flagship transform (`camera_rotation`) at N ∈ {15, 40,
100} scenes, 2 seeds per point, holding the protocol otherwise fixed.

### Temporal experiments

Task 12 uses `generation.motion.generate_trajectory` +
`generation.bpy_renderer.render_trajectory` to render a continuous
camera-orbit clip per scene, splits it into two temporal windows, and fits
`Z_{t2} ≈ W_T Z_{t1} + b_T` — plus a required static-clip control
(`camera_motion=None`).

### Occlusion experiments

Task 13 constructs scenes where a tracked object (by
`generation.scene.ObjectState.instance_id`) becomes occluded for a
documented sub-sequence of frames, **verified** via segmentation-mask pixel
counts, then reappears; a linear probe predicts post-occlusion object state
from the post-occlusion representation window, compared against a
scrambled-identity control and a no-occlusion upper bound.

### Counterfactual experiments

Task 14 compares two magnitude-matched counterfactual interventions
(`object_translation` vs. `object_rotation`, both scaled to 0.4 scene
units of SE(3) displacement) via a discrimination probe on
`(Z_before, Z_after)` pairs, against a label-scrambled control, a
pre-declared chance threshold, a representation-norm-only control (required
to rule out trivial magnitude separability), and a fixed-variable leakage
check.

---

## Results

### Headline results table (every result with its baseline comparison, per Global Invariant 19)

| Task | Experiment (N, flagship transform where applicable) | Primary metric | Best applicable baseline/control | Exceeds baseline? |
|---|---|---|---|---|
| 6 | `camera_rotation` equivariance, N=40 (`state/task_06_result.json`) | learned `W_T` R²=**-0.313** | mean-baseline R²=**-0.107** (best of 3) | **No** |
| 7 | 6-transform equivariance, N=40, same split (`state/task_07_result.json`) | 0/4 geometric, 0/2 appearance beat best baseline | camera_translation best control R²=0.638; camera_rotation 0.079; object_translation 0.807; object_rotation 0.385; lighting_change 0.467; texture_change 0.870 | **No (0/6)** |
| 7B | `camera_rotation` (yaw-only) equivariance, N_train≤1024 (`state/task_07b_result.json`, `experiments/task7b_latent_transformation_discovery/final_report.md`) | sealed test R²=**0.5571** | persistence R²=**0.157** (best control) | **Yes** (+0.400 margin), but **scale-limited** (learning curve `NOT_STABILIZED`) |
| 8 | physical-state linear probes, N=40, 4 variables (`state/task_08_result.json`) | 2/4 variables exceed baseline | camera_elevation_deg: real R²=0.227 vs. best baseline R²=-0.378; camera_distance: real R²=0.349 vs. best baseline R²=0.253 | **Mixed (2/4)** |
| 9 | appearance vs. geometric invariance, N=40, no fit (`state/task_09_result.json`) | mean cosine similarity 0.948-0.991 across 6 transforms | null-transform sanity check: cosine=1.000000 (PASSED) | **No clean separation** (geometric `camera_rotation` is the *least* invariant of all 6) |
| 10 | 5-baseline flagship comparison, `camera_rotation`, N=40 (`state/task_10_result.json`) | learned `W_T` R²=**-0.3130** | random_encoder R²=**0.7809** (best of 5) | **No** (below 4/5 baselines) |
| 11 | scale ladder, `camera_rotation`, N=15/40/100 (`state/task_11_result.json`) | R²=-0.0504 / -0.1626 / -0.0204 | random_encoder_baseline R²≈0.75/0.67/0.71 | **No at any point**; margin trend classified **"strengthens"** (-0.798→-0.836→-0.730) but stays deeply negative |
| 12 | temporal window transition, camera-orbit motion (`state/task_12_result.json`) | real-motion R²=**0.306** | static-clip control R²=**0.609**; best of 3 baselines R²=0.383 | **No** |
| 13 | post-occlusion state recovery, 2 variables (`state/task_13_result.json`) | real probe R²=**-0.493** | scrambled-identity control R²=**-0.035** | **No (0/2)**; no-occlusion upper bound (R²=-0.637) also below baseline |
| 14 | counterfactual discrimination, `object_translation` vs. `object_rotation` (`state/task_14_result.json`) | discrimination accuracy=**0.625** | pre-declared chance threshold=**0.750** | **No** (below chance) **and invalidated** (norm-only control=0.938 exceeded chance) |
| 15 | schema unification (infrastructure, `state/task_15_result.json`) | n/a — no new science | n/a | n/a |

### Per-task narrative

**Task 6** (`state/task_06_result.json`): "On held-out test scenes, the
linear map W_T fit on train scenes achieved R^2=-0.313 predicting the frozen
encoder's mean-pooled representation after a fixed-magnitude camera
rotation, which did not exceed the best of the three required controls
(persistence R^2=-0.128, mean-transformed-representation R^2=-0.107,
shuffled-pairing R^2=-0.860)." A valid negative result (269/269 tests
passed).

**Task 7** (`state/task_07_result.json`): running the identical protocol
across all four geometric transforms and both appearance controls on the
same 40-scene, 32/8 split found "0/4 geometric transforms showed the learned
W_T exceeding the best of the three baselines on held-out test scenes
(none), and 0/2 appearance-only controls did the same (none)." An explicit
confound check (Pearson correlation between each transform's mean absolute
pixel change and its learned R², across the six transforms) found r=-0.226
— weak and in the wrong direction to explain the R² pattern by pixel-change
magnitude alone (`confound_analysis.pearson_correlation_pixel_diff_vs_learned_r2`).
364/364 tests passed.

**Task 7B** (`state/task_07b_result.json`, full detail in
`experiments/task7b_latent_transformation_discovery/final_report.md`): a
preregistered, independent follow-up restricted to `camera_rotation` at
fixed elevation 0.0° (pure yaw), using a fresh 1463-scene population (never
Task 7's 40) split 1024/219/220 by scene. A model-selection hierarchy (M0
persistence through M6 residual MLP) selected **M4 (full linear map)** at
N_train=1024. The sealed test (run exactly once, on the untouched 220-scene
split) found **R²=0.5571** (95% bootstrap CI [0.528, 0.580]), versus
train-mean R²=-0.007, persistence R²=0.157, shuffled-pair R²=-0.137 — a
margin of **+0.400** over the best control. However, the N_train ∈
{64,128,256,512,1024} learning curve rose monotonically (mean val R²:
0.329→0.406→0.470→0.524→0.562) with **no sign of flattening**, so per the
preregistered stopping rule this is classified `COMPLETE_SCALE_LIMITED`
(status field), not `COMPLETE` — no claim is made about R² beyond N=1024.
Structure tests (composition/inverse/identity) showed output-space
predictions above control levels (R² 0.44-0.92) but the literal fitted
matrices did **not** closely satisfy the corresponding exact algebraic
identities (normalized Frobenius discrepancy 0.65-1.003, i.e. as large as or
larger than the target operator itself) — the calibrated claim recorded is
"predictions are consistent with compositional structure at the output
level under tested conditions, without close agreement at the operator
level," not "approximate compositional latent transformation family."

**Task 8** (`state/task_08_result.json`): of 4 physical-state variables
probed, "2/4 variables showed the real probe's held-out R^2 exceeding the
best of the required shuffled-label and mean-prediction baselines
(camera_elevation_deg, camera_distance)." Per-variable: `camera_azimuth_deg`
(sin/cos encoding) real R²=-0.781 vs. best baseline R²=-0.036 (not
exceeded); `camera_elevation_deg` real R²=0.227 vs. best baseline R²=-0.378
(exceeded); `camera_distance` real R²=0.349 vs. best baseline R²=0.253
(exceeded, margin ≈0.10); `primary_object_position_xy` real R²=-1.653 vs.
best baseline R²=-0.204 (not exceeded). A scene/render alignment check
confirmed the probed labels and the encoded clips describe the identical
physical configuration (`scene_render_alignment_check.max_camera_position_abs_diff=0.0`).
578/578 tests passed.

**Task 9** (`state/task_09_result.json`): the null-transform sanity check
(same scene rendered twice, `T=identity`) scored cosine=1.000000,
relative-L2=0.000000, passing its pre-declared threshold (cosine≥0.999,
rel_l2≤0.05), confirming the pipeline is deterministic. But across the six
transforms, invariance did **not** cleanly separate geometric from
appearance-only: `lighting_change` (appearance) cosine=0.9736,
`texture_change` (appearance) cosine=0.9915, `camera_translation`
(geometric) cosine=0.9761, `camera_rotation` (geometric) cosine=**0.9481**
(the *lowest* of all six — i.e. camera rotation moves `Z` *more* than either
appearance control), `object_translation` (geometric) cosine=0.9871,
`object_rotation` (geometric) cosine=0.9637 (also below `lighting_change`).
"This is a meaningful negative finding that directly undercuts confidence in
Tasks 6-7's geometric interpretation" (quoted directly from
`scientific_result`).

**Task 10** (`state/task_10_result.json`): the full five-baseline
comparison on the flagship transform found the primary pretrained encoder's
`W_T` R²=-0.3130, below persistence (-0.1281), mean (-0.1073),
pixel_statistics (0.2212), and random_encoder (0.7809, the best of the
five). Numerical parity with Task 6's previously-recorded numbers was
confirmed to within 8.34e-07 (`numerical_parity_with_task6`), and the
refactor changed no historical result. The task's own `scientific_qa`
field records `pixel_statistics_comparable_to_primary: true` and
`random_encoder_comparable_to_primary: true`, using a declared "within 0.05
R²" threshold — but the raw R² gap between the primary encoder (-0.3130)
and the pixel-statistics baseline (0.2212) is ≈0.53, which is **not** within
0.05 by direct arithmetic on the numbers in the same result file. This
internal inconsistency is flagged here (see "Failure analysis" below) rather
than silently repeated or resolved by this report. The directionally
unambiguous part of the finding — that a non-learned pixel-statistics
representation and an untrained, matched-architecture encoder both scored
*higher* R² than the pretrained encoder on this transform — stands regardless
of how "comparable" is defined, and is itself a major negative finding
against any claim that Tasks 6/7's signal on this transform is specific to
pretraining.

**Task 11** (`state/task_11_result.json`): the flagship transform run at
N ∈ {15, 40, 100} (2 seeds each) found learned `W_T` mean R²=-0.0504 (N=15,
std=0.1243), -0.1626 (N=40, std=0.0575), -0.0204 (N=100, std=0.0454) —
against a best baseline (`random_encoder_baseline`) of R²≈0.75, 0.67, 0.71
respectively. The equivariance margin (learned minus best baseline) across
the ladder is [-0.798, -0.836, -0.730], classified `"strengthens"` under a
pre-declared ±0.03 R² tolerance — but the learned map's absolute R² never
approaches, let alone exceeds, zero, let alone the baseline, at any tested
point. The task's own interpretation-limits note this is "a trend across 3
scale points on ONE synthetic scene distribution and ONE transform... not an
asymptotic scaling law."

**Task 12** (`state/task_12_result.json`): on the real-motion (camera-orbit)
condition, the temporal-window map achieved R²=0.306, which "did not exceed
the best of the three Task-6-style controls (R^2=0.383; margin=-0.077) and
did not exceed the static-clip control (R^2=0.609; margin=-0.303)." The
required static-clip control (ruling out that temporal windowing alone,
without real motion, produces spurious consistency) scored *higher* than
the real-motion condition — meaning temporal adjacency/positional structure
alone predicts the representation transition better than real motion does
under this windowing scheme. 677/677 tests passed.

**Task 13** (`state/task_13_result.json`): 0 of 2 post-occlusion variables
(`post_occlusion_position_x`, `post_occlusion_velocity_x`) exceeded the
object-identity-scrambled control (real R²=-0.493 vs. scrambled R²=-0.035
for both variables). Critically, the required no-occlusion upper-bound
control **also** scored below the mean baseline (R²=-0.637), so — per the
task's own recorded caveat — "this run's negative result should be read as
'no measurable linear recoverability of this state parameterization from
mean-pooled Z under this protocol at all' rather than as 'occlusion
specifically destroys otherwise-recoverable information' — the two are not
distinguishable from this result alone."

**Task 14** (`state/task_14_result.json`): the discrimination probe
(`object_translation` vs. `object_rotation`, magnitude-matched to 0.4 scene
units per scene — confirmed: object_translation mean displacement=0.4000,
object_rotation mean=0.4000, mean |difference|=0.000000) scored accuracy=0.625
against a pre-declared chance threshold of 0.750 — already a negative
result. The task's own required representation-norm-only control (checking
whether the two conditions are trivially separable by `‖Z_after -
Z_before‖` alone, independent of any intervention-specific structure) scored
0.938, **exceeding** the chance threshold — meaning the two conditions *are*
trivially separable by magnitude alone. Per the task's design requirement,
this **invalidates** interpreting any discrimination signal as
intervention-specific; the result is recorded as "NEGATIVE/INVALIDATED." The
separately-required fixed-variable leakage check (probing whether an
unrelated variable pair predicts the real condition) passed at
accuracy=0.562 (below chance), finding no such confound. 723/723 tests
passed.

**Task 15** (`state/task_15_result.json`): a pure infrastructure/schema-
unification task. It validated all nine prior result files against a new
shared schema, found zero validation errors and zero required migrations
(files byte-identical before/after, sha256-verified), and built one CLI
dispatcher over Tasks 6-14's existing experiment code (`metrics.num_prior_results_validated=9`,
`metrics.num_tasks_dispatchable_through_unified_cli=9`; 808/808 of its own
tests passed, `tests.passed=808`). It makes, and supports, no new
scientific claim.

### Reconciling conflicting results (required per this task's protocol)

The starkest conflict in this pipeline is **camera rotation's equivariance
signal across scale and parameterization**: Task 6 (N=40, azimuth fixed at
30°, elevation range ±10°) found R²=-0.313; Task 7 (same scene/split,
azimuth+elevation *ranges* rather than a fixed magnitude) found R²=-0.176;
Task 11's scale ladder up to N=100 (same protocol as Task 6) stayed
negative throughout (R² from -0.050 to -0.163 to -0.020); but Task 7B — a
different, much larger (N_train up to 1024), narrower (pure yaw only, no
elevation variation), and independently preregistered study of the *same*
named transform — found R²=0.557, far above its own controls. These are
**not the same experiment**: Task 7B changed scene count by roughly 25x
*and* narrowed the transform's degrees of freedom simultaneously, so scale
and parameterization are confounded in this comparison and cannot be
separately attributed from the evidence available. This report does not
pick the more favorable number to headline: it states both, together with
their protocol differences, as directly conflicting evidence about the same
named transform under different experimental conditions. Task 11's
finding that the (negative) margin "strengthens" from N=15 to N=100 is
*directionally* consistent with Task 7B's much larger positive result at
N_train=1024, but Task 11's own absolute numbers never leave deeply negative
territory within the range it tested (N≤100), so Task 11 alone provides only
weak, inconclusive support for extrapolating toward Task 7B's regime — the
extrapolation is not made here.

A second conflict: Task 8 found camera-related physical state (elevation,
distance) *is* linearly accessible above baseline, while Task 6/7 found
camera rotation's *effect on the representation* is not linearly
predictable above baseline. These are DESIGN.md §13's two explicitly
distinct claims (accessibility of absolute state vs. predictability of a
transform's effect) and are not required to agree — this is exactly the
"related but distinct... itself scientifically informative" combination
`research/CANONICAL_RESEARCH_PROTOCOL.md`'s Task 8 section anticipates.

A third conflict: Task 9 found geometric transforms are *not* more
representation-changing than appearance-only controls (in fact
`camera_rotation`, geometric, is the single least-invariant transform
tested) — this directly undercuts, rather than supports, any reading of
Task 6/7's numbers as reflecting geometry-specific sensitivity, and is
reported here as a negative finding that weakens confidence in the
geometric framing of the whole pipeline, not minimized.

---

## Negative results (collected explicitly, not omitted)

- Task 6: camera rotation's effect on `Z` is not linearly predictable above
  any of 3 controls at N=40 (R²=-0.313, best control -0.107).
- Task 7: 0/6 transforms (4 geometric + 2 appearance) beat their best
  control; no clean geometric/appearance separation.
- Task 8: 2/4 physical-state variables (camera azimuth, object xy-position)
  are not linearly accessible above baseline.
- Task 9: geometric transforms are not more representation-changing than
  appearance controls; `camera_rotation` is the least-invariant transform of
  all six tested.
- Task 10: the pretrained encoder's flagship-transform `W_T` does not beat
  4 of 5 baselines, including a non-learned pixel-statistics encoder and an
  untrained matched-architecture encoder.
- Task 11: the flagship transform's `W_T` stays deeply negative R² at every
  tested scale point (N=15, 40, 100), despite a "strengthening" margin
  trend.
- Task 12: real camera-orbit motion's temporal-window transition is *less*
  predictable than the static-clip control (R²=0.306 vs. 0.609).
- Task 13: post-occlusion object state is not more recoverable than a
  scrambled-identity control for either variable tested; the no-occlusion
  upper bound also failed to beat the mean baseline.
- Task 14: counterfactual discrimination accuracy (0.625) is below the
  pre-declared chance threshold (0.750), and the required norm-only control
  shows the two conditions are trivially separable by magnitude alone,
  invalidating any interpretation as intervention-specific structure.

None of these are software failures — every task's own test suite passed
(see the per-task test counts above and the "Reproducibility" section), and
every negative result above was produced by an implementation that also
passed its own leakage/scientific-validity checks. Per Global Invariants 16
and 17 (`research/RESEARCH_INVARIANTS.md`), these are valid, complete
scientific outcomes, not deficiencies to be corrected.

## Failure analysis (distinguishing software failure from scientific negative result, per §5)

No task in 6-15 was reclassified from a scientific negative result to a
software failure: every one reports `implementation_status: "COMPLETE"`
with 0 failed tests in its own `tests` field, and each negative result above
was reached through a protocol whose own leakage/scientific-validity checks
passed (scene-level splits verified disjoint in every task; frozen-encoder
checks passed in every task using the encoder; Task 14's own required
norm-confound check is a control *working as designed*, not a defect).

One genuine anomaly was found during this synthesis and is recorded here,
not silently repeated: **Task 10's `scientific_qa.pixel_statistics_comparable_to_primary`
field is `true` under a declared "within 0.05 R²" threshold, but the two
R² values it compares (pixel_statistics=0.2212, primary encoder=-0.3130,
`state/task_10_result.json`'s own `metrics` block) differ by ≈0.53, not
≤0.05.** This report does not modify Task 10's file (a completed, protected
task result) or attempt to silently correct or re-derive the "comparable"
determination. The underlying, unambiguous numbers (which are internally
consistent, and which this discrepancy does not call into question) still
support the finding stated in "Results" above: both non-learned baselines
score a *higher* R² than the pretrained encoder on this transform. The
labeling/threshold-application discrepancy itself is exactly the kind of
finding Task 18's adversarial audit (`research/RESEARCH_INVARIANTS.md`
invariant 18; Global Invariant 30) exists to catch, and is flagged here for
that purpose.

## Alternative explanations (required scientific-QA step, per Tasks 7/9/13/14)

- **Pixel-magnitude confound (Task 7)**: could transform-to-transform
  differences in learned R² simply reflect which transform perturbs more
  pixels? Task 7's own confound check found a weak, wrong-signed Pearson
  correlation (r=-0.226) between mean pixel difference and learned R²
  across the six transforms — this alternative explanation is not
  supported by the data, though the check is weak (6 data points).
- **Scale vs. parameterization confound (Task 7B vs. Task 6/7/11)**: Task
  7B's positive result could be attributable to its ~25x larger training
  set, to its narrower (pure-yaw-only) transform definition, or to both
  simultaneously — the study design does not allow these to be separated,
  and no claim attributing the effect to one or the other is made here.
- **Pretraining-specificity confound (Task 10)**: an untrained,
  matched-architecture encoder and a non-learned pixel-statistics
  representation both score higher R² than the pretrained encoder on the
  flagship transform — an alternative explanation for any apparent
  geometric-consistency signal on this transform is that it reflects
  architecture and/or low-level pixel statistics rather than anything
  learned through pretraining specifically.
- **State-parameterization vs. occlusion confound (Task 13)**: because the
  no-occlusion upper-bound control also failed to beat the mean baseline,
  an equally consistent alternative explanation for Task 13's null result
  is that the probed state parameterization (narrow per-scene label range,
  small held-out test set) is not linearly recoverable at all under this
  protocol, independent of occlusion.
- **Trivial-magnitude confound (Task 14)**: the representation-norm-only
  control positively identifies this alternative explanation for any
  apparent discrimination signal (the two conditions are separable by
  `‖ΔZ‖` alone) — this is not a hypothetical alternative explanation but a
  measured, confirmed one, which is exactly why the task's own result is
  recorded as invalidated rather than positive.
- **Temporal-position confound (Task 12)**: the static-clip control scoring
  *higher* R² than the real-motion condition suggests that whatever
  predictability exists between temporal windows in this protocol may be
  driven by positional/temporal-encoding structure in the encoder rather
  than by genuine motion tracking.

## Limitations (project-wide, inherited from every task's own interpretation-limits section)

- **Scale**: the great majority of results (Tasks 6-10, 12-14) are measured
  on 40 scenes (32 train / 8 test); Task 11 extends to 100; only Task 7B
  reaches N_train≈1024, and only for one transform in one narrow
  parameterization, and even then without a stabilized learning curve.
- **Single synthetic scene distribution**: one procedural generator, 1-3
  objects per scene, one renderer (Blender/Cycles); no claim about real
  (non-synthetic) video is supported anywhere in this pipeline.
- **One encoder, one pooling scheme**: `facebook/vjepa2-vitl-fpc64-256`,
  frozen, mean-pooled only. No task performed a token-level (unpooled)
  analysis; Task 7B explicitly reserves that as out-of-scope future work.
- **Linear probes/maps only**: per Global Invariant 13, no nonlinear probe
  was substituted for a null linear result anywhere in this pipeline. A
  negative linear result is evidence of non-*linear*-accessibility only,
  never evidence the quantity is absent from `Z` in some other form
  (DESIGN.md §13).
- **Transform-magnitude ranges as tested**: results do not generalize to
  magnitudes, axes, or transform types outside what was actually rendered
  in each task.
- **Comparative, not causal**: every positive or negative finding above is a
  statistical property of a fitted linear model relative to specific
  baselines under a specific protocol — none of it is evidence about the
  encoder's internal computation, mechanism, or "understanding."
- **Ridge `alpha` fixed, not tuned per task/transform** (α=10.0 throughout
  Tasks 6-14, α=100.0 for Task 7B's selected model) — never chosen by
  looking at test-split performance (verified per task), but also never
  swept, so no task's negative result rules out that a different
  regularization strength would perform differently.

## Conclusions

Restated against §1's six operationally-testable components
(`research/CANONICAL_RESEARCH_PROTOCOL.md` §1), strictly bounded to what was
actually measured:

1. **Information present in `Z`** — not directly, separately tested by any
   task in this pipeline; Task 10's finding that a non-learned
   pixel-statistics baseline and an untrained encoder both score
   competitively on the flagship transform is suggestive that *some*
   information relevant to this specific transform/metric is present even
   without learned pretraining, but this is not a general "information
   present" test.
2. **Linear accessibility** — mixed (Task 8): supported for camera
   elevation and camera distance; not supported for camera azimuth or
   object xy-position, under this scene distribution and probe design.
3. **Consistently represented geometric transformations** — not supported
   at this project's primary scene-count floor (Tasks 6, 7, 11, all
   N≤100: negative across all six transforms and all tested scales);
   supported, with an explicit scale-limited caveat and a narrower
   transform definition, by the independent Task 7B study (N_train≤1024,
   camera rotation/pure-yaw only). These two bodies of evidence describe
   different experimental conditions and are reported together, not
   reconciled into one number.
4. **Temporal dynamics** — not supported (Task 12): the real-motion
   condition did not exceed its own required static-clip control.
5. **Object identity/state persistence under occlusion** — not supported
   for either variable tested (Task 13); the negative result is itself
   only partially interpretable, since the no-occlusion upper bound also
   failed, so this pipeline cannot distinguish "occlusion destroys
   information" from "this state parameterization is not linearly
   recoverable at all" from the evidence gathered.
6. **Counterfactual intervention structure** — not supported (Task 14):
   discrimination accuracy was below the pre-declared chance threshold, and
   the required control confirmed the two tested conditions are separable
   by representation-magnitude alone, invalidating any interpretation of a
   positive signal as intervention-specific.
7. **"Understands 3D"** — never claimed, anywhere in this pipeline, and not
   supported by any combination of the above. Per Global Invariant 30, this
   report does not collapse 1-6 into 7.

## Future work

- A full six-transform matrix rerun at Task 7B's scale (N_train≈1000+),
  not just `camera_rotation`, to separate the scale confound from the
  transform-parameterization confound identified above.
- A token-level (unpooled) follow-up study, explicitly reserved by Task 7B
  and never attempted by any task in this pipeline.
- Re-run Task 13 with a wider per-scene label range and/or larger held-out
  test set to determine whether its null result reflects occlusion
  specifically or general non-recoverability of the probed state
  parameterization.
- Investigate and resolve the Task 10 `scientific_qa` threshold-application
  discrepancy identified in "Failure analysis" above.
- Extend Task 14 with a broader magnitude ladder or an explicit
  magnitude-conditioning step to avoid the trivial-norm confound its own
  required control identified.
- Extend Task 12 to additional motion types and windowing schemes beyond
  the single camera-orbit condition tested.
- A ridge-`alpha` sweep (reported honestly, not tuned against test-split
  performance) to check whether any task's negative result is sensitive to
  regularization strength.

---

## Supported vs. not-supported claims (mirroring DESIGN.md §13's structure)

**Can support, for the specific encoder, pooling scheme, scene
distribution, and transform-parameter ranges actually tested in Tasks
6-15/7B:**

- Camera elevation and camera distance are linearly accessible from `Z`
  above baseline, at N=40 (Task 8).
- Under a large-scale (N_train≤1024), narrow (pure-yaw), preregistered
  protocol, `camera_rotation`'s effect on `Z` is linearly predictable and
  generalizes to held-out scenes, far above controls — with an explicit
  scale-limited caveat (Task 7B).
- Comparative statements: the pretrained encoder is *not* more predictable
  than a non-learned pixel-statistics representation or an untrained
  matched-architecture encoder, on the flagship transform at N=40 (Task
  10).
- A deterministic rendering+encoding pipeline (Task 9's null-transform
  sanity check).

**Cannot support, under any result in this pipeline:**

- Any claim that the model "understands 3D," has an internal camera model,
  or performs geometric reasoning — never made, never supported, by any of
  Tasks 6-15 or 7B.
- That camera rotation's effect on `Z` is linearly predictable at this
  project's primary scene-count floor (N≤100) — directly contradicted by
  Tasks 6, 7, and 11.
- That geometric transforms are more representation-changing than
  appearance-only controls — directly contradicted by Task 9.
- That physical camera azimuth or object position are linearly accessible
  from `Z` — not supported by Task 8.
- That temporal dynamics, object persistence under occlusion, or
  counterfactual intervention specificity are supported by this
  representation under any tested protocol — not supported by Tasks 12, 13,
  14 respectively.
- Any generalization beyond the tested encoder, pooling scheme, scene
  distribution, or transform-magnitude ranges (DESIGN.md §13).
- Absence of a capability from any negative *linear*-probe/map result —
  per DESIGN.md §13 and Hewitt & Liang/Belinkov, a null linear result is
  evidence of non-linear-inaccessibility only, never evidence the quantity
  is wholly absent from `Z`.

---

## Claim-evidence matrix

| # | CLAIM | EXPERIMENT | METRIC | CONTROL | ACTUAL RESULT | LIMITATION |
|---|---|---|---|---|---|---|
| 1 | Camera rotation induces a linearly predictable representation change generalizing to unseen scenes, at N=40. | Task 6 | held-out R² of `W_T` | persistence, mean, shuffled-pair | R²=-0.313, below all 3 controls (best -0.107) | Single fixed 30° magnitude; N=40; no claim beyond this regime. |
| 2 | This effect extends across geometric transforms as a class, and distinguishes geometry from appearance-only change. | Task 7 | per-transform held-out R² vs. 3 controls, 6 transforms | persistence/mean/shuffled-pair per transform | 0/6 transforms beat best baseline; no clean geometric/appearance separation | Single scene set/split; ranged (not fixed) magnitudes per transform. |
| 3 | At larger scale and a narrower transform definition, camera rotation *is* linearly predictable, generalizing to held-out scenes. | Task 7B | sealed multi-output R² on 220 held-out test scenes | persistence, train-mean, shuffled-pair | R²=0.557 (CI [0.528,0.580]) vs. best control 0.157; learning curve NOT stabilized by N_train=1024 | Scale-limited, no asymptotic claim; yaw-only; operator-level composition/inverse identities not closely satisfied. |
| 4 | Scaling from ~15 to ~100 scenes (Task 6's exact protocol) shows the equivariance margin strengthening. | Task 11 | R² margin (learned − best baseline) across N=15,40,100 | full Task 10 baseline set at every point | margin -0.798→-0.836→-0.730 ("strengthens" by pre-declared ±0.03 tolerance); absolute R² stays deeply negative throughout | Only 3 points to N=100 (not Task 7B's N≈1000); one transform only. |
| 5 | Physical camera/object state is linearly decodable from `Z`. | Task 8 | held-out probe R² per variable vs. shuffled-label/mean baseline | shuffled-label, mean-prediction | 2/4 variables exceed baseline (elevation 0.227, distance 0.349); azimuth and xy-position do not | Accessibility ≠ representation; N=40; camera_distance margin thin (≈0.10). |
| 6 | Appearance-only changes leave `Z` comparatively unchanged relative to geometric transforms. | Task 9 | direct cosine similarity/relative L2 (no fit); null-transform sanity check | null-transform (T=identity) | Null check passed (cosine=1.0); no clean ordering — geometric `camera_rotation` is *least* invariant of all 6 | Only 2 appearance factors tested; undercuts, does not support, Tasks 6/7's geometric framing. |
| 7 | The pretrained encoder's advantage over trivial baselines reflects pretraining/learned structure specifically. | Task 10 | 5-baseline R² comparison on flagship transform | persistence, mean, shuffled-pair, pixel-statistics, random-init encoder | Learned R²=-0.313, below 4/5 baselines (pixel-stats 0.221, random-encoder 0.781, the best) | Flagship transform only; Task 10's own "comparable" (≤0.05 R²) flag is numerically inconsistent with its own reported values (see "Failure analysis"), flagged not resolved. |
| 8 | Representation transitions across continuous camera-orbit motion are linearly predictable beyond a static-clip control. | Task 12 | held-out R² between temporal windows | static-clip control; 3 Task-6-style baselines | Real-motion R²=0.306, below static-clip control (0.609) and below best baseline (0.383) | One motion type/windowing scheme; does not rule out other windowings. |
| 9 | Object-level state persists through occlusion, linearly recoverable beyond a scrambled-identity control. | Task 13 | held-out probe R²: real vs. scrambled-identity vs. no-occlusion upper bound | scrambled-identity, no-occlusion upper bound, mean baseline | 0/2 variables beat scrambled control (real -0.493 vs. scrambled -0.035); no-occlusion upper bound (-0.637) also below baseline | Cannot distinguish "occlusion destroys information" from "state parameterization not linearly recoverable at all." |
| 10 | The representation distinguishes two magnitude-matched counterfactual interventions. | Task 14 | discrimination accuracy vs. scrambled-label control and pre-declared chance threshold; norm-only control | label-scrambled, chance=0.750, representation-norm-only | accuracy=0.625, below chance; norm-only control=0.938 exceeded chance → **invalidated** | Negative *and* invalidated by its own required confound check; no evidence of intervention-specific structure. |
| 11 | Tasks 6-15 together support "V-JEPA 2 understands 3D." | — (no experiment tests this) | — | — | Not claimed by, and not supported by, any task in this pipeline | This row documents the claim this report explicitly does **not** make (Global Invariant 30). |

---

## Reproducibility

Every task above recorded `seed`, `config`, and `software_versions`
(`python: 3.11.15`, `torch: 2.14.0+cu130`, `transformers: 5.17.0`,
consistent across Tasks 6-15) in its own result JSON; `base_seed=0`
throughout Tasks 6-14, and Task 7B's own independent protocol/seed schedule
recorded in `experiments/task7b_latent_transformation_discovery/protocol.json`.
Task 10 additionally re-verified Task 6's recorded numbers to within
8.34e-07 after its baseline-framework refactor. This report cites no number
that is not present, verbatim or via a direct arithmetic combination stated
above (e.g. a margin computed as one recorded value minus another), in its
source `state/task_NN_result.json`.
