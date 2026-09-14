# Task 12 — Temporal geometric consistency

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 12 — TEMPORAL GEOMETRIC CONSISTENCY" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Extend the static geometric consistency framework into temporal
dynamics.

**Why this task exists.** Tasks 6–11 all operate on a single discrete before/after transform pair
per scene (a static-clip or single-transform-application design,
explicitly noted in DESIGN.md §13's "cannot support" list as "any claim
about ... temporal dynamics beyond a single before/after pair"). Task
12 exists to close exactly that gap, using `generation.motion`'s
already-implemented real multi-frame trajectory rendering (orbit
camera motion, constant-velocity object motion) that Tasks 6–11
deliberately did not need.

**Relationship to previous tasks.** Uses `generation.motion.generate_trajectory`
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

## Scientific question

**Scientific question.** Does representation change over time in a way that corresponds to
controlled physical changes in the underlying world?

**Hypothesis.** For a clip with real, continuous motion (camera orbiting, or an object
moving at constant velocity), the representation at one temporal window
relates to the representation at another temporal window through a
transform that is predictable from the known relative
camera/object displacement between the two windows, generalizing across
scenes the same way Task 6 tested for a single discrete transform.

**Mathematical formulation.** For temporal windows `t1`, `t2` within one clip, with known relative
transform `T_{t1→t2}` derived from the trajectory's recorded per-frame
poses:

```
Z_{t1} = E(V[window t1]),   Z_{t2} = E(V[window t2])
Z_{t2} ≈ W_{T} Z_{t1} + b_T      (fit on TRAIN scenes only)
```

structurally identical to Task 6's formulation, with `T` now derived
from continuous motion instead of a single `apply_transform` call.

## Implementation requirements

**Inputs.** Clips rendered with a non-trivial `camera_motion` (mode `"orbit"`) or
`object_motions` via `generation.motion.generate_trajectory` +
`generation.bpy_renderer.render_trajectory`; at least two distinct
temporal windows per clip (e.g. frames 0–3 vs. frames 4–7, or two
overlapping windows — the exact windowing scheme decided and documented
explicitly, not left implicit); the ground-truth relative
camera/object transform between the two windows, read from the
trajectory's per-frame pose records (reused, not re-derived).

**Outputs.** `state/task_12_result.json`, following Task 6's schema shape, with an
added `motion_config` field (exact `ObjectMotion`/`CameraMotion`
parameters used) and a `static_clip_control` metrics block.

**Experimental protocol.** 1. Render clips with real motion (orbit camera or moving object) via
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

**Dataset requirements.** At least one real-motion condition (orbit camera or moving object) and
the static-clip control, each with its own scene set meeting the
scene-count floor established in Task 6 (≥40), and — per the leakage
requirement below — kept disjoint from each other or explicitly
documented as sharing scenes.

**Train/test protocol.** Standard scene-level split (Global Invariants 7–10), applied
independently to the real-motion condition and the static-clip control;
if the same scenes are reused across both (motion-enabled vs.
motion-disabled variant of the same scene), this must not leak train/
test assignment across the two conditions — documented explicitly
either way.

**Controls.** **Static-clip control** (no real motion) is required, not optional,
for this task specifically — it is the only way to rule out that
temporal-windowing itself, independent of genuine motion, produces
spurious apparent consistency.

**Baselines.** The same three from Task 6 (persistence, mean, random-pair), reported
for both the real-motion condition and the static-clip control.

**Metrics.** Same as Task 6 (R², cosine similarity, relative L2 error) applied to
`(Z_{t1}, Z_{t2})` pairs instead of `(Z, Z')` pairs.

## Required artifacts

`state/task_12_result.json` per the Outputs section.

## Tests required

**Required software tests.** - Frame ordering and indices correct; adjacent/chosen-window pairing
  correct; physical-transition ground truth correctly derived from the
  trajectory; representation-transition computation correct.
- Regression (existing suite, including `tests/test_motion.py`,
  `tests/test_bpy_renderer.py`).
- No NaN/Inf; reproducibility.

**Required scientific-validity tests.** - A leakage test that the two temporal windows are computed from the
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

**Required research-alignment checks.** Same frozen-encoder/no-ground-truth-input/linear-map-first discipline
as Task 6, now applied to a temporal pair instead of a transform pair.

## Leakage checks

Standard scene-level split (invariants 4–5-equivalent, i.e. Global
Invariants 7–10), plus: the static-clip control must use a different,
disjoint scene set from the real-motion experiment, or, if the same
scenes' motion-disabled variant is used, an explicit check that this
does not leak train/test assignment across the two conditions.

## Acceptance criteria

**Failure conditions.** Any of: temporal windows drawn from mismatched scenes; the static-clip
control omitted; windowing scheme chosen after seeing which gives the
best score; ground-truth trajectory poses fed into the encoder as
input.

**Acceptance criteria.** 1. At least one real-motion condition run end-to-end with the
   two-window protocol.
2. Static-clip control run under the identical protocol.
3. All three Task 6-style baselines reported for both conditions.
4. No NaN/Inf.

## Prohibited shortcuts

Using ground-truth trajectory poses as encoder input; picking temporal
windows after seeing which windowing gives the best score; omitting the
static-clip control.

## Scientific interpretation limits

**Interpretation rules.** A positive result here is evidence of consistency under one motion
type (orbit / constant velocity) and one windowing scheme — it does
not establish general temporal-dynamics understanding, and does not
imply the encoder tracks arbitrary motion patterns.

**What a positive result means.** The tested motion type induces a temporally-local, linearly predictable
representation transition that generalizes across scenes, beyond what
the static-clip control alone produces.

**What a negative result means.** No measurable temporal consistency beyond the static-clip control, or
no advantage over Task 6-style baselines — a valid, complete finding
for this motion type and windowing scheme.

**What this task does NOT establish.** Anything about motion types not tested; general temporal-dynamics
understanding; object permanence or identity persistence specifically
(Task 13's distinct question); causal or predictive-world-model claims.

---
