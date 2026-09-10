# Task 7 — Multiple geometric transformations

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 7 — MULTIPLE GEOMETRIC TRANSFORMATIONS" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Determine whether the phenomenon tested in Task 6 generalizes beyond a
single camera rotation.

**Why this task exists.** Task 6 establishes the protocol on one transform. A single positive or
negative result on `camera_rotation` alone cannot support any claim
about "geometric transformations" as a class — it could be an artifact
specific to rotation's particular visual signature. Task 7 exists to
turn Task 6's anecdote into a comparison across the full set of
transforms this project defines as geometric, using the appearance-only
controls as an explicit contrast class.

**Relationship to previous tasks.** Directly reuses Task 6's scene generation, splitting, encoding, fitting,
and metric code as a library — Task 6's script must be refactored into
a per-transform function if it was not already, rather than
copy-pasted per transform. Inherits Task 6's baseline set unchanged.

## Scientific question

**Scientific question.** Does the representation exhibit predictable, scene-generalizable
responses to multiple distinct physical geometric transformations?

**Hypothesis.** The four members of `transforms.scene_transform.GEOMETRIC_TRANSFORMS`
(`camera_rotation`, `camera_translation`, `object_rotation`,
`object_translation`) each induce a linearly predictable,
scene-generalizing change in `Z`, to varying degrees; the two members of
`CONTROL_TRANSFORMS` (`lighting_change`, `texture_change`) — which are
not rigid transforms (`transform_matrix` is `null` for them) — show
markedly weaker equivariance under the identical protocol.

**Mathematical formulation.** For each `T` in `GEOMETRIC_TRANSFORMS ∪ CONTROL_TRANSFORMS`:
`S' = T(S)`, `V=R(S)`, `V'=R(S')`, `Z=E(V)`, `Z'=E(V')`,
`Z' ≈ W_T Z + b_T` fit on train, evaluated on test — identical
formulation to Task 6, run once per transform, on the **same scene set
and same train/test split across all six transforms** (so results are
comparable transform-to-transform, not confounded by different scene
samples).

## Implementation requirements

**Inputs.** The same scene set used in Task 6 (or a fresh, equally-sized set,
explicitly recorded either way), rendered through all four
`GEOMETRIC_TRANSFORMS` plus both `CONTROL_TRANSFORMS`, using transform
magnitudes drawn from `TransformConfig`'s existing ranges unless a
specific fixed magnitude is scientifically motivated and recorded.

**Outputs.** `state/task_07_result.json` with a `results: {transform_name: {...}}`
map (one entry per transform, each shaped like Task 6's top-level
metrics block), plus a comparison table/report.

**Experimental protocol.** 1. Reuse Task 6's scene generation and the identical train/test split
   for all six transforms in one run.
2. Render every transform's original/transformed pair for every scene
   (reusing `transforms.pairs.generate_pair`).
3. Encode, pool, fit, and evaluate exactly as in Task 6, independently
   per transform — each transformation must be independently evaluated;
   transformations must not be pooled into one result in a way that
   hides transformation-specific failures.
4. Run Task 6's three baselines for every one of the six transforms.

**Dataset requirements.** Same scene count/seed discipline as Task 6 (≥40 scenes), all six
transforms rendered for every scene.

**Train/test protocol.** Identical scene-level split as Task 6, reused verbatim across all six
transforms in the same run (a scene's split assignment does not vary by
transform).

**Controls.** The same three as Task 6 (persistence, mean, random-pair), applied
per-transform.

**Baselines.** Same as Task 6's; DESIGN.md's full four-baseline set remains Task 10's
responsibility to formalize, not Task 7's to add.

**Metrics.** R², mean cosine similarity, mean relative L2 error, per transform, for
the learned `W_T` and all three baselines — a 6×4-metric comparison
table at minimum, using the same metric implementations as Task 6
(no transform-specific metric substitution).

## Required artifacts

`state/task_07_result.json` (per-transform breakdown as above); a
comparison table/report (markdown or the result JSON itself).

## Tests required

**Required software tests.** - Every transformation executes; the renderer correctly reflects the
  transformation (checked against `apply_transform`'s own
  `changed_variables`/`fixed_variables`/`transform_matrix` ground
  truth); representations generated correctly; `W_T` fits; evaluation
  executes; outputs generated for all six transforms.
- No NaN/Inf anywhere.
- Regression: Task 6 remains valid (its own tests, and ideally its own
  recorded numbers, still reproduce under the shared library code).

**Required scientific-validity tests.** - Expected physical properties actually change under each transform,
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

**Required research-alignment checks.** Same as Task 6, applied per transform; additionally, confirm the
identical scene/split is used for all six (a dedicated check comparing
`train_scene_ids`/`test_scene_ids` across every transform's entry).

## Leakage checks

Same as Task 6, checked independently per transform, plus: the shared
split's `train_scene_ids`/`test_scene_ids` are identical across every
transform's entry in the result JSON.

## Acceptance criteria

**Failure conditions.** Same categories as Task 6, plus: using six different train/test splits
(confounding transform effect with sample effect); pooling
transform-level results in a way that obscures a transform-specific
leakage or failure; silently overwriting Task 6's own recorded result
when reusing its scene set (a numeric difference from re-sampling must
be recorded as an explicit, intentional protocol note per Global
Invariant 27, not silently accepted).

**Acceptance criteria.** 1. All four geometric transforms and both appearance controls run
   through the identical protocol on the identical scene split.
2. All three baselines present for every transform.
3. No NaN/Inf anywhere in the six-transform results.
4. Task 6's own recorded result is not silently contradicted without an
   explicit, recorded protocol note.

## Prohibited shortcuts

Cherry-picking which transforms to report based on which look best; using
different train/test splits per transform; adjusting a transform's
magnitude after seeing its score to improve it.

## Scientific interpretation limits

**Interpretation rules.** A ranking across transforms is evidence about *this frozen encoder,
this scene distribution, this transform-magnitude range* — not a
general claim about geometric transformations in video representations.
If appearance controls score similarly to geometric transforms, that is
a meaningful negative finding (weak evidence `Z` is not distinguishing
geometry from any change), to be reported plainly, not fixed until it
goes away.

**What a positive result means.** For the transforms that show it: this frozen encoder's representation
responds to that transform type in a way that is linearly predictable
and generalizes across scenes, under the tested magnitude range.

**What a negative result means.** For the transforms that do not show it: no evidence, under this
protocol, that this transform type's effect is linearly predictable in
`Z` — a valid, complete, reportable finding, potentially different
per transform.

**What this task does NOT establish.** Any claim about transforms not tested; universal geometric
understanding from success on one or a few transforms; whether the
underlying physical parameters (not just the representation's response)
are themselves accessible (Task 8); robustness to appearance confounds
beyond the two controls tested here (Task 9 goes deeper); behavior at
scale (Task 11).

---
