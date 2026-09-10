# Task 14 — Counterfactual representation consistency

**Depends on:** Tasks 6–13's full experiment/probe/baseline
infrastructure; `transforms/scene_transform.py`'s explicit
`changed_variables`/`fixed_variables` ground truth (Task 3), which is
exactly what a counterfactual-edit claim needs to be checkable.

## Objective

Test whether a *counterfactual* edit to a scene (changing one physical
variable while holding others fixed) moves the representation in the
direction implied by that specific variable's physical effect — a
stronger, more targeted claim than Tasks 6–7's "some transform changes
Z predictably."

## Scientific question

If we counterfactually change exactly one physical variable (e.g. move
one object further from the camera, holding camera and every other
object fixed — using `transforms.scene_transform.apply_transform`'s
existing `changed_variables`/`fixed_variables` contract to guarantee
isolation), does the representation change in a way that (a) is
consistent in direction/magnitude with that variable specifically
(distinguishable from changing a *different* variable by a comparable
amount), and (b) does not change in ways attributable to variables that
were held fixed?

## Implementation requirements

1. Choose at least two isolatable counterfactual edits already
   supported by `transforms.scene_transform.py`'s per-variable isolation
   guarantees (e.g. `object_translation` on one object vs.
   `object_rotation` on the same object) — reuse the existing
   `changed_variables`/`fixed_variables` output directly as the ground
   truth of what was and wasn't touched, rather than re-deriving it.
2. For each counterfactual condition, compute `Z` before and after, and
   test whether a probe/classifier trained to distinguish "which
   variable changed" from `(Z_before, Z_after)` pairs (or their
   difference) generalizes to held-out scenes — this reuses Task 6/8's
   linear-probe machinery, now applied to a discrimination task between
   two (or more) counterfactual conditions rather than a regression onto
   one transform.
3. Include a **magnitude-matched** design: the two counterfactual edits
   being distinguished should be scaled so their effect isn't trivially
   separable by representation-norm alone (document how magnitudes were
   matched, e.g. by a comparable pixel-level change or a comparable
   ground-truth SE(3) displacement magnitude).

## Required controls

- **Label-scrambled control**: same discrimination task with
  scene-to-condition labels permuted, as in every prior task's
  shuffled-pairing-style control.
- **Fixed-variable leakage check**: verify that a probe trained only on
  representations from scenes where a *held-fixed* variable was
  (independently, in a separate control run) varied does NOT predict
  the counterfactual condition — i.e. the discrimination signal isn't
  coming from a confound correlated with, but not equal to, the intended
  variable.

## Required artifacts

`state/task_14_result.json`: which counterfactual conditions were
compared, magnitude-matching method, discrimination accuracy for real
vs. scrambled labels, and the fixed-variable leakage check's result.

## Tests required

- A unit test verifying the magnitude-matching computation itself (given
  two known transform params, correctly computes/compares their
  displacement magnitude).
- Regression (existing suite).

## Leakage checks

Standard scene-level split; plus the fixed-variable leakage check above
is itself a required leakage check for this task specifically (not
optional).

## Acceptance criteria

1. At least two counterfactual conditions compared, magnitude-matched,
   with `changed_variables`/`fixed_variables` used as ground truth of
   isolation (not re-derived by hand).
2. Real discrimination probe and label-scrambled control both reported.
3. Fixed-variable leakage check performed and reported (pass or fail —
   a fail here is a real finding that must block the task, not be
   averaged away).
4. No NaN/Inf.

## Prohibited shortcuts

- Do not compare two counterfactual edits of wildly different magnitude
  and claim the resulting easy discrimination is a meaningful positive
  result.
- Do not feed the `changed_variables`/`fixed_variables` labels into the
  encoder (invariant 7) — they are ground truth for constructing
  conditions and for the leakage check only.

## Scientific interpretation limits

A positive discrimination result shows the representation carries
*some* information distinguishing these two specific counterfactual
edits under this magnitude-matching scheme — it is not evidence of a
causal or compositional world model, and does not generalize to
edits/variables not tested here.
