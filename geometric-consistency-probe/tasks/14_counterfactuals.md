# Task 14 — Controlled counterfactual representation consistency

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 14 — CONTROLLED COUNTERFACTUAL REPRESENTATION CONSISTENCY" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Test whether controlled interventions generate structured and
reproducible representation changes.

**Why this task exists.** Tasks 6–7 test whether a *given* transform's effect is predictable.
Task 14 asks a sharper question: whether the representation
*distinguishes* between two different counterfactual interventions of
comparable magnitude, holding everything else fixed — a test more
directly probing whether `Z` carries intervention-specific structure
rather than only "some change happened." This uses Task 3's
already-established per-variable isolation guarantee
(`changed_variables`/`fixed_variables` from `apply_transform`), which
is exactly the ground truth a counterfactual-isolation claim needs to
be checkable.

**Relationship to previous tasks.** Reuses Tasks 6–8's full experiment/probe/baseline infrastructure and
`transforms.scene_transform.apply_transform`'s explicit
`changed_variables`/`fixed_variables` contract (Task 3) directly, as
ground truth of isolation, rather than re-deriving it.

## Scientific question

**Scientific question.** When two matched worlds differ only in a controlled intervention, does
the representation change in a way specific to that intervention?

**Hypothesis.** For two isolatable counterfactual edits already supported by
`transforms/scene_transform.py`'s per-variable isolation guarantees
(e.g. `object_translation` vs. `object_rotation` on the same object),
scaled to comparable magnitude, a probe trained to distinguish "which
variable changed" from `(Z_before, Z_after)` pairs (or their
difference) generalizes to held-out scenes, beyond a label-scrambled
control.

**Mathematical formulation.** Construct `S`, `S_A = intervention_A(S)`, `S_B = intervention_B(S)`,
with `changed_variables`/`fixed_variables` confirming only the intended
variable differs between `S` and each of `S_A`/`S_B`. Fit a
discrimination probe (reusing Task 6/8's linear-probe machinery) on
`(Z, Z_A)` / `(Z, Z_B)` pairs or their differences, on train scenes;
evaluate generalization to test scenes, against a label-scrambled
control.

## Implementation requirements

**Inputs.** At least two counterfactual conditions per scene, constructed via
`transforms.scene_transform.apply_transform` with isolation guarantees
confirmed by its own `changed_variables`/`fixed_variables` output;
magnitudes scaled to be comparable by an explicitly documented method
(e.g. matched by a comparable pixel-level change or a comparable
ground-truth SE(3) displacement magnitude, computed via `transforms.
se3`), so the two conditions are not trivially separable by
representation-norm alone.

**Outputs.** `state/task_14_result.json`: which counterfactual conditions were
compared, the magnitude-matching method, discrimination accuracy for
real vs. scrambled labels, and the fixed-variable leakage check's
result.

**Experimental protocol.** 1. Choose at least two isolatable counterfactual edits already
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

**Dataset requirements.** At least two counterfactual conditions, magnitude-matched, meeting the
scene-count floor established in Task 6.

**Train/test protocol.** Standard scene-level split (Global Invariants 7–10).

**Controls.** - **Label-scrambled control** (same shuffled-pairing-style pattern as
  every prior task).
- **Fixed-variable leakage check** (this task's own, specifically
  required control — not optional): confirm the discrimination signal
  is not attributable to a confound correlated with, but distinct from,
  the intended variable.

**Baselines.** The label-scrambled control above; Task 6/8's linear-probe machinery is
reused, not a new probe family invented.

**Metrics.** Discrimination accuracy (real vs. scrambled labels); the
magnitude-matching method's own displacement-magnitude comparison is
reported alongside, not just the resulting accuracy.

## Required artifacts

`state/task_14_result.json` per the Outputs section.

## Tests required

**Required software tests.** - Unit test verifying the magnitude-matching computation itself (given
  two known transform params, correctly computes/compares their
  displacement magnitude).
- Correct pairing between counterfactual conditions and scenes; no
  NaN/Inf; reproducibility; regression (existing suite).

**Required scientific-validity tests.** - Ask, explicitly, whether representation differences could arise from
  unintended differences rather than the intended intervention — this
  is exactly what the fixed-variable leakage check tests, and it must
  be performed, not merely discussed.
- Magnitude-matching must not leave the two conditions trivially
  separable by representation-norm alone (checked, not assumed).

**Required research-alignment checks.** No ground-truth `changed_variables`/`fixed_variables` labels fed into
the encoder (invariant 6) — they are ground truth for constructing
conditions and for the leakage check only, never encoder input.

## Leakage checks

Standard scene-level split; the fixed-variable leakage check above is
itself a required leakage check specific to this task, not optional.

## Acceptance criteria

**Failure conditions.** Comparing two counterfactual edits of wildly different, unmatched
magnitude and treating easy discrimination as a meaningful positive
result; the fixed-variable leakage check failing (a real finding that
must block the task, not be averaged away); ground-truth
`changed_variables`/`fixed_variables` fed to the encoder.

**Acceptance criteria.** 1. At least two counterfactual conditions compared, magnitude-matched,
   with `changed_variables`/`fixed_variables` used as ground truth of
   isolation.
2. Real discrimination probe and label-scrambled control both
   reported.
3. Fixed-variable leakage check performed and reported (pass or fail).
4. No NaN/Inf.

## Prohibited shortcuts

Comparing magnitude-mismatched conditions; feeding
`changed_variables`/`fixed_variables` labels into the encoder.

## Scientific interpretation limits

**Interpretation rules.** Do not use causal language unless the experimental design actually
supports a causal claim — a positive discrimination result shows the
representation carries *some* information distinguishing the two
tested counterfactual edits under this magnitude-matching scheme, not a
causal or compositional world model.

**What a positive result means.** The representation carries information distinguishing these two
specific counterfactual edits, under this magnitude-matching scheme,
beyond the label-scrambled control, and the fixed-variable leakage
check did not reveal a confound.

**What a negative result means.** No discrimination advantage over the scrambled control, or the
fixed-variable leakage check reveals a confound — both are valid,
complete, and (in the leakage-check case) important findings requiring
prominent reporting.

**What this task does NOT establish.** A causal or compositional world model; generalization to
edits/variables not tested here; anything beyond the two (or few)
specific counterfactual conditions actually compared.

---
