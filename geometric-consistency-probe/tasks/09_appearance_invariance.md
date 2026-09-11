# Task 9 — Appearance invariance controls

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 9 — APPEARANCE INVARIANCE CONTROLS" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Test whether representation variation is specifically related to
physical state or whether superficial appearance changes can produce
comparable effects.

**Why this task exists.** Tasks 6–7 already include `lighting_change`/`texture_change` as
within-task appearance controls, but only as a side comparison to the
geometric transforms. Task 9 exists to make that comparison the primary
object of a dedicated experiment, deepen it with a fixed sanity-check
(the null-transform case), and rule out that Tasks 6–7's apparent
geometric sensitivity is actually explainable by lighting/texture/
material/rendering-pipeline artifacts rather than genuine geometric
change.

**Relationship to previous tasks.** Directly reuses Task 7's `(Z, Z')` pairs for the appearance controls
and, for comparison, the geometric transforms — no need to re-render if
Task 7's outputs are available and unmodified. Uses `metrics.invariance
.evaluate_invariance` (DESIGN.md §7's canonical implementation:
`Z' ≈ Z` measured directly, with **no map fit**, deliberately distinct
from "equivariance with `ρ(T)=Identity`," which is the identity
baseline in Tasks 6/7/10).

## Scientific question

**Scientific question.** When physical state is held constant, how sensitive is `Z` to changes
in lighting, texture, material, or color?

**Hypothesis.** `Z` stays comparatively invariant (high cosine similarity, low relative
L2 movement) under `lighting_change`/`texture_change`, in contrast to
its behavior under the four geometric transforms.

**Mathematical formulation.** For `T_a` in `CONTROL_TRANSFORMS`:

```
E(R(T_a(S))) ≈ E(R(S))     i.e.    Z' ≈ Z
```

measured directly via `metrics.invariance.evaluate_invariance(transform
_name, Z, Z_prime)` — no `W_T` fit at all, per DESIGN.md §7.

## Implementation requirements

**Inputs.** `(Z, Z')` pairs for `lighting_change` and `texture_change` (reused from
Task 7, or freshly rendered under the identical protocol); a
"null-transform" pair per scene (re-render the identical `SceneState`
with no transform applied at all, `T = identity`).

**Outputs.** `state/task_09_result.json`: invariance metrics for `lighting_change`,
`texture_change`, the null-transform sanity check, and (for comparison)
the four geometric transforms' invariance scores.

**Experimental protocol.** 1. For each `CONTROL_TRANSFORMS` entry, compute invariance metrics
   between `Z` (original) and `Z'` (transformed).
2. Compute the null-transform sanity check: re-render the same
   `SceneState` with no transform, compute invariance between the two
   renders' representations. If this does not score near-perfect
   invariance, that indicates a pipeline determinism bug (e.g.
   nondeterministic rendering), which is a software-correctness
   finding, not a scientific one, and must block this task until fixed.
3. Compute invariance metrics for the four geometric transforms (reused
   from Task 7's `(Z, Z')` pairs) for direct comparison.

**Dataset requirements.** Same scene set as Task 7 where reused; otherwise a fresh, equally-sized
set under the identical scene-level split discipline.

**Train/test protocol.** No fitting occurs in this task (invariance is measured directly, not
via a fit map), so there is no train/test split to contaminate for the
invariance computation itself; the underlying scene set nonetheless
retains its scene-level split labeling for consistency with Tasks 6–7
and for any control that does involve fitting (see below).

**Controls.** **Null-transform sanity check** (identity re-render) — required, not
optional; it is the upper-bound/sanity-check control this task adds
beyond Tasks 6–7's use of the same transforms.

**Baselines.** Geometric transforms' invariance scores (reused from Task 7) serve as
the contrastive comparison baseline for this task specifically — the
appearance controls are expected to be markedly *more* invariant.

**Metrics.** Mean cosine similarity and mean relative L2 error between `Z` and `Z'`
(via `metrics.invariance.evaluate_invariance`) — the same primitives as
Tasks 6–7, applied without a fitted map.

## Required artifacts

`state/task_09_result.json` per the Outputs section.

## Tests required

**Required software tests.** - Physical-state equality verified for every appearance intervention
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

**Required scientific-validity tests.** - Verify the intended physical equality rather than assuming it (every
  appearance intervention explicitly recorded and checked, per above).
- Investigate, explicitly, whether apparent geometric sensitivity in
  Tasks 6–7 could actually be caused by lighting, texture, material,
  color, or rendering-pipeline artifacts rather than genuine geometric
  change — a required scientific-QA step for this task.
- Scene-level split respected even though no fitting occurs here (for
  cross-task consistency and any future extension of this task that
  does fit something).

**Required research-alignment checks.** Consistent with DESIGN.md §7's explicit distinction between invariance
(no map fit) and equivariance-with-identity-baseline (the identity
baseline in Tasks 6/7/10) — this task's result JSON must not conflate
the two.

## Leakage checks

Same scene-level split discipline as prior tasks, for the underlying
scene set.

## Acceptance criteria

**Failure conditions.** Any of: an appearance intervention that inadvertently changes geometry,
pose, or object identity (per `changed_variables`); the null-transform
sanity check failing its threshold (pipeline determinism bug); tuning
the invariance threshold after seeing results to make appearance
controls look more invariant than they are.

**Acceptance criteria.** 1. Both appearance controls and the null-transform sanity check
   evaluated with `metrics.invariance`.
2. Geometric transforms' invariance scores included for direct
   comparison.
3. Null-transform sanity check passes its documented threshold, or the
   task is marked blocked with the anomaly explained (a pipeline bug,
   not a valid negative result).

## Prohibited shortcuts

Assuming physical-state equality for an appearance intervention instead
of verifying it; tuning the invariance metric's threshold after seeing
the appearance-control results.

## Scientific interpretation limits

**Interpretation rules.** Invariance under appearance change is evidence the representation is
not merely reacting to *any* pixel-level change. It does **not** by
itself prove the representation encodes geometry (that inference needs
Tasks 6–8's positive results too, taken together, still bounded by
their own limits). Do not claim complete appearance invariance — only
claim what was tested (lighting, texture; not every possible appearance
factor).

**What a positive result means.** Appearance-only change moves `Z` markedly less than the tested
geometric transforms, under this scene distribution — supporting
(not proving) that `Z`'s sensitivity in Tasks 6–7 tracks geometry
rather than arbitrary visual change.

**What a negative result means.** Appearance changes move `Z` almost as much as geometric ones — a valid,
meaningful negative finding that directly undercuts confidence in
Tasks 6–7's geometric interpretation, and must be reported prominently,
not minimized.

**What this task does NOT establish.** Invariance to appearance factors not tested (e.g. novel textures/
materials never sampled); any claim about *which* geometric property
drives Tasks 6–7's results (only that appearance alone does not fully
explain it, if the result is positive).

---
