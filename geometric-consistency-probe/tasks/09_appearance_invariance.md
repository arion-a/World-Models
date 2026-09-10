# Task 9 — Appearance invariance controls

**Depends on:** Task 7's appearance-control results (`lighting_change`,
`texture_change`) as a starting point; this task deepens that into its
own dedicated experiment rather than a side note.

## Objective

Directly test the complementary prediction to Tasks 6–7: `Z` should be
comparatively **invariant** (not equivariant) to transforms that change
appearance but not physical geometry.

## Scientific question

Does the frozen encoder's representation stay relatively stable
(`Z' ≈ Z`, high cosine similarity, low relative L2 movement) under
`lighting_change`/`texture_change`, in contrast to its behavior under
the geometric transforms in Tasks 6–7? Invariance here is the mirror
image of equivariance — evaluated with `metrics.invariance` (inspect
its existing interface before adding a new one).

## Implementation requirements

1. For each `CONTROL_TRANSFORMS` entry, compute `Z` (original) and `Z'`
   (transformed) as in Task 6, on the same scene split conventions.
2. Use `metrics.invariance.evaluate_invariance` (reuse) to quantify how
   close `Z'` stays to `Z` — do not conflate this with Task 7's
   equivariance R² against `W_T`; invariance asks "did it move at all,"
   equivariance asks "if it moved, was the movement predictable."
3. Directly compare invariance scores for the two appearance controls
   against invariance scores for the four geometric transforms (which
   should show markedly *lower* invariance, i.e. `Z` does move under
   real geometric change).

## Required controls

- A "null transform" control: apply no transform at all (`T = identity`,
  re-render the same `SceneState`) and measure the corresponding
  invariance metric as an upper bound / sanity check — if the null
  transform doesn't score essentially perfect invariance, something is
  wrong with the pipeline (e.g. rendering nondeterminism), and that is
  a software-correctness finding, not a scientific one.

## Required artifacts

`state/task_09_result.json`: invariance metrics for `lighting_change`,
`texture_change`, the null-transform sanity check, and (for comparison)
the four geometric transforms' invariance scores.

## Tests required

- Regression (existing suite, plus `tests/test_metrics.py`'s existing
  invariance coverage if present).
- The null-transform sanity check must itself be asserted in a test
  (e.g. invariance score above a documented near-1.0 threshold), not
  just reported informally.

## Leakage checks

Same scene-level split discipline as prior tasks.

## Acceptance criteria

1. Both appearance controls and the null-transform sanity check
   evaluated with `metrics.invariance`.
2. Geometric transforms' invariance scores included for direct
   comparison (reuse Task 7's `Z`/`Z'` pairs rather than re-rendering).
3. Null-transform sanity check passes its documented threshold, or the
   task is marked blocked with the anomaly explained (this would be a
   pipeline determinism bug, not a valid negative result).

## Prohibited shortcuts

- Do not tune the invariance metric's threshold after seeing the
  appearance-control results to make them look more invariant than they
  are.

## Scientific interpretation limits

Invariance under appearance change is evidence the representation is
not merely reacting to *any* pixel-level change — it does not by itself
prove the representation encodes geometry (that inference needs Tasks
6–8's positive equivariance/accessibility results too). A weak
invariance result (appearance changes move `Z` almost as much as
geometric ones) is a valid, meaningful negative finding.
