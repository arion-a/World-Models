# Task 7 — Multiple geometric transformations

**Depends on:** Task 6's experiment script and result schema (extend,
do not fork a parallel implementation), plus all four geometric
transforms already defined in `transforms.scene_transform.
GEOMETRIC_TRANSFORMS` (`camera_translation`, `camera_rotation`,
`object_translation`, `object_rotation`).

## Objective

Generalize Task 6's single-transform experiment to all four geometric
transforms, under the same protocol, so the result is a comparison
across transform types rather than a single anecdote.

## Scientific question

Is generalizing, linear geometric consistency (`Z' ≈ W_T Z`, fit on
train, evaluated on test) specific to camera rotation, or does it hold
(to varying degrees) across all four geometric transform types? Do the
two appearance-only controls (`lighting_change`, `texture_change`) show
markedly *weaker* consistency than the geometric ones — the pattern
`DESIGN.md` predicts if `Z` is tracking geometry rather than any change
at all?

## Implementation requirements

1. Reuse Task 6's scene generation, splitting, encoding, fitting, and
   metric code as a library — refactor Task 6's script into a
   per-transform function if it wasn't already, rather than copy-pasting
   it four (or six) times.
2. Run the identical protocol for all four `GEOMETRIC_TRANSFORMS`, using
   the **same scene set and same train/test split** across all four (so
   results are comparable transform-to-transform, not confounded by
   different scene samples).
3. Also run the two `CONTROL_TRANSFORMS` (`lighting_change`,
   `texture_change`) through the identical pipeline, as within-task
   controls: these should show relatively weak equivariance R² (since
   they're not rigid transforms — `transform_matrix` is `null` for
   them) and should be reported and compared directly against the
   geometric transforms' scores, not run separately and ignored.
4. Reuse Task 6's three baselines (persistence, mean, random-pair) for
   every one of the six transforms.

## Metrics

Per transform: R², mean cosine similarity, mean relative L2 error, for
the learned `W_T` and all three baselines — a 6×4-metric comparison
table at minimum.

## Required artifacts

- `state/task_07_result.json` extending Task 6's schema with a
  per-transform breakdown (a `results: {transform_name: {...}}` map
  instead of Task 6's single-transform top level).
- A comparison table/report (markdown or the result JSON itself is
  sufficient for QA; a human-readable summary is encouraged).

## Tests required

- Confirm the same train/test scene split is used across all six
  transforms in one run (a test asserting the recorded
  `train_scene_ids`/`test_scene_ids` are identical across every
  transform's entry in the result JSON).
- Regression: full existing suite plus Task 6's tests still pass.

## Leakage checks

Same as Task 6, checked independently per transform.

## Acceptance criteria

1. All four geometric transforms and both appearance controls run
   through the identical protocol on the identical scene split.
2. All three baselines present for every transform.
3. No NaN/Inf anywhere in the six-transform results.
4. Regression: Task 6's own result is not silently changed/re-run in a
   way that contradicts its recorded numbers (re-running Task 6's
   transform here should reuse or reproduce, not quietly overwrite,
   its prior recorded result — if numbers differ because of a fixed
   scene set now shared across all 6, record that explicitly as an
   intentional protocol note, invariant 15).

## Prohibited shortcuts

- Do not cherry-pick which transforms to report based on which look
  best.
- Do not use six different train/test splits to make comparison
  claims — that would confound "transform effect" with "sample effect."

## Scientific interpretation limits

A ranking across transforms is evidence about *this frozen encoder,
this scene distribution, this magnitude of transform* — not a general
claim about geometric transforms in video representations. If
appearance controls score similarly to geometric transforms, that is a
meaningful negative finding (weak evidence that `Z` is not
distinguishing geometry from any change), not a bug to be fixed until
it goes away.
