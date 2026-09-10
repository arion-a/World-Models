# Task 11 — Scale experiment

**Depends on:** Tasks 6–10's consolidated experiment + baseline
framework.

## Objective

Determine how Task 6/7's geometric-consistency findings change as the
number of training scenes grows, rather than relying on one scene-count
data point.

## Scientific question

Does the learned `W_T`'s held-out equivariance (R², cosine similarity)
improve, plateau, or stay flat as `num_scenes` increases (e.g. 40, 100,
250, 500 — choose a concrete, documented ladder appropriate to
available compute/render time)? A flat curve near baseline performance
at every scale is itself informative (evidence the effect isn't a
small-sample fitting artifact, in either direction).

## Implementation requirements

1. Reuse Task 10's consolidated experiment/baseline code; do not
   reimplement fitting/evaluation.
2. Run the flagship transform (`camera_rotation`, per `configs/config.py`'s
   `flagship_transform`) at each scene-count point on the ladder, each
   with its own fresh scene-level train/test split (same split
   fraction/seed convention as Task 6).
3. Hold the train/test *fraction* fixed across the ladder (e.g. always
   80/20) so what varies is absolute scene count on both sides, not the
   ratio.
4. Record wall-clock render+encode time per scale point — this is a
   genuine practical constraint on the project (rendering is the
   bottleneck), and future tasks/readers need this to judge what scale
   is feasible.

## Required controls

Run all three Task 6/10 baselines at every scale point too — a
baseline's score changing with scale (e.g. the mean-baseline's R²
naturally rising as the training mean estimate stabilizes) is expected
and must be shown alongside the learned map's curve, not omitted.

## Required artifacts

`state/task_11_result.json`: a `scale_points: [{num_scenes, metrics for
learned map and 3 baselines, wall_clock_seconds}, ...]` list.

## Tests required

- Regression (existing suite).
- A test that every scale point used a disjoint, correctly-sized
  train/test split (reusing Task 6's leakage-check pattern).

## Leakage checks

Per scale point, same as Task 6 — disjoint scene sets, no test-scene
influence on any fitted parameter.

## Acceptance criteria

1. At least 3 distinct scene-count points run end-to-end.
2. Learned map and all baselines reported at every point.
3. Wall-clock timing recorded per point.
4. No NaN/Inf at any point.

## Prohibited shortcuts

- Do not stop the ladder early and extrapolate/claim a trend from fewer
  than 3 points.
- Do not reuse the exact same train/test split across scale points by
  just adding more scenes to one side only — regenerate the split fresh
  at the documented fraction for each point so scale isn't confounded
  with "which specific scenes happened to be in test."

## Scientific interpretation limits

A trend across 3–5 scale points on one synthetic scene distribution and
one transform is suggestive, not a asymptotic scaling law. Do not
extrapolate beyond the tested range (e.g. do not claim what would
happen at 10,000 scenes from a 40–500 ladder).
