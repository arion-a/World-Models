# Task 11 — Scale experiment

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 11 — SCALE EXPERIMENT" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Determine whether observed effects survive increasing dataset size.

**Why this task exists.** Every prior task (6–10) has operated on scene counts in the tens. A
result that only appears (or only fails to appear) at that scale could
be a small-sample artifact in either direction. Task 11 exists to test
this directly, using the flagship transform as a tractable
representative case rather than re-running the entire Task 7 matrix at
every scale (which would be computationally prohibitive given
rendering is this project's bottleneck).

**Relationship to previous tasks.** Reuses Task 10's consolidated experiment/baseline framework; targets
the flagship transform (`camera_rotation`, per `configs.config
.ExperimentConfig.flagship_transform`) as established by the existing
V0 config convention.

## Scientific question

**Scientific question.** Are the observed representation properties robust to dataset scale, or
are they artifacts of a small controlled dataset?

**Hypothesis.** Where appropriate: the flagship transform's equivariance margin over
baselines (Task 6/7's finding) persists, strengthens, weakens, or
saturates as scene count grows from tens to (at minimum) approximately
100, and, if computationally feasible, approximately 1000 scenes.

**Mathematical formulation.** Identical to Task 6's `Z' ≈ W_T Z + b_T` for `T = camera_rotation`, run
independently at each scale point on a scale-specific train/test split,
with multiple random seeds per scale point.

## Implementation requirements

**Inputs.** Scene counts on a documented ladder (at minimum ~100 scenes; ~1000 if
feasible), each with its own fresh scene-level train/test split at a
fixed fraction (e.g. always 80/20), and multiple seeds per scale point.

**Outputs.** `state/task_11_result.json`: a `scale_points: [{num_scenes, seed,
metrics for learned map and all Task 10 baselines, mean, std,
per-seed results, wall_clock_seconds}, ...]` list.

**Experimental protocol.** 1. For each scale point on the ladder, sample a fresh scene-level
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

**Dataset requirements.** At minimum 3 distinct scene-count points on the ladder (e.g. ~40, ~100,
~1000, budget permitting), each with multiple seeds.

**Train/test protocol.** Fresh scene-level split per scale point, at a fixed train/test
fraction; scale is varied on absolute scene count on both sides, not on
the split ratio.

**Controls.** Multiple seeds per scale point (a "control" against seed-specific
noise, not a baseline in the Task 6/10 sense).

**Baselines.** Task 10's full consolidated baseline set, run at every scale point.

**Metrics.** Same as Task 6 (R², cosine similarity, relative L2 error), aggregated
as mean ± standard deviation across seeds per scale point, with every
per-seed value also recorded (not just the aggregate).

## Required artifacts

`state/task_11_result.json` per the Outputs section; wall-clock timing
per scale point (a genuine practical constraint on this project, given
rendering is the bottleneck).

## Tests required

**Required software tests.** - The requested scale was actually used (scene counts verified, not
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

**Required scientific-validity tests.** - Determine, explicitly, whether the effect persists, strengthens,
  weakens, saturates, or disappears with scale — a required conclusion
  in the result JSON, not left implicit in raw numbers.
- Report all seeds/scale points, not only the best seed or best scale.

**Required research-alignment checks.** Global Invariants 21–24 (seed, config, provenance, machine-readable
results) checked per scale point, not just once for the whole task.

## Leakage checks

Per scale point: disjoint scene sets, no test-scene influence on any
fitted parameter — same discipline as Task 6, independently re-verified
at each point.

## Acceptance criteria

**Failure conditions.** Fewer than 3 scale points completed; a scale point's split not
disjoint; only the best seed/scale reported; methodology silently
changed between scale points (e.g. a different pooling scheme at a
larger scale) without an explicit, recorded justification.

**Acceptance criteria.** 1. At least 3 distinct scene-count points run end-to-end.
2. Learned map and all baselines reported at every point.
3. Wall-clock timing recorded per point.
4. No NaN/Inf at any point.

## Prohibited shortcuts

Stopping the ladder early and extrapolating a trend from fewer than 3
points; reusing the exact same train/test split across scale points by
only adding scenes to one side; reporting only the best-performing
seed or scale point.

## Scientific interpretation limits

**Interpretation rules.** A trend across 3–5 scale points on one synthetic scene distribution and
one transform is suggestive, not an asymptotic scaling law. Do not
extrapolate beyond the tested range (e.g. do not claim what would
happen at 10,000 scenes from a 40–1000 ladder).

**What a positive result means.** The flagship transform's advantage over baselines persists or
strengthens with scale, under this scene distribution — evidence the
effect is not a small-sample artifact, bounded to the tested range.

**What a negative result means.** The effect weakens, saturates, or disappears with scale — a valid,
important finding that directly qualifies (and may substantially
weaken confidence in) Tasks 6–7's conclusions, and must be reported
prominently.

**What this task does NOT establish.** Behavior beyond the tested scale range; behavior for transforms other
than the flagship one tested here (a full Task 7 matrix at scale is
explicitly out of scope given rendering cost, unless a documented
protocol change extends it).

---
