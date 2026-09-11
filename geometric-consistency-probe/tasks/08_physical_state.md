# Task 8 — Physical-state accessibility

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 8 — PHYSICAL-STATE ACCESSIBILITY" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Test whether physically meaningful variables are linearly accessible
from `Z`.

**Why this task exists.** Equivariance (Tasks 6–7) asks whether a transform's *effect* on `Z` is
predictable. Accessibility asks a different, complementary question:
whether the *absolute* physical state of the (untransformed) scene can
be read out of `Z` at all. DESIGN.md §13 is explicit that these are
related but distinct claims, and a project answering "does a world
model understand 3D" cannot skip either — a representation could be
highly equivariant under a transform while the underlying physical
quantity itself remains linearly inaccessible, or vice versa, and that
combination is itself scientifically informative.

**Relationship to previous tasks.** Reuses the scene/encoding pipeline from Tasks 6–7 and, ideally, the
same scenes, so results are comparable across tasks. Reuses
`representations.dataset.build_state_probe_split`'s pattern for
assembling `(Z_train, y_train, Z_test, y_test)`, and
`probes.linear_state_probe.LinearStateProbe` if it already covers the
needed target — inspected and extended, not reimplemented, before
adding anything new. Reuses `generation.state_features
.{camera_azimuth_deg, camera_elevation_deg, camera_distance,
primary_object_position_xy}` as probe targets (DESIGN.md §10);
`velocity` remains explicitly out of scope until Task 12 defines `v_i`
via real multi-frame motion.

## Scientific question

**Scientific question.** Even if representation transformations are not perfectly equivariant
(Tasks 6–7), does the latent representation contain linearly
recoverable information about physical scene state?

**Hypothesis.** At least camera azimuth (continuity with the V0 pipeline's existing
`camera_azimuth_deg` probe) and one additional physical-state variable
are linearly decodable from `Z` (of the original, untransformed scene)
above chance/mean-baseline level, under this scene distribution.

**Mathematical formulation.** For target `y` (a physical-state scalar or vector):

```
y ≈ W_probe Z + b_probe     (linear ridge probe, fit on TRAIN scenes only)
```

Evaluated by held-out R² (regression) on test-split scenes.

## Implementation requirements

**Inputs.** `Z` for the original (untransformed) render of each scene in the
train/test split; ground-truth physical-state labels computed directly
from each scene's `SceneState` via `generation.state_features` feature
functions (`feature_fn(SceneState) -> float | np.ndarray`).

**Outputs.** `state/task_08_result.json`: which state variables were probed,
per-variable R² for the real probe / shuffled-label control / mean
baseline, train/test scene IDs, seed, config.

**Experimental protocol.** 1. Define/select at least two `feature_fn`s, one of which is
   `camera_azimuth_deg` for continuity with V0.
2. For a rotation-based variable (azimuth), explicitly decide and
   record whether raw degrees or `(sin, cos)` components are probed —
   an angle must not be probed linearly across its wrap-around
   discontinuity without addressing it.
3. Assemble `(Z_train, y_train, Z_test, y_test)` via the scene-level
   split (reusing `representations.dataset.build_state_probe_split`'s
   pattern).
4. Fit a **linear** ridge probe per variable on train only; evaluate
   held-out R² on test.
5. Run the shuffled-label control (fit the identical probe on
   scene-shuffled `(Z_train, y_train_shuffled)`) and the mean-prediction
   baseline (`y_hat = mean(y_train)`).

**Dataset requirements.** Reuses Tasks 6–7's scene set where practical; if a fresh set is
sampled, it must follow the same scene-level split discipline and scene
count floor.

**Train/test protocol.** Scene-level split (Global Invariants 7–10), ideally identical to Tasks
6–7's for cross-task comparability.

**Controls.** - **Shuffled-label control**: fit the same probe on scene-shuffled
  labels — if this scores nearly as well as the real fit, the
  probe/metric setup itself is suspect. The shuffled-label control's
  permutation must not be re-derivable to the identity by construction
  (guard against a trivial/identity permutation for small `N`, mirroring
  the existing guard in `baselines/shuffled_pairing_baseline.py`).
- **Mean-prediction baseline** (`y_hat = mean(y_train)`).

**Baselines.** The shuffled-label control and mean-prediction baseline above are this
task's required baseline set (a different baseline family than Tasks
6–7/10's representation-transform baselines, since the target here is a
scalar physical quantity, not a representation vector).

**Metrics.** R² (primary); MAE and RMSE reported for interpretability; angular error
for any orientation-valued quantity probed via `(sin, cos)` decoding.

## Required artifacts

`state/task_08_result.json` per the Outputs section above.

## Tests required

**Required software tests.** - Correct labels (feature functions verified against known `SceneState`
  fixtures); correct dimensions; correct metric implementation; no
  NaN/Inf; reproducibility.
- Unit test that the shuffled-label control uses a genuine permutation
  (not an identity permutation for small N).
- Regression (existing suite).

**Required scientific-validity tests.** - **Linear probes only** — nonlinear probes must not replace linear
  probes in this task (Global Invariant 13); reaching for a nonlinear
  probe to explain away a null linear result is prohibited without
  first exhausting the linear-probe design space (regularization
  strength, pooling choice) and recording why linear was insufficient.
- Correct alignment between representation and label (the label
  computed from exactly the same `SceneState` whose render produced the
  probed `Z`).
- Verify that high probe performance cannot be trivially explained by
  scene identity, camera identity, a near-constant object position
  across the sampled scenes, leakage, or duplicate scenes — an explicit
  scientific-QA step, not assumed away.
- Pose/angle convention documented (which axis, which sign, degrees vs.
  radians, raw vs. `(sin, cos)`).
- No ground-truth state fed into the encoder (invariant 6) — state is
  the probe *target*, never encoder input.

**Required research-alignment checks.** Consistent with Global Invariants 12–13 (linear probes first, no
nonlinear substitution) and §3's general mapping.

## Leakage checks

Scene-level train/test disjointness (Global Invariants 7–10); the
shuffled-label control's permutation must not accidentally leak the
real correspondence back in.

## Acceptance criteria

**Failure conditions.** Any of: a nonlinear probe substituted for the required linear one; an
angle probed naively across its discontinuity with no stated
convention; train/test overlap; a missing shuffled-label or mean
baseline; feature-function/label misalignment with the probed `Z`.

**Acceptance criteria.** 1. At least two physical-state variables probed, one of them
   `camera_azimuth_deg`.
2. Real probe, shuffled-label control, and mean baseline all reported
   per variable.
3. Train/test split disjoint and consistent with Tasks 6–7's scenes
   where practical.
4. No NaN/Inf.

## Prohibited shortcuts

Feeding ground-truth `SceneState` fields into the encoder; selecting
which state variables to report based on which probe performs best
after the fact; replacing a null linear result with a nonlinear probe
instead of reporting the null result.

## Scientific interpretation limits

**Interpretation rules.** Allowed: "Variable X is linearly accessible from Z, under this scene
distribution." **Not allowed**: "The model explicitly represents X," or
"the model understands X" — accessibility is an operational, statistical
property of a fitted linear model (DESIGN.md §13), not a claim about
internal computation.

**What a positive result means.** The named variable is linearly recoverable from `Z` for this scene
distribution and pooling scheme — an accessibility finding, not a
representational or causal one.

**What a negative result means.** Per DESIGN.md §13 (echoing Hewitt & Liang / Belinkov, §0): a low score
from a linear probe is evidence the quantity is not *linearly*
accessible under this setup — it is **not** evidence the quantity is
absent from `Z` in some non-linearly-decodable form. A null result here
alongside a positive equivariance result in Task 6/7 (or vice versa) is
itself a valid, informative combination and must be reported as such,
not forced to agree.

**What this task does NOT establish.** That the encoder "represents" or "knows" the variable in any stronger
sense than linear accessibility; anything about non-linear
accessibility (explicitly out of scope, per Global Invariant 13); any
causal role of the variable in the encoder's computation.

---
