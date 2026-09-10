# Task 8 — Physical-state accessibility

**Depends on:** Tasks 6–7's scene/encoding pipeline; `generation.
state_features` (already has `camera_azimuth_deg`, per
`experiments/run.py`'s V0 usage — inspect and extend it rather than
reimplementing feature extraction from `SceneState`).

## Objective

Test whether physical state variables are **linearly decodable** from
the frozen encoder's representation `Z` of the *original*, untransformed
scene — a different question from Tasks 6–7's "does a transform's
*effect* on `Z` generalize."

## Scientific question

Is physical state — camera azimuth/elevation/distance, object
position, object rotation (whichever are meaningfully variable in the
scene sampler) — linearly accessible from `Z`? This is the project's
second pillar alongside equivariance (`DESIGN.md` §10, "physical-state
probes"): even perfect equivariance under a transform's *effect* would
not by itself prove state is *represented*; a probe against the
absolute state is the complementary check.

## Implementation requirements

1. For each physical-state variable being probed, define a
   `feature_fn(SceneState) -> float | np.ndarray`, following the
   existing `camera_azimuth_deg` convention.
2. Reuse (or adapt) `representations.dataset.build_state_probe_split`'s
   pattern for assembling `(Z_train, y_train, Z_test, y_test)` — same
   scene-level split as Tasks 6–7, ideally the *same* scenes, so results
   are comparable across tasks.
3. Fit a **linear** probe (ridge regression, `sklearn.linear_model.
   Ridge` or `probes.linear_state_probe.LinearStateProbe` if it already
   covers this — inspect before adding a new one) per state variable, on
   train only.
4. Evaluate held-out R² (regression) per variable on test.
5. Add at least one **rotation-based** variable requiring circular
   handling (e.g. azimuth) — decide explicitly whether to probe raw
   degrees, or `(sin, cos)` components, and record which and why; do not
   silently probe an angle linearly across its wrap-around discontinuity
   without addressing it.

## Required controls

- A shuffled-label control: fit the same probe on
  scene-shuffled `(Z_train, y_train_shuffled)` — if this scores nearly
  as well as the real fit, the probe/metric setup itself is suspect
  (mirrors `baselines/shuffled_pairing_baseline.py`'s logic, applied
  here to labels instead of transform targets).
- A mean-prediction baseline (`y_hat = mean(y_train)`).

## Required artifacts

`state/task_08_result.json` with, at minimum: which state variables
were probed, per-variable R² for the real probe / shuffled-label
control / mean baseline, train/test scene IDs, seed, config.

## Tests required

- Regression (existing suite).
- A unit test that the shuffled-label control uses a genuine permutation
  (not an identity permutation for small N — mirror the existing guard
  in `shuffled_pairing_baseline.py`).

## Leakage checks

- Train/test scene disjointness (invariant 4).
- The shuffled-label control's permutation must not be re-derivable to
  the identity by construction (same check as invariant 17's spirit:
  the control must not accidentally leak the real correspondence back
  in).

## Acceptance criteria

1. At least two physical-state variables probed, one of them the
   existing `camera_azimuth_deg` for continuity with V0.
2. Real probe, shuffled-label control, and mean baseline all reported
   per variable.
3. Train/test split disjoint and consistent with Tasks 6–7's scenes
   where practical.
4. No NaN/Inf.

## Prohibited shortcuts

- Do not feed any ground-truth `SceneState` field into the encoder
  itself (invariant 7) — state is the probe *target*, never encoder
  input.
- Do not select which state variables to report based on which probe
  best after the fact.

## Scientific interpretation limits

A high probe R² shows the variable is *linearly accessible* from `Z` —
not that the encoder "knows" or "represents" that variable in any
stronger sense, and not that it is causally used by the encoder for
anything. A null result here alongside a positive equivariance result
in Task 6/7 (or vice versa) is itself an interesting, valid, reportable
combination — do not force these two pillars to agree.
