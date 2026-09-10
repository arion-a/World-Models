# Task 12 — Temporal geometric consistency

**Depends on:** `generation/motion.py` (`ObjectMotion`, `CameraMotion`,
`generate_trajectory`) and `generation/bpy_renderer.py`'s
`render_trajectory` — Task 2's real multi-frame trajectory renderer,
already capable of constant-velocity object/camera motion (orbit mode
included), which Tasks 6–11 deliberately did not need (they used
short, mostly-static clips per pair).

## Objective

Test geometric consistency under **genuine motion within a clip**,
rather than a single static-scene transform applied once between two
otherwise-static renders.

## Scientific question

When a scene has real, continuous motion (camera orbiting, or an object
moving at constant velocity), does the frozen encoder's representation
trajectory `Z(t)` change in a way that is consistent with the known
motion — e.g. does the representation at time `t2` relate to the
representation at time `t1` through a transform predictable from the
known camera/object displacement between `t1` and `t2`, generalizing
across scenes the same way Task 6 tested for a single discrete
transform?

## Implementation requirements

1. Use `generation.motion.generate_trajectory` with a non-trivial
   `camera_motion` (mode `"orbit"`) or `object_motions` to render clips
   with real per-frame motion, via `generation.bpy_renderer.
   render_trajectory` (reuse; this is exactly what it was built for).
2. Encode each clip with `VJEPAEncoder` as before, but now extract
   representations at (at least) two distinct temporal windows within
   the same clip (e.g. frames 0–3 vs frames 4–7, or two overlapping
   windows), rather than one pooled vector for the whole clip — decide
   and document the exact windowing scheme.
3. Compute the known ground-truth relative transform between the two
   windows' camera/object poses (available from the trajectory's
   per-frame pose records — reuse, do not re-derive).
4. Fit and evaluate `W_T` between the two windows' representations, on
   train scenes, exactly as in Task 6, using the ground-truth relative
   transform as the transform identity being tested (this is the same
   protocol as Task 6, with `T` now derived from continuous motion
   instead of a single discrete `apply_transform` call).

## Required controls

Same three as Task 6 (persistence, mean, random-pair), plus: a
**static-clip control** — repeat the same experiment with
`camera_motion=None`/no object motion (a static clip) to confirm the
temporal-window split alone (without real motion) does not itself
produce spurious "consistency" (e.g. from encoder positional biases
across the token sequence).

## Required artifacts

`state/task_12_result.json`, following Task 6's schema shape, with an
added `motion_config` field recording exactly what
`ObjectMotion`/`CameraMotion` parameters were used, and a
`static_clip_control` metrics block.

## Tests required

- Regression (existing suite, including `tests/test_motion.py`,
  `tests/test_bpy_renderer.py`).
- A leakage test that the two temporal windows are computed from the
  same scene's single trajectory (not accidentally mixing frames across
  different scenes).

## Leakage checks

Standard scene-level split (invariants 4–5), plus: the static-clip
control must use a **different, disjoint** scene set from the
real-motion experiment (or the same scenes' motion-disabled variant —
document which, and if the same scenes, ensure this doesn't leak train/
test assignment across the two conditions).

## Acceptance criteria

1. At least one real-motion condition (orbit camera or moving object)
   run end-to-end with the two-window protocol.
2. Static-clip control run under the identical protocol.
3. All three Task 6-style baselines reported for both conditions.
4. No NaN/Inf.

## Prohibited shortcuts

- Do not use the ground-truth trajectory poses as encoder input
  (invariant 7) — only as the label for what `T` is between windows.
- Do not pick temporal windows after seeing which windowing gives the
  best score.

## Scientific interpretation limits

This tests consistency under continuous motion for one motion type
(orbit / constant velocity) and one windowing scheme — it does not
establish general temporal dynamics understanding, and a positive
result here does not imply the encoder tracks *arbitrary* motion
patterns.
