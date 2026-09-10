# Task 13 — Object persistence under occlusion

**Depends on:** `generation/scene.py`'s `ObjectState.instance_id` and
`generation/bpy_renderer.py`'s segmentation-pass rendering (Task 2),
Task 12's temporal/motion infrastructure.

## Objective

Test whether object-level information persists in the frozen
representation through a period where the object is occluded (out of
camera view or behind another object), rather than only asking about
whole-scene/whole-frame consistency as in Tasks 6–12.

## Scientific question

If an object is visible, then occluded for some frames, then visible
again, is the object's state (e.g. position, identity) more accurately
recoverable from the representation immediately after re-appearance
than from a matched control where a *different* object occupies that
role — i.e. does the representation carry object-level continuity
through occlusion, beyond what's visible in the current frame alone?

## Implementation requirements

1. Construct scenes where object motion (reuse `generation.motion`) or
   camera framing causes one tracked object (identified by its stable
   `instance_id`, per `generation/scene.py`) to become occluded (by
   another object, or by leaving the camera frustum) for a documented
   sub-sequence of frames, then reappear.
2. Use the segmentation ground truth (`ClipGroundTruth.segmentation`,
   Task 2) to programmatically **verify** occlusion actually occurred
   (the tracked instance's segmentation mask pixel count drops to zero,
   or below a documented threshold, for the intended frames) — do not
   assume a scripted trajectory produces occlusion without checking the
   rendered ground truth.
3. Extract representations from the pre-occlusion, occluded, and
   post-occlusion windows (reuse Task 12's windowing approach).
4. Define and justify a concrete "object state recoverable" metric —
   e.g. a linear probe (reuse `probes.linear_state_probe`/Task 8's
   pattern) predicting the tracked object's post-occlusion position
   from the post-occlusion representation window, compared against the
   same probe's accuracy when the object's identity is scrambled
   (matched control, see below).

## Required controls

- **Object-identity-scrambled control**: repeat the recoverability
  probe but with object identity/position labels permuted across scenes
  (same spirit as `shuffled_pairing_baseline.py`) — if scrambled labels
  score nearly as well, the "recoverability" signal is not really about
  that object.
- **No-occlusion control**: the same scene setup but with a trajectory
  that keeps the object visible throughout, as the upper-bound
  comparison.

## Required artifacts

`state/task_13_result.json`: occlusion verification evidence (frame
ranges + segmentation-based confirmation), the recoverability metric
for the real, scrambled, and no-occlusion conditions.

## Tests required

- A unit test on the occlusion-verification logic itself (given a
  synthetic segmentation array, correctly detects occlusion/no
  occlusion).
- Regression (existing suite).

## Leakage checks

Standard scene-level split; additionally, the scrambled-identity
control's permutation must be a real derangement over scenes (mirroring
the existing guard against accidental identity permutations elsewhere
in `baselines/`).

## Acceptance criteria

1. At least one occlusion scenario constructed and **verified via
   ground-truth segmentation**, not assumed from the trajectory alone.
2. Recoverability metric computed for real, scrambled, and
   no-occlusion conditions.
3. No NaN/Inf.

## Prohibited shortcuts

- Do not claim occlusion occurred without the segmentation-based check
  passing.
- Do not feed segmentation masks or instance IDs into the encoder
  (invariant 7) — ground truth here is for verification and probe
  labels only.

## Scientific interpretation limits

A positive result shows *some* object-level continuity signal is
linearly recoverable after occlusion, for this occlusion pattern and
scene distribution — it does not establish object permanence in any
general sense, and does not distinguish "the encoder tracks the
object" from "the encoder reconstructs a plausible continuation from
scene-level statistics." A negative result (no better than the
scrambled control) is a valid, complete outcome.
