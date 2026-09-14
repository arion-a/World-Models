# Task 13 — Object persistence under occlusion

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 13 — OBJECT PERSISTENCE UNDER OCCLUSION" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Test whether object-related information persists through controlled
visual occlusion and reappearance.

**Why this task exists.** Task 12 establishes that temporal representation transitions can be
studied at all. Task 13 asks a qualitatively different temporal
question: not "is the transition between two visible states
predictable," but "does object-level information survive a period of
zero direct visual evidence." This is a distinct, harder claim
(component E in this project's six operationally-testable claims, §1)
that neither Tasks 6–11's static-pair design nor Task 12's
continuously-visible-motion design can address.

**Relationship to previous tasks.** Uses `generation.scene.ObjectState.instance_id` (the persistent,
segmentation-mask-tied per-object identity already defined in Task 2)
and `generation.bpy_renderer`'s segmentation-pass rendering (Task 2) to
both construct and **verify** occlusion, and Task 12's motion/windowing
infrastructure to construct the pre-occlusion/occluded/post-occlusion
temporal structure.

## Scientific question

**Scientific question.** When an object becomes visually unavailable and later reappears, does
the representation preserve information sufficient to associate the
reappearing object with the same underlying physical object?

**Hypothesis.** For a tracked object (by `instance_id`) that becomes occluded for a
documented sub-sequence of frames and then reappears, the
representation's post-occlusion window carries information about that
object's state (e.g. position) that is more accurately recoverable by a
linear probe than the same probe applied under a scrambled-identity
control, as a function of occlusion duration.

**Mathematical formulation.** A linear probe (reusing Task 8's `y ≈ W_probe Z + b_probe` pattern) is
fit to predict the tracked object's post-occlusion physical state from
the post-occlusion representation window, compared against the
identical probe under: (a) an object-identity-scrambled control (labels
permuted across scenes), and (b) a no-occlusion control (upper bound).

## Implementation requirements

**Inputs.** Scenes constructed (via `generation.motion` trajectories or camera
framing) so that one tracked object (by `instance_id`) becomes occluded
— by another object or by leaving the camera frustum — for a documented
sub-sequence of frames, then reappears; the segmentation ground truth
(`ClipGroundTruth.segmentation`, Task 2) for **verifying** occlusion
actually occurred.

**Outputs.** `state/task_13_result.json`: occlusion verification evidence (frame
ranges + segmentation-based confirmation), the recoverability metric
for the real, scrambled, and no-occlusion conditions.

**Experimental protocol.** 1. Construct at least one occlusion scenario via object motion or
   camera framing.
2. **Verify** occlusion programmatically using segmentation ground
   truth: the tracked instance's segmentation mask pixel count drops to
   zero (or below a documented threshold) for the intended frames — do
   not assume a scripted trajectory produces occlusion without checking
   the rendered ground truth.
3. Extract representations from pre-occlusion, occluded, and
   post-occlusion windows (reusing Task 12's windowing approach).
4. Fit the recoverability probe (post-occlusion representation →
   post-occlusion object state) on train scenes; evaluate on test
   scenes, for the real, scrambled-identity, and no-occlusion
   conditions.
5. Where feasible, vary occlusion duration and report recoverability as
   a function of duration.

**Dataset requirements.** At least one occlusion scenario, verified via segmentation, meeting the
scene-count floor established in Task 6; ideally multiple occlusion
durations.

**Train/test protocol.** Standard scene-level split (Global Invariants 7–10).

**Controls.** - **Object-identity-scrambled control**: permute object identity/
  position labels across scenes (same spirit as
  `shuffled_pairing_baseline.py`) — its permutation must be a real
  derangement over scenes, not accidentally close to the identity
  permutation.
- **No-occlusion control**: identical scene setup with a trajectory
  that keeps the object visible throughout, as the upper-bound
  comparison.
- A chance/random-identity baseline is required at minimum, per the
  task brief.

**Baselines.** The scrambled-identity and no-occlusion controls above serve as this
task's baseline set; Task 8's linear-probe machinery is reused, not
a new probe family invented.

**Metrics.** Held-out R²/MAE/RMSE for the recoverability probe (matching Task 8's
metric family), compared across real/scrambled/no-occlusion conditions
and, where feasible, across occlusion durations.

## Required artifacts

`state/task_13_result.json` per the Outputs section.

## Tests required

**Required software tests.** - Unit test on the occlusion-verification logic itself (given a
  synthetic segmentation array, correctly detects occlusion vs. no
  occlusion).
- Persistent IDs correctly assigned/tracked; occlusion duration
  correctly computed; trajectory correctness; reappearance genuinely
  corresponds to the same physical object (`instance_id` match).
- No NaN/Inf; reproducibility; regression (existing suite).

**Required scientific-validity tests.** - Occlusion claimed only when the segmentation-based check actually
  passes (not assumed from the trajectory alone).
- Investigate, explicitly, trivial cues that could explain successful
  re-identification without genuine persistence: object color, object
  location continuity, background, or scene identity — a required
  scientific-QA step.
- No identity-label leakage (the scrambled control must not
  accidentally preserve the true correspondence).

**Required research-alignment checks.** No ground-truth segmentation/instance-ID data fed into the encoder
(invariant 6) — used only for verification and probe labels.

## Leakage checks

Standard scene-level split; the scrambled-identity control's
permutation must be a genuine derangement, not trivially close to
identity for small `N`.

## Acceptance criteria

**Failure conditions.** Occlusion claimed without a passing segmentation-based verification;
the scrambled or no-occlusion control omitted; identity labels leaking
into the scrambled control; ground-truth segmentation/instance IDs fed
to the encoder.

**Acceptance criteria.** 1. At least one occlusion scenario constructed and verified via
   ground-truth segmentation.
2. Recoverability metric computed for real, scrambled, and
   no-occlusion conditions.
3. No NaN/Inf.

## Prohibited shortcuts

Claiming occlusion occurred without the segmentation-based check
passing; feeding segmentation masks or instance IDs into the encoder.

## Scientific interpretation limits

**Interpretation rules.** Successful re-identification/recoverability alone does not prove an
internal object model — it must be reported alongside the explicit
ruling-out of trivial cues (color, location, background, scene
identity) required above.

**What a positive result means.** Some object-level continuity signal is linearly recoverable after
occlusion, for this occlusion pattern and scene distribution, beyond
what the scrambled-identity control achieves.

**What a negative result means.** No recoverability advantage over the scrambled control — a valid,
complete outcome, and evidence against object-level persistence in this
representation under this protocol.

**What this task does NOT establish.** Object permanence in any general sense; that "the encoder tracks the
object" as opposed to "the encoder reconstructs a plausible
continuation from scene-level statistics" (both remain consistent with
a positive result unless further distinguished); behavior for occlusion
patterns, durations, or object types not tested.

---
