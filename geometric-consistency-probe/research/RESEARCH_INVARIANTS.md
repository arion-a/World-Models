# RESEARCH_INVARIANTS.md — durable rules for every task from 6 onward

**Status: binding, orchestrator-enforced.** These are the rules
`orchestrator/qa.py`'s RESEARCH ALIGNMENT and DATA LEAKAGE layers check
against for every task. A task that violates one of these without an
explicit, recorded protocol-change justification (invariant 15) fails
QA regardless of how good its numbers look. `tests/research/
test_research_alignment.py` checks that this file itself stays intact
and that every task spec references it.

These restate, for tasks 6–18, principles already established in
`DESIGN.md` and `IMPLEMENTATION_NOTES.md` for tasks 1–5 — nothing here
should contradict those documents; where a future task's implementation
seems to require contradicting one, invariant 15 applies: stop and
report, do not silently change the design.

1. **The central object of study is the learned representation `Z`.**
   Every experiment's core question is about what `Z` (or a declared,
   documented pooling of it) does under a transform or physical-state
   probe — not about pixels, not about ground-truth state directly.

2. **Physical scene state `S` must be explicitly represented where
   required.** Every scene that participates in an experiment must have
   a serializable `SceneState` (`generation/scene.py`) or equivalent
   ground truth on disk, not an implicit/undocumented state.

3. **Physical transformations `T` must be known and controllable.** Any
   transform applied to a scene must be one of `transforms/
   scene_transform.py`'s named transforms (or a new one added with the
   same explicit-parameters-and-ground-truth-matrix discipline as
   `transforms/se3.py` already established), never an ad hoc pixel-space
   perturbation.

4. **Training and test scenes must be disjoint.** No `scene_id` may
   appear in both the train and test split of any experiment.

5. **All transformed variants of one scene remain in the same split.**
   A scene's original render and every transformed render derived from
   it share that scene's split label. (This is the existing V0
   guarantee in `representations/dataset.py` / `tests/
   test_scene_split.py` generalized to every later task's own splitting
   code.)

6. **The primary V-JEPA encoder remains frozen unless a future task
   explicitly changes this.** `requires_grad` stays `False` on every
   encoder parameter; no task may introduce an optimizer over encoder
   weights without an explicit, recorded protocol change (invariant 15).

7. **No ground-truth 3D state may secretly be fed into the
   representation model.** The encoder's only input is rendered video
   (`(T, H, W, 3)` uint8 RGB — see `encoders/REPRESENTATION_FORMAT.md`).
   Ground-truth `SceneState`/depth/segmentation may be used to *label*
   or *evaluate* representations, never to *compute* them.

8. **Linear probes are the initial accessibility test before introducing
   nonlinear probes.** A task may not reach for an MLP/nonlinear probe
   to explain away a null linear result without first exhausting the
   linear-probe design space (regularization strength, pooling choice)
   and recording why linear was insufficient.

9. **Every major scientific claim must correspond to an actual
   experiment.** No claim in a task's `scientific_result` field may
   describe a result that isn't backed by a metric actually computed
   and recorded in that task's `state/task_NN_result.json`.

10. **Every experiment must have a plausible failure/negative outcome.**
    If a task's design cannot, even in principle, produce a result that
    would count as evidence *against* geometric consistency, the
    experiment is not well-posed and QA's SCIENTIFIC VALIDITY layer
    fails it regardless of what it reports.

11. **Negative results are scientifically valid.** "Weak equivariance",
    "no better than baseline", or "not linearly accessible" are valid,
    completable task outcomes — see `orchestrator/qa.py`'s explicit
    SOFTWARE FAILURE vs. SCIENTIFIC NEGATIVE RESULT distinction. QA must
    never fail a task for reporting a negative result honestly, and must
    never reward a task for reporting a positive one without evidence.

12. **No experiment may be added merely because it produces impressive
    visuals.** Every experiment must map to a `tasks/NN_*.md`
    objective/scientific question; visualization is a reporting aid, not
    a justification for scope.

13. **No "understanding" claim may be made without an operational
    metric.** Any report (task result, `reports/`, or the final report
    in Task 16) that uses a phrase like "the model understands 3D" must
    immediately cite the specific metric and threshold behind it, per
    `DESIGN.md` §13's "what this framework can and cannot support."

14. **Baselines and controls are mandatory where scientifically
    relevant.** At minimum: a persistence baseline (`Z_hat' = Z`), a
    mean/constant-prediction baseline, and a shuffled-pairing / random
    correspondence control, evaluated with the exact same metrics and
    exact same test split as the learned map (see `baselines/` for the
    ones tasks 1–5 already established: `identity_baseline.py`,
    `shuffled_pairing_baseline.py`).

15. **Protocol changes must be explicit and recorded.** If completing a
    task honestly requires deviating from its spec, from
    `DESIGN.md`, or from another invariant on this page, the task must
    stop, record the conflict, and surface it — via the task's QA
    result and, if the orchestrator's fix loop cannot resolve it,
    `BLOCKED` status — rather than silently changing course. (This is
    the same rule already in effect for tasks 1–5's QA protocol.)

16. **No test-set fitting.** No test-split scene, representation, or
    label may influence a fitted parameter (a `W_T`, a probe's
    coefficients, a baseline's mean, a hyperparameter chosen by looking
    at test performance). Fitting happens on train; test is read only at
    evaluation time.

17. **No accidental scene identity leakage.** A representation, feature,
    or metric must not implicitly encode *which scene this is* in a way
    that lets a probe "cheat" by scene-identity lookup rather than
    generalizing — e.g. a state probe achieving high accuracy only
    because train and test scenes share near-duplicate seeds/geometry.
    When in doubt, this is checked by confirming the probe's train/test
    performance gap is consistent with genuine generalization, not
    memorization of a small closed scene set.

18. **The orchestrator must not mark a task PASS solely because Claude
    reports success.** `orchestrator/qa.py` independently executes
    tests, inspects files, parses result JSON, and checks git diffs; a
    task's own self-reported `"implementation_status": "COMPLETE"` is
    one input among many, never sufficient by itself. See
    `orchestrator/qa.py`'s module docstring.
