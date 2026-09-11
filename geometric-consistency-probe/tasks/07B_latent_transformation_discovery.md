# Task 7B — Discovery of Latent Geometric Transformation Structure

**Status.** A new, preregistered discovery experiment. It does not alter,
replace, reinterpret as a rerun, or write into historical Task 7 or Task 7
Diagnostic artifacts. Historical results remain immutable:
`state/task_07_result.json`, `experiments/geometric_consistency/`,
`experiments/task7_forensic_audit/`, `experiments/task7_diagnostic/`.

**Basis.** The Task 7 forensic audit reproduced Task 7 exactly and found no
pairing/transform/rendering/encoding/baseline/metric/leakage defect, but
found `N_train=32, N_test=8` against `D=1024`, anisotropic pooled
representations (effective rank ≈17–19), and a raw `(512,1024)` token
tensor collapsed to one `(1024,)` vector by global mean pooling. The Task 7
diagnostic found the camera-rotation learning curve improved monotonically
from N=32 through 256 without plateauing, and that naive dense-token
flattening was worse than pooling at small N. Task 7B is a direct,
preregistered response: a global-pooled, learning-curve-based,
model-hierarchy discovery study, run to a scientifically interpretable
conclusion, before any separate token-level study.

## Scientific question

For a known physical geometric transformation, is there a stable,
scene-generalizable latent transformation, and what is the simplest
supported mathematical form? It must not claim a result before the fixed
held-out evaluation is completed; a supported null result is valid.

## Mathematical formulation

For scene `i`: `S_i` (full known physical state), `T_θ` (a known,
deterministic transform — signed camera yaw, elevation fixed at 0.0°),
`S_i^(θ)=T_θ(S_i)`, `V_i=R(S_i)`, `Z_i=P(E(V_i))∈R^1024` (frozen encoder,
mean-pooled). `N` counts unique source scenes, never clips/frames/tokens.
Multi-output `R²=1 − Σ‖Z'_i−Ẑ'_i‖² / Σ‖Z'_i−mean_eval(Z')‖²`, reported as
the string `"undefined"` (never coerced to 0) when the denominator is
zero. MSE, relative L2 error, cosine similarity, and 95% bootstrap CIs
over seeds/scenes are also reported. See
`experiments/task7b_latent_transformation_discovery/metrics.py`.

## Implementation requirements

**Inputs.** Task 7B's own freshly-sampled master scene population (never
Task 7's 40 scenes), the frozen `VJEPAEncoder`
(`facebook/vjepa2-vitl-fpc64-256`), `camera_rotation` at fixed elevation
0.0° (verified to satisfy an exact composition law — see
`tests/test_task7b_composition_law.py`).

**Protocol freeze.** Before any test-scene rendering, a content-hashed
`protocol.json`/`protocol.md` fixes the encoder config, transform family,
magnitude set, master population/split, N-train schedule, seeds, model
grid, and every acceptance/stopping rule
(`experiments/task7b_latent_transformation_discovery/protocol.py`).
Re-freezing with different content after a sealed test result exists is
refused (`protocol.freeze_protocol`'s guard).

**Split.** Deterministic 70/15/15 (by scene ID) master population sized
so the training pool ≥1024 and the test split ≥64 scenes. Every scene's
original and every transformed variant stay in exactly one split.

**Stage 1 — model hierarchy.** M0 persistence, M1 scalar, M2 diagonal, M3
low-rank residual (`r∈{2,4,8,16,32}`), M4 full linear, M5 affine, M6
one-hidden-layer residual MLP (`h∈{8,16,32}`) — fixed order, every
candidate fit on train only, regularization/early-stopping selected on
validation only, compared against train-mean/persistence/shuffled-pair
controls. `N_train=[64,128,256,512,1024]`, ≥5 seeds per point, nested
subsets. Stop once the final two points agree within 0.02 mean R² with
overlapping 95% CIs, or the full schedule has run (`NOT_STABILIZED`).

**Selection & sealed test.** Highest mean validation R² wins; ties within
0.01 favor fewer parameters, then earlier hierarchy position. Exactly one
sealed test run, once, on the fixed test split, with all controls,
per-scene metrics, and bootstrap CIs.

**Stage 2 — magnitude sweep.** The SAME selected functional form refit
per nonzero magnitude in `{-60,-30,-20,-10,10,20,30,60}` (20° reserved for
interpolation), same fixed splits (a bounded, documented subset for
compute-budget reasons — see `run_stage2_structure_tests.py`'s module
docstring).

**Stage 3 — structure tests.** Composition, inverse, identity,
interpolation — all no-refit, on held-out test scenes, only for magnitude
pairs whose physical composition law was verified numerically first.

**Token-level boundary.** Stage 1-3 are global-pooled only. Raw tokens may
be saved for reproducibility but never flattened/aligned/fit as the
primary analysis — a later `Task 7B-token` study is explicitly
out of scope here.

## Required artifacts

`experiments/task7b_latent_transformation_discovery/`: `protocol.json`,
`protocol.md`, `split_manifest.json`, `stage1_data_manifest.json`,
`z_cache_primary.npz`, `z_cache_stage2.npz`, `data_validation_qa_report.{md,json}`,
`learning_curve_all_points.json`, `learning_curve_aggregated.json`,
`model_selection_record.json`, `sealed_test_result.json`,
`structure_test_results.json`, `qa_report.{md,json}`,
`final_report.md`. `state/task_07b_result.json` (own state machine).

## Tests required

`tests/test_task7b_composition_law.py` (physical composition/inverse/
identity verification, matrix-algebra + real-scene levels),
`tests/test_task7b_fast_render.py` (bit-identical fast-render
verification), `tests/test_task7b_metrics.py`, `tests/test_task7b_models.py`
(synthetic recovery for every M0-M6 model + controls, no-NaN/Inf sweep),
`tests/test_task7b_protocol_and_data.py` (split exclusivity, nested
subsets, protocol content-hash immutability), `tests/test_task7b_learning_curve.py`
(selection/tie-break/stabilization logic, structural no-test-leak check),
`tests/test_task7b_sealed_test.py` (contamination guard), `tests/test_task7b_structure_tests.py`
(operator-level math against known synthetic operators), `tests/test_task7b_state.py`.

## Leakage checks

Every regression's fitting/regularization-selection/early-stopping uses
train or train+validation only, never test (structurally verified in
`tests/test_task7b_learning_curve.py::test_run_point_never_touches_a_provided_test_only_id_set`
and `qa_diagnostic`-style source inspection). The sealed test evaluates
exactly once; a second call without `force=True` raises. Every scene ID
appears in exactly one split (`data.three_way_split`'s own assertions,
re-verified in `qa_gates.check_split_exclusivity`).

## Acceptance criteria

All QA gates pass; every feasible scheduled `(N_train, seed)` ran,
including failures; the selection record proves validation-only choice;
the sealed test artifact includes all controls, CIs, and a contamination
check; every mathematically valid Stage 3 check has a result or a
`NOT_APPLICABLE` reason; `state/task_07b_result.json` reaches `COMPLETE`
or `COMPLETE_SCALE_LIMITED`.

## Prohibited shortcuts

No model/rank/regularization/pooling/frame-count/scene selection using
test data. No silent protocol change after a sealed test exists. No
claim of a result before the sealed evaluation completes. No token-level
claim from this global-pooled study. No modification of any historical
Task 7/audit/diagnostic artifact.

## Scientific interpretation limits

If a model generalizes above controls: "measurable predictive
consistency under the tested conditions" only. If composition/inverse/
identity also hold: "approximate compositional latent transformation
family under tested conditions" only. If no candidate beats controls
after a stabilized curve: "no scene-independent relationship of the
tested forms was supported." If the curve does not stabilize:
scale-limited, no asymptotic claim. Never "V-JEPA understands 3D,"
universality across transformations, or a token-level conclusion from
this global-pooled study.
