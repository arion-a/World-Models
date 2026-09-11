# Task 7B Protocol (frozen, hash `44bff5c8c1aea7a5`)

**Master population:** 1463 scenes (base_seed=0)
**Split:** train_pool=1024, val=219, test=220
**N_train schedule:** [64, 128, 256, 512, 1024] (5 seeds each)
**Transform:** camera_rotation, magnitudes [-60.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 60.0] deg, primary=30.0 deg

## Selection & stopping rules

- Selection: At each N, choose the candidate/hyperparameter combo with highest mean VALIDATION R^2. Ties within 0.01 -> lower parameter count wins, then earlier position in the M0-M6 hierarchy.
- Stopping: Adequately powered if the final two scheduled N's mean validation R^2 differ by <0.02 AND their 95% seed-bootstrap CIs overlap. Otherwise the full schedule is run and the curve is labeled NOT_STABILIZED.
- Sealed test: The sealed test command evaluates ONLY the single already-selected (model, hyperparameter) configuration plus all pre-specified controls, exactly once, on the fixed test split. A rerun is permitted only for a documented software failure discovered BEFORE result inspection.

## Compute budget

N_train=1024 requires a 1463-scene master population (70% train pool = 1024, per compute_population_size()). Measured throughput this session: ~5s/scene render+encode at 4 frames/128px. Stage 1's full population was rendered as a single ~2 hour background job -- an explicit choice confirmed with the user (not a silent scope cut) given the alternative of capping the schedule at N_train=512. Stage 2/3 (the 7 additional magnitudes' structure tests) use a much smaller, separately fixed scene subset (not the full 1024-scene pool) since algebraic-property verification does not need learning-curve-scale N -- see STRUCTURE_TEST_SUBSET_SIZE in run_structure_tests.py.

## Protocol notes

- Task 7B is a new, independent discovery study -- it does not reuse Task 7's 40-scene population or its state/task_07_result.json in any fitting or evaluation step. Its own master population (see master_population_size) is sampled fresh via the same deterministic sampler (geometric_consistency_lib.sample_scenes) Task 6/7 use, with Task 7B's own base_seed, so its scene set is disjoint in general from Task 7's (both start numbering at scene_0000, but are independent draws from the sampler and are never cross-referenced).
- camera_rotation's azimuth is pinned to a single fixed elevation of 0.0 degrees for the entire Task 7B study (transforms.scene_transform.TransformConfig.fixed_elevation_deg), making it a pure single-axis (world +Z) yaw -- verified, not assumed, to satisfy an exact one-parameter rotation-group composition law in tests/test_task7b_composition_law.py before this protocol relies on it for Stage 3's structure tests.
- N_train=1024 (the full training pool) was included in the schedule per an explicit, session-compute-budget decision made WITH the user before any rendering began -- not silently dropped. See compute_budget_note below.

## Immutable historical paths (never modified by Task 7B)

- `state/task_07_result.json`
- `experiments/geometric_consistency/`
- `experiments/task7_forensic_audit/`
- `experiments/task7_diagnostic/`
