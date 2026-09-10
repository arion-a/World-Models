# Task 6 — First camera-rotation geometric consistency experiment

**Depends on (do not redo, do not modify without explicit justification
recorded in the task result — invariant 15):** Tasks 1–5. Concretely:
`generation/generate.py` (`generate_scene`, `generate_trajectory`),
`transforms/pairs.py` (`generate_pair`), `transforms/scene_transform.py`
(`camera_rotation`), `encoders/vjepa.py` (`VJEPAEncoder`, `mean_pool`),
`encoders/extract.py`.

## Objective

Build and run the first real geometric-consistency experiment using the
Task 5 frozen-encoder pipeline (not V0's older pooled `FrozenEncoder` /
`encoders/vjepa2.py` path — this is a new, independent experiment
script under `experiments/` or a new `research_experiments/` module,
your choice, as long as it imports the Task 2/3/5 interfaces rather than
duplicating them).

## Scientific question

Does a fixed, known camera rotation applied to a rendered 3D scene
induce a **predictable, linear** transformation in the frozen encoder's
representation space — one that **generalizes to scenes never seen while
fitting that linear map**? This is the first direct test of the
project's central hypothesis:

```
Z' ≈ W_T Z
```

for `T = camera_rotation`, `W_T` fit on TRAIN scenes only, evaluated on
TEST scenes.

## Implementation requirements

1. **Scenes.** Sample at least 40 scenes with `generation.scene_sampler.
   sample_scene` (or `generation.generate.generate_scene`), each with a
   distinct seed. Split scene IDs into train/test **before** any
   rendering or encoding happens (scene-level split — invariants 4–5).
   An 80/20 or similar split is fine; record the exact fraction and seed
   used to compute it.
2. **Transform.** For each scene, apply `camera_rotation` via
   `transforms.scene_transform.apply_transform` (through `transforms.
   pairs.generate_pair`, which renders both the original and transformed
   side with full ground truth). Use a **fixed** rotation magnitude for
   this first experiment — 30 degrees azimuth, per the task brief, using
   whatever `TransformConfig` field already controls `camera_rotation`'s
   magnitude in `transforms/scene_transform.py` (do not invent a second,
   parallel way to specify the angle — inspect and reuse the existing
   one, only setting the magnitude explicitly instead of leaving it at
   its sampled default if it is currently randomized).
3. **Rendering.** Reuse `transforms.pairs.generate_pair` — do not write
   a new renderer call path. A handful of frames per clip (matching
   Task 2/3's existing defaults, e.g. 4–8) is enough; this experiment
   does not require long clips.
4. **Encoding.** Encode both the original and transformed clip with
   `encoders.vjepa.VJEPAEncoder` (`pretrained=True` — use real weights;
   if the environment cannot reach Hugging Face Hub when this task runs,
   the encoder's own documented fallback applies and **the resulting
   `pretrained: false` must be recorded and surfaced**, not hidden).
   Reduce each clip's token sequence to a single vector with `encoders.
   vjepa.mean_pool` before fitting anything — ridge regression over the
   raw unpooled `(num_tokens, hidden_size)` tensor is both
   dimensionally unreasonable for ~40 scenes and not what "predictable
   transformation of the representation" should mean here; pooling
   first is a deliberate, documented choice, not an oversight.
5. **Fitting `W_T`.** Use `probes.linear_rep_transform.LinearRepTransform.
   fit` (ridge regression, reuse — do not reimplement) on
   `(Z_train, Z'_train)` only.
6. **Evaluation.** Use `metrics.equivariance.evaluate_equivariance`
   (reuse) on `(Z_test, Z'_test)` for R², mean cosine similarity, mean
   relative L2 error.

## Required controls (all evaluated on the exact same test split, exact
same metrics, as the learned `W_T` — invariant 14)

1. **Persistence baseline** (`Z_hat' = Z`) — reuse `baselines.
   identity_baseline.evaluate_identity_baseline`.
2. **Mean transformed-representation baseline**
   (`Z_hat' = mean(Z'_train)`, constant prediction for every test scene)
   — this does not exist yet in `baselines/`; add it as
   `baselines/mean_baseline.py`, following the exact structure/style of
   `baselines/identity_baseline.py` (a dataclass result + one function),
   computing the metrics with `metrics.common`'s existing primitives.
3. **Random-pair control** — reuse `baselines.
   shuffled_pairing_baseline.evaluate_shuffled_pairing_baseline` (fits
   `W_T` on deliberately mismatched train pairs; a real test of whether
   the fitting procedure itself is finding genuine correspondence).

## Metrics (same three for the learned map and every control, per scene split)

- Held-out R² (`metrics.common.r_squared` via `evaluate_equivariance`)
- Mean cosine similarity between predicted and actual `Z'`
- Mean relative L2 error

## Required artifacts

- A runnable script/module (e.g. `experiments/task6_camera_rotation.py`
  or `research_experiments/task6_camera_rotation.py`) with a `main()`
  callable via `python -m ...` and a `--config` flag for reproducibility.
- Rendered scene data on disk (reuse `transforms.pairs.generate_pair`'s
  own output layout; do not invent a new one).
- `state/task_06_result.json` (see schema below).
- A short human-readable report (markdown) is optional but encouraged;
  `state/task_06_result.json` is the QA-checked artifact of record.

## `state/task_06_result.json` schema (minimum required fields)

```json
{
  "task": 6,
  "implementation_status": "COMPLETE",
  "scientific_result": "<one paragraph, plain language, no unsupported claims>",
  "transform": "camera_rotation",
  "transform_params": { "...": "exact magnitude/config used" },
  "encoder": {
    "name": "VJEPAEncoder",
    "checkpoint": "facebook/vjepa2-vitl-fpc64-256",
    "pretrained": true,
    "frozen": true,
    "pooling": "mean_pool"
  },
  "dataset": {
    "num_scenes": 0,
    "train_scene_ids": [],
    "test_scene_ids": [],
    "train_fraction": 0.0,
    "base_seed": 0
  },
  "fitting": { "method": "ridge", "alpha": 10.0 },
  "metrics": {
    "learned_W_T": { "r2": 0.0, "mean_cosine_similarity": 0.0, "mean_relative_l2_error": 0.0 },
    "persistence_baseline": { "r2": 0.0, "mean_cosine_similarity": 0.0, "mean_relative_l2_error": 0.0 },
    "mean_baseline": { "r2": 0.0, "mean_cosine_similarity": 0.0, "mean_relative_l2_error": 0.0 },
    "random_pair_control": { "r2": 0.0, "mean_cosine_similarity": 0.0, "mean_relative_l2_error": 0.0 }
  },
  "tests": { "passed": 0, "failed": 0 },
  "artifacts": ["path/to/manifest-or-dataset", "path/to/report.md"],
  "config": "path/to/config used or inline dict",
  "seed": 0,
  "software_versions": { "transformers": "...", "torch": "...", "python": "..." }
}
```

`train_scene_ids`/`test_scene_ids` are required precisely so QA's DATA
LEAKAGE layer can independently check disjointness without re-running
the experiment.

## Tests required

- Unit test(s) for the new `baselines/mean_baseline.py` (mirroring the
  style of existing `tests/` files for `identity_baseline`/
  `shuffled_pairing_baseline` if those exist, or added alongside
  `tests/test_metrics.py`/`tests/test_probes.py`'s existing coverage).
- A test (fast, `pytest -m "not slow"`-safe, using a small synthetic
  in-memory example — no real render/encode needed) that the experiment
  script's scene-level split logic never puts a scene's original and
  transformed representation in different splits, and that train/test
  scene ID sets are disjoint.
- Run the full existing suite (`pytest -m "not slow"`) and confirm no
  regression.

## Leakage checks (independently verified by orchestrator QA, layer E)

- `train_scene_ids` and `test_scene_ids` in the result JSON are disjoint
  sets.
- Every `test_scene_ids` entry's representation was excluded from
  `LinearRepTransform.fit`'s training data — i.e. `Z_train`/`Z'_train`'s
  scene provenance matches `train_scene_ids` exactly.
- No test-split scene contributed to the mean-baseline's computed mean.

## Acceptance criteria

1. At least 40 scenes generated and rendered through both original and
   `camera_rotation`-transformed sides.
2. Real `VJEPAEncoder` (pretrained, per requirement 4) used for both
   sides of every scene.
3. `W_T` fit on train only, evaluated on test only.
4. All three required controls computed on the identical test split.
5. `state/task_06_result.json` present and matches the schema above.
6. No NaN/Inf in any reported metric.
7. Full existing test suite still passes (`pytest -m "not slow"`).
8. Scene-level split integrity holds (see leakage checks).

## Prohibited shortcuts

- Do not feed ground-truth camera pose/rotation matrices into the
  encoder as input (invariant 7). They may be used to *construct* the
  transform and to *label* metadata, never as model input.
- Do not fit `W_T` (or choose its ridge `alpha`) by looking at test
  metrics.
- Do not reduce scene count to make ridge regression trivially "work" —
  40 is a floor, not a target to shrink from if results look weak.
- Do not silently swap `pretrained=False` in to dodge slow/networked
  runs without recording it plainly in the result JSON — see
  requirement 4.
- Do not modify `transforms/`, `generation/`, or `encoders/` core
  modules from Tasks 1–5 to make this experiment's numbers look better;
  if a genuine bug is found there, fix it minimally and record why in
  the result JSON's `protected_files_justification` field.

## Scientific interpretation limits

A high R² / cosine similarity for the learned `W_T`, clearly above all
three baselines, is evidence that camera-rotation's effect on this
frozen encoder's representation is **linearly predictable and
generalizes** — nothing more. It is **not** evidence that the model
"understands 3D," has an internal camera model, or generalizes to
rotations of other magnitudes/axes (that is Task 7's question). A weak
or null result is a valid, complete outcome for this task — see
`research/RESEARCH_INVARIANTS.md` invariants 10–11 — and must be
reported as such rather than reframed as a software failure.
