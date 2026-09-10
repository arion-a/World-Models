# Task 6 — First camera-rotation geometric consistency experiment

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`,
"TASK 6 — First camera-rotation geometric consistency experiment"
section.** This file is that section's content, reorganized under the
header taxonomy the orchestrator's QA (`tests/research/
test_research_alignment.py`) checks against, so it can be sent to
Claude Code verbatim as this task's prompt without any loss of
substance versus the canonical document. If the two ever disagree,
`research/CANONICAL_RESEARCH_PROTOCOL.md` is authoritative — treat any
discrepancy as a defect in this file, to be re-synced, not resolved by
editing the canonical document to match.

**Depends on (do not redo, do not modify without explicit justification
recorded in the task result — Global Invariant 27):** Tasks 1–5.
Concretely: `generation/generate.py` (`generate_scene`,
`generate_trajectory`), `transforms/pairs.py` (`generate_pair`),
`transforms/scene_transform.py` (`camera_rotation`), `encoders/vjepa.py`
(`VJEPAEncoder`, `mean_pool`), `encoders/extract.py`.

## Objective

**Purpose.** Establish the first controlled test of whether a known
physical geometric transformation produces a predictable transformation
in latent representation space.

**Why this task exists.** This is the first experiment directly testing
the central geometric consistency hypothesis with the real Task 4/5
encoder (`encoders.vjepa.VJEPAEncoder`, real pretrained ViT-L/16
weights) and the real Task 2/3 scene/transform pipeline, rather than the
exploratory V0 pipeline that predates the formal task numbering.
Everything later (Tasks 7–18) either generalizes this protocol (Task 7),
asks a related but distinct question about it (Task 8), controls for
confounds in it (Task 9), formalizes its baselines (Task 10),
stress-tests it at scale (Task 11), or extends it into new regimes
(temporal — Task 12, occlusion — Task 13, counterfactual — Task 14).
Getting Task 6's protocol right is a precondition for all of them.

**Relationship to previous tasks.** Inherits directly, without
modification: `generation.scene_sampler.sample_scene`/`generation
.generate.generate_scene` (Task 2 scene sampling),
`transforms.scene_transform.apply_transform` and `transforms.pairs
.generate_pair` (Task 3's transform + matched-pair rendering),
`encoders.vjepa.VJEPAEncoder`/`encoders.extract` (Task 4/5's frozen
encoder). Does **not** modify these modules except for a minimal,
justified bug fix (recorded per Global Invariant 27 /
`protected_files_justification` in the result JSON).

## Scientific question

If the camera undergoes a known rotation, does the frozen V-JEPA
representation change according to a predictable transformation that
generalizes across unseen scenes?

**Hypothesis.** A fixed-magnitude camera rotation applied to a
controlled synthetic scene induces a change in the frozen encoder's
pooled representation that is well-approximated by a single linear map
`W_T`, fit on a set of training scenes, and that this map generalizes
to unseen test scenes better than the persistence baseline, the
mean-transformed-representation baseline, and the shuffled-pairing
(random-pair) control.

**Mathematical formulation.**

```
S' = camera_rotation(S)        (transforms.scene_transform.apply_transform)
V  = R(S),   V' = R(S')        (transforms.pairs.generate_pair, reusing generation's renderer)
Z  = E(V),   Z' = E(V')        (encoders.vjepa.VJEPAEncoder.encode, then mean_pool)

Z' ≈ W_T Z + b_T                (probes.linear_rep_transform.LinearRepTransform.fit, TRAIN scenes only)
```

Evaluated via `metrics.equivariance.evaluate_equivariance` on
`(Z_test, Z'_test)`, per DESIGN.md §6's weaker, aggregate,
parameter-marginalized notion of equivariance (not strict group
equivariance; the homomorphism property is not tested).

## Implementation requirements

**Inputs.**

- A set of at least 40 sampled `SceneState`s (`generation.scene_sampler
  .sample_scene`, one seed per scene), split into train/test at the
  **scene level**, before any rendering.
- A fixed camera-rotation magnitude (e.g. 30 degrees azimuth), set
  explicitly via `TransformConfig`'s existing
  `camera_rotation_azimuth_deg_range` field (e.g. `(30.0, 30.0)`) rather
  than left at its sampled `(10.0, 35.0)` default or specified through a
  new, parallel mechanism — the exact convention (azimuth vs.
  elevation, sign, units) must be inspected from
  `transforms/scene_transform.py` and `transforms/se3.py`, never
  invented.
- `encoders.vjepa.VJEPAEncoder(pretrained=True)` — real pretrained
  weights; if unreachable at run time, the encoder's own documented
  fallback applies and `pretrained: false` **must** be recorded
  plainly, never hidden.

**Outputs.**

- Rendered original/transformed clip pairs for every scene (reusing
  `transforms.pairs.generate_pair`'s output layout — no new renderer
  output convention).
- `state/task_06_result.json` (schema below).
- A runnable script/module under `experiments/` (e.g.
  `experiments/task6_camera_rotation.py`) with a `main()` callable via
  `python -m ...` and a `--config` flag.

**Experimental protocol.**

1. Sample ≥40 scenes with distinct seeds.
2. Assign train/test split at the scene level, before rendering.
3. For every scene, render the original and the fixed-magnitude
   `camera_rotation`-transformed clip via `transforms.pairs
   .generate_pair`.
4. Encode both clips with `VJEPAEncoder`, reduce to a single vector per
   clip with `mean_pool` (pooling choice stated explicitly, not an
   unstated default — ridge regression over the raw unpooled
   `(num_tokens, hidden_size)` tensor is dimensionally unreasonable at
   this scene count).
5. Fit `W_T` on `(Z_train, Z'_train)` only.
6. Evaluate `W_T`, and every baseline (below), on `(Z_test, Z'_test)`.

**Dataset requirements.** ≥40 scenes, each with a distinct seed, each
rendered on both the original and `camera_rotation`-transformed side.
An 80/20 (or similar, explicitly recorded) train/test fraction.

**Train/test protocol.** Scene-level, assigned once before any
rendering. Every variant of `scene_001` (original, rotated) belongs to
the same split as every other variant of `scene_001`. It is forbidden
to place different variants of the same underlying scene into train
and test.

**Controls.**

1. **Persistence** (`Z_hat' = Z`) — `baselines.identity_baseline
   .evaluate_identity_baseline`.
2. **Mean transformed representation** (`Z_hat' = mean(Z'_train)`) —
   does not yet exist; add `baselines/mean_baseline.py`, matching
   `identity_baseline.py`'s structure (a dataclass result + one
   function), using `metrics.common`'s existing primitives.
3. **Random-pair control**, which must destroy the true scene
   correspondence — `baselines.shuffled_pairing_baseline
   .evaluate_shuffled_pairing_baseline`.

**Baselines.** The three controls above are this task's required
baseline set. Task 10 will later formalize DESIGN.md §12's full
four-baseline set (identity, shuffled-pairing, pixel-statistics
encoder, randomly-initialized encoder) as reusable infrastructure; Task
6 is not required to run the pixel-statistics or
randomly-initialized-encoder baselines itself, but must not report
results in a way that would conflict with their later addition (e.g.
must not claim "beats every reasonable baseline" — only "beats
persistence, mean, and random-pair").

**Metrics.** Held-out R² (`metrics.common.r_squared`), mean cosine
similarity, mean relative L2 error — via `metrics.equivariance
.evaluate_equivariance`, computed identically for the learned `W_T` and
all three controls. Metric definitions are fixed by this document and
by `metrics.equivariance`/`metrics.common` before any result is
inspected; the metric must not be chosen or changed after inspecting
results to produce a favorable conclusion.

## Required artifacts

- `experiments/camera_rotation/` (or the task's own output directory,
  consistent with `transforms.pairs.generate_pair`'s layout) containing
  the rendered scene data.
- `state/task_06_result.json`, containing at minimum:

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

**Required software tests.**

- Transformation runs; rendering runs; encoding runs; `W_T` fitting
  runs; prediction runs; metrics run; result files are generated.
- No NaN/Inf anywhere in the result.
- Expected tensor dimensions at every stage (pooled `Z`/`Z'` are
  `(1024,)`; stacked train/test arrays are `(N, 1024)`).
- Unit test(s) for the new `baselines/mean_baseline.py`.
- Full existing suite (`pytest -m "not slow"`) still passes — no
  regression in Tasks 1–5.

**Required scientific-validity tests.**

- The transformation is actually applied (the rendered transformed
  video's ground truth — `changed_variables`/`fixed_variables`/
  `transform_matrix` from `apply_transform` — reflects exactly the
  intended camera rotation and nothing else).
- The transformed video genuinely corresponds to the transformed scene,
  and `Z`/`Z'` genuinely correspond to `V`/`V'` (no accidental
  mismatch/misindexing between scene, render, and representation).
- `W_T` was fit on train scenes only; test scenes were never seen
  during fitting.
- Transformed variants never cross the train/test split.
- The encoder is frozen throughout (parameters unchanged before/after
  encoding, per `tests/test_vjepa_encoder.py`'s existing pattern).
- No ground-truth coordinates (rotation matrices, camera pose) enter
  the encoder as input — only as labels/config for constructing `T` and
  as provenance metadata.

**Required research-alignment checks.** Consistent with
`research/RESEARCH_INVARIANTS.md` and `research
/CANONICAL_RESEARCH_PROTOCOL.md`'s Global Scientific Invariants: frozen
encoder declared and verified, scene-level split declared, baselines
present, seed/config recorded, no fabricated/stub metrics
(`orchestrator/qa.py` layers A/C/D/F).

## Leakage checks

- `train_scene_ids` and `test_scene_ids` are disjoint sets.
- Every `test_scene_ids` entry's representation was excluded from
  `LinearRepTransform.fit`'s training data.
- No test-split scene contributed to the mean-baseline's computed mean.

**Reproducibility requirements.** Fixed seed produces a
bit-reproducible (or documented-tolerance) result across independent
runs; `base_seed`, the fixed rotation magnitude, ridge `alpha`, encoder
checkpoint/`pretrained` flag, and software versions (`transformers`,
`torch`, `python`) are all recorded in `state/task_06_result.json`.

## Acceptance criteria

**Failure conditions** (any of the following is a SOFTWARE FAILURE —
the orchestrator must retry/fix, not accept): missing/invalid result
JSON; NaN/Inf; train/test scene overlap; a baseline silently skipped;
encoder parameters changed by the run; ground-truth coordinates fed to
the encoder; scene count below 40; `alpha` or any hyperparameter chosen
by inspecting test metrics; unjustified modification of Task 1–5
modules.

**Acceptance criteria.**

1. ≥40 scenes generated and rendered through both original and
   `camera_rotation`-transformed sides.
2. Real `VJEPAEncoder` (pretrained per the Inputs section) used for
   both sides of every scene.
3. `W_T` fit on train only, evaluated on test only.
4. All three required controls computed on the identical test split.
5. `state/task_06_result.json` present and matches the schema above.
6. No NaN/Inf in any reported metric.
7. Full existing test suite still passes.
8. Scene-level split integrity holds (leakage checks above).

## Prohibited shortcuts

- Feeding ground-truth camera pose/rotation matrices into the encoder
  as input.
- Fitting `W_T` or choosing its ridge `alpha` by looking at test
  metrics.
- Reducing scene count to make ridge regression trivially "work."
- Silently swapping `pretrained=False` to dodge slow/networked runs
  without recording it.
- Modifying `transforms/`, `generation/`, or `encoders/` to make this
  experiment's numbers look better (a genuine bug fix is allowed, must
  be minimal and justified in `protected_files_justification`).

## Scientific interpretation limits

**Interpretation rules.** Allowed: "The representation exhibits
measurable predictive consistency under the tested camera rotation,
exceeding [baseline] by [margin]." Allowed: "No measurable predictive
consistency was found under the tested protocol." **Not allowed, under
any result:** "V-JEPA understands 3D," or any claim beyond DESIGN.md
§13's "can support" list.

**What a positive result means.** Camera-rotation's effect on this
frozen encoder's pooled representation is linearly predictable and
generalizes to unseen scenes, under this scene distribution, this
pooling scheme, and this rotation magnitude — nothing more.

**What a negative result means.** Under this protocol, no linear map
recovers camera-rotation's effect on `Z` better than trivial controls —
a real, complete, reportable finding, not a task failure.

**What this task does NOT establish.** Generalization to other rotation
magnitudes or axes (Task 7's question); that physical camera state is
*itself* decodable from `Z` (Task 8's question, related but distinct —
DESIGN.md §13); robustness to appearance confounds (Task 9); behavior
at scale (Task 11); anything about temporal dynamics, occlusion, or
counterfactual structure (Tasks 12–14); and, under any outcome, any
claim that the model "understands" 3D, has an internal camera model, or
performs geometric reasoning.
