# Task 7 Forensic Audit

**Ordered by:** user, before allowing the pipeline to proceed to Task 8.
**Scope:** independent, from-scratch verification of Task 7's negative
result (`state/task_07_result.json`, checkpoint `0367ff8ee585`). This
audit never modified `state/task_07_result.json`, the scientific
protocol, or any file under `experiments/geometric_consistency/`; it
only reads them and writes new artifacts under
`experiments/task7_forensic_audit/`.

**Recorded result being audited** (unchanged, for reference):

| transform | kind | learned W_T R² | best baseline R² |
|---|---|---|---|
| camera_translation | geometric | 0.198 | 0.638 |
| camera_rotation | geometric | -0.176 | 0.079 |
| object_translation | geometric | 0.381 | 0.807 |
| object_rotation | geometric | -0.195 | 0.385 |
| lighting_change | control | 0.070 | 0.467 |
| texture_change | control | 0.426 | 0.870 |

0 of 4 geometric transforms and 0 of 2 controls had the learned map beat
its best baseline.

---

## 1. Pairing integrity — LIKELY_NOT_A_PROBLEM

**Evidence.** 24 randomly sampled (transform, scene) pairs were checked
directly against on-disk artifacts (`pairing_integrity_sample.json`):
directory scene_id, `metadata.json`'s own `scene_id`, and both sides'
`seed` all agreed in all 24/24 cases; `transformation.json`'s recorded
`type` matched its own directory's transform name in all 24/24 cases.
Extended to **all 480 clips** (`full_reproduction_stdout.log`): every
`metadata.json`'s `scene_id` matches its directory, every
`transformation.json`'s `type` matches its directory. A new regression
test, `tests/test_task7_pairing_integrity.py` (6 tests, all passing),
checks this structurally against the pipeline's own on-disk output and
directly exercises the actual production pairing code
(`geometric_consistency_lib.build_arrays`) against a synthetic
per-scene-identity-encoding fake, proving rows are assembled by
`scene_id` dictionary lookup — never by list position.

**Affected code.** `experiments/geometric_consistency_lib.py`'s
`encode_all_pairs`/`build_arrays`; `transforms/pairs.py`'s
`generate_pair`/`load_pair`.

**Why this is not a bug by construction.** Every representation is
stored and consumed as `dict[scene_id]`, never as a positionally-ordered
list zipped against another list — there is no code path where an
independent-sort, shuffled-loader, or batch-order bug could enter.

**Scientific consequence if it were a bug.** Would fully explain a
negative result (comparing scene A's before-representation to scene B's
after-representation is meaningless). Ruled out.

**Confidence.** High (full-population check + new structural test + the
end-to-end reproduction below, which would not reproduce recorded
numbers exactly under a pairing bug).

**Recommended action.** None required. Keep
`tests/test_task7_pairing_integrity.py` in the suite as a standing
regression guard.

---

## 2. Transformation integrity — LIKELY_NOT_A_PROBLEM

**Evidence** (`transform_integrity_sample.json`, 24 sampled pairs): for
every sample, every variable the transform declares `fixed` was
bit-identical (`atol=1e-9`) between original and transformed metadata,
and every variable it declares `changed` actually differed. 0 failures
in 24/24 samples, covering 7–18 fixed-variable checks and 1–3
changed-variable checks per sample depending on transform type.

**Affected code.** `transforms/scene_transform.py`'s `apply_transform`
(computes `T @ old_pose_matrix` once, and the same matrix is what gets
recorded — see that module's own docstring); cross-checked here against
the *rendered metadata*, not just the transform's self-report.

**Scientific consequence if it were a bug.** A transform silently
touching more than its declared variables would produce an uncontrolled
confound; touching less/nothing would make "geometric transform" and
"appearance control" indistinguishable. Ruled out for the sample checked.

**Confidence.** High for the sampled 24; not exhaustively re-checked for
all 480 (would be redundant with `verify_transform_ground_truth`, which
already runs this exact check for every single pair at render time as
part of the actual pipeline, per `geometric_consistency_lib.py`'s
`render_transform_pairs`, and would have raised `ValueError` and aborted
the run had it ever failed).

**Recommended action.** None required.

---

## 3. Video pair integrity — LIKELY_NOT_A_PROBLEM

**Evidence.** All 480 clips checked (`full_reproduction_stdout.log`):
single distinct resolution `{128}`, fps `{4.0}`, frame count `{4}`,
dtype `{uint8}`, shape `{(4, 128, 128, 3)}`. 0 issues. Original and
transformed sides of every pair use identical preprocessing by
construction (the same `render_transform_pairs`/`generate_pair` call
path, same `resolution`/`num_frames`/`fps` arguments, for both sides in
one call).

**Affected code.** `generation/bpy_renderer.py`, `transforms/pairs.py`.

**Confidence.** High (full population, not a sample).

**Recommended action.** None required.

---

## 4. Render determinism — LIKELY_NOT_A_PROBLEM

**Evidence** (`render_determinism.json`): the same scene rendered twice,
independently, for two different transforms (`camera_rotation`,
`lighting_change`) produced bit-identical RGB arrays (`max_abs_pixel_diff:
0`, `bit_identical: true`) for both original and transformed sides, and
the 4 frames within one clip are bit-identical to each other (expected —
these are static, zero-motion clips). Independently corroborated at full
scale: `full_reproduction.py` found the `original` side's rendering
**bit-identical across all 6 transform folders for every one of the 40
scenes** (240-way cross-check, `original_render_cross_transform_identity.json`)
— i.e., re-rendering the same scene 6 separate times (once per transform
run) never produced a different image.

**Affected code.** `generation/bpy_renderer.py` (`scene.cycles.samples =
32`, `use_denoising = True`, no explicit RNG seed set for Cycles —
apparently not needed for reproducibility on this CPU rendering path, but
this is empirical, not a documented Cycles guarantee — see Recommended
action).

**Confidence.** High.

**Recommended action.** None required for Task 7's conclusions. For
future robustness, consider explicitly documenting/pinning Cycles'
sampling determinism (e.g. `scene.cycles.seed`) rather than relying on
its default behavior remaining deterministic across Blender versions.

---

## 5. Encoder determinism — LIKELY_NOT_A_PROBLEM, with one MAJOR side finding

**Evidence** (`encoder_determinism.json`): encoding the same clip twice
produced bit-identical token tensors (`relative_l2_diff: 0.0`,
`cosine_similarity: 1.0`). `model.training == False` (eval mode) and
`any_param_requires_grad == False` confirmed directly on the live model
object, not just inferred from code reading. Cross-scene sanity: cosine
similarity between two different scenes' representations (0.495) is
clearly lower than the same-clip repeat-encode similarity (1.0),
confirming the encoder is not collapsed to a constant output.

**MAJOR side finding: severe temporal-context mismatch.** The loaded
checkpoint's own `config.json` declares `frames_per_clip: 64,
tubelet_size: 2` (32 native temporal tubelets) — the checkpoint's own
name, `vjepa2-vitl-fpc64-256`, literally encodes "frames per clip 64."
This project renders 4-frame clips
(`configs/experiments/task7_geometric_consistency.yaml`'s
`num_frames: 4`), i.e. **2 of the 32 temporal tubelets the model was
pretrained with — 6.25% of its native temporal context.** Traced the
actual `transformers` source
(`transformers/models/vjepa2/modeling_vjepa2.py`): `VJEPA2RopeAttention`
computes position ids from the *actual* input token count at inference
time (`get_position_ids`), not a fixed-size learned positional-embedding
buffer, so this does not crash, truncate, or silently misindex — encoding
verifiably produces the correct `(512, 1024)` token shape for a 4-frame,
128×128 input at this checkpoint's native `crop_size=256`. But it is
still a large out-of-distribution shift in temporal sequence length
relative to pretraining, which is a plausible, real contributor to
degraded representation quality, independent of any code bug.

**Affected code.** `encoders/vjepa.py`; upstream
`transformers.models.vjepa2`.

**Scientific consequence.** Running the model this far outside its
pretrained temporal-context regime could plausibly produce lower-quality
or differently-structured representations than encoding a full 64-frame
(16-second-equivalent) clip would. This is not something this audit can
quantify without re-running the experiment (out of scope — "do not
change the experiment yet"), so it is reported as a limitation, not
falsified or confirmed as *the* explanation.

**Confidence.** High that the mismatch exists and does not crash;
moderate/untested that it materially degrades representation quality for
this specific static-scene, single-instant-in-time use case (unlike
video classification, "temporal context" may matter less for encoding
one static instant repeated across 4 frames).

**Recommended action.** For a future task, consider rendering `num_frames`
closer to (or interpolated/repeated up to) 64 and comparing R² — this is
a testable, falsifiable follow-up, not something to guess at here.

---

## 6. Representation inspection — factual, no issue

**Traced directly from code + a live model object**, not assumed:

- Model/checkpoint: `VJEPA2Model`, `facebook/vjepa2-vitl-fpc64-256`.
- Path used: `encoders/vjepa.py`'s `VJEPAEncoder.encode()` →
  `model.get_vision_features(pixel_values)` → `forward(...,
  skip_predictor=True).last_hidden_state`. The mask/predictor path
  (`VJEPA2PredictorEmbeddings`, used only for JEPA-style masked-prediction
  pretraining) is **not** exercised by this project at all.
- Output tensor shape for this project's actual clips: `(512, 1024)`
  (verified via live `encoder.encode()` call, not inferred) — 512 = 2
  temporal tubelets × 16×16 spatial patches (`crop_size=256`,
  `patch_size=16`), 1024 = `hidden_size`.
- Both temporal and spatial tokens are retained in the raw (unpooled)
  output; `encoders/vjepa.py`'s separate `mean_pool()` function (called
  explicitly by `geometric_consistency_lib.py`, not inside `encode()`
  itself) collapses **both** dimensions together into one `(1024,)`
  vector.
- Normalization: ImageNet mean/std, applied identically to every clip —
  a fixed constant, not fit from data (no leakage surface).
- dtype: float32 throughout.

No issue here per se — this section exists to make section 7/14's
findings below traceable to exact code, not to flag a problem on its own.

---

## 7. Representation statistics — MAJOR (anisotropy) + informative negative correlation

**Evidence** (`representation_statistics.json`, computed on the
independently re-derived real Z/Z' arrays, N=40 per transform):

- **Effective rank ≈ 17–19** out of an ambient dimension of 1024, for
  both `Z` and every `Z'`. The top single PCA component explains ~20% of
  variance; the top 10 explain ~75–77%; the top 50 explain ~100%.
- **Mean pairwise cosine similarity ≈ 0.86–0.88** across all six
  representation sets (original and every transformed side) — the
  representation cloud occupies a narrow cone, not an isotropic sphere.
  This is a well-documented phenomenon in pretrained transformer
  embeddings ("representation anisotropy"), not unique to this pipeline.
- **Transform effect size is real, not negligible**: mean
  `||Z' - Z|| / ||Z||` ranges from 0.121 (texture_change) to 0.308
  (camera_rotation) — every transform measurably moves the pooled
  representation, by double-digit percentages of its own norm.
- **Striking, unexpected negative correlation (r = −0.89)** between a
  transform's effect *size* (`delta_vs_Z_norm_ratio_mean`) and its
  learned-W_T test R² across the six transforms:
  `camera_rotation` has the *largest* average effect (0.308) and the
  *worst* R² (−0.176); `texture_change` has one of the *smallest*
  average effects (0.121) and the best R² (0.426, still below baseline).
  This means the transforms that move Z the most are the *least*
  consistent/predictable across scenes, not the most — the opposite of
  what a simple "bigger effect = easier to fit" story would predict.

**Scientific consequence.** The anisotropy means the *effective*
dimensionality of the fitting problem is closer to ~20–50 than to the
ambient 1024 (see section 8's revised analysis, which uses this directly
rather than an idealized isotropic assumption). The negative correlation
is itself a substantive, reportable finding: it is evidence *against* "Z
just doesn't move under any of these transforms" (it clearly does move)
and instead suggests the transforms' effects on Z are scene-idiosyncratic
rather than governed by one shared global linear rule — which is
precisely what a genuine "no shared linear equivariance" result would
look like, as opposed to a power/pooling artifact that would show *no*
consistent effect direction at all.

**Confidence.** High (full N=40 per transform, exact re-derivation, not
a subsample).

**Recommended action.** Report the anisotropy and the effect-size/R²
anti-correlation explicitly in any interpretation of Task 7's negative
result — both are informative, and neither was reported in the original
`state/task_07_result.json`.

---

## 8. Regression conditioning — MAJOR, but more nuanced than initially assumed (see revision below)

**Evidence.**

- `N_train = 32`, `D = 1024` (hidden size). Naive parameter count for a
  full affine map `W_T ∈ R^{1024×1024}, b ∈ R^{1024}`: **1,049,600
  parameters fit from 32 samples** — `D² ≈ 1.05M ≫ N_train`.
- Fitting method: `sklearn.linear_model.Ridge(alpha=10.0,
  fit_intercept=True)` (`probes/linear_rep_transform.py`) — **not** raw
  OLS; ridge always has a unique, well-posed solution regardless of
  `D` vs. `N`, so this is not a literal singularity/crash risk, only a
  statistical-power one.
- **Synthetic sanity test, idealized isotropic case**
  (`synthetic_sanity.json`): with a KNOWN, exactly linear, low-noise
  (`noise_std=0.01`) synthetic ground truth `Z' = AZ` at the exact real
  shape (`N_train=32, D=1024`), the identical `evaluate_transform` code
  **fails to recover the known mapping** (test R² = −0.149), regardless
  of ridge `alpha` (tested 0.01 and 10.0, both catastrophic — alpha is
  not the bottleneck). A control at an easy, well-determined shape
  (`D=8`) with the same code recovers the known mapping almost perfectly
  (R² = 0.998), proving the *fitting/evaluation code itself* is correct
  — the isotropic real-shape failure is a genuine sample-size effect, not
  a bug. A sample-size sweep at `D=1024` shows recovery only becomes
  reliable around **N_train ≈ 1000–2000** for an isotropic input
  distribution.
- **Revised, more realistic test using the REAL Z data's actual (anisotropic,
  effective-rank≈18) distribution** (`real_Z_synthetic_A_test.json`): applying a
  KNOWN synthetic linear map to the *real* re-derived `Z` array (not
  idealized isotropic noise) and fitting at the exact same `N_train=32`
  recovers the known mapping with **test R² = 0.614, cosine similarity
  0.978** — clearly above every baseline (best baseline R² = −0.14 in
  this control). **This substantially revises the isotropic-only
  finding**: because real V-JEPA representations are highly anisotropic
  (effective rank ≈ 18, not 1024), this exact `N_train=32` protocol
  *can* detect a strong, exact linear signal when one is actually
  present in representations shaped like the real ones — R²=0.614 is
  well short of the ~0.998 ceiling seen in the easy D=8 control, so
  detection power is still measurably reduced relative to an
  ideal/large-sample regime, but "32 samples can never work at D=1024"
  is **not** an accurate characterization once the real data's structure
  is accounted for.

**Affected code.** `probes/linear_rep_transform.py`,
`metrics/equivariance.py`, `experiments/geometric_consistency_lib.py`'s
`evaluate_transform`.

**Also relevant: research/RESEARCH_INVARIANTS.md invariant 8** ("exhausting
the linear-probe design space (regularization strength, pooling choice)
... before ... recording why linear was insufficient"). Task 7 used a
single fixed `alpha=10.0` and a single pooling choice (mean-pool) with no
ablation recorded. This audit's alpha sweep (0.01 vs. 10.0, isotropic
case) found alpha is not the limiting factor here, partially addressing
this invariant's intent — but pooling choice was never varied, which
section 14 flags separately.

**Scientific consequence.** The regression has genuinely reduced
statistical power relative to an ideal large-sample regime (R²≈0.61,
not ≈1.0, ceiling for a *known, exact, generalizing* linear signal at
this N/D/anisotropy) — enough to plausibly miss a **partial or noisy**
real equivariance effect, but not so little power that *any* real
signal comparable in strength to what was synthetically tested would go
undetected entirely. This moves the interpretation from "the experiment
is underpowered to the point of being uninformative" to "the experiment
has real, quantifiable, but not totally disqualifying, power
limitations" — a materially different and more precise conclusion than
either extreme.

**Confidence.** High for both synthetic results (exact code path,
directly measured); the real-world implication (how strong V-JEPA's
"true" equivariance signal would have to be to survive this power level)
is not directly measurable without ground truth and is stated as a
bound, not a certainty.

**Recommended action.** For any task that revisits this question,
increase `num_scenes` substantially (the R²=0.614 real-data control at
N_train=32 vs. ≈0.9+ typically achievable with more data suggests real
headroom) and/or report a held-out-fit learning curve (R² vs. N_train)
alongside the single-N_train number, so a reader can see where on this
curve the actual result sits.

---

## 9. Normalization — LIKELY_NOT_A_PROBLEM

**Evidence (traced through code, not assumed).**

- Video preprocessing normalization (`encoders/vjepa.py`'s
  `_preprocess`): fixed ImageNet mean/std constants, identical for every
  clip regardless of split — no data-dependent fitting, hence no
  train/test leakage surface at all here.
- Regression-level "normalization": `sklearn.Ridge(fit_intercept=True)`
  centers `X`/`y` internally using **training-data statistics only**
  (standard, well-documented sklearn behavior) and stores the resulting
  intercept; `LinearRepTransform.predict()`
  (`probes/linear_rep_transform.py`) applies the stored `W`/`b` to new
  (test) `Z` directly — no separate test-time re-fitting or
  re-normalization step exists anywhere in this path.
- `mean_baseline.py`'s mean is explicitly documented and implemented as
  `Z_prime_train.mean(axis=0)` — train-only, per its own docstring citing
  research/RESEARCH_INVARIANTS.md invariant 16 directly.
- No separate/duplicate normalization step (e.g. a Z-score standardization
  of `Z` before ridge) exists anywhere in this pipeline outside of
  Ridge's own internal training-only centering.

**Confidence.** High (full code trace, not sampled).

**Recommended action.** None required.

---

## 10. Metric audit — LIKELY_NOT_A_PROBLEM, one documented subtlety

**Evidence (traced through `metrics/common.py`).** `r_squared(pred,
target)` computes `SS_tot` using **the evaluation set's own mean**
(`target.mean(axis=0)`), not a training-set mean — documented explicitly
in that function's own docstring as the "simple total-sum-of-squares
form, ... exact for a single held-out evaluation set." This is a
legitimate, standard definition of R² for a fixed evaluation set (`R² > 0`
means "beats predicting this set's own mean"), verified correct by the
synthetic identity test (section 12/13: persistence baseline scores
exactly R²=1.0 when `Z'=Z`, as it must) and by the `_as_samples_by_features`
1-D-vs-2-D reshape fix documented in that module (confirmed present and
correctly applied — `metrics/common.py`'s own docstring records this was
a real historical bug, now fixed and covered by
`tests/test_metrics.py`). `R² > 0` in every reported number was verified
to mean "beat predicting the target's own mean," consistent throughout
every one of the four result blocks (`learned_W_T`, `persistence_baseline`,
`mean_baseline`, `random_pair_control`) — all four call the exact same
`r_squared`/`cosine_similarity`/`relative_l2_error` functions
(`experiments/geometric_consistency_lib.py`'s `evaluate_transform`),
so the comparison in the report ("R² exceeded/did not exceed best
baseline") is apples-to-apples.

**Affected code.** `metrics/common.py`.

**Confidence.** High.

**Recommended action.** None required; the docstring is already
explicit enough that this is not a silent trap, but a reader unfamiliar
with this specific R² convention could mistakenly expect a training-mean
baseline — worth a one-line mention in future reports for readers outside
this codebase.

---

## 11. Baseline audit — LIKELY_NOT_A_PROBLEM

**Evidence (traced through code).**

- **Persistence** (`baselines/identity_baseline.py`): `pred = Z_test`
  (no fitting at all) — correctly receives *only* test-split data, as it
  must (it has no fitted parameters to leak into).
- **Mean** (`baselines/mean_baseline.py`): `pred =
  broadcast(mean(Z_prime_train))` — explicitly documented and verified to
  use **only** `Z_prime_train`, never touching `Z_prime_test` in fitting
  (matches research/RESEARCH_INVARIANTS.md invariant 16, cited directly in
  that file's own docstring).
- **Random-pair (shuffled-pairing) control**
  (`baselines/shuffled_pairing_baseline.py`): fits `LinearRepTransform`
  on `(Z_train, Z_prime_train[shuffled_permutation])` — same
  fitting/evaluation code as the real learned map, only with
  deliberately mismatched training correspondences — a real negative
  control for the *fitting procedure*, not a strawman (uses ridge, same
  alpha, same code path).
- All three, plus the learned `W_T`, are computed from the exact same
  `Z_train/Zp_train/Z_test/Zp_test` arrays passed into
  `evaluate_transform` in one call — no separate preprocessing path per
  baseline that could introduce an apples-to-oranges comparison.

**Confidence.** High (full code trace).

**Recommended action.** None required.

---

## 12 & 13. Synthetic sanity / known-transformation tests — DECISIVE, see section 8

Covered in full under section 8 above (the recovery test *is* the
regression-conditioning finding). Summary of what these tests establish
on their own terms:

- The fitting/prediction/metric code correctly recovers a known linear
  map when the problem is well-determined (`D=8`: R²=0.998).
- The persistence baseline correctly scores R²=1.0 exactly when `Z'=Z`.
- The constant-target edge case (`Z'` independent of `Z`, and literally
  constant across all samples) correctly returns `nan` for every method,
  per `r_squared`'s own documented behavior when `SS_tot≈0` — an
  expected degenerate edge case, not a bug (flagged here only so it
  isn't mistaken for one).
- At the real experiment's exact shape with idealized isotropic input,
  recovery fails (R²=−0.15); using the real anisotropic `Z` data instead,
  recovery partially succeeds (R²=0.61) — see section 8.

**Verdict for these two sections on their own: LIKELY_NOT_A_PROBLEM** —
the pipeline code is correct; whatever the negative result reflects, it
is not a bug in the fitting/evaluation machinery.

---

## 14. Pooled vs. dense representation — MAJOR (methodological limitation, not changed)

**Evidence.** `encoders/vjepa.py`'s `mean_pool()` collapses the full
`(512, 1024)` token sequence — 2 temporal tubelets × 16×16 spatial
patches — to a single `(1024,)` vector via an unweighted mean over **all
512 token positions simultaneously** (temporal and spatial together, not
separately). 511 of 512 tokens' individual spatial/temporal identity is
discarded; only their average survives.

**Scientific consequence (not a claim that pooling is wrong, per the
audit instructions — a documented limitation).** A camera rotation's
dominant effect is a rearrangement of *which spatial patch sees which
scene content* (parallax, occlusion changes, content shifting across the
16×16 grid) more than a change in the *marginal distribution* of
patch-level content. Mean-pooling is specifically insensitive to this
class of effect: summing over all patch positions destroys the very
positional information a purely geometric rearrangement acts on. This is
a directly plausible, specific (not generic) methodological reason a
real per-token geometric signal could exist in the dense representation
while being substantially attenuated in the pooled one used for fitting.

**Affected code.** `encoders/vjepa.py`'s `mean_pool`;
`experiments/geometric_consistency_lib.py`'s `encode_all_pairs`, which
calls it unconditionally.

**Confidence.** High that pooling discards this information as a
mathematical fact; untested (out of scope for this audit, which does not
re-run the experiment) whether it materially changes Task 7's
conclusion.

**Recommended action.** A follow-up task, not this audit, could compare
against a spatially-aware pooling (e.g. keep the two temporal tubelets
separate, or fit per-spatial-position and aggregate) as a properly
controlled variant — never as a way to "fix" a disliked negative number,
but as a distinct, pre-registered scientific question ("does token-level
structure carry the signal that mean-pooling discards?").

---

## 15. Training sample size — MAJOR (same finding as section 8, stated plainly)

**Evidence.** `N_train = 32`, `N_test = 8`, `D = 1024`. Directly reported
in `state/task_07_result.json`'s `dataset` block, confirmed against the
40 unique scene directories on disk.

**Scientific consequence.** See section 8: this is not, by itself,
sufficient to make the pipeline blind to a real strong linear signal in
representations shaped like V-JEPA's actual output (R²=0.61 recovery
demonstrated), but it does measurably reduce power relative to a larger
sample, and no learning curve or power analysis was reported alongside
the single N=32 result.

**Recommended action.** Same as section 8: a larger `num_scenes` and/or
a reported learning curve would let a future result state its
statistical power explicitly rather than leaving it implicit.

---

## 16. Compliance: no result-optimization was performed

This audit did not change the metric, cherry-pick scenes, change the
test split, remove a control, alter transform magnitudes, remove
examples, unfreeze the encoder, introduce a nonlinear probe, or use test
data for tuning. Every check above either (a) re-derives Task 7's own
recorded numbers independently and compares, or (b) runs a wholly
synthetic control with a known ground truth, entirely separate from
`experiments/geometric_consistency/`'s real data. `state/task_07_result.json`
was read-only throughout.

---

## Reproducibility check (bonus, not one of the 16 sections but load-bearing for several verdicts above)

A complete, independent re-derivation was performed:
`experiments/task7_forensic_audit/full_reproduction.py` built a **fresh**
`VJEPAEncoder` instance, re-encoded **every one of the 480 real cached
`rgb.npy` files** from scratch (never reading Task 7's in-memory Z arrays
— those were never persisted to disk in the first place, see note below),
and re-ran the exact same `evaluate_transform` call Task 7 used, for
every one of the six transforms.

**Result: max absolute difference across every transform × method ×
metric = 0.0 (exact).** Every one of the 6×4×3 = 72 recomputed numbers
matches `state/task_07_result.json` bit-for-bit
(`reproduction_diff.json`). This is strong, direct evidence against a
pairing bug, a caching bug, or run-to-run non-determinism: an entirely
independent process, run over a day later, with a freshly-loaded model
and freshly-read files, reproduced the original numbers exactly.

**Process note, not a bug:** Task 7's own pipeline
(`geometric_consistency_lib.encode_all_pairs`) computes `Z`/`Z'` in
memory and never writes them to disk — only the final aggregated
metrics land in `state/task_07_result.json`. This audit had to
re-encode from the cached `rgb.npy` files to get raw representations for
sections 7/12/13 at all. Worth noting for future tasks: persisting raw
`Z`/`Z'` alongside the result JSON would make this kind of audit
(and any future re-analysis) faster and require no re-encoding.

---

## Final classification of every issue

| # | Section | Classification |
|---|---|---|
| 1 | Pairing integrity | LIKELY_NOT_A_PROBLEM |
| 2 | Transformation integrity | LIKELY_NOT_A_PROBLEM |
| 3 | Video pair integrity | LIKELY_NOT_A_PROBLEM |
| 4 | Render determinism | LIKELY_NOT_A_PROBLEM |
| 5a | Encoder determinism | LIKELY_NOT_A_PROBLEM |
| 5b | Temporal-context mismatch (6.25% of native) | MAJOR |
| 6 | Representation inspection | (factual; no issue) |
| 7 | Representation statistics (anisotropy) | MAJOR |
| 8 | Regression conditioning (N=32, D=1024) | MAJOR |
| 9 | Normalization | LIKELY_NOT_A_PROBLEM |
| 10 | Metric audit | LIKELY_NOT_A_PROBLEM |
| 11 | Baseline audit | LIKELY_NOT_A_PROBLEM |
| 12/13 | Synthetic sanity / known-transform recovery | LIKELY_NOT_A_PROBLEM (pipeline code) |
| 14 | Pooled vs. dense representation | MAJOR |
| 15 | Training sample size | MAJOR (same root cause as #8) |
| — | Reproducibility (bonus) | LIKELY_NOT_A_PROBLEM (exact match) |

**No BLOCKER-level issue was found.** No pairing bug, no transform bug,
no rendering non-determinism, no encoder non-determinism, no metric bug,
no baseline bug, and no normalization/leakage bug. The from-scratch
reproduction matches exactly. The four MAJOR findings (temporal-context
mismatch, representation anisotropy, regression conditioning/sample
size, and mean-pooling's loss of spatial/temporal structure) are all
genuine, quantified, real methodological limitations of the *power* and
*representational granularity* of this specific protocol — none of them
is a defect that produced an incorrect or fabricated number; all of them
bound how much a null result here can be trusted to generalize.

---

## FINAL DECISION

**C. INCONCLUSIVE** — with an important qualification: this is not
"the audit found nothing decisive." It found, decisively, that:

1. **The pipeline is implemented correctly.** No bug was found in
   pairing, transform application, rendering, encoding, metrics, or
   baselines, across every check performed, including a from-scratch,
   bit-exact reproduction of all 72 recorded numbers and a synthetic
   ground-truth recovery test that isolates the fitting code from the
   real data entirely.
2. **The protocol has real, quantified power limitations** that a valid
   negative result should be read alongside, not instead of: mean-pooling
   discards the spatial/temporal structure a geometric transform most
   directly acts on (section 14); the encoder ran at 6.25% of its native
   temporal context (section 5); and the N_train=32/D=1024 regression,
   while empirically shown to detect a strong exact linear signal in
   representations shaped like the real data (R²=0.61, section 8), has
   real, non-trivial headroom below the ~0.95+ ceiling a larger sample
   would give — enough to plausibly miss a **partial** true effect even
   though it is not so underpowered as to explain away the *entire*
   observed result on its own.

This is why **A (valid negative result)** would overstate confidence —
the audit did not, and structurally could not without re-running the
experiment at more scenes and/or with dense (unpooled) representations,
rule out that these specific power/pooling limits are hiding a real,
weaker signal. It is also why **B (implementation/methodology failure)**
would overstate the opposite — no defect was found that fabricates or
invalidates the reported numbers; they are an honest, correctly-computed
output of a correctly-implemented pipeline running with known,
quantified constraints.

**Recommended next action (not performed by this audit, and explicitly
not a re-run of Task 7 to "fix" the number):** a follow-up task that (a)
substantially increases `num_scenes` and reports a learning curve
(R² vs. N_train) rather than a single N, and (b) as a separate,
pre-registered comparison, evaluates equivariance on a
spatially/temporally-structure-preserving representation (not full
mean-pooling) alongside the pooled one — both changes targeted
specifically at the two MAJOR limitations this audit identified as most
load-bearing (sample size and pooling), not at manufacturing a more
positive-looking number.
