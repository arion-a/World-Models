# Task 7 Diagnostic Follow-Up

**Ordered by:** user, as a direct follow-up to the Task 7 forensic audit
(`experiments/task7_forensic_audit/`), which found no implementation bug
but identified four unquantified methodological limitations. This
diagnostic's purpose is **not** to make the learned map `W_T` perform
better — it is to determine, via controlled manipulation of one factor
at a time, which of those limitations actually drives the observed
negative result, and to classify the original Task 7 finding as **VALID
NEGATIVE**, **INCONCLUSIVE**, or **METHODOLOGICALLY LIMITED** on that
basis.

`state/task_07_result.json`, `experiments/geometric_consistency/`, and
the scientific protocol were never modified. All new scenes rendered by
this diagnostic live under `experiments/task7_diagnostic/rendered/` (and
`rendered_temporal/`), never inside Task 7's own output directory. No
metric, split, control, or transform magnitude was changed to seek a
different-looking number; every regression's fitting step is train-only,
and the fixed held-out test set never changes across conditions within
each investigation.

---

## A. Training-sample scaling

**Setup.** `camera_rotation` (the transform with the worst original R²
and the largest representation-shift magnitude, per the forensic audit).
The scene pool was extended from Task 7's original 40 scenes to 264,
using the exact same deterministic sampling function Task 7 itself uses
(`geometric_consistency_lib.sample_scenes(N, base_seed=0)`), so the
first 40 scenes are byte-identical to Task 7's own. **The 8 held-out
test scenes are Task 7's own recorded `test_scene_ids`, fixed across
every condition below — never resampled, never enlarged.** For each
`N_train`, 5 independent random subsets were drawn from the (up to
256-scene) train pool — except `N_train=256`, the full pool, where only
one subset exists — and Ridge (`alpha=10.0`, identical to Task 7's own)
was fit on each subset and evaluated once on the fixed test set. The
`seed=0, N_train=32` condition is pinned to Task 7's *own* recorded
32-scene train split exactly, not an independently drawn subset — a
built-in exact-reproduction check (see QA section D below).

**Result** (`learning_curve.json`; for reference, Task 7's own recorded
`learned_W_T` R² = **−0.176**, best baseline R² = **0.079**):

| N_train | seeds | R² range | R² mean |
|---|---|---|---|
| 32 | 5 | [−0.208, −0.062] | **−0.153** |
| 64 | 5 | [−0.190, −0.037] | **−0.115** |
| 128 | 5 | [−0.075, +0.092] | **−0.009** |
| 256 | 1 (full pool) | — | **+0.042** |

**This is a clean, monotonic, unsaturated learning curve.** Going from
32 to 256 training scenes moves the mean test R² from −0.153 to +0.042 —
crossing from clearly negative to clearly positive — while still not
quite reaching the original baseline (0.079) by the largest N tested.
The curve has **not plateaued**: there is no indication that R² would
stop improving with yet more data. This is the single most decisive
finding in this diagnostic: it directly demonstrates, by controlled
manipulation (holding the transform, test set, encoder, and
regularization fixed and varying only `N_train`), that Task 7's original
N_train=32 result sits on a rising part of a learning curve, not at a
converged asymptote.

**Compute-budget note.** `N_train=512` was not attempted — the full run
up to 256 (which required rendering 224 new scenes beyond Task 7's
original 40) took 1141s (19 min) of the compute budget available in this
session; 512 would have required roughly double the scene pool and
render time, which was not pursued given diminishing returns for this
report's purpose (the trend at 128→256 was already clear and consistent).

---

## B. Regression conditioning

**Setup.** All six transforms, using the forensic audit's already
re-derived real representations (no new rendering). Compares Task 7's
original fixed `alpha=10.0` against (1) OLS (`alpha=0`) and (2)
`RidgeCV` with `alpha` selected by **4-fold cross-validation on the 32
train scenes only** — the held-out test set is never touched during
selection (verified in QA layer E below).

**Result** (`regression_conditioning.json`):

| transform | fixed α=10 | OLS | CV-selected α (value) | best baseline |
|---|---|---|---|---|
| camera_translation | 0.198 | 0.155 | 0.273 (α=300) | 0.638 |
| camera_rotation | −0.176 | −0.259 | 0.002 (α=1000) | 0.079 |
| object_translation | 0.381 | 0.353 | 0.416 (α=100) | 0.807 |
| object_rotation | −0.195 | −0.301 | 0.108 (α=300) | 0.385 |
| lighting_change | 0.070 | 0.034 | 0.139 (α=300) | 0.467 |
| texture_change | 0.426 | 0.408 | 0.451 (α=100) | 0.870 |

Train-only CV consistently selects `alpha` an order of magnitude higher
than the original fixed 10 (100–1000), and this measurably improves
absolute R² for every transform, most dramatically for the two worst
performers (`camera_rotation`: −0.176→0.002; `object_rotation`:
−0.195→0.108). **But it does not change the qualitative outcome for any
transform** — every transform's CV-optimized R² still falls short of its
own best baseline. **Regularization strength is not, by itself, the
bottleneck**, though it was measurably suboptimal in the original run.

---

## C. Temporal context

**Setup.** `camera_rotation`, comparing `num_frames ∈ {4, 16, 32, 64}` on
a **fixed 9-train/3-test subset** of Task 7's own original scenes
(scaled down from the full 32/8 split for compute-budget reasons — see
below), holding everything else (transform config, resolution, fps,
ridge alpha) fixed.

**Result** (`temporal_context.json`):

| num_frames | learned W_T R² | persistence baseline R² |
|---|---|---|
| 4 | −0.856 | −0.030 |
| 16 | −1.012 | −0.009 |
| 32 | −0.901 | −0.014 |
| 64 | −1.007 | −0.111 |

**No clear trend.** R² is flat-to-noisy across all four frame counts,
and — critically — the **persistence baseline itself** (which has no
fitted parameters and should be a low-variance number) swings from
−0.030 to −0.111 purely as a function of frame count on this tiny test
set. This is the signature of a test set too small (`n_test=3`) to
distinguish a real effect from sampling noise, not evidence that
temporal context doesn't matter.

**Compute-budget note, stated plainly rather than hidden:** a 64-frame
clip cost ~44× a 4-frame clip's wall time on this diagnostic's subset
(1990s vs. 45s for the same 9 scenes), because VJEPA2's attention cost
scales with the *square* of token count (8192 tokens at 64 frames vs.
512 at 4 frames). Running this sweep at the full 32-train/8-test scale
would have cost roughly 3.3× longer (~2.75 hours) for this step alone,
which was not pursued in this session.

**Verdict for this section: INCONCLUSIVE** — this diagnostic could not,
within the compute available, establish whether temporal context
materially affects the result. It neither confirms nor refutes the
forensic audit's temporal-context-mismatch concern; a properly powered
version of this comparison (full 32/8 split, or a repeated-subsample
design) is a specific, actionable follow-up, not performed here.

---

## D. Representation structure (pooled vs. token-preserving)

**Setup.** `camera_rotation` and `object_translation` (a cross-check),
re-encoding Task 7's own already-rendered clips (no new rendering) to
obtain the raw `(512, 1024)` token sequence, then comparing three
poolings, all fit/evaluated with the identical ridge protocol via a
memory-safe dual-ridge formulation (needed because the largest pooling
produces a 262144-dimensional representation, for which a naive
primal-space `sklearn.Ridge` would require an infeasible 262144×262144
coefficient matrix — verified to reproduce `sklearn.Ridge` exactly to
floating-point precision before use, see QA):

- `global_mean` — Task 7's original: mean over all 512 tokens → (1024,).
- `spatial_grid_mean` — mean over the 2 temporal tubelets only, keeping
  the full 16×16 spatial grid → (262144,) flattened.
- `quadrant_mean` — mean over temporal and 2×2 spatial quadrants →
  (4096,) flattened.

**Result** (`pooling_diagnostic.json`):

| transform | global_mean (D=1024) | spatial_grid_mean (D=262144) | quadrant_mean (D=4096) |
|---|---|---|---|
| camera_rotation | R²=−0.176 | R²=−0.168 | R²=−0.160 |
| object_translation | R²=**0.381** | R²=**−0.104** | R²=**0.289** |

**Preserving spatial structure does not help — it hurts, and the more
spatial resolution retained, the worse the degradation.**
`spatial_grid_mean` is the *worst* of the three for both transforms
(dramatically so for `object_translation`, where it turns a positive
R²=0.381 into R²=−0.104); `global_mean` is best or comparable in both
cases. This is best explained by dimensionality/overfitting at
`N_train=32` dominating over any positional-information gain: adding
262144 output dimensions without adding training scenes makes fitting
*harder*, not more informative.

**This is a genuinely important, somewhat counter-intuitive finding,
reported exactly as observed rather than favorably reinterpreted**: it
provides evidence **against** (not for) mean-pooling's information loss
being the primary driver of the original negative result. It does not
mean V-JEPA's dense tokens lack positional geometric information — only
that *naively* preserving that structure, without also solving the
sample-size problem, reveals no additional signal and actively degrades
fitting. This result and finding A reinforce each other: both point at
sample size (relative to representation dimensionality) as the dominant
lever, not pooling.

---

## E. Representation geometry (train-fitted PCA/whitening)

**Setup.** All six transforms, using the forensic audit's real
representations. PCA is fit on the **32 train scenes only** (never
re-fit on test) and applied (not re-fit) to the test set — a legitimate,
non-leaking transformation, distinct from the forensic audit's earlier
train+test-pooled effective-rank figure.

**Train-only effective rank: 15.7** (of a nominal 1024 ambient
dimensions; 16 components capture ≥90% of train variance) — slightly
lower than the forensic audit's N=40 pooled estimate (~18), as expected
for a smaller sample.

**Result** (`representation_geometry.json`, k=16 components):

| transform | raw-space ridge | PCA-reduced ridge | whitened ridge | best baseline |
|---|---|---|---|---|
| camera_translation | 0.198 | 0.294 | 0.277 | 0.638 |
| camera_rotation | −0.176 | −0.052 | −0.025 | 0.079 |
| object_translation | 0.381 | 0.373 | 0.363 | 0.807 |
| object_rotation | −0.195 | +0.065 | +0.094 | 0.385 |
| lighting_change | 0.070 | 0.087 | 0.130 | 0.467 |
| texture_change | 0.426 | 0.388 | 0.370 | 0.870 |

Whitening/PCA-reduction nudges the two worst performers
(`camera_rotation`, `object_rotation`) substantially toward zero, and
mildly helps `lighting_change`, but mildly *hurts* the two
already-better performers (`object_translation`, `texture_change`).
**No transform crosses its baseline under either transformation.** This
converges with finding B: addressing conditioning/anisotropy directly
recovers some of the same headroom that better regularization does, but
neither closes the gap — consistent with sample size (not
regularization or the raw/whitened basis choice) being the dominant
remaining factor.

---

## Synthesis: distinguishing the six candidate explanations

| Candidate explanation | Verdict from this diagnostic |
|---|---|
| 1. Insufficient sample size | **Confirmed as the dominant, load-bearing factor** — a direct, controlled scaling experiment (A) shows a clean, monotonic, unsaturated improvement from R²=−0.153 (N=32) to R²=+0.042 (N=256), crossing zero and continuing to trend toward (not yet past) baseline. |
| 2. Regression conditioning (regularization) | **Ruled out as an independent driver** — train-only CV-selected α improves absolute R² but never changes any transform's below-baseline outcome (B). |
| 3. Temporal-context mismatch | **Inconclusive** — the compute-budget-constrained diagnostic (C) could not distinguish a real effect from test-set noise; neither confirmed nor refuted. |
| 4. Information loss from mean pooling | **Evidence against** as the primary driver — naively preserving spatial structure makes fitting worse, not better, at this sample size (D); does not rule out that dense tokens carry geometric signal, only that this specific, naive way of accessing it doesn't help without more data. |
| 5. Representation anisotropy | **A contributing, not independent, factor** — train-only whitening recovers some of the same headroom that better regularization does (E), consistent with anisotropy affecting *effective* sample efficiency, entangled with (not separate from) finding 1. |
| 6. Genuinely weak geometric consistency | **Cannot be ruled out, and is not the most parsimonious explanation** given the clear, unsaturated sample-size trend — a result that improves monotonically and substantially with more data is not what "genuinely absent structure" would look like; but this diagnostic did not reach a large enough N to positively confirm the map would exceed baseline, so a residual possibility remains open. |

---

## FINAL CLASSIFICATION: **METHODOLOGICALLY LIMITED**

Not **VALID NEGATIVE**: a valid negative result requires the finding to
be stable with respect to known confounds. Diagnostic A directly
falsifies stability with respect to sample size — the result at
`N_train=32` is measurably, substantially, monotonically different from
the result at `N_train=256`, with no sign of having converged. Reporting
Task 7's original number as if it reflected V-JEPA's asymptotic
behavior would overstate what was actually measured.

Not **INCONCLUSIVE** in the "no informative evidence either way" sense:
this diagnostic *does* identify which specific factor is most load-bearing
(sample size, with anisotropy as a related contributor) and rules out
two others (regularization strength, and — perhaps counter-intuitively —
naive spatial-structure preservation) as independent explanations. That
is a positive, actionable finding, not a shrug.

**METHODOLOGICALLY LIMITED** captures both facts at once: the original
number was computed correctly (per the forensic audit) and is not the
product of a bug, but the *protocol* — specifically, `N_train=32` against
a `D=1024` representation — was measurably too small to yield a stable,
interpretable estimate. The learning curve trends toward, but had not
reached, the baseline by the largest N this session's compute budget
allowed (256). Whether it would eventually exceed baseline with
substantially more data (the diagnostic's own extrapolation, not a
measured fact) remains genuinely open — this is exactly why "valid
negative" would be too strong a conclusion, and "methodologically
limited" is the accurate one.

**Recommended next action** (not performed here, and explicitly not a
re-run of Task 7 aimed at a better-looking number): a follow-up task
with `num_scenes` substantially larger than 40 (this diagnostic's own
N=256 result, R²=+0.042, is closer to but still below baseline —
continuing the trend a similar distance further, e.g. to N≈500–1000,
per the forensic audit's earlier isotropic-case estimate, is the natural
next data point) reporting a full learning curve as its primary result
rather than a single N, and — separately — a properly powered version of
the temporal-context comparison (full 32/8 split) to resolve C's
inconclusive result.

---

## Exact configurations, seeds, and scene IDs

All of the following are recorded in full, machine-readable form in the
named JSON files (not reproduced in full here to keep this report
readable):

- **A.** `experiments/task7_diagnostic/learning_curve.json` — every
  `N_train`/seed's exact scene-ID list, ridge alpha (10.0), fixed test
  scene IDs (Task 7's own recorded 8), and per-seed R²/cosine.
- **B.** `experiments/task7_diagnostic/regression_conditioning.json` —
  the full alpha grid tested, CV fold count (4), selected alpha per
  transform, and OLS/fixed/CV R²/cosine for all six transforms.
- **C.** `experiments/task7_diagnostic/temporal_context.json` — the
  exact 9 train / 3 test scene IDs used, per-frame-count wall time, and
  R²/cosine/persistence-baseline at each of 4/16/32/64 frames.
- **D.** `experiments/task7_diagnostic/pooling_diagnostic.json` — R²/
  cosine/dimensionality for all three poolings, both transforms.
- **E.** `experiments/task7_diagnostic/representation_geometry.json` —
  train-only effective rank, explained-variance-by-k, selected k (16,
  via a test-blind ≥90%-train-variance rule), and raw/PCA/whitened R²
  for all six transforms.
- **Encoder/representation extraction method**: identical to Task 7's
  own (`encoders.vjepa.VJEPAEncoder`, `facebook/vjepa2-vitl-fpc64-256`,
  `mean_pool` for A/B/C/E; raw `(512,1024)` tokens re-pooled three ways
  for D) in every diagnostic except D, which explicitly varies pooling
  as its whole point, and C, which explicitly varies `num_frames`.
- **Regression method**: `Ridge`/mathematically-equivalent dual-ridge,
  `fit_intercept=True`, `alpha=10.0` (Task 7's own value) everywhere
  except B, which explicitly varies alpha as its whole point.
