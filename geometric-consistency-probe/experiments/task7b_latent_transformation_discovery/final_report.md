# Task 7B — Discovery of Latent Geometric Transformation Structure: Final Report

**Status: `COMPLETE_SCALE_LIMITED`** (see [Completion status](#completion-status) below for why this is not `COMPLETE`, and why that is not a failure).

## 1. What was tested

A fresh, preregistered study, independent of Task 7 / the forensic audit / the diagnostic (whose artifacts remain byte-for-byte untouched — verified by `qa_final.py` directly from git, not assumed). Transform: `camera_rotation` at fixed elevation 0.0° (pure yaw), verified beforehand to satisfy an exact one-parameter rotation-group composition law (`tests/test_task7b_composition_law.py`). Encoder: frozen `facebook/vjepa2-vitl-fpc64-256` (V-JEPA2 ViT-L/16), mean-pooled, `D=1024`, real pretrained weights (`pretrained: true` throughout). Master population: 1463 freshly-sampled scenes (never Task 7's 40), split deterministically 1024/219/220 (train pool / val / test) by scene ID, no scene's original or transformed render crossing a split. Protocol frozen and content-hashed before any test-scene rendering (`protocol_hash: 44bff5c8...`).

## 2. Stage 1 — model hierarchy learning curve

Fit M0 (persistence) through M6 (one-hidden-layer residual MLP), every candidate on train only, regularization/early-stopping chosen on validation only, at `N_train ∈ {64,128,256,512,1024}`, 5 seeds per point (nested subsets):

| N_train | mean val R² | 95% CI |
|---|---|---|
| 64 | 0.329 | [0.319, 0.338] |
| 128 | 0.406 | [0.402, 0.409] |
| 256 | 0.470 | [0.467, 0.473] |
| 512 | 0.524 | [0.521, 0.527] |
| 1024 | 0.562 | [0.562, 0.562] |

The curve rose **monotonically across the entire schedule** with no sign of flattening. Per the preregistered stopping rule (final two points within 0.02 mean R² with overlapping 95% CIs), this is **`NOT_STABILIZED`** — the full schedule ran anyway, as required, rather than stopping early or picking an arbitrary endpoint.

Best validation R² per model family at N=1024 (full detail in `model_selection_record.json`):

| Model | Best val R² |
|---|---|
| M0 persistence | 0.171 |
| M1 scalar | 0.192 |
| M2 diagonal | 0.246 |
| M3 low-rank (r=32) | 0.508 |
| **M4 full linear (selected)** | **0.562** |
| M5 affine | 0.562 (tied within 0.01; fewer-params tie-break selects M4) |
| M6 residual MLP (h=32) | 0.470 |

**Selected: M4 full linear, α=100.0**, at N_train=1024 (seed=1). Notably, the added flexibility of M5 (bias term) and M6 (nonlinearity) did **not** outperform the plain full linear map — no evidence a nonlinear or affine correction is needed over what a linear map already captures.

## 3. Sealed test (run exactly once)

Evaluated on the untouched 220-scene test split:

| Metric | Selected (M4) | Best control |
|---|---|---|
| R² | **0.5571** | 0.1574 (persistence) |
| MSE | 0.1964 | — |
| Relative L2 error | 0.2321 | — |
| Cosine similarity | 0.9713 | — |

Controls: train-mean R²=−0.007, persistence R²=0.157, shuffled-pair R²=−0.137. **Difference vs. best control: +0.400.**

95% bootstrap CI over scenes (scene-resampling, not per-scene R² — see §5): **[0.528, 0.580]**, point estimate 0.557.

Contamination status: **CLEAN** — test scenes were never touched by fitting, regularization/early-stopping selection, or model selection at any point before this evaluation (verified structurally in `tests/test_task7b_learning_curve.py` and `tests/test_task7b_sealed_test.py`).

## 4. Stage 2/3 — magnitude sweep and structure tests

The selected form (M4, α=100.0) was refit per magnitude on the same fixed 256-scene structure-test subset, evaluated on the full 220-scene test split, no re-selection.

**Output-space agreement** (how close predictions are to the true target, using the standard multi-output R²):

| Test | Result |
|---|---|
| Identity (F₀(Z) vs Z) | R²=0.917, cosine=0.995 |
| Composition (5 pairs, e.g. F₋₁₀∘F₋₁₀ vs F₋₂₀) | R² 0.62–0.87 (best at small magnitude sums) |
| Inverse (3 pairs, e.g. F₋₁₀∘F₁₀ vs identity) | R² 0.44–0.75 (degrades with magnitude) |
| Interpolation (predicted vs. true, never-fit +20°) | R²=0.570, cosine=0.972 |

**Operator-level agreement** (does the literal fitted D×D matrix satisfy the exact algebraic identity, via normalized Frobenius discrepancy — 0 is exact agreement, ≥1 means the discrepancy is as large as, or larger than, the target operator itself):

| Test | Normalized Frobenius discrepancy |
|---|---|
| Identity (‖W₀ − I‖ / ‖I‖) | 0.949 |
| Composition (5 pairs) | 0.65 – 1.00 |
| Inverse (3 pairs) | 0.986 – 1.003 |

**This is the key nuance of Stage 2/3**: output-space predictions are meaningfully better than chance and show real, above-control structure under composition/inverse/identity — but the underlying fitted **operators themselves do not closely satisfy the exact matrix identities** (composed/inverse/identity operators deviate from their theoretical targets by an amount comparable to, or exceeding, the target operator's own scale). This is consistent with strong ridge regularization (α=100 against D²≈1M parameters) shrinking every fitted operator well away from any "clean" rotation matrix, even when its predictions on real data remain reasonably accurate in aggregate.

## 5. A bug found and fixed during this task (documented per contract Sec. 7's rerun exception)

`sealed_test.py` originally computed "per-scene R²" by slicing one test row and calling the standard multi-output R² formula on it. This is **mathematically always `"undefined"`**, for any model or dataset: a 1-sample baseline mean equals that sample, so the denominator is identically zero. This silently produced a useless `n=0` bootstrap CI. Fixed by (a) reporting per-scene relative-L2-error and cosine-similarity instead (both well-defined per single scene), and (b) adding `bootstrap_ci_over_scenes`, which resamples *scenes* and recomputes the same set-level R² statistic per resample — the correct way to bootstrap a metric that requires more than one sample. The sealed test was rerun once with `force=True`; the core fit-derived numbers (R²=0.5571096447462491, etc.) were verified bit-for-bit identical before and after, confirming this was a metrics-formatting fix, not a result-driven redo.

A second bug (sealed_test.py passing the test set itself as the early-stopping validation set — would only have mattered had M6 been selected) and a rendering inefficiency (Stage 2 re-rendering the magnitude-independent original side on every sweep) were also found and fixed pre-execution / mid-execution respectively; neither affected the results reported here (M4 was selected, not M6; the rendering fix only changed wall-clock time).

## 6. Scientific interpretation (calibrated per contract Sec. 13)

- **A scene-independent, learnable relationship between Z and Z' clearly exists and generalizes**: the sealed R² (0.557) is far above every control, on scenes never touched during fitting or selection. Calibrated claim: **"measurable predictive consistency under the tested conditions."**
- **The simplest supported form is a plain full linear map** (M4) — no evidence more flexibility (affine bias, or a nonlinear residual MLP) helps.
- **Composition/inverse/identity show real but qualified support**: output-space predictions are consistent with the algebraic structure above control levels, but the literal fitted operators do not closely match the exact group-theoretic identities (large Frobenius discrepancies). The calibrated claim here is therefore **not** a clean "approximate compositional latent transformation family" — it is closer to: *"predictions are consistent with compositional structure at the output level under tested conditions, without close agreement at the operator level."*
- **The learning curve did not stabilize.** Per contract Sec. 5/13, this result is **scale-limited: no claim is made about what R² would be at N>1024, or that 0.557 is any kind of ceiling.**
- None of the following is supported by this study and is explicitly **not** claimed: "V-JEPA understands 3D," any universal geometric-understanding claim, or any token-level conclusion (this study is global-pooled only, per the contract's explicit token-level boundary).

## 7. Completion status

`state/task_07b_result.json` reached **`COMPLETE_SCALE_LIMITED`**, not `COMPLETE`, solely because the learning curve did not stabilize within the preregistered `N_train≤1024` schedule — every other acceptance criterion (all QA gates, sealed test cleanliness, every structure test resolved, historical artifacts untouched, full repo test suite green) passed. Per the contract, a scale-limited, non-stabilized result is an explicitly valid, non-failing outcome, not a defect to be papered over.
