# RESEARCH_PLAN.md — Task pipeline for the Geometric Consistency Probe

**Status: living plan, orchestrator-facing.** This document tracks *which
task the project is on* and *what each task is for*, at a level the
orchestrator (`orchestrator/`) and its QA layer can check against
mechanically. It is not a restatement of the science — that is
`DESIGN.md` (binding) and `IMPLEMENTATION_NOTES.md` (engineering
history). This document is subordinate to both wherever they disagree.

## Research question (restated, unchanged)

> Does a world model actually understand 3D? When a learned video/world-
> model representation predicts or represents the future, does its
> latent representation encode underlying 3D state and geometric
> structure of the world, or is it primarily exploiting appearance /
> statistical correlations?

Operationalized as:

```
S --renderer R--> V --frozen encoder E--> Z
S' = T(S) --R--> V' --E--> Z'

question: is Z' ≈ W_T Z, for some simple W_T (linear, fit on TRAIN
scenes only, evaluated on unseen TEST scenes)?
```

The project is not training a foundation model. It is running
scientifically controlled experiments designed to be able to
distinguish genuine geometric consistency from appearance similarity,
low-level visual similarity, memorization, scene-identity leakage,
camera/rendering artifacts, dataset leakage, weak baselines, and
accidental correlation. See `research/RESEARCH_INVARIANTS.md` for the
durable rules this implies, and `DESIGN.md` §13 for exactly what
claims a good result can and cannot support.

## Task pipeline

Tasks 1–5 are **complete** and must not be redone. They established the
foundation every later task builds on:

| # | Task | What it produced |
|---|------|-------------------|
| 1 | Research specification | `DESIGN.md` |
| 2 | Controlled 3D scene generation | `generation/` (`SceneState`, trajectories, Blender/Cycles renderer, `generation/generate.py` CLI, ground truth RGB+depth+segmentation) |
| 3 | Controlled geometric transformations | `transforms/` (`se3.py` SE(3) engine, `scene_transform.py`'s 6 named transforms, `pairs.py` matched original/transformed video-pair rendering) |
| 4 | Frozen video representation extraction | `encoders/vjepa.py` (`VJEPAEncoder`, wraps `facebook/vjepa2-vitl-fpc64-256` via `transformers.VJEPA2Model`, frozen, deterministic fallback), `encoders/extract.py` |
| 5 | Verification of frozen representation extraction | `tests/test_vjepa_encoder.py` (T5.1–T5.9-style: loads correctly, no NaN/Inf, frozen, no optimizer, deterministic, real pretrained weights now verified to load — see `IMPLEMENTATION_NOTES.md`'s "Task 4 — real pretrained weights") |

Tasks 6–18 are **pending**, executed one at a time by the orchestrator,
each gated on independent QA (`orchestrator/qa.py`) before the next one
starts:

| # | Task | One-line objective |
|---|------|---------------------|
| 6 | First camera-rotation geometric consistency experiment | Does a fixed camera rotation induce a predictable, generalizing linear map in Z-space? |
| 7 | Multiple geometric transformations | Does the finding in 6 hold across all four geometric transforms, not just rotation? |
| 8 | Physical-state accessibility | Is physical state (e.g. camera azimuth, object position) linearly decodable from Z? |
| 9 | Appearance invariance controls | Does Z stay stable under transforms that change appearance but not physical geometry? |
| 10 | Baseline framework | Consolidate and harden the baseline/control suite used by every experiment so far. |
| 11 | Scale experiment | Does the effect strengthen, weaken, or stay flat as scene count grows? |
| 12 | Temporal geometric consistency | Does consistency hold across real motion (not just static-pair renders)? |
| 13 | Object persistence under occlusion | Is object identity/position preserved in Z through occlusion? |
| 14 | Counterfactual representation consistency | Do counterfactual scene edits change Z in the direction physically implied? |
| 15 | Unified evaluation framework | Consolidate tasks 6–14's metrics into one reusable evaluation harness. |
| 16 | Final research report | Synthesize all findings, with explicit claims/limits per `DESIGN.md` §13. |
| 17 | Open-source release | Prepare the repository for external use/review. |
| 18 | Adversarial independent research audit | An independent adversarial pass looking for leakage, confounds, or unsupported claims in everything above. |

Each task's full specification lives in `tasks/{NN}_{name}.md`, written
one at a time — **the task file for task N+1 is only added once task N
has passed QA**, so the orchestrator naturally cannot get ahead of
itself: with no spec file to read, it stops and waits (see
`orchestrator/run.py`).

## How this plan is used mechanically

- `state/progress.json` is the single source of truth for "which task
  are we on" — see `orchestrator/state.py`. This document is the human-
  and Claude-readable companion to it, not a replacement.
- A task is "done" only when `orchestrator/qa.py` independently verifies
  it (software correctness, task acceptance, scientific validity,
  research alignment, data leakage, reproducibility, regression) — never
  because an implementation run merely claimed success. See
  `research/RESEARCH_INVARIANTS.md` invariant 18.
- A scientifically negative result (e.g. "camera rotation consistency is
  weak") is not a QA failure by itself — see invariants 10–11 and
  `orchestrator/qa.py`'s SOFTWARE FAILURE vs. SCIENTIFIC NEGATIVE RESULT
  distinction.
