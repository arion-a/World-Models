# World-Models

**Does a frozen, pretrained video model actually understand 3D geometry, or does it just recognize appearance and temporal patterns?**

This repository holds one project, the **Geometric Consistency Probe** (`geometric-consistency-probe/`) — a controlled-experiment testbed that answers this question empirically for [V-JEPA 2](https://huggingface.co/docs/transformers/en/model_doc/vjepa2) ViT-L/16, without training or fine-tuning anything.

## The idea, in one diagram

```
Physical state S --render--> video V --frozen encoder E--> representation Z

S' = T(S) --render--> video V' --E--> Z'

Question: is Z' predictable from Z, i.e. Z' ≈ rho(T)·Z for a simple rho(T)?
```

Because every scene is a synthetic 3D render (Blender/Cycles), we know the *exact* ground-truth physical state behind every frame — camera pose, object positions, lighting — something no probe of real video can assume. That lets us ask precise questions a pixel-similarity check can't: is the encoder's representation *equivariant* to a known transform (camera rotation, object translation, …), *invariant* to transforms that shouldn't matter (a lighting change), and is physical state *linearly decodable* from the representation at all?

Every answer here comes from **linear ridge regression** on a **frozen** encoder, evaluated on **scene-level held-out** splits, against a standard set of trivial baselines (predict no change, predict the mean, predict from a deliberately mismatched pairing). No neural probe is ever trained, and the encoder's weights never move — see `geometric-consistency-probe/DESIGN.md` for the full specification and `IMPLEMENTATION_NOTES.md` for the engineering history.

## Start here

```bash
cd geometric-consistency-probe
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -m "not slow"      # fast checks, no bpy render calls, no network
```

`geometric-consistency-probe/README.md` has the full per-feature command reference (scene generation, transform pairs, encoder extraction, running experiments). `geometric-consistency-probe/CLAUDE.md`'s parent, this repo's own `CLAUDE.md`, documents the **autonomous orchestrator** that drives Tasks 6–18 (below) — it invokes Claude Code per task and independently re-verifies every result from disk before advancing, rather than trusting a task's own self-report.

## Key findings so far

The honest headline: **most of the individual, single-condition tests came back negative** — the frozen encoder's representation usually did *not* show the hypothesized linear structure, under the exact conditions each task tested. That is treated as a valid result throughout this project, not a failure (`research/RESEARCH_INVARIANTS.md`, invariant: negative results are never discarded or reframed as success). But two cross-cutting patterns emerged once several of these negative results were pushed further:

| # | Question tested | Result |
|---|---|---|
| Task 6/7 | Does a linear map predict the representation change under camera/object rotation or translation? | **No**, for any of the 4 geometric transforms or 2 appearance controls tested, at the original scale and motion magnitude. |
| Task 8 | Is physical state (azimuth, elevation, distance, object position) linearly decodable from the representation? | **Partially** — 2 of 4 variables (elevation, distance) were; azimuth and object position were not. |
| Task 9 | Does the representation stay invariant under appearance-only changes (lighting, texture) the way it should? | **No clean separation** — appearance controls disturbed the representation about as much as genuine geometric transforms did. |
| Task 11 | Does the Task 6 negative result hold as sample size scales (15→100 scenes)? | **Yes** — the equivariance gap *strengthens*, not an artifact of too little data. |
| Task 12 | Is the representation temporally consistent across a moving camera's video window? | **No**, at the original fixed ~30°/window motion magnitude. |
| Task 13 | Does post-occlusion object state stay linearly decodable? | **No**, for either variable tested (position, velocity) — though a same-rig follow-up at N=100 flipped this from ambiguous-negative to a small positive margin, i.e. this one *was* partly a small-sample artifact. |

**The most interesting result came from pushing Task 12's negative finding along a different axis than sample size.** Instead of asking "is there more data?", a follow-up asked "was the camera rotation just too small to move the representation in the first place?" — and swept the rotation magnitude itself (θ = 0° → 30°) at N=800 scenes:

| θ (rotation) | learned map R² | "assume nothing changed" R² |
|---|---|---|
| 0° (no rotation — literally the same frame twice) | 0.99 | **1.00** (exact, by construction) |
| 5° | 0.85 | 0.86 |
| 10° | 0.75 | 0.71 — **learned map overtakes** |
| 20° | 0.63 | 0.43 |
| 30° | 0.52 | **0.19** |

At small rotations, "assume nothing changed" wins trivially, because almost nothing *did* change — a near-perfect trivial score there is not evidence the representation is well-behaved, just evidence the test was too gentle to tell. Once the rotation is large enough to meaningfully move the representation, the learned map overtakes the trivial baseline and the gap widens sharply, while beating a shuffled-pairing control throughout. **The frozen encoder does encode real, learnable camera-rotation structure — Task 12's original negative result was measuring at a magnitude too small to see it, not measuring the wrong thing entirely.**

Two lessons this project keeps re-learning: a negative result can hide behind too little data (Task 13) *or* too small an effect size (Task 12) — and the fix in each case is a different, explicit follow-up experiment, never a re-interpretation of the original one. Every follow-up here lives in its own `experiments/*_followup_*/` directory with its own config and result file; the original task's sealed checkpoint and `state/task_NN_result.json` are never modified.

## Repository layout

```
CLAUDE.md                      Guidance for Claude Code sessions working in this repo
geometric-consistency-probe/   The project itself
  DESIGN.md                    Research specification: transform/metric definitions, what this can't support
  IMPLEMENTATION_NOTES.md      Engineering decisions and history, including corrected assumptions
  README.md                    Full per-feature usage commands
  research/                    RESEARCH_PLAN.md (Task 6-18 roadmap), RESEARCH_INVARIANTS.md (18 durable rules)
  orchestrator/                The autonomous Task 6-18 driver + its independent QA layers
  tasks/                       One spec file per orchestrator task
  generation/ transforms/ encoders/ representations/ probes/ metrics/ baselines/
                                The pipeline stages themselves (see DESIGN.md's architecture diagram)
  experiments/                 Task runners, plus every follow-up analysis in its own subdirectory
  state/                       task_NN_result.json per completed task — the source of truth for results
```

## Status

Tasks 1–13 are complete and checkpointed (`state/progress.json`); Task 14 (counterfactual representation consistency) is next. See `research/RESEARCH_PLAN.md` for the full Task 6–18 roadmap and `CLAUDE.md` for how to run the orchestrator yourself.
