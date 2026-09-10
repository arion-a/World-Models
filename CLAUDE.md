# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repo currently contains one project: `geometric-consistency-probe/` (the "Geometric Consistency Probe", GCP). All commands below assume that directory as the working directory.

## What this project is

GCP tests whether a **frozen, pretrained** video representation model (V-JEPA 2 ViT-L/16, `facebook/vjepa2-vitl-fpc64-256`) encodes 3D geometric structure or just appearance/temporal correlation. It renders synthetic 3D scenes with Blender, applies a known physical transform `T` (or a non-geometric control) to the scene, and checks whether the change in the frozen encoder's representation is *predictable* — fitting a linear `rho(T)` on train scenes and evaluating equivariance/invariance on held-out scenes, plus a linear probe for whether physical state (e.g. camera azimuth) is linearly accessible.

Read `DESIGN.md` first for any research-methodology question — it is the binding specification (transform definitions, equivariance/invariance definitions, leakage-prevention protocol, baselines, and an explicit "what this framework can and cannot support" section) and is authoritative over `IMPLEMENTATION_NOTES.md` wherever they'd disagree. `IMPLEMENTATION_NOTES.md` records engineering decisions and history (including corrections — e.g. its "Task 4" section documents why the encoder is ViT-L/16 and not the originally-planned ViT-B/16 or "V-JEPA 2.1", once real checkpoint availability was verified against the Hub). `README.md` has all per-feature usage commands.

## Setup

```bash
cd geometric-consistency-probe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`bpy` (Blender as a Python module, pinned to `5.0.1`) is a large (~1GB), Linux-focused wheel. If it's unavailable, everything except rendering (transforms, metrics, probes) still works, via `pytest -m "not slow"`.

## Commands

```bash
# Full test suite (includes real-Blender-render and real-model-forward-pass tests)
pytest

# Fast unit tests only — no bpy render calls, no model forward passes, no network
pytest -m "not slow"

# Single test
pytest tests/test_se3_matrices.py::test_orbit_rotation_matrix_matches_look_at -v

# Whole file
pytest tests/test_scene_transform_matrices.py -v
```

There is no separate lint/typecheck command configured in this repo — validate changes with the test suite above.

### Running the V0 experiment

```bash
python -m experiments.run --config configs/experiments/camera_rotation_v0.yaml   # fast (~2 min, CPU, no network)
python -m experiments.run --config configs/default.yaml                          # full scale, real pretrained weights
```

`experiments/run.py` calls scene generation and representation extraction automatically if their outputs don't exist yet, so this one command is enough for a first run. Set `encoder.name: pixel_baseline` in a config to swap in the non-learned pixel-statistics baseline.

### Task 2/3/4 entry points (independent of the V0 experiment above)

```bash
# Controlled scene generation with full ground truth (rgb/depth/segmentation + poses)
python -m generation.generate --config configs/generation.yaml

# Frozen representation extraction over a directory of {video_id}/rgb.npy files
python -m encoders.extract --input_dir data/generation_v0 --output_dir data/representations_v0
```

`transforms.pairs.generate_pair(scene, transform_name, out_dir)` renders a matched original/transformed pair (see `README.md`'s "Task 3" section for the Python usage example).

## Architecture

Pipeline: `Physical state S --renderer R--> video V = R(S) --frozen encoder E--> Z`, then `S' = T(S) --R--> V' --E--> Z'`, asking whether `Z' ≈ rho(T) Z` for a simple `rho(T)`.

- **`generation/`** — `SceneState`/`ObjectState`/`CameraState`/`LightState` dataclasses (`scene.py`), scene sampling (`scene_sampler.py`), object/camera trajectories (`motion.py`, pure math, no bpy dependency), and the Blender/Cycles renderer (`bpy_renderer.py`, builds the compositor graph for RGB+depth+segmentation passes). `generate.py` is the canonical Task 2 CLI; `generate_dataset.py` is what V0's `experiments/run.py` calls. See `generation/COORDINATE_SYSTEM.md` before consuming any rendered output — it documents (empirically verified, not assumed) the world frame, camera convention, and that depth is **camera-space Z, not Euclidean ray distance**.
- **`transforms/`** — `se3.py` is pure SE(3) matrix math (rotation/translation composition, rigid inverse, orbit rotation, look-at). `scene_transform.py` applies one of six named transforms to a `SceneState`, returning the transformed scene plus exact ground truth: a 4x4 matrix for the four geometric transforms, `null` for the two appearance-only controls (which are not rigid transforms), and which `SceneState` fields changed vs. stayed fixed. `pairs.py` renders a matched original/transformed pair reusing `generation`'s renderer.
  - `GEOMETRIC_TRANSFORMS = (camera_translation, camera_rotation, object_translation, object_rotation)`
  - `CONTROL_TRANSFORMS = (lighting_change, texture_change)` — appearance-only, used as controls (should show invariance, not equivariance)
- **`encoders/`** — two parallel interfaces that intentionally coexist (see `IMPLEMENTATION_NOTES.md`'s "Two encoder interfaces, on purpose"): `FrozenEncoder` (`base.py`, `encode_video -> pooled vector`, used by V0's `encoders/vjepa2.py` + `representations/`/`experiments/run.py`) and `VideoEncoder` (`base.py`, `encode -> native unpooled token sequence`, used by `encoders/vjepa.py` + `encoders/extract.py`). `encoders/vjepa.py` is the actively-developed, tested encoder wrapper; both it and the older `vjepa2.py` load `facebook/vjepa2-vitl-fpc64-256` through `transformers.VJEPA2Model` directly (no `trust_remote_code`). When pretrained weights can't be loaded, both fall back to the same architecture with **seeded** random weights (`fallback_seed`, default 0) so the fallback is reproducible across processes — and loudly warn that results are pipeline-validation only. See `encoders/REPRESENTATION_FORMAT.md` for the exact unpooled tensor shape/token-ordering contract and on-disk `metadata.json` schema.
- **`representations/`** — extracts and caches `Z = E(V)` (`extract.py`) and assembles train/test splits (`dataset.py`), enforcing a strict **scene-level** split: a scene's original and transformed renders are never split across train and test (checked by `tests/test_scene_split.py`).
- **`probes/`** — `linear_rep_transform.py` fits `rho(T)` (ridge regression, not a trained network); `linear_state_probe.py` probes linear accessibility of physical state (e.g. camera azimuth) from `Z`.
- **`metrics/`** — `equivariance.py`, `invariance.py`, plus shared numeric primitives in `common.py` (see `_as_samples_by_features()` there — a `(N,)` vs `(N,1)` `np.atleast_2d` reshape bug silently produced wrong R² scores here once; always route 1D arrays through that helper rather than `np.atleast_2d` directly).
- **`baselines/`** — identity map and shuffled-pairing controls, plus the non-learned pixel-statistics encoder (`encoders/pixel_baseline.py`).
- **`experiments/run.py`** — the end-to-end V0 runner tying generation → extraction → probe fitting → evaluation → report together, run over every transform in the config (not just one), for a like-for-like comparison between geometric transforms (expected to show equivariance) and appearance controls (expected not to).
- **`configs/`** — dataclass-based config (`config.py`) loaded from YAML; `default.yaml` is full-scale (100 scenes, real pretrained weights), `experiments/camera_rotation_v0.yaml` is a fast no-network dev config (`pretrained: false`).

## Autonomous task orchestrator (Tasks 6-18)

Tasks 1-5 (above) were built interactively. From Task 6 onward, the
project is driven by an **autonomous orchestrator** that supervises
Claude Code's implementation of each task and independently verifies
the result before allowing the pipeline to advance — see
`research/RESEARCH_PLAN.md` for the full Task 6-18 list and
`research/RESEARCH_INVARIANTS.md` for the 18 durable rules every task
is checked against (frozen encoder, disjoint train/test splits, no
ground-truth-into-encoder shortcuts, mandatory baselines, negative
results are valid, etc.).

**Critical design point: the orchestrator never trusts Claude's own
self-report.** "Task complete, all tests pass" from a Claude Code
invocation is one input, not evidence of completion — every task is
independently re-verified from disk/git/subprocess by
`orchestrator/qa.py` (research/RESEARCH_INVARIANTS.md invariant 18).

```bash
# See current state, next task, planned Claude command and QA layers --
# makes NO changes.
python -m orchestrator.run --dry-run

# Actually run the next incomplete task once (invoke Claude, QA it, fix
# loop up to MAX_FIX_ATTEMPTS on failure, git-checkpoint and advance on
# success).
python -m orchestrator.run

# Keep going across consecutive tasks, but only as long as the NEXT
# task's tasks/{NN}_*.md spec file already exists -- it stops and waits
# otherwise (see "Adding a new task" below).
python -m orchestrator.run --loop

# After investigating and fixing a BLOCKED/FAILED task by hand, allow it
# to be retried from scratch (does not touch any files, just the state).
python -m orchestrator.run --reset-task 6
```

Configuration is via environment variables, all optional:
`CLAUDE_COMMAND` (default `claude`), `CLAUDE_PERMISSION_MODE` (default
`acceptEdits` — never a permission-bypass mode by default),
`CLAUDE_TIMEOUT_SECONDS`, `CLAUDE_EXTRA_ARGS`, `MAX_FIX_ATTEMPTS`
(default 3).

**Adding a new task**: this project's workflow is that a task's
`tasks/{NN}_*.md` specification is written and added only *after* the
previous task has passed QA — the orchestrator's `--loop` mode is built
around this: it stops cleanly and waits (no error, no state change)
whenever the next task's spec file doesn't exist yet, rather than
guessing at scope. To hand the orchestrator a new task, add
`tasks/{NN}_{name}.md` (see any existing file in `tasks/` for the
required section structure: Objective, Scientific question,
Implementation requirements, Required artifacts, Tests required,
Leakage checks, Acceptance criteria, Prohibited shortcuts, Scientific
interpretation limits — `tests/research/test_research_alignment.py`
checks every task file has these) and then run `python -m
orchestrator.run` (or let a running `--loop` pick it up).

**State and logs**: `state/progress.json` is the single source of truth
for which task is current/completed — safe to interrupt and resume at
any point (see `orchestrator/state.py`'s module docstring for exactly
how). `state/task_{NN}_result.json` is each task's own machine-readable
result. `logs/task_{NN}.log` / `logs/task_{NN}_qa.log` record what
prompt was sent, what Claude returned, and the full QA report per
attempt (gitignored — operational, not part of the project's history).
A passing task gets a `task-{NN}-pass` git commit; nothing is ever
force-pushed, rebased, or amended.

**Architecture**: `orchestrator/state.py` (persistent state machine —
`READY → RUNNING → QA → {PASSED | FIXING | BLOCKED}`, `current_task`
only ever advances via `advance_task()`, which requires `status ==
PASSED`), `orchestrator/qa.py` (seven independent QA layers: software
correctness, task acceptance, scientific validity, research alignment,
data leakage, reproducibility, regression — see its module docstring
for exactly what's mechanically checked vs. heuristic, and how a
scientific *negative* result is deliberately never treated as a
failure), `orchestrator/claude_client.py` (non-interactive `claude -p
... --output-format json` invocation via argument-array subprocess
calls only — no shell strings, no `--allow-dangerously-skip-permissions`
by default), `orchestrator/git_ops.py` (safe checkpoint commits — never
force-pushes/resets/amends, refuses to start a task on a dirty tree so
a checkpoint's `git add -A` can never fold in unrelated work),
`orchestrator/prompts.py` (renders the initial task prompt from
`tasks/{NN}_*.md`, and the fix prompt from a failed QA report),
`orchestrator/run.py` (the CLI wiring all of the above together, plus
`--dry-run` and interruption/resume handling). `tests/orchestrator/`
unit-tests every piece with the `claude` CLI always mocked;
`tests/research/` checks the research documents and task specs
themselves stay intact.

## Project principles that shape the code

- No new renderer, no new foundation model: rendering is 100% Blender/Cycles (`bpy`); the encoder's weights are always frozen (never fine-tuned).
- Linear before neural: every probe and every `rho(T)` is ridge regression, not a trained network.
- Every experiment has a baseline/control and a strict scene-level train/test split.
- Additive, backward-compatible extension pattern: new task modules add parallel names/functions instead of rewriting what came before (e.g. `generate_scene`/`sample_scene`, `render_trajectory`/`render_scene`, `FrozenEncoder`/`VideoEncoder`, `vjepa2.py`/`vjepa.py`) — each documented in `IMPLEMENTATION_NOTES.md`.
- Every non-trivial API assumption (Blender compositor API, `transformers.VJEPA2Model` I/O contract, depth-pass semantics, which checkpoints actually exist on the Hub) is verified empirically against the installed version/live source before being trusted in code, not assumed from memory or documentation — and that verification is written down in `IMPLEMENTATION_NOTES.md`, not just left in scrollback.
- A good probe score is evidence of *accessibility*/*predictability*, not "understanding" — see `DESIGN.md` §13 ("what this framework can and cannot support") before drawing conclusions from any report in `reports/`.
- `pretrained: true/false` provenance must never be silently dropped or aggregated across — it's threaded through every report, result dict, and `metadata.json`.
