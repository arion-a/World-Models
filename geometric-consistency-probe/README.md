# Geometric Consistency Probe (GCP)

Do self-supervised video representations encode the geometry of the
underlying physical world, or do they primarily encode appearance and
temporal correlations?

GCP is a small, controlled-experiment testbed for asking this question
of a **frozen, pretrained** video representation model -- initially
[V-JEPA 2](https://huggingface.co/docs/transformers/en/model_doc/vjepa2)
ViT-L/16 (`facebook/vjepa2-vitl-fpc64-256`) -- without training any new
model. We render synthetic 3D
scenes, apply a known physical transformation `T` to the scene, and check
whether the resulting change in the encoder's representation is
*predictable*:

```
Physical state S --renderer R--> video V = R(S) --frozen encoder E--> Z

S' = T(S) --renderer R--> V' = R(S') --E--> Z'

question: is Z' ≈ rho(T) Z, for some simple rho(T)?
```

Because scenes are synthetic, we have the ground-truth physical state
`S` for every render -- something no probe of a real-video encoder can
assume. That lets us go beyond "does this correlate with pixels" and ask
scientifically specific questions: is the change in representation
*equivariant* to a known transform, is it *invariant* to transforms that
shouldn't matter, and is physical state *linearly accessible* from Z at
all.

**Current status: V0.** The end-to-end pipeline (generate scenes -> apply
transforms -> render -> encode -> fit a linear `rho(T)` -> evaluate on
held-out scenes -> report) works and is unit tested. It has been
exercised on ~24 scenes with the real `VJEPA2Model` architecture; **real
pretrained weights now load successfully** (see "Pretrained weights"
below) but the numbers in `reports/` predate that and still reflect the
untrained-fallback caveat stated in each report -- the V0-scale run has
not yet been re-executed with real weights.

## Project principles (why the code looks the way it does)

- Start small: V0 covers exactly one pipeline, six transform types, and
  one physical-state probe. See `IMPLEMENTATION_NOTES.md` for what's
  deliberately deferred, and `DESIGN.md` for the research specification
  this pipeline implements.
- No new renderer, no new foundation model: rendering is 100% Blender/
  Cycles (`bpy`); the encoder's weights are always frozen.
- Linear before neural: every probe and every `rho(T)` here is ridge
  regression, not a trained network.
- Every experiment has a baseline (identity map, shuffled-pairing
  control, and a non-learned pixel-statistics encoder), and a strict
  scene-level train/test split (a scene's original and transformed
  renders are never split across train and test -- enforced in
  `representations/dataset.py` and checked by
  `tests/test_scene_split.py`).
- Every metric has a textbook definition, unit tested in isolation
  (`tests/test_metrics.py`).
- Results are reported with explicit interpretation guardrails: a good
  probe score is evidence of *accessibility* or *predictability*, not of
  "understanding" -- see `DESIGN.md`, "13. What this framework can and
  cannot support."

## Pretrained weights

`encoders/vjepa2.py` (V0) and `encoders/vjepa.py` (Task 4) both load
`facebook/vjepa2-vitl-fpc64-256` from Hugging Face Hub -- the official
Meta checkpoint, loaded through `transformers`' built-in `VJEPA2Model`
(no `trust_remote_code`). This repo was first built in a
network-sandboxed environment that blocked `huggingface.co`, so every
result currently in `reports/` was produced with the identical model
*architecture* but **randomly initialized weights** (the encoder detects
a failed download and falls back automatically, with a loud warning, so
the pipeline was still fully exercisable). That block has since been
lifted for this environment, and real pretrained weights have been
verified to load and produce a valid forward pass
(`tests/test_vjepa_encoder.py::test_pretrained_checkpoint_loads_real_weights`).
The V0-scale experiment in `reports/` has not yet been re-run with real
weights -- it still reflects the untrained-fallback caveat stated
explicitly in each report and in each result dict's `encoder_pretrained`
field.

Two checkpoints originally planned here turned out not to exist on the
Hub once real access was available: `facebook/vjepa2-vitb-fpc64-256`
(the ViT-B-sized V-JEPA 2 checkpoint DESIGN.md and this project's
original code assumed) is not hosted under any namespace, and "V-JEPA
2.1" has no official `facebook/`-namespaced checkpoint at all -- only
unverified individual/community conversions requiring
`trust_remote_code=True`. See `encoders/vjepa.py`'s module docstring and
`IMPLEMENTATION_NOTES.md`'s "Task 4 -- real pretrained weights" section
for the full account; the encoder is now V-JEPA 2 ViT-L/16, not V-JEPA
2.1 ViT-B/16 as originally planned.

## Repository layout

```
configs/          YAML configs (+ a small dataclass-based loader)
generation/       SceneState, trajectories, the Blender/Cycles renderer,
                  and Task 2's controlled-scene-generation entry point
                  (see generation/COORDINATE_SYSTEM.md)
transforms/       The 6 physical transforms T, and SE(3) math helpers
encoders/         Frozen encoder interface; V-JEPA 2 wrapper; pixel-statistics baseline
representations/  Extract & cache Z = E(V); assemble train/test splits
probes/           Linear rho(T) fitting; linear physical-state probe
metrics/          Equivariance, invariance, and the shared numeric primitives
baselines/        Identity and shuffled-pairing controls
experiments/      The end-to-end experiment runner
reports/          Generated result reports (see the V0 smoke test below)
tests/            pytest suite (fast unit tests + a few `@pytest.mark.slow` renderer smoke tests)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`bpy` (Blender as a Python module) is a large (~1 GB), Linux-focused
wheel; if it's unavailable for your platform, everything except
rendering (transforms, metrics, probes) still works and is still tested
via `pytest -m "not slow"`.

### Optional: RunPod GPU pods via MCP

Full-scale runs (`configs/default.yaml` -- 100 scenes, real pretrained
`facebook/vjepa2-vitl-fpc64-256` weights) are much faster with a GPU than
on CPU. This repo checks in a project-scoped `.mcp.json` (at the repo
root, alongside this project directory) that registers RunPod's hosted
MCP server, so a Claude Code session opened against this repo can
provision/manage a RunPod GPU pod directly instead of you doing it by
hand in the RunPod console.

One-time, per machine (the server config is shared via `.mcp.json`, but
the OAuth login itself is local and not something that can be checked
into the repo):

```bash
claude mcp login runpod
```

This is unrelated to the `bpy` renderer, which stays CPU-bound Blender
regardless of where it runs -- the GPU pod is only useful for the
`encoders`/`representations` forward-pass steps.

## Running the V0 experiment

```bash
# 1. Generate ~24 scenes with all 6 transforms rendered (fast dev config)
python -m generation.generate_dataset --config configs/experiments/camera_rotation_v0.yaml

# 2. Extract frozen-encoder representations (cached to disk after first run)
python -m representations.extract --config configs/experiments/camera_rotation_v0.yaml

# 3. Fit rho(T) per transform, evaluate on held-out scenes, write a report
python -m experiments.run --config configs/experiments/camera_rotation_v0.yaml
```

`experiments/run.py` also runs steps 1-2 automatically if their outputs
don't exist yet, so `python -m experiments.run --config ...` alone is
enough for a first run. Use `configs/default.yaml` (100 scenes, real
pretrained weights by default) for the actual V0-scale run once Hub
access is available; `configs/experiments/camera_rotation_v0.yaml` is a
small, fast (~2 minutes on CPU), no-network-required config for
development and CI-style smoke testing.

To try the non-learned pixel-statistics baseline encoder instead, set
`encoder.name: pixel_baseline` in a config (see
`encoders/pixel_baseline.py`).

## Task 2: controlled 3D scene generation with complete ground truth

A separate, independently-runnable generator (no transforms `T`
involved -- see "Task 3" below for those):

```bash
python -m generation.generate --config configs/generation.yaml
```

Produces ~10 scenes (per `configs/generation.yaml`), each with a
constant-velocity object/camera trajectory, under
`data/generation_v0/{scene_id}/`: `rgb.npy`, `depth.npy`,
`segmentation.npy`, and a `metadata.json` with camera intrinsics and
every frame's camera/object pose and velocity. **Read
`generation/COORDINATE_SYSTEM.md` before consuming any of this** --
it documents, with the empirical checks behind each claim, exactly what
world frame, camera convention, depth encoding, and segmentation
encoding are used, several of which differ from common
computer-vision defaults (e.g. depth is *not* Euclidean ray distance).

## Task 3: controlled geometric transformations (S' = T(S))

The transformation engine (`transforms/scene_transform.py`,
`transforms/se3.py`) applies one of six named transforms to a
`SceneState` and returns both the transformed scene and its exact
ground truth: an explicit 4x4 SE(3) matrix for the four geometric
transforms (camera/object translation and rotation), and, for every
transform, which `SceneState` fields changed and which were held fixed.
Generate a rendered original/transformed pair (reusing Task 2's
renderer, so each side gets full RGB + depth + segmentation ground
truth):

```python
from generation.scene_sampler import generate_scene, SceneSamplerConfig
from transforms.pairs import generate_pair

scene = generate_scene("demo", seed=0, cfg=SceneSamplerConfig())
pair_dir = generate_pair(scene, "camera_rotation", "data/pairs/demo_camera_rotation")
# -> data/pairs/demo_camera_rotation/{original,transformed}/{rgb,depth,segmentation}.npy + metadata.json
#    data/pairs/demo_camera_rotation/transformation.json
```

`transformation.json` records the transform's own parameters, its exact
`transform_matrix` (`null` for the two appearance-only controls, which
are not rigid transforms), and `changed_variables`/`fixed_variables`.
See `IMPLEMENTATION_NOTES.md`'s "Task 3" section for how the SE(3)
engine is built (in particular, why `camera_rotation`'s azimuth+
elevation orbit is expressible as a single rotation matrix, and the
world-to-camera vs. camera-to-world distinction) and
`tests/test_se3_matrices.py` / `tests/test_scene_transform_matrices.py`
/ `tests/test_pairs.py` for the identity/composition/inverse/isolation
tests this is built to satisfy.

## Task 4: frozen V-JEPA 2 representation extraction

`encoders/vjepa.py`'s `VJEPAEncoder` wraps V-JEPA 2 ViT-L/16
(`facebook/vjepa2-vitl-fpc64-256`, the official Meta checkpoint -- see
`IMPLEMENTATION_NOTES.md`'s "Task 4 -- real pretrained weights" section
for why this replaced the originally-planned V-JEPA 2.1 ViT-B/16, and
`encoders/vjepa.py`'s module docstring for the full verification
account). `encoders/extract.py` runs it and saves the result:

```python
from encoders.vjepa import VJEPAEncoder
from encoders.extract import extract_and_save, load_representation

encoder = VJEPAEncoder()  # pretrained=True by default; auto-uses a GPU if one is available
representation = encoder.encode(video)  # (num_tokens, 1024) float32 -- not pooled, see below
extract_and_save(encoder, video, "my_video", "data/representations")
representation, metadata = load_representation("data/representations", "my_video")
```

Or batch-process a Task 2/3-style dataset (any directory of
`{video_id}/rgb.npy` files):

```bash
python -m encoders.extract --input_dir data/generation_v0 --output_dir data/representations_v0
```

`encode()` returns the encoder's native token sequence, not a pooled
vector -- see `encoders/REPRESENTATION_FORMAT.md` for the exact shape
formula and on-disk `metadata.json` schema, and
`encoders/base.py`'s `VideoEncoder` docstring for why pooling is left to
whoever consumes these representations rather than done here.

## Tests

```bash
pytest                    # full suite, including real-renderer smoke tests
pytest -m "not slow"      # fast unit tests only (math, metrics, probes, splits) -- no bpy/torch model calls
```

## Results so far

See `reports/v0_smoke_test_untrained_encoder.md` for the first
end-to-end run (24 scenes, untrained-weights caveat as above) and what
it does and does not show. In short: the pipeline runs correctly
end-to-end and produces internally consistent, sensible numbers (e.g.
the appearance-only lighting control is the one transform where every
method's score drops), but no claim about V-JEPA 2's actual geometric
sensitivity can be made until it is re-run with real pretrained weights.
