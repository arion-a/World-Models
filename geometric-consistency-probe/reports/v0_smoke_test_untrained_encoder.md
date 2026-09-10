# V0 smoke test: end-to-end pipeline check (untrained encoder)

**Status: pipeline validation only.** This report exists to show the
full pipeline runs correctly and produces numbers that respond
sensibly to the experimental manipulation. It is **not** a claim about
what V-JEPA 2 represents, because the encoder here has randomly
initialized weights, not the pretrained ones -- see "Why untrained?"
below. Treat every number as a check on the code, not on the model.

Reproduce with:

```
python -m generation.generate_dataset --config configs/experiments/camera_rotation_v0.yaml
python -m representations.extract --config configs/experiments/camera_rotation_v0.yaml
python -m experiments.run --config configs/experiments/camera_rotation_v0.yaml
```

Raw output: `reports/v0_smoke_test_untrained_encoder.json`.

## Setup

- 24 scenes, 1-3 primitives each (cube/cone/cylinder/monkey), scene-level
  75/25 train/test split (18 train, 6 test), seeded and reproducible.
- Renderer: Blender/Cycles (`bpy`), 128x128, CPU.
- Encoder: `VJEPA2Model` (ViT-B/16 architecture: hidden size 768, 12
  layers), **random initialization** -- see below.
- rho(T): ridge regression, alpha=10, fit on train scenes, evaluated on
  held-out test scenes.
- Baselines: identity (predict Z'=Z), shuffled-pairing (fit on
  deliberately mismatched scene pairs).

## Why untrained?

`encoders/vjepa2.py` calls `VJEPA2Model.from_pretrained(...)`, which
downloads weights from Hugging Face Hub. The sandbox this repo was
first built in blocks outbound access to `huggingface.co` at the
network-policy level (confirmed via the proxy's diagnostic endpoint,
not just a timeout). The encoder detects the failed download, prints a
loud `RuntimeWarning`, and falls back to the same architecture with
random weights so the rest of the pipeline can still be exercised.
**Anyone running this outside that sandbox should get real pretrained
weights automatically** -- set `encoder.pretrained: true` (the default)
and nothing else needs to change.

## Results

| transform | learned rho(T) R^2 | identity R^2 | shuffled-pairing R^2 | raw cosine sim |
|---|---:|---:|---:|---:|
| camera_translation | 0.682 | 0.834 | 0.414 | 0.993 |
| camera_rotation | 0.558 | 0.744 | 0.507 | 0.991 |
| object_translation | 0.815 | 0.982 | 0.450 | 0.999 |
| object_rotation | 0.834 | 0.991 | 0.485 | 1.000 |
| lighting_change | -0.500 | -0.364 | -0.028 | 0.930 |
| texture_change | 0.807 | 0.970 | 0.454 | 0.999 |

Physical-state probe (camera azimuth, degrees, linear ridge on Z):
held-out R^2 = **-0.583** (worse than predicting the mean).

## Reading these numbers correctly

1. **Identity beats or ties the learned map on every geometric
   transform.** With random weights, the pooled representation of a
   scene barely moves under any of these transforms (raw cosine
   similarity >= 0.99 across the board) -- an untrained deep net's
   mean-pooled output is dominated by initialization statistics, not by
   fine-grained input content. So "predict Z' = Z" is nearly optimal,
   and there is very little residual structure left for a *learned*
   map to explain. This is expected for random weights and is exactly
   why the identity baseline must always be reported alongside the
   learned rho(T): a bare R^2 number for the learned map, without it,
   would look far more impressive than it is.

2. **The shuffled-pairing baseline scores 0.41-0.51 R^2 on the
   geometric transforms** -- fitting a ridge map on *deliberately wrong*
   scene correspondences still explains about half the test variance.
   Combined with (1), this says most of what the learned map "explains"
   on this run is dataset-level structure (e.g., everything shrinking
   toward a similar mean under ridge regularization with only 18
   training pairs in 768 dimensions), not scene-specific correspondence.
   With random weights and this little training data, that is the
   expected failure mode of an underdetermined linear fit, not evidence
   about geometry.

3. **`lighting_change` is the one transform where every method,
   including identity, scores negative R^2.** Raw cosine similarity is
   still high (0.93) but visibly the lowest of the six -- consistent
   with lighting being the transform most likely to change pixel-level
   shading substantially. The negative R^2 values themselves are likely
   dominated by the small test set (n=6) rather than a real effect; this
   is flagged, not concluded.

4. **The physical-state probe (component C) fails outright** (R^2 <
   0), i.e. a linear map cannot recover camera azimuth from these
   representations better than guessing the mean. With random weights
   this is unsurprising and uninformative either way.

## What this run does and does not show

- **Does show:** scene generation, transform application, rendering,
  encoding (with the real `VJEPA2Model` architecture and its verified
  input/output contract), ridge-regression fitting, the equivariance /
  invariance / accessibility metrics, and the baselines all work
  correctly together, with a strict scene-level train/test split
  (enforced in code and unit tested).
- **Does not show:** anything about whether V-JEPA 2 encodes geometry.
  That question needs the pretrained checkpoint. Re-running with
  `encoder.pretrained: true` on a machine with Hugging Face Hub access
  requires no code changes -- only network access this sandbox did not
  have.
- **Also worth noting as a genuine V0 design caveat** (not a result):
  every clip in this run is a single rendered frame repeated across
  time (see DESIGN.md, "Static-clip videos"), which is out-of-distribution
  input for a model pretrained on natural motion. Once real weights are
  available, this is a variable worth ablating (e.g., compare against
  clips with a small synthetic camera pan) before trusting equivariance
  numbers from the pretrained model at face value.
