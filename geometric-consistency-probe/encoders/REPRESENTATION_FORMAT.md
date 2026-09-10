# Representation tensor format (Task 4)

This is the reference for exactly what `encoders/vjepa.py:VJEPAEncoder.encode()`
returns and what `encoders/extract.py` saves to disk. Read this before
writing anything that consumes a saved representation.

## In memory: `VJEPAEncoder.encode(video)`

```
input:  video, shape (T, H, W, 3), dtype uint8, RGB
output: representation, shape (num_tokens, hidden_size), dtype float32
```

- **Not pooled.** This is the encoder's native output token sequence
  (`VJEPA2Model.get_vision_features(...)`'s `last_hidden_state`, batch
  dimension squeezed out) -- one row per spatiotemporal patch, not a
  single vector per clip. Pooling is deliberately left to whatever
  consumes this (see `encoders/base.py`'s `VideoEncoder` docstring for
  why); `encoders/vjepa.py:mean_pool(representation)` is provided as the
  obvious first reduction if a single vector is wanted, but is not
  applied automatically.
- **`hidden_size`** is `768` for the ViT-B/16 configuration this project
  uses (`encoders/vjepa.py:VITB16_CONFIG_KWARGS`), whether or not
  pretrained weights loaded (the untrained fallback uses the same
  architecture size).
- **`num_tokens`** depends on the input video's shape and the encoder's
  `tubelet_size`/`patch_size`/`crop_size`:

  ```
  num_tokens = (T_used / tubelet_size) * (crop_size / patch_size) ** 2
  ```

  where `T_used` is `T` rounded up to a multiple of `tubelet_size` (the
  underlying `VJEPA2Embeddings` repeats frames if `T < tubelet_size`;
  see `transformers`' `modeling_vjepa2.py`), and `crop_size`/`patch_size`
  are the encoder's configured values (`crop_size` defaults to 384 for
  V-JEPA 2.1; `patch_size` is fixed at 16). Concretely, for the default
  8-frame, `tubelet_size=2` configuration: `T_used/tubelet_size = 4`
  temporal groups, times `(crop_size/16)**2` spatial patches.
- **Token ordering** matches `Conv3d`'s flattened output order in
  `VJEPA2PatchEmbeddings3D.forward` (`.flatten(2).transpose(1, 2)`):
  row-major over `(temporal_group, patch_row, patch_col)`, i.e. all
  spatial patches of temporal group 0, then all of temporal group 1,
  etc. This is a direct consequence of the verified upstream API
  (`encoders/vjepa.py`'s docstring), not independently re-derived here.

## On disk: `encoders/extract.py`

```
out_dir/{video_id}/representation.npy   -- the array above, as saved by np.save
out_dir/{video_id}/metadata.json        -- provenance (see below)
```

`metadata.json` schema (all fields always present):

```json
{
  "video_id": "scene_0000",
  "encoder": "VJEPAEncoder",
  "checkpoint": "apiantonio/vjepa2.1-vit-base-384",
  "pretrained": false,
  "preprocessing_config": {
    "crop_size": 384,
    "resize_size": 439,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "resize_mode": "bilinear"
  },
  "video_shape": [8, 128, 128, 3],
  "num_frames": 8,
  "frame_height": 128,
  "frame_width": 128,
  "representation_shape": [64, 768],
  "representation_dtype": "float32",
  "device": "cpu"
}
```

- `checkpoint` / `pretrained` are the single most important fields to
  check before trusting any downstream result: `pretrained: false` means
  the representation came from a randomly-initialized (though seeded and
  reproducible -- see `encoders/vjepa.py`) architecture, not real
  V-JEPA 2.1 weights. **Never aggregate or compare representations
  across a `pretrained: true` / `pretrained: false` boundary as if they
  meant the same thing.**
- `preprocessing_config` plus `video_shape` together let you reconstruct
  exactly what pixels the model actually saw, without re-running
  inference.
- `representation_shape`/`representation_dtype` are redundant with
  `representation.npy`'s own `.shape`/`.dtype` once loaded -- they are
  saved anyway so a `metadata.json` on its own (e.g. before deciding
  whether to load a large `.npy`) already answers "what shape would this
  be."

Load a saved representation with `encoders.extract.load_representation(out_dir, video_id) -> (representation, metadata)`.
