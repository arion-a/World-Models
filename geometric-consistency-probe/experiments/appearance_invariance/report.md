# Task 9 -- appearance invariance controls

- scenes: 40
- encoder pretrained: True
- frozen verified (params bit-identical before/after all encoding): True

## Invariance comparison (direct Z vs Z', no map fit)

| transform | kind | mean cosine similarity | mean relative L2 error |
|---|---|---|---|
| lighting_change | appearance control | 0.9736 | 0.2200 |
| texture_change | appearance control | 0.9915 | 0.1213 |
| camera_translation | geometric | 0.9761 | 0.2084 |
| camera_rotation | geometric | 0.9481 | 0.3080 |
| object_translation | geometric | 0.9871 | 0.1530 |
| object_rotation | geometric | 0.9637 | 0.2281 |
| null_transform | sanity check | 1.0000 | 0.0000 |

## Null-transform sanity check

{
  "transform_name": "null_transform",
  "is_geometric": false,
  "is_appearance_control": false,
  "n": 40,
  "mean_cosine_similarity": 1.0,
  "mean_relative_l2_error": 0.0,
  "is_null_sanity_check": true,
  "cosine_threshold": 0.999,
  "relative_l2_threshold": 0.05,
  "passed_threshold": true
}

## Geometric-vs-appearance confound investigation

No clean separation: at least one appearance control is at least as invariant-violating as at least one geometric transform (or vice-versa is not uniformly true). This is a meaningful negative finding that directly undercuts confidence in Tasks 6-7's geometric interpretation and must be reported prominently, not minimized (tasks/09_appearance_invariance.md's 'What a negative result means').
