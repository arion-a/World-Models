# Task 7 -- multiple geometric transformations vs. appearance controls

- scenes: 40 (train=32, test=8)
- encoder pretrained: True
- frozen verified (params bit-identical before/after all six transforms' encoding): True

## R^2 comparison (learned W_T vs. all three baselines)

| transform | kind | learned W_T R^2 | persistence R^2 | mean R^2 | random-pair R^2 |
|---|---|---|---|---|---|
| camera_translation | geometric | 0.1976 | 0.6378 | -0.1644 | -1.0099 |
| camera_rotation | geometric | -0.1757 | 0.0789 | -0.2254 | -0.8977 |
| object_translation | geometric | 0.3812 | 0.8065 | -0.1244 | -0.9250 |
| object_rotation | geometric | -0.1948 | 0.3852 | -0.2184 | -0.9808 |
| lighting_change | control | 0.0700 | 0.4666 | -0.1608 | -1.1924 |
| texture_change | control | 0.4259 | 0.8700 | -0.1422 | -0.8333 |

## Visual-confound check

Pearson correlation between mean absolute pixel diff and learned W_T R^2 across the six transforms: -0.2258007342479408

## Task 6 reproduction check

{
  "task6_result_found": true,
  "task6_camera_rotation_r2": -0.31295549869537354,
  "task7_camera_rotation_r2": -0.1757209300994873,
  "difference": 0.13723456859588623,
  "identical_scene_split_as_task6": true,
  "note": "A numeric difference here is EXPECTED and intentional -- see this result's 'protocol_notes' field and transform_config_for()'s docstring (Task 7 uses camera_rotation's default azimuth RANGE for cross-transform comparability, not Task 6's fixed 30-degree magnitude). Task 6's own state/task_06_result.json is left unmodified on disk."
}
