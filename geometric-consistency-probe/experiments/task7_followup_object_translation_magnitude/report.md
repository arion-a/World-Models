# Task 7 follow-up -- object_translation magnitude fix (N=100)
- pilot candidates tested: [2.0] (fallback used: False)
- chosen magnitude: 2.0 (smallest candidate clearing ratio >= 0.3)
- scenes: 100 (train=80, test=20)
- encoder pretrained: True, frozen verified: True

## Pilot detectability ratios

| magnitude | ratio | object visible fraction |
|---|---|---|
| 2.0 | 0.5132 | 1.00 |

## Full N=100 result

- detectability ratio (N=100, all scenes): 0.3535
- learned W_T R^2 (held-out test): 0.1027
- persistence baseline R^2: 0.2883
- mean baseline R^2: -0.0788
- random-pair (shuffled) control R^2: -0.9865
- object visible after transform (fraction): 0.97

## Comparison to Task 7's original object_translation result

{
  "found": true,
  "original_n_scenes": 40,
  "original_learned_W_T_r2": 0.381194531917572,
  "original_transform_config": {
    "camera_translation_range": [
      0.5,
      1.5
    ],
    "camera_rotation_azimuth_deg_range": [
      10.0,
      35.0
    ],
    "camera_rotation_elevation_deg_range": [
      -10.0,
      10.0
    ],
    "object_translation_range": [
      0.4,
      1.0
    ],
    "object_rotation_deg_range": [
      30.0,
      150.0
    ],
    "light_energy_scale_range": [
      0.3,
      2.5
    ],
    "light_azimuth_deg_range": [
      60.0,
      180.0
    ],
    "texture_color_jitter": 0.6
  }
}
