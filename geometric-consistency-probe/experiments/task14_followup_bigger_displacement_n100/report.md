# Task 14 -- controlled counterfactual representation consistency

- conditions: ('object_translation', 'object_rotation'), target_displacement=2.0
- confound (leakage-check) conditions: ('camera_translation', 'lighting_change')

| check | accuracy | chance threshold | result |
|---|---|---|---|
| real discrimination | 0.7250 | 0.6581 | above chance |
| scrambled-label control | 0.4500 | 0.6581 | -- |
| representation-norm-only | 0.5500 | 0.6581 | PASS (not trivial) |
| fixed-variable leakage check | 0.5750 | 0.6581 | PASS |

## Magnitude matching

{
  "target_displacement": 2.0,
  "object_translation_measured_displacement": {
    "mean": 2.0,
    "std": 1.6467268631127714e-16,
    "min": 1.9999999999999996,
    "max": 2.0000000000000004
  },
  "object_rotation_measured_displacement": {
    "mean": 0.7225398880420751,
    "std": 0.13425953455343506,
    "min": 0.5004678913422407,
    "max": 0.9993353644814016
  },
  "mean_absolute_difference_between_conditions": 1.2774601119579247,
  "any_rotation_angle_clipped": true
}
