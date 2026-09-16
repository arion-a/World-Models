# Task 14 -- controlled counterfactual representation consistency

- conditions: ('object_translation', 'object_rotation'), target_displacement=0.4
- confound (leakage-check) conditions: ('camera_translation', 'lighting_change')

| check | accuracy | chance threshold | result |
|---|---|---|---|
| real discrimination | 0.6250 | 0.7500 | at/below chance |
| scrambled-label control | 0.3750 | 0.7500 | -- |
| representation-norm-only | 0.9375 | 0.7500 | FAIL (trivial) |
| fixed-variable leakage check | 0.5625 | 0.7500 | PASS |

## Magnitude matching

{
  "target_displacement": 0.4,
  "object_translation_measured_displacement": {
    "mean": 0.4,
    "std": 5.755518659813419e-17,
    "min": 0.3999999999999999,
    "max": 0.40000000000000013
  },
  "object_rotation_measured_displacement": {
    "mean": 0.4,
    "std": 1.1970346188077034e-16,
    "min": 0.39999999999999974,
    "max": 0.40000000000000047
  },
  "mean_absolute_difference_between_conditions": 9.853229343548264e-17,
  "any_rotation_angle_clipped": false
}
