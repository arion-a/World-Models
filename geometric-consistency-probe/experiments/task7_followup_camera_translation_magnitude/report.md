# Task 7 follow-up -- camera_translation magnitude fix (N=100)

- pilot scenes: 10 (base_seed=101)
- candidate magnitudes tried: [3.0, 4.5, 6.0]
- chosen magnitude: 3
- full run scenes: 100 (train=80, test=20)
- encoder pretrained: True
- frozen verified (params bit-identical before/after pilot+full encoding): True

## Pilot detectability ratios

| magnitude | ratio | frac_below_floor | frac_too_close |
|---|---|---|---|
| 3 | 1.0948 | 0.00 | 0.00 |
| 4.5 | 1.8853 | 0.10 | 0.00 |
| 6 | 2.3548 | 0.10 | 0.10 |

## Full N=100 result

- detectability ratio (full set): 1.0179
- learned W_T held-out R^2: -0.5244
- persistence baseline R^2: -0.9147
- mean baseline R^2: -0.1084
- random-pair control R^2: -1.5266
