# Task 6 -- camera_rotation geometric consistency

- scenes: 40 (train=32, test=8)
- fixed azimuth magnitude: 30.0 deg
- encoder pretrained: True
- frozen verified (params bit-identical before/after encoding): True

| method | R^2 | mean cosine sim | mean rel. L2 err |
|---|---|---|---|
| learned W_T | -0.3130 | 0.9201 | 0.3828 |
| persistence | -0.1281 | 0.9318 | 0.3542 |
| mean baseline | -0.1073 | 0.9309 | 0.3618 |
| random-pair control | -0.8599 | 0.8888 | 0.4672 |
