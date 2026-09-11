# Task 10 -- baseline framework: expanded flagship comparison

- transform: camera_rotation (Task 6/7's flagship)
- scenes: 40 (train=32, test=8)
- primary encoder pretrained: True, frozen verified: True
- random-encoder frozen verified: True
- numerical parity with state/task_06_result.json: CONFIRMED

## Five-baseline comparison

| baseline | R^2 | mean cosine sim | mean rel. L2 err |
|---|---|---|---|
| learned W_T (primary, pretrained) | -0.3130 | 0.9201 | 0.3828 |
| persistence | -0.1281 | 0.9318 | 0.3542 |
| mean | -0.1073 | 0.9309 | 0.3618 |
| shuffled-pairing | -0.8599 | 0.8888 | 0.4672 |
| pixel-statistics encoder | 0.2212 | 0.9904 | 0.1385 |
| randomly-initialized encoder | 0.7809 | 0.9922 | 0.1046 |

## Parity check detail

{
  "learned_W_T.r2": {
    "recomputed": -0.3129563331604004,
    "recorded": -0.31295549869537354,
    "abs_diff": 8.344650268554688e-07
  },
  "learned_W_T.mean_cosine_similarity": {
    "recomputed": 0.9201258420944214,
    "recorded": 0.9201258420944214,
    "abs_diff": 0.0
  },
  "learned_W_T.mean_relative_l2_error": {
    "recomputed": 0.38277357816696167,
    "recorded": 0.3827735185623169,
    "abs_diff": 5.960464477539063e-08
  },
  "persistence_baseline.r2": {
    "recomputed": -0.12811720371246338,
    "recorded": -0.12811720371246338,
    "abs_diff": 0.0
  },
  "persistence_baseline.mean_cosine_similarity": {
    "recomputed": 0.9318380355834961,
    "recorded": 0.9318380355834961,
    "abs_diff": 0.0
  },
  "persistence_baseline.mean_relative_l2_error": {
    "recomputed": 0.35424643754959106,
    "recorded": 0.35424643754959106,
    "abs_diff": 0.0
  },
  "mean_baseline.r2": {
    "recomputed": -0.10729348659515381,
    "recorded": -0.10729348659515381,
    "abs_diff": 0.0
  },
  "mean_baseline.mean_cosine_similarity": {
    "recomputed": 0.930877685546875,
    "recorded": 0.930877685546875,
    "abs_diff": 0.0
  },
  "mean_baseline.mean_relative_l2_error": {
    "recomputed": 0.36177998781204224,
    "recorded": 0.36177998781204224,
    "abs_diff": 0.0
  },
  "random_pair_control.r2": {
    "recomputed": -0.8598562479019165,
    "recorded": -0.8598562479019165,
    "abs_diff": 0.0
  },
  "random_pair_control.mean_cosine_similarity": {
    "recomputed": 0.8888447284698486,
    "recorded": 0.8888447284698486,
    "abs_diff": 0.0
  },
  "random_pair_control.mean_relative_l2_error": {
    "recomputed": 0.46717655658721924,
    "recorded": 0.46717655658721924,
    "abs_diff": 0.0
  }
}
