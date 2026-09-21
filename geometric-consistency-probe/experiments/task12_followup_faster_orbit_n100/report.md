# Task 12 -- temporal geometric consistency

- window_frames=4, fps=4.0, orbit_deg_per_sec=90.0
- real_motion: 100 scenes (train=80, test=20)
- static_control: 100 scenes (train=80, test=20)

| condition | method | R^2 | mean cosine sim | mean rel. L2 err |
|---|---|---|---|---|
| real_motion | learned_W_T | 0.5043 | 0.9860 | 0.1628 |
| real_motion | persistence_baseline | 0.3556 | 0.9813 | 0.1904 |
| real_motion | mean_baseline | -0.0509 | 0.9697 | 0.2376 |
| real_motion | random_pair_control | -0.6273 | 0.9534 | 0.2887 |
| static_control | learned_W_T | 0.8780 | 0.9919 | 0.1256 |
| static_control | persistence_baseline | 1.0000 | 1.0000 | 0.0000 |
| static_control | mean_baseline | -0.0404 | 0.9294 | 0.3659 |
| static_control | random_pair_control | -0.9794 | 0.8716 | 0.4928 |

margins: {
  "real_motion_learned_W_T_minus_own_best_baseline": 0.14871472120285034,
  "real_motion_learned_W_T_minus_static_control_learned_W_T": -0.3736518621444702
}
