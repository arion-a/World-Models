# Task 12 -- temporal geometric consistency

- window_frames=4, fps=4.0, orbit_deg_per_sec=30.0
- real_motion: 40 scenes (train=32, test=8)
- static_control: 40 scenes (train=32, test=8)

| condition | method | R^2 | mean cosine sim | mean rel. L2 err |
|---|---|---|---|---|
| real_motion | learned_W_T | 0.3060 | 0.9841 | 0.1767 |
| real_motion | persistence_baseline | 0.3829 | 0.9859 | 0.1649 |
| real_motion | mean_baseline | -0.1682 | 0.9732 | 0.2282 |
| real_motion | random_pair_control | -0.4130 | 0.9679 | 0.2487 |
| static_control | learned_W_T | 0.6090 | 0.9823 | 0.1859 |
| static_control | persistence_baseline | 1.0000 | 1.0000 | 0.0000 |
| static_control | mean_baseline | -0.3121 | 0.9396 | 0.3394 |
| static_control | random_pair_control | -1.6605 | 0.8822 | 0.4809 |

margins: {
  "real_motion_learned_W_T_minus_own_best_baseline": -0.07689493894577026,
  "real_motion_learned_W_T_minus_static_control_learned_W_T": -0.30298686027526855
}
