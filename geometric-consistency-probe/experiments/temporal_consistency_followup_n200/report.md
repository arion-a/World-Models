# Task 12 -- temporal geometric consistency

- window_frames=4, fps=4.0, orbit_deg_per_sec=30.0
- real_motion: 200 scenes (train=160, test=40)
- static_control: 200 scenes (train=160, test=40)

| condition | method | R^2 | mean cosine sim | mean rel. L2 err |
|---|---|---|---|---|
| real_motion | learned_W_T | 0.5150 | 0.9861 | 0.1628 |
| real_motion | persistence_baseline | 0.5327 | 0.9866 | 0.1615 |
| real_motion | mean_baseline | -0.0196 | 0.9703 | 0.2393 |
| real_motion | random_pair_control | -0.6523 | 0.9520 | 0.3022 |
| static_control | learned_W_T | 0.9405 | 0.9962 | 0.0859 |
| static_control | persistence_baseline | 1.0000 | 1.0000 | 0.0000 |
| static_control | mean_baseline | -0.0389 | 0.9328 | 0.3562 |
| static_control | random_pair_control | -0.7094 | 0.8918 | 0.4544 |

margins: {
  "real_motion_learned_W_T_minus_own_best_baseline": -0.01770082116127014,
  "real_motion_learned_W_T_minus_static_control_learned_W_T": -0.4255039878189564
}
