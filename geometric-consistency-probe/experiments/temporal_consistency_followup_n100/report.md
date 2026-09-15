# Task 12 -- temporal geometric consistency

- window_frames=4, fps=4.0, orbit_deg_per_sec=30.0
- real_motion: 100 scenes (train=80, test=20)
- static_control: 100 scenes (train=80, test=20)

| condition | method | R^2 | mean cosine sim | mean rel. L2 err |
|---|---|---|---|---|
| real_motion | learned_W_T | 0.4579 | 0.9843 | 0.1746 |
| real_motion | persistence_baseline | 0.5366 | 0.9864 | 0.1625 |
| real_motion | mean_baseline | -0.0451 | 0.9695 | 0.2403 |
| real_motion | random_pair_control | -0.4430 | 0.9578 | 0.2849 |
| static_control | learned_W_T | 0.8649 | 0.9930 | 0.1177 |
| static_control | persistence_baseline | 1.0000 | 1.0000 | 0.0000 |
| static_control | mean_baseline | -0.0656 | 0.9433 | 0.3290 |
| static_control | random_pair_control | -1.1665 | 0.8873 | 0.4669 |

margins: {
  "real_motion_learned_W_T_minus_own_best_baseline": -0.07869124412536621,
  "real_motion_learned_W_T_minus_static_control_learned_W_T": -0.4069926589727402
}
