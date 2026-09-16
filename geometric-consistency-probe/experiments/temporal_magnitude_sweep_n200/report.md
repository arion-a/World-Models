# Task 12 follow-up #3 -- camera-rotation magnitude sweep (N=400)

num_scenes=200, thetas_deg=[0.0, 5.0, 10.0, 20.0, 30.0], window_frames=4, fps=4.0

| theta (deg) | R2_learned | R2_persistence | R2_shuffled | delta_R2 | beats_shuffled | change_ratio | MSE_learned | alpha |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.9309 | 1.0000 | -1.3011 | -0.0691 | True | 0.0000 | 30.80 | 1 |
| 5 | 0.7670 | 0.8579 | -0.3597 | -0.0909 | True | 0.0770 | 102.82 | 100 |
| 10 | 0.6854 | 0.7537 | -0.3239 | -0.0683 | True | 0.1495 | 138.40 | 100 |
| 20 | 0.4826 | 0.4887 | -0.3473 | -0.0061 | True | 0.3124 | 239.14 | 100 |
| 30 | 0.3817 | 0.2416 | -0.3430 | +0.1401 | True | 0.4276 | 285.79 | 100 |
