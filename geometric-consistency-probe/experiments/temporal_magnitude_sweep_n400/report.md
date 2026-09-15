# Task 12 follow-up #3 -- camera-rotation magnitude sweep (N=400)

num_scenes=400, thetas_deg=[0.0, 5.0, 10.0, 20.0, 30.0], window_frames=4, fps=4.0

| theta (deg) | R2_learned | R2_persistence | R2_shuffled | delta_R2 | beats_shuffled | change_ratio | MSE_learned | alpha |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.9745 | 1.0000 | -1.3515 | -0.0255 | True | 0.0000 | 12.73 | 1 |
| 5 | 0.8042 | 0.8257 | -0.2292 | -0.0215 | True | 0.0814 | 94.45 | 100 |
| 10 | 0.6801 | 0.6683 | -0.2361 | +0.0118 | True | 0.1605 | 150.06 | 100 |
| 20 | 0.5505 | 0.3726 | -0.2042 | +0.1778 | True | 0.3140 | 216.60 | 100 |
| 30 | 0.4821 | 0.1307 | -0.2100 | +0.3514 | True | 0.4290 | 257.49 | 100 |
