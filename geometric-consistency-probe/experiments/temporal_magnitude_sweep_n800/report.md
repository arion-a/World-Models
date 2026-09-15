# Task 12 follow-up #3 -- camera-rotation magnitude sweep (N=400)

num_scenes=800, thetas_deg=[0.0, 5.0, 10.0, 20.0, 30.0], window_frames=4, fps=4.0

| theta (deg) | R2_learned | R2_persistence | R2_shuffled | delta_R2 | beats_shuffled | change_ratio | MSE_learned | alpha |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.9927 | 1.0000 | -1.3385 | -0.0073 | True | 0.0000 | 3.52 | 1 |
| 5 | 0.8475 | 0.8609 | -0.1692 | -0.0135 | True | 0.0768 | 72.70 | 100 |
| 10 | 0.7495 | 0.7143 | -0.1957 | +0.0352 | True | 0.1495 | 117.81 | 100 |
| 20 | 0.6306 | 0.4309 | -0.1992 | +0.1997 | True | 0.2980 | 173.58 | 100 |
| 30 | 0.5229 | 0.1875 | -0.1943 | +0.3354 | True | 0.4230 | 237.92 | 100 |
