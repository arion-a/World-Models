# Task 13 -- object persistence under occlusion
- window_frames=4, num_frames=12, fps=4.0
- occlusion_pixel_threshold=8 (resolution=128)
- measured occlusion duration (frames): [4]

| variable | condition | R^2 | MAE | RMSE |
|---|---|---|---|---|
| post_occlusion_position_x | occlusion (real) | -0.0069 | 0.1181 | 0.1389 |
| post_occlusion_position_x | occlusion (scrambled-identity) | -0.3593 | 0.1168 | 0.1367 |
| post_occlusion_position_x | occlusion (mean baseline) | -0.0221 | 0.1132 | 0.1265 |
| post_occlusion_position_x | no_occlusion (upper bound) | 0.0574 | 0.0719 | 0.0859 |
| post_occlusion_position_x | diagnostic: occluded-window probe | -0.5433 | 0.1212 | 0.1427 |
| post_occlusion_position_x | diagnostic: pre-window probe | 0.2303 | 0.1209 | 0.1429 |
| post_occlusion_velocity_x | occlusion (real) | -0.0069 | 0.1890 | 0.2222 |
| post_occlusion_velocity_x | occlusion (scrambled-identity) | -0.3593 | 0.1868 | 0.2187 |
| post_occlusion_velocity_x | occlusion (mean baseline) | -0.0221 | 0.1811 | 0.2025 |
| post_occlusion_velocity_x | no_occlusion (upper bound) | 0.0574 | 0.1150 | 0.1375 |
| post_occlusion_velocity_x | diagnostic: occluded-window probe | -0.5433 | 0.1938 | 0.2283 |
| post_occlusion_velocity_x | diagnostic: pre-window probe | 0.2303 | 0.1934 | 0.2286 |

## Trivial-cue investigation

max |r| = 0.1670

{
  "position_x_vs_color_r": [
    0.09683048526229726,
    -0.16698572735417175,
    -0.12296890064654714
  ],
  "position_x_vs_floor_shade_r": -0.06848926423097207,
  "position_x_vs_light_energy_r": -0.02229104815704409,
  "position_x_vs_tracked_scale_r": 0.046300402905213756,
  "velocity_x_vs_color_r": [
    0.09683048526229719,
    -0.16698572735417191,
    -0.12296890064654722
  ],
  "velocity_x_vs_floor_shade_r": -0.06848926423097193,
  "velocity_x_vs_light_energy_r": -0.02229104815704414,
  "velocity_x_vs_tracked_scale_r": 0.046300402905213846
}
