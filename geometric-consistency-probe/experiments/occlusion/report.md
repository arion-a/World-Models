# Task 13 -- object persistence under occlusion
- window_frames=4, num_frames=12, fps=4.0
- occlusion_pixel_threshold=8 (resolution=128)
- measured occlusion duration (frames): [4]

| variable | condition | R^2 | MAE | RMSE |
|---|---|---|---|---|
| post_occlusion_position_x | occlusion (real) | -0.4930 | 0.1418 | 0.1634 |
| post_occlusion_position_x | occlusion (scrambled-identity) | -0.0351 | 0.1513 | 0.1790 |
| post_occlusion_position_x | occlusion (mean baseline) | -0.0071 | 0.1308 | 0.1475 |
| post_occlusion_position_x | no_occlusion (upper bound) | -0.6375 | 0.0872 | 0.1020 |
| post_occlusion_position_x | diagnostic: occluded-window probe | -0.4536 | 0.1498 | 0.1799 |
| post_occlusion_position_x | diagnostic: pre-window probe | 0.3737 | 0.1380 | 0.1575 |
| post_occlusion_velocity_x | occlusion (real) | -0.4930 | 0.2269 | 0.2614 |
| post_occlusion_velocity_x | occlusion (scrambled-identity) | -0.0351 | 0.2421 | 0.2864 |
| post_occlusion_velocity_x | occlusion (mean baseline) | -0.0071 | 0.2092 | 0.2360 |
| post_occlusion_velocity_x | no_occlusion (upper bound) | -0.6375 | 0.1394 | 0.1632 |
| post_occlusion_velocity_x | diagnostic: occluded-window probe | -0.4536 | 0.2397 | 0.2878 |
| post_occlusion_velocity_x | diagnostic: pre-window probe | 0.3737 | 0.2208 | 0.2521 |

## Trivial-cue investigation

max |r| = 0.3202

{
  "position_x_vs_color_r": [
    0.16560693293018047,
    -0.32024679796001615,
    -0.2710534815735905
  ],
  "position_x_vs_floor_shade_r": 0.04274717212188145,
  "position_x_vs_light_energy_r": 0.06832507898240898,
  "position_x_vs_tracked_scale_r": 0.08220118318250105,
  "velocity_x_vs_color_r": [
    0.16560693293018042,
    -0.3202467979600163,
    -0.2710534815735904
  ],
  "velocity_x_vs_floor_shade_r": 0.04274717212188126,
  "velocity_x_vs_light_energy_r": 0.06832507898240883,
  "velocity_x_vs_tracked_scale_r": 0.08220118318250133
}
