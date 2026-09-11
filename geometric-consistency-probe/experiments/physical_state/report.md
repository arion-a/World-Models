# Task 8 -- physical-state accessibility

- scenes: 40 (train=32, test=8)
- render source: reused_task7_cache (experiments/geometric_consistency/camera_rotation)
- encoder pretrained: True
- frozen verified (params bit-identical before/after encoding): True
- max scene/render alignment diff: 0.0

## R^2 comparison (real probe vs. required controls)

| variable | encoding | real probe R^2 | shuffled-label R^2 | mean-baseline R^2 |
|---|---|---|---|---|
| camera_azimuth_deg | sin_cos | -0.7811 | -0.9797 | -0.0364 |
| camera_elevation_deg | raw | 0.2272 | -5.3429 | -0.3783 |
| camera_distance | raw | 0.3490 | 0.2526 | -0.0052 |
| primary_object_position_xy | raw | -1.6532 | -1.1823 | -0.2043 |

## Pose/angle convention

World frame is right-handed, Z-up (Blender convention; see generation/COORDINATE_SYSTEM.md). camera_azimuth_deg = degrees(atan2(camera.position.y, camera.position.x)), i.e. the angle of the camera's position in the world XY-plane, range (-180, 180]; camera_elevation_deg = degrees(atan2(z, hypot(x, y))); camera_distance = ||camera.position||_2, scene units; primary_object_position_xy = (x, y) of objects[0] (the object always present regardless of per-scene object count), scene units. camera_azimuth_deg is sampled uniformly over the FULL (0, 360) degree range (generation/scene_sampler.py's camera_azimuth_range_deg default), i.e. genuinely wraps around -- it is therefore probed via its (sin, cos) components (sin(radians(azimuth)), cos(radians(azimuth))), decoded back with atan2(sin, cos) only to report an interpretable angular error in degrees. The other three variables are sampled over ranges with no wraparound (elevation (20, 50) deg, distance (6, 8) scene units, object position +-1.6 scene units) and are probed directly as raw scalars/vectors.
