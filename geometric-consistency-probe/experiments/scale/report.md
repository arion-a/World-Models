# Task 11 -- scale experiment (camera_rotation)

- scale ladder: [15, 40, 100]
- seeds per point: [0, 1]
- train_fraction: 0.8
- primary encoder pretrained: True
- total wall clock: 2689.4s

## Per-scale-point results

| n_scenes | seeds ok | learned W_T R^2 (mean+/-std) | persistence | mean | shuffled-pairing | pixel-stats | random-encoder | wall clock (s) |
|---|---|---|---|---|---|---|---|---|
| 15 | 2/2 | -0.0504+/-0.1243 | 0.4443+/-0.1082 | -0.2205+/-0.0272 | -0.7879+/-0.1376 | 0.3929+/-0.0109 | 0.7479+/-0.0946 | 269.8 |
| 40 | 2/2 | -0.1626+/-0.0575 | -0.0151+/-0.0699 | -0.1718+/-0.0130 | -0.9368+/-0.0900 | 0.3385+/-0.0155 | 0.6737+/-0.0241 | 697.2 |
| 100 | 2/2 | -0.0204+/-0.0454 | 0.0824+/-0.0835 | -0.0536+/-0.0010 | -0.8134+/-0.2117 | 0.4584+/-0.0051 | 0.7093+/-0.0050 | 1722.4 |

## Trend

learned_W_T R^2 minus best-baseline R^2 (the 'margin') across the tested scale ladder [15, 40, 100]: [-0.7983, -0.8363, -0.7297]. Classified as 'strengthens' using a fixed +/-0.03 R^2 tolerance for 'materially changed', decided before computing these numbers.

Classification: **strengthens**
