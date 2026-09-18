# Neural Mackerel Road results

## Run

- Device: `NVIDIA GeForce RTX 5090`
- Training maps: `64`
- Validation maps: `16`
- Held-out maps per model seed: `100`
- Held-out map set shared across model seeds: `yes`
- Across-seed summary: `mean ± population SD` (not a map-level CI)
- Steps: `120`
- Events: `NONE, LANDSLIDE, FALLEN_TREE, CART_BLOCKAGE`
- Transition targets: `none`
- Teacher engine: `none`

## Optimization

| Variant | Seed | Best step | Validation loss | Seconds |
|---|---:|---:|---:|---:|
| viability | 0 | 60 | 0.0000 | 112.1 |
| viability | 1 | 30 | 0.0000 | 102.5 |
| viability | 2 | 30 | 0.0000 | 103.1 |
| +balance | 0 | 70 | 0.0000 | 110.4 |
| +balance | 1 | 120 | 0.0000 | 111.4 |
| +balance | 2 | 110 | 0.0000 | 110.3 |
| +specialization | 0 | 90 | 0.7253 | 110.6 |
| +specialization | 1 | 120 | 0.8318 | 110.9 |
| +specialization | 2 | 120 | 0.6844 | 110.4 |
| +skill_weather | 0 | 90 | 1.4610 | 135.8 |
| +skill_weather | 1 | 120 | 1.4877 | 133.4 |
| +skill_weather | 2 | 120 | 1.5246 | 160.9 |

## Selected run

- Variant: `+skill_weather`
- Seed: `0`
- Validation loss: `1.4610`

## Held-out hard games

| Variant | Light win | Strong win | Failure | Freshness | Events | Skill leverage |
|---|---:|---:|---:|---:|---:|---:|
| viability | 0.667 ± 0.031 | 0.333 ± 0.031 | 0.000 ± 0.000 | 0.523 ± 0.054 | 0.258 ± 0.011 | 0.432 ± 0.027 |
| +balance | 0.470 ± 0.022 | 0.530 ± 0.022 | 0.000 ± 0.000 | 0.644 ± 0.023 | 0.233 ± 0.014 | 0.247 ± 0.023 |
| +specialization | 0.500 ± 0.008 | 0.500 ± 0.008 | 0.000 ± 0.000 | 0.434 ± 0.016 | 0.182 ± 0.028 | 0.493 ± 0.006 |
| +skill_weather | 0.560 ± 0.029 | 0.440 ± 0.029 | 0.000 ± 0.000 | 0.514 ± 0.039 | 0.172 ± 0.008 | 0.475 ± 0.004 |

## Selected held-out diagnostics

- Equal-skill light/strong/draw: `0.600/0.400/0.000`
- Exact-choice light/strong/draw: `0.550/0.450/0.000`
- Skill tolerance: `0.339`
- Mean freshness margin: `0.033`
- Event frequencies: `NONE=0.825, LANDSLIDE=0.064, FALLEN_TREE=0.048, CART_BLOCKAGE=0.064`
- Semantic penalties: `distance=0.0000, difficulty=0.0552, rain=0.0000, strong_role=0.0000, light_role=0.0000`

## Selected CGQ profile

- Epsilon to weaker-player score: `0.00:0.500, 0.25:0.250, 0.50:0.150, 0.75:0.070, 1.00:0.020`

## Plots

- [Objective ablation](../report/figures/neural-ablation.png)
- [Held-out skill profile](../report/figures/neural-skill-profile.png)
- [Event frequencies](../report/figures/neural-event-frequencies.png)
- [Validation history](../report/figures/neural-training.png)
- [Compact report overview](../report/figures/neural-report-overview.png)
