# Trajectory Data Augmentation Test

This repository contains a reference implementation of trajectory augmentation for a diffusion-based planner.

The augmentation simulates lateral tracking or localization errors by shifting the ego pose sideways at `t0`, then generating:

- a past bridge that smoothly connects the original past trajectory to the offset state over `M` seconds
- a future bridge that smoothly returns from the offset state to the original GT trajectory over `N` seconds
- once the bridge ends, the trajectory follows the original GT again

The implementation is designed to avoid two common failure modes:

- speed inconsistency caused by forcing the augmented path to reach an infeasible GT point in the same amount of time
- oscillatory trajectories caused by naive interpolation

## Horizon Setup

The repository now uses two different horizons:

- full GT context: past `5 s`, future `10 s`
- actual augmented output window: past `3 s`, future `8 s`

The augmentation is computed on the full GT trajectory so that cases with longer augmented paths still have extra trajectory context available. The final training-style output and the main visualization focus on the `[-3 s, +8 s]` window.

## Test Pattern CSVs

Trajectory patterns are stored as CSV files under `test_patterns/`.

Each CSV contains:

- `t`
- `x`
- `y`
- `yaw`
- `speed_mps`

The repository supports these geometry patterns:

- `straight`
- `curve`
- `s_curve`

and these speed profiles:

- `constant`
- `decelerating`
- `accelerating`
- `stopping`

This gives 12 combinations in total, for example:

- `straight_constant.csv`
- `curve_decelerating.csv`
- `s_curve_accelerating.csv`
- `straight_stopping.csv`

## Main Idea

The GT trajectory is treated as a centerline. The augmented path is constructed in a Frenet-like manner:

- the lateral offset is generated with a monotonic time-polynomial bridge
- the bridge is state-matched to the GT at the merge time
- after the merge time, the augmented trajectory copies the original GT exactly
- the same logic is applied backward in time for the past bridge

This keeps the path smooth while preserving continuity at `t0` and exact GT tracking after the merge.

## Bridge Feasibility

The repository evaluates three feasibility constraints:

- `a_lat = v^2 * kappa`
- bridge speed gap: `|v_aug - v_gt|`
- bridge longitudinal jerk: `|dj/dt|`

If a requested augmentation exceeds any configured limit, the visualization highlights the violation and also searches for the lowest passing bridge-time candidate:

1. find the minimum feasible `N`
2. if that is still insufficient, find the minimum feasible pair `(M, N)`

When `--recover-time` and `--past-connect-time` are omitted, the demo starts from an auto-search seed of `0.1 s` and directly selects the lowest feasible bridge times it can find within the available full GT horizon.

## Files

- `diffusion_planner_augmentation.py`: augmentation implementation, CSV pattern generation/loading, visualization utilities, and CLI
- `test_patterns/*.csv`: trajectory test patterns
- `tests/test_augmentation.py`: unit tests for continuity, merge behavior, feasibility search, horizon handling, and curvature bounds

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

## Write the CSV Pattern Files

```bash
python3 diffusion_planner_augmentation.py --write-pattern-csvs
```

## List Available Pattern Names

```bash
python3 diffusion_planner_augmentation.py --list-patterns
```

## Run a Single Example

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --offset -3.0 \
  --recover-time 1.5 \
  --past-connect-time 1.0 \
  --max-lateral-accel 3.0 \
  --max-bridge-speed-gap 0.5 \
  --max-bridge-jerk 5.0 \
  --output augmentation_demo.png
```

This generates a 6-panel figure with:

- trajectory
- speed
- curvature
- lateral acceleration
- bridge speed gap
- bridge longitudinal jerk

The trajectory panel shows:

- full GT context in light gray
- the actual `[-3 s, +8 s]` output window
- a triangle marker at every `0.1 s` pose for both GT and augmented trajectories, so temporal spacing and heading are visible

The feasibility panels show:

- GT lateral acceleration
- the requested augmentation, or the auto-search seed if `M/N` were omitted
- red markers where the requested trajectory exceeds the limit
- the lowest passing candidate, if one is found

## Run a Single Example With Automatic Bridge-Time Search

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --offset 3.0 \
  --yaw-offset-deg 10.0 \
  --max-lateral-accel 3.0 \
  --max-bridge-speed-gap 0.5 \
  --max-bridge-jerk 5.0 \
  --output augmentation_demo_auto_bridge.png
```

In this mode:

- the search starts from `M = 0.1 s`, `N = 0.1 s`
- it first searches for the minimum feasible `N`
- if that still fails, it searches for the minimum feasible `(M, N)` pair

If you want a fixed `M/N` visualization with diagnostics only, keep `--recover-time` and `--past-connect-time` explicitly set.

## Run the Full Sweep

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --sweep \
  --output-prefix augmentation_demo
```

This generates one figure per parameter combination using:

- `offset = [-3, -2, -1, +1, +2, +3] m`
- `N = [0.5, 1.0, 1.5, 2.0] s`
- `M = [0.5, 1.0, 1.5, 2.0] s`

That results in 96 images for the selected pattern, with filenames like:

- `augmentation_demo_curve_decelerating_offsetm3p0m_N0p5s_M0p5s.png`
- `augmentation_demo_curve_decelerating_offsetp1p0m_N1p5s_M1p0s.png`
- `augmentation_demo_curve_decelerating_offsetp3p0m_N2p0s_M2p0s.png`

## Run Tests

```bash
python3 -m unittest discover -s tests -v
```

The test suite checks:

- all 12 CSV patterns can be generated and loaded
- the full GT horizon is `[-5 s, +10 s]`
- the output window is `[-3 s, +8 s]`
- the augmented future matches the GT exactly after the merge time
- recovery pose consistency with the intended merge point
- continuity at `t0`
- bounded curvature and curvature variation
- exact endpoint agreement at `+8 s` after state-matched merge
- composite feasibility search for both the `N-only` and `(M, N)` fallback paths
- stopping-pattern behavior

## Notes

- The feasibility search currently uses sampled trajectory kinematics inside the bridge windows rather than a full vehicle model.
- Low-speed or stopping patterns are supported as synthetic test cases, but no steering-rate or tire-force limits are modeled yet.
