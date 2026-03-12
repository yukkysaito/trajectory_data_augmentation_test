# Trajectory Data Augmentation Test

This repository contains a reference implementation of a trajectory augmentation method for a diffusion-based planner.

The augmentation simulates lateral tracking or localization errors by shifting the ego pose sideways at `t0`, then generating:

- a past bridge that smoothly connects the original past trajectory to the offset state over `M` seconds
- a future bridge that smoothly returns from the offset state to the original ground-truth trajectory over `N` seconds

The implementation is designed to avoid two common failure modes:

- speed inconsistency caused by forcing the augmented path to reach an infeasible GT point in the same amount of time
- oscillatory trajectories caused by naive interpolation

## Main Idea

The future GT trajectory is treated as a centerline. The augmented path is constructed in a Frenet-like manner:

- the lateral offset decays with a monotonic quintic profile
- the merge point on the GT is adjusted if needed so the augmented path does not require higher speed than the original GT
- the same logic is applied backward in time for the past bridge

This keeps the path smooth while preserving feasible longitudinal progress.

## Files

- `diffusion_planner_augmentation.py`: augmentation implementation, synthetic data generator, visualization utilities, and CLI
- `tests/test_augmentation.py`: unit tests for continuity, speed feasibility, and curvature bounds

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

## Run a Single Example

```bash
python3 diffusion_planner_augmentation.py \
  --offset 1.0 \
  --recover-time 1.5 \
  --past-connect-time 1.0 \
  --output augmentation_demo.png
```

This generates a 3-panel figure with:

- trajectory
- speed
- curvature

The trajectory panel also shows a triangle marker at every 0.1 s pose for both GT and augmented trajectories so temporal spacing and heading are visible.

## Run the Full Sweep

```bash
python3 diffusion_planner_augmentation.py --sweep --output-prefix augmentation_demo
```

This generates one figure per parameter combination using:

- `offset = [-3, -2, -1, +1, +2, +3] m`
- `N = [0.5, 1.0, 1.5, 2.0] s`
- `M = [0.5, 1.0, 1.5, 2.0] s`

That results in 96 images with filenames like:

- `augmentation_demo_offsetm3p0m_N0p5s_M0p5s.png`
- `augmentation_demo_offsetp1p0m_N1p5s_M1p0s.png`
- `augmentation_demo_offsetp3p0m_N2p0s_M2p0s.png`

## Run Tests

```bash
python3 -m unittest discover -s tests -v
```

The test suite checks:

- recovery pose consistency with the intended merge point
- no speed overshoot during the past and future bridge windows
- continuity at `t0`
- bounded curvature and curvature variation

## Notes

- The repository uses synthetic GT data generated at 0.1 s resolution over `[-3 s, +8 s]`.
- Depending on the sign of the lateral offset and local curvature, the augmented path can be either longer or shorter than the original local GT path.
- If the augmented bridge is longer, the implementation delays the merge point on the GT rather than demanding infeasible speed.
