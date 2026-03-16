# Trajectory Data Augmentation Test

This repository contains a reference implementation of trajectory augmentation for a diffusion-based planner.

The augmentation simulates lateral tracking or localization errors by shifting the ego pose sideways at `t0`, then generating:

- a past bridge that smoothly connects the original past trajectory to the offset state over `M` seconds
- a future bridge that smoothly returns from the offset state to the original GT trajectory over `N` seconds

## Horizon Setup

- full GT context: past `5 s`, future `10 s`
- augmented output window: past `3 s`, future `8 s`

The augmentation is computed on the full GT trajectory so larger offsets still have extra context available. The main plots focus on the `[-3 s, +8 s]` window.

## Test Pattern CSVs

Trajectory patterns are stored under `test_patterns/`.

Supported geometry patterns:

- `straight`
- `curve`
- `s_curve`

Supported speed profiles:

- `constant`
- `decelerating`
- `accelerating`
- `stopping`
- `stop8s`

This gives 15 combinations such as:

- `straight_constant.csv`
- `curve_decelerating.csv`
- `straight_stop8s.csv`
- `s_curve_accelerating.csv`

## Package Layout

The code is now split into reusable modules:

- `trajectory_augmentation/core.py`: augmentation logic, CSV I/O, diagnostics, reusable case runner
- `trajectory_augmentation/visualization.py`: Plotly figure builders
- `trajectory_augmentation/cli.py`: command-line entrypoint and sweep rendering
- `diffusion_planner_augmentation.py`: backward-compatible wrapper script

This makes it easier to reuse the augmentation logic and visualization helpers from other repositories such as `Diffusion-Planner`.

## Requirements

- Python 3.10+
- `numpy`
- `plotly`

## Write the CSV Pattern Files

```bash
python3 diffusion_planner_augmentation.py --write-pattern-csvs
```

## List Available Pattern Names

```bash
python3 diffusion_planner_augmentation.py --list-patterns
```

## Run a Single Interactive Example

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --offset -3.0 \
  --recover-time 1.5 \
  --past-connect-time 1.0 \
  --max-lateral-accel 3.0 \
  --output augmentation_demo.html
```

The HTML output uses Plotly, so you can zoom, pan, hide traces from the legend, and inspect points with hover tooltips.

## Run a Single Example With Automatic Bridge-Time Search

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --offset 3.0 \
  --yaw-offset-deg 10.0 \
  --max-lateral-accel 3.0 \
  --output augmentation_demo_auto_bridge.html
```

In this mode:

- the search starts from `M = 0.1 s`, `N = 0.1 s`
- it first searches for the minimum feasible `N`
- if that still fails, it searches for the minimum feasible `(M, N)` pair

If you want fixed `M/N` values with diagnostics only, keep `--recover-time` and `--past-connect-time` explicitly set.

## Run the Full Sweep

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --sweep \
  --output-prefix augmentation_demo
```

This generates one Plotly HTML per parameter combination using:

- `offset = [-3, -2, -1, +1, +2, +3] m`
- `N = [0.5, 1.0, 1.5, 2.0] s`
- `M = [0.5, 1.0, 1.5, 2.0] s`

## Reusable Python API

Example:

```python
from pathlib import Path

from trajectory_augmentation import run_demo_case, write_plotly_demo_figure

artifacts = run_demo_case(
    pattern_name="curve_decelerating",
    pattern_dir=Path("test_patterns"),
    seed=7,
    offset_m=2.0,
    yaw_offset_deg=5.0,
    recover_time_s=1.5,
    past_connect_time_s=1.0,
    max_lateral_accel_mps2=3.0,
    adaptive_bridge_search=True,
)

write_plotly_demo_figure(artifacts, Path("demo.html"))
```

`run_demo_case(...)` returns a `DemoArtifacts` object containing:

- the seed trajectory result
- the best trajectory result
- feasibility diagnostics
- the initial `M/N` values used for the run

## Diagnostics

The implementation evaluates lateral acceleration in the output window:

- `a_lat = v^2 * kappa`
- `v` comes from the exact arc-length progress used by the augmentation logic
- `kappa` is computed from the sampled trajectory

The plots show:

- GT trajectory context
- seed trajectory
- best trajectory
- best-trajectory offset point and merge points
- best-trajectory pose triangles at 0.1 s spacing
- exact-arc and chord-based speed
- curvature
- lateral acceleration and feasibility status

## Run Tests

```bash
python3 -m unittest discover -s tests -v
```

The test suite checks:

- all 9 CSV patterns can be generated and loaded
- the full GT horizon is `[-5 s, +10 s]`
- the output window is `[-3 s, +8 s]`
- recovery pose consistency with the merge point
- no speed overshoot during the bridge windows
- continuity at `t0`
- bounded curvature and curvature variation
- visible endpoint lag at `+8 s` for a representative large-offset case
- lateral-acceleration feasibility search behavior
- reusable API and Plotly HTML output
