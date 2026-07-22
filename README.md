# Trajectory Data Augmentation Test

This repository contains a reference implementation of trajectory augmentation for a diffusion-based planner.

A detailed explanation of the algorithm and a full parameter reference (in Japanese) is available in [docs/algorithm_and_parameters.md](docs/algorithm_and_parameters.md).

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

## Choosing the Bridge-Time Search Mode (Required)

Every run (single example or sweep) must state explicitly whether the automatic
bridge-time search is enabled:

- `--adaptive-bridge-search`: search for a feasible `M/N` pair when the initial
  values violate the constraints
- `--no-adaptive-bridge-search`: keep the requested `M/N` values and only report
  constraint diagnostics

Omitting both is an error. Only `--list-patterns` and `--write-pattern-csvs`
work without this flag.

## Run a Single Interactive Example

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --offset -3.0 \
  --recover-time 1.5 \
  --past-connect-time 1.0 \
  --max-lateral-accel 3.0 \
  --no-adaptive-bridge-search \
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
  --adaptive-bridge-search \
  --output augmentation_demo_auto_bridge.html
```

In this mode:

- the search starts from `M = 0.1 s`, `N = 0.1 s` (when `--recover-time` /
  `--past-connect-time` are omitted)
- it first searches for the minimum feasible `N`
- if that still fails, it searches for the minimum feasible `(M, N)` pair

If you want fixed `M/N` values with diagnostics only, keep `--recover-time` and
`--past-connect-time` explicitly set and use `--no-adaptive-bridge-search`.

## Decorrelating History and Recovery

The past bridge encodes the offset and its timing in the history, so a model
could learn to predict the recovery by extrapolating the ego history instead of
reasoning about the map. Two options break that shortcut:

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --offset 2.0 \
  --adaptive-bridge-search \
  --randomize-bridge-times \
  --bridge-time-extra-range 1.5 \
  --past-bump \
  --output augmentation_demo_decorrelated.html
```

Both options require `--adaptive-bridge-search`; with
`--no-adaptive-bridge-search` they are skipped.

- `--randomize-bridge-times`: after finding the minimal feasible `M/N`, sample
  random bridge times from `[minimum, minimum + extra range]` so the same past
  can correspond to different recoveries. Every candidate is re-checked against
  the constraints.
- `--past-bump`: inject one random lateral bump into the past history. Its
  amplitude (`0.05-0.20 m`, random sign), duration (`1.0-2.0 s`), and start
  time are sampled from fixed internal ranges (`BUMP_AMPLITUDE_RANGE_M`,
  `BUMP_DURATION_RANGE_S` in `trajectory_augmentation/core.py`). The bump uses
  a `sin^3` window, which joins the surrounding path with C2 continuity (zero
  value, slope, and second derivative at both edges), so `t0` continuity is
  preserved. Bump geometry perturbs the path; yaw and the exact-arc speed are
  recomputed from the perturbed geometry so all channels stay consistent. The
  bump is only applied to the past (conditioning) side, never to the future
  target, and sampled bumps that violate the lateral acceleration, bridge speed
  gap, or jerk constraints are rejected and re-sampled.

## Run the Full Sweep

```bash
python3 diffusion_planner_augmentation.py \
  --pattern curve_decelerating \
  --sweep \
  --adaptive-bridge-search \
  --output-prefix augmentation_demo
```

This generates one Plotly HTML per parameter combination using:

- `offset = [-3, -2, -1, +1, +2, +3] m`
- `yaw offset = [-15, -10, -5, 0, +5, +10, +15] deg`
- `N = [0.5, 1.0, 1.5, 2.0] s`
- `M = [0.5, 1.0, 1.5, 2.0] s`
- `past bump = [off, on]`

Each output filename ends with a `_bump0` / `_bump1` suffix. Passing
`--randomize-bridge-times` applies the random bridge-time extension on top of
each grid `M/N` value. Note that the full grid is large (over a thousand HTML
files), so consider narrowing the pattern or editing the arrays in
`trajectory_augmentation/cli.py` for quick experiments.

## Run the Bump Sweep

```bash
python3 diffusion_planner_augmentation.py \
  --pattern straight_constant \
  --offset 1.0 \
  --bump-sweep \
  --adaptive-bridge-search \
  --output-prefix bump_sweep
```

This sweeps deterministic past-history bump parameters with a fixed offset.
With `--adaptive-bridge-search` (recommended) the baseline bridge times are
first adapted so the no-bump baseline passes the constraints, and every bump is
applied on top of that feasible baseline; with `--no-adaptive-bridge-search`
the requested bridge times (`--recover-time` / `--past-connect-time`,
defaulting to `N = 1.5 s` / `M = 1.0 s`) are used as-is, which may already
violate the constraints before any bump is added. The swept bump grid is:

- `amplitude = [-0.2, -0.1, +0.1, +0.2] m`
- `duration = [0.8, 1.2, 1.6, 2.0] s`
- `start time = [0.3, 0.8, 1.3, 1.8] s before t0`

Combinations whose bump would extend beyond the `3 s` past window are skipped.
In each figure the seed trajectory is the no-bump baseline and the best
trajectory is the bumped variant, so the effect of each parameter combination
is directly visible, together with its constraint PASS/FAIL status.

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
- past-history bumps perturb the history while preserving the future target and `t0`
- randomized bridge times stay feasible and vary across seeds
- reusable API and Plotly HTML output
