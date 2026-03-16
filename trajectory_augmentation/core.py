from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FULL_PAST_HORIZON_S = 5.0
FULL_FUTURE_HORIZON_S = 10.0
OUTPUT_PAST_HORIZON_S = 3.0
OUTPUT_FUTURE_HORIZON_S = 8.0
DEFAULT_DT = 0.1
MIN_BRIDGE_TIME_S = 0.1

SHAPE_NAMES = ("straight", "curve", "s_curve")
SPEED_PROFILE_NAMES = ("constant", "decelerating", "accelerating", "stopping", "stop8s")

DEFAULT_PATTERN_NAME = "s_curve_constant"
DEFAULT_PATTERN_DIR = Path(__file__).resolve().parent.parent / "test_patterns"


@dataclass
class Trajectory2D:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray

    def slice(self, start: int, end: int | None = None) -> "Trajectory2D":
        return Trajectory2D(
            t=self.t[start:end].copy(),
            x=self.x[start:end].copy(),
            y=self.y[start:end].copy(),
            yaw=self.yaw[start:end].copy(),
        )


@dataclass
class Centerline:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


@dataclass
class DensePath:
    sigma: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


@dataclass
class SegmentAugmentationResult:
    original_segment: Trajectory2D
    augmented_segment: Trajectory2D
    centerline: Centerline
    dense_augmented_path: DensePath
    distance_profile: np.ndarray
    progress_profile: np.ndarray
    exact_speed_profile: np.ndarray
    connect_time_s: float
    merge_centerline_s: float
    merge_path_length_m: float
    connect_distance_budget_m: float
    connect_speed_scale: float
    lateral_offset_m: float
    heading_offset_rad: float


@dataclass
class BidirectionalAugmentationResult:
    pattern_name: str
    original_full: Trajectory2D
    augmented_full: Trajectory2D
    original_past: Trajectory2D
    original_future: Trajectory2D
    augmented_past: Trajectory2D
    augmented_future: Trajectory2D
    original_window: Trajectory2D
    augmented_window: Trajectory2D
    past_segment: SegmentAugmentationResult
    future_segment: SegmentAugmentationResult
    current_index: int
    window_start_index: int
    window_end_index: int
    lateral_offset_m: float
    heading_offset_rad: float
    past_connect_time_s: float
    future_recover_time_s: float


@dataclass
class ConstraintDiagnostics:
    time: np.ndarray
    bridge_mask: np.ndarray
    lateral_accel_limit_mps2: float
    speed_gap_limit_mps: float
    jerk_limit_mps3: float
    gt_lateral_accel_mps2: np.ndarray
    augmented_lateral_accel_mps2: np.ndarray
    max_abs_augmented_lateral_accel_mps2: float
    gt_arc_speed_mps: np.ndarray
    augmented_arc_speed_mps: np.ndarray
    bridge_speed_gap_mps: np.ndarray
    max_bridge_speed_gap_mps: float
    gt_longitudinal_jerk_mps3: np.ndarray
    augmented_longitudinal_jerk_mps3: np.ndarray
    max_abs_bridge_jerk_mps3: float
    lateral_accel_passes: bool
    speed_gap_passes: bool
    jerk_passes: bool
    passes: bool

    @property
    def limit_mps2(self) -> float:
        return self.lateral_accel_limit_mps2


LateralAccelDiagnostics = ConstraintDiagnostics


@dataclass
class FeasibilitySearchDiagnostics:
    initial: ConstraintDiagnostics
    adapted_result: BidirectionalAugmentationResult | None
    adapted: ConstraintDiagnostics | None
    adaptation_strategy: str | None


@dataclass
class DemoArtifacts:
    seed_result: BidirectionalAugmentationResult
    selected_result: BidirectionalAugmentationResult
    feasibility: FeasibilitySearchDiagnostics
    initial_recover_time_s: float
    initial_past_connect_time_s: float
    adaptive_bridge_search: bool


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=float)
    deltas = np.column_stack((np.diff(x), np.diff(y)))
    segment_lengths = np.linalg.norm(deltas, axis=1)
    s = np.zeros_like(x, dtype=float)
    s[1:] = np.cumsum(segment_lengths)
    return s


def strictly_increasing_param(param: np.ndarray) -> np.ndarray:
    fixed = np.asarray(param, dtype=float).copy()
    for idx in range(1, len(fixed)):
        if fixed[idx] <= fixed[idx - 1]:
            fixed[idx] = fixed[idx - 1] + 1.0e-6
    return fixed


def heading_from_xy(x: np.ndarray, y: np.ndarray, param: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        if len(x) == 0:
            return np.array([], dtype=float)
        if len(x) == 1:
            return np.array([0.0], dtype=float)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        yaw = np.arctan2(dy, dx)
        return np.array([yaw, yaw], dtype=float)
    param = strictly_increasing_param(param)
    dx = np.gradient(x, param, edge_order=2)
    dy = np.gradient(y, param, edge_order=2)
    return np.unwrap(np.arctan2(dy, dx))


def curvature_from_xy(x: np.ndarray, y: np.ndarray, param: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.zeros_like(x, dtype=float)
    param = strictly_increasing_param(param)
    dx = np.gradient(x, param, edge_order=2)
    dy = np.gradient(y, param, edge_order=2)
    ddx = np.gradient(dx, param, edge_order=2)
    ddy = np.gradient(dy, param, edge_order=2)
    denom = np.maximum(dx * dx + dy * dy, 1.0e-9) ** 1.5
    return (dx * ddy - dy * ddx) / denom


def rotate_points(x: np.ndarray, y: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return c * x - s * y, s * x + c * y


def compute_forward_yaw(x: np.ndarray, y: np.ndarray, fallback_yaw: np.ndarray | None = None) -> np.ndarray:
    sigma = cumulative_distance(x, y)
    if len(x) >= 3 and sigma[-1] > 1.0e-6:
        return wrap_angle(heading_from_xy(x, y, sigma))
    if fallback_yaw is not None:
        return wrap_angle(np.asarray(fallback_yaw, dtype=float))
    return np.zeros_like(x, dtype=float)


def speed_from_trajectory(traj: Trajectory2D) -> np.ndarray:
    distance = cumulative_distance(traj.x, traj.y)
    if len(traj.t) < 3:
        return np.zeros_like(traj.t, dtype=float)
    return np.gradient(distance, strictly_increasing_param(traj.t), edge_order=2)


def speed_from_progress(progress: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(time) < 3:
        return np.zeros_like(time, dtype=float)
    return np.gradient(progress, strictly_increasing_param(time), edge_order=2)


def build_progress_speed_lookup(progress: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    progress = np.asarray(progress, dtype=float)
    speed = np.clip(np.asarray(speed, dtype=float), 0.0, None)
    unique_progress, inverse = np.unique(progress, return_inverse=True)
    lookup_speed = np.zeros_like(unique_progress, dtype=float)
    for idx in range(len(unique_progress)):
        lookup_speed[idx] = float(np.min(speed[inverse == idx]))
    return unique_progress, lookup_speed


def build_progress_time_lookup(progress: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    progress = np.asarray(progress, dtype=float)
    time = np.asarray(time, dtype=float)
    unique_progress, first_indices = np.unique(progress, return_index=True)
    return unique_progress, time[first_indices]


def sample_speed_by_progress(progress_samples: np.ndarray, speed_samples: np.ndarray, progress_query: np.ndarray) -> np.ndarray:
    clamped_progress = np.clip(progress_query, progress_samples[0], progress_samples[-1])
    return np.interp(clamped_progress, progress_samples, speed_samples)


def integrate_progress_with_speed_lookup(
    time: np.ndarray,
    start_progress: float,
    progress_samples: np.ndarray,
    speed_samples: np.ndarray,
    max_progress: float,
) -> np.ndarray:
    integrated = np.zeros_like(time, dtype=float)
    if len(time) == 0:
        return integrated

    integrated[0] = float(start_progress)
    for idx in range(1, len(time)):
        dt = float(time[idx] - time[idx - 1])
        prev_progress = integrated[idx - 1]
        prev_speed = float(sample_speed_by_progress(progress_samples, speed_samples, np.array([prev_progress]))[0])
        next_progress = prev_progress + prev_speed * dt
        next_progress = min(next_progress, max_progress)
        for _ in range(2):
            next_speed = float(sample_speed_by_progress(progress_samples, speed_samples, np.array([next_progress]))[0])
            next_progress = prev_progress + 0.5 * (prev_speed + next_speed) * dt
            next_progress = min(next_progress, max_progress)
        integrated[idx] = next_progress
    return integrated


def acceleration_from_speed(speed: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(time) < 3:
        return np.zeros_like(time, dtype=float)
    return np.gradient(speed, strictly_increasing_param(time), edge_order=2)


def jerk_from_acceleration(acceleration: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(time) < 3:
        return np.zeros_like(time, dtype=float)
    return np.gradient(acceleration, strictly_increasing_param(time), edge_order=2)


def chord_speed_from_trajectory(traj: Trajectory2D) -> np.ndarray:
    if len(traj.t) < 2:
        return np.zeros_like(traj.t, dtype=float)

    time = strictly_increasing_param(traj.t)
    speed = np.zeros_like(traj.t, dtype=float)

    if len(traj.t) == 2:
        value = np.hypot(traj.x[1] - traj.x[0], traj.y[1] - traj.y[0]) / (time[1] - time[0])
        speed[:] = value
        return speed

    speed[0] = np.hypot(traj.x[1] - traj.x[0], traj.y[1] - traj.y[0]) / (time[1] - time[0])
    speed[-1] = np.hypot(traj.x[-1] - traj.x[-2], traj.y[-1] - traj.y[-2]) / (time[-1] - time[-2])

    span_x = traj.x[2:] - traj.x[:-2]
    span_y = traj.y[2:] - traj.y[:-2]
    span_t = time[2:] - time[:-2]
    speed[1:-1] = np.hypot(span_x, span_y) / span_t
    return speed


def smoothstep(unit_value: np.ndarray) -> np.ndarray:
    u = np.clip(unit_value, 0.0, 1.0)
    return 3.0 * u**2 - 2.0 * u**3


def split_pattern_name(pattern_name: str) -> tuple[str, str]:
    if "_" not in pattern_name:
        raise ValueError(f"Invalid pattern name: {pattern_name}")
    shape_name, speed_profile_name = pattern_name.rsplit("_", 1)
    if shape_name not in SHAPE_NAMES:
        raise ValueError(f"Unsupported shape: {shape_name}")
    if speed_profile_name not in SPEED_PROFILE_NAMES:
        raise ValueError(f"Unsupported speed profile: {speed_profile_name}")
    return shape_name, speed_profile_name


def list_pattern_names() -> list[str]:
    return [f"{shape}_{speed_profile}" for shape in SHAPE_NAMES for speed_profile in SPEED_PROFILE_NAMES]


def build_speed_profile(times: np.ndarray, speed_profile_name: str) -> np.ndarray:
    u = (times - times[0]) / (times[-1] - times[0])
    blend = smoothstep(u)
    if speed_profile_name == "constant":
        speed = np.full_like(times, 8.0, dtype=float)
    elif speed_profile_name == "decelerating":
        speed = 10.0 - 4.0 * blend
    elif speed_profile_name == "accelerating":
        speed = 6.0 + 4.0 * blend
    elif speed_profile_name == "stopping":
        speed = np.full_like(times, 8.0, dtype=float)
        stop_time_s = 5.0
        future_blend = smoothstep(np.clip(times / stop_time_s, 0.0, 1.0))
        future_mask = times > 0.0
        speed[future_mask] = 8.0 * (1.0 - future_blend[future_mask])
        speed[times >= stop_time_s] = 0.0
    elif speed_profile_name == "stop8s":
        speed = np.full_like(times, 8.0, dtype=float)
        decel_start_s = 6.0
        stop_time_s = OUTPUT_FUTURE_HORIZON_S
        future_blend = smoothstep(
            np.clip((times - decel_start_s) / (stop_time_s - decel_start_s), 0.0, 1.0)
        )
        future_mask = times > decel_start_s
        speed[future_mask] = 8.0 * (1.0 - future_blend[future_mask])
        speed[times >= stop_time_s] = 0.0
    else:
        raise ValueError(f"Unsupported speed profile: {speed_profile_name}")
    return np.clip(speed, 0.0, None)


def build_curvature_profile(s_rel: np.ndarray, shape_name: str) -> np.ndarray:
    if shape_name == "straight":
        curvature = np.zeros_like(s_rel)
    elif shape_name == "curve":
        curvature = 0.010 + 0.0015 * np.sin(0.030 * s_rel + 0.2)
    elif shape_name == "s_curve":
        curvature = (
            0.014 * np.exp(-((s_rel + 12.0) / 18.0) ** 2)
            - 0.014 * np.exp(-((s_rel - 22.0) / 18.0) ** 2)
        )
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")
    return curvature


def integrate_trajectory(
    times: np.ndarray,
    speed: np.ndarray,
    curvature: np.ndarray,
    current_index: int,
) -> Trajectory2D:
    ds = np.zeros_like(times)
    ds[1:] = 0.5 * (speed[1:] + speed[:-1]) * np.diff(times)

    yaw = np.zeros_like(times)
    x = np.zeros_like(times)
    y = np.zeros_like(times)
    for idx in range(1, len(times)):
        yaw[idx] = yaw[idx - 1] + 0.5 * (curvature[idx - 1] + curvature[idx]) * ds[idx]
        yaw_mid = 0.5 * (yaw[idx - 1] + yaw[idx])
        x[idx] = x[idx - 1] + ds[idx] * np.cos(yaw_mid)
        y[idx] = y[idx - 1] + ds[idx] * np.sin(yaw_mid)

    x -= x[current_index]
    y -= y[current_index]
    x, y = rotate_points(x, y, -yaw[current_index])
    yaw = wrap_angle(yaw - yaw[current_index])
    return Trajectory2D(t=times.copy(), x=x, y=y, yaw=yaw)


def generate_synthetic_gt(
    shape_name: str = "s_curve",
    speed_profile_name: str = "constant",
    past_horizon_s: float = FULL_PAST_HORIZON_S,
    future_horizon_s: float = FULL_FUTURE_HORIZON_S,
    dt: float = DEFAULT_DT,
) -> tuple[Trajectory2D, int]:
    times = np.arange(-past_horizon_s, future_horizon_s + 1.0e-9, dt)
    current_index = int(round(past_horizon_s / dt))

    speed = build_speed_profile(times, speed_profile_name)
    ds = np.zeros_like(times)
    ds[1:] = 0.5 * (speed[1:] + speed[:-1]) * dt
    s = np.cumsum(ds)
    s_rel = s - s[current_index]
    curvature = build_curvature_profile(s_rel, shape_name)

    return integrate_trajectory(times, speed, curvature, current_index), current_index


def write_trajectory_csv(traj: Trajectory2D, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    speed = speed_from_trajectory(traj)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "yaw", "speed_mps"])
        for row in zip(traj.t, traj.x, traj.y, traj.yaw, speed):
            writer.writerow([f"{value:.8f}" for value in row])


def load_trajectory_csv(path: Path) -> tuple[Trajectory2D, int]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    traj = Trajectory2D(
        t=np.asarray(data["t"], dtype=float),
        x=np.asarray(data["x"], dtype=float),
        y=np.asarray(data["y"], dtype=float),
        yaw=np.asarray(data["yaw"], dtype=float),
    )
    current_matches = np.where(np.isclose(traj.t, 0.0, atol=1.0e-9))[0]
    if len(current_matches) != 1:
        raise ValueError(f"Expected exactly one t=0.0 sample in {path}")
    return traj, int(current_matches[0])


def write_test_pattern_csvs(
    output_dir: Path = DEFAULT_PATTERN_DIR,
    dt: float = DEFAULT_DT,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for pattern_name in list_pattern_names():
        shape_name, speed_profile_name = split_pattern_name(pattern_name)
        traj, _ = generate_synthetic_gt(
            shape_name=shape_name,
            speed_profile_name=speed_profile_name,
            past_horizon_s=FULL_PAST_HORIZON_S,
            future_horizon_s=FULL_FUTURE_HORIZON_S,
            dt=dt,
        )
        path = output_dir / f"{pattern_name}.csv"
        write_trajectory_csv(traj, path)
        written_paths.append(path)
    return written_paths


def ensure_test_pattern_csvs(pattern_dir: Path = DEFAULT_PATTERN_DIR) -> list[Path]:
    existing_paths = [pattern_dir / f"{pattern_name}.csv" for pattern_name in list_pattern_names()]
    if all(path.exists() for path in existing_paths):
        return existing_paths
    return write_test_pattern_csvs(pattern_dir)


def load_test_pattern(
    pattern_name: str = DEFAULT_PATTERN_NAME,
    pattern_dir: Path = DEFAULT_PATTERN_DIR,
) -> tuple[Trajectory2D, int]:
    split_pattern_name(pattern_name)
    ensure_test_pattern_csvs(pattern_dir)
    return load_trajectory_csv(pattern_dir / f"{pattern_name}.csv")


def build_centerline(segment: Trajectory2D, dense_ds: float = 0.05) -> Centerline:
    s_samples = cumulative_distance(segment.x, segment.y)
    dense_s = np.arange(0.0, s_samples[-1] + dense_ds * 0.5, dense_ds)
    yaw_samples = np.unwrap(segment.yaw)
    dense_x = np.interp(dense_s, s_samples, segment.x)
    dense_y = np.interp(dense_s, s_samples, segment.y)
    dense_yaw = np.interp(dense_s, s_samples, yaw_samples)
    return Centerline(s=dense_s, x=dense_x, y=dense_y, yaw=dense_yaw)


def sample_centerline(centerline: Centerline, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_clamped = np.clip(s_query, centerline.s[0], centerline.s[-1])
    x = np.interp(s_clamped, centerline.s, centerline.x)
    y = np.interp(s_clamped, centerline.s, centerline.y)
    yaw = np.interp(s_clamped, centerline.s, centerline.yaw)
    return x, y, yaw


def quintic_decay(unit_s: np.ndarray) -> np.ndarray:
    u = np.clip(unit_s, 0.0, 1.0)
    return 1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5


def solve_lateral_profile_coeffs(
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float,
) -> np.ndarray:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")

    # Use a quintic profile in arc length so the path can satisfy:
    # l(0), l'(0), l''(0), l(L), l'(L), l''(L).
    #
    # l'(0) controls the initial heading error. For small offsets on a smooth
    # centerline, heading error is approximately atan(l'(0)).
    L = float(s_merge)
    a0 = float(lateral_offset_m)
    a1 = float(np.tan(heading_offset_rad))
    a2 = 0.0

    system = np.array(
        [
            [L**3, L**4, L**5],
            [3.0 * L**2, 4.0 * L**3, 5.0 * L**4],
            [6.0 * L, 12.0 * L**2, 20.0 * L**3],
        ],
        dtype=float,
    )
    rhs = np.array(
        [
            -(a0 + a1 * L + a2 * L**2),
            -(a1 + 2.0 * a2 * L),
            -(2.0 * a2),
        ],
        dtype=float,
    )
    a3, a4, a5 = np.linalg.solve(system, rhs)
    return np.array([a0, a1, a2, a3, a4, a5], dtype=float)


def lateral_offset_profile(
    s: np.ndarray,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
) -> np.ndarray:
    coeffs = solve_lateral_profile_coeffs(
        s_merge=s_merge,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
    )
    powers = np.stack([s**idx for idx in range(6)], axis=-1)
    return powers @ coeffs


def build_merge_path(
    centerline: Centerline,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
) -> DensePath:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")

    s_segment = np.arange(0.0, s_merge, dense_ds)
    if len(s_segment) == 0 or not np.isclose(s_segment[-1], s_merge):
        s_segment = np.append(s_segment, s_merge)

    base_x, base_y, base_yaw = sample_centerline(centerline, s_segment)
    offset = lateral_offset_profile(
        s_segment,
        s_merge,
        lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
    )
    normal_x = -np.sin(base_yaw)
    normal_y = np.cos(base_yaw)

    x = base_x + offset * normal_x
    y = base_y + offset * normal_y
    sigma = cumulative_distance(x, y)
    yaw = heading_from_xy(x, y, sigma)
    return DensePath(sigma=sigma, x=x, y=y, yaw=yaw)


def solve_merge_centerline_s(
    centerline: Centerline,
    distance_budget_m: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
    tol_m: float = 1.0e-3,
) -> tuple[float, DensePath]:
    if abs(lateral_offset_m) < 1.0e-9 and abs(heading_offset_rad) < 1.0e-9:
        merge_path = build_merge_path(
            centerline,
            distance_budget_m,
            lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            dense_ds=dense_ds,
        )
        return distance_budget_m, merge_path

    upper = min(distance_budget_m, centerline.s[-1])
    lower = min(max(dense_ds, abs(lateral_offset_m) * 0.25), upper * 0.5)

    def length_for(s_merge: float) -> float:
        return build_merge_path(
            centerline,
            s_merge,
            lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            dense_ds=dense_ds,
        ).sigma[-1]

    while lower > dense_ds * 1.0e-3 and length_for(lower) >= distance_budget_m:
        lower *= 0.5

    if length_for(upper) < distance_budget_m:
        raise RuntimeError("Unable to find a feasible merge point within the distance budget.")

    for _ in range(50):
        mid = 0.5 * (lower + upper)
        if length_for(mid) < distance_budget_m:
            lower = mid
        else:
            upper = mid
        if upper - lower < tol_m:
            break

    s_merge = 0.5 * (lower + upper)
    merge_path = build_merge_path(
        centerline,
        s_merge,
        lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )
    return s_merge, merge_path


def plan_recovery_path(
    centerline: Centerline,
    distance_budget_m: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
) -> tuple[float, DensePath, float]:
    candidate_path = build_merge_path(
        centerline=centerline,
        s_merge=distance_budget_m,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )
    candidate_length = candidate_path.sigma[-1]
    if candidate_length <= distance_budget_m + 1.0e-3:
        speed_scale = candidate_length / max(distance_budget_m, 1.0e-9)
        return distance_budget_m, candidate_path, speed_scale

    s_merge, merge_path = solve_merge_centerline_s(
        centerline=centerline,
        distance_budget_m=distance_budget_m,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )
    return s_merge, merge_path, 1.0


def build_full_augmented_path(
    centerline: Centerline,
    merge_path: DensePath,
    s_merge: float,
    total_distance_m: float,
    dense_ds: float = 0.05,
) -> DensePath:
    continuation_end_s = total_distance_m
    continuation_s = np.arange(s_merge, continuation_end_s, dense_ds)
    if len(continuation_s) == 0 or not np.isclose(continuation_s[-1], continuation_end_s):
        continuation_s = np.append(continuation_s, continuation_end_s)

    cont_x, cont_y, _ = sample_centerline(centerline, continuation_s)
    continuation_sigma = merge_path.sigma[-1] + (continuation_s - s_merge)

    full_sigma = np.concatenate((merge_path.sigma, continuation_sigma[1:]))
    full_x = np.concatenate((merge_path.x, cont_x[1:]))
    full_y = np.concatenate((merge_path.y, cont_y[1:]))
    full_yaw = heading_from_xy(full_x, full_y, full_sigma)
    return DensePath(sigma=full_sigma, x=full_x, y=full_y, yaw=full_yaw)


def sample_dense_path(path: DensePath, sigma_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(path.sigma) == 0:
        raise ValueError("DensePath must contain at least one sample.")
    sigma = strictly_increasing_param(path.sigma)
    sigma_clamped = np.clip(sigma_query, sigma[0], sigma[-1])
    x = np.interp(sigma_clamped, sigma, path.x)
    y = np.interp(sigma_clamped, sigma, path.y)
    yaw = wrap_angle(np.interp(sigma_clamped, sigma, np.unwrap(path.yaw)))
    return x, y, yaw


def extract_future_segment(gt: Trajectory2D, current_index: int) -> Trajectory2D:
    future = gt.slice(current_index, None)
    return Trajectory2D(
        t=future.t - future.t[0],
        x=future.x.copy(),
        y=future.y.copy(),
        yaw=future.yaw.copy(),
    )


def extract_reversed_past_segment(gt: Trajectory2D, current_index: int) -> Trajectory2D:
    past = gt.slice(0, current_index + 1)
    return Trajectory2D(
        t=-past.t[::-1],
        x=past.x[::-1].copy(),
        y=past.y[::-1].copy(),
        yaw=past.yaw[::-1].copy(),
    )


def augment_directed_segment(
    segment: Trajectory2D,
    lateral_offset_m: float,
    heading_offset_rad: float,
    connect_time_s: float,
    dense_ds: float = 0.05,
) -> SegmentAugmentationResult:
    if not 0.0 < connect_time_s <= segment.t[-1]:
        raise ValueError("connect_time_s must be within the segment horizon.")

    centerline = build_centerline(segment, dense_ds=dense_ds)
    distance_profile = cumulative_distance(segment.x, segment.y)
    progress_time_samples, progress_time_lookup = build_progress_time_lookup(distance_profile, segment.t)
    connect_budget_m = float(np.interp(connect_time_s, segment.t, distance_profile))
    total_distance_m = float(distance_profile[-1])
    s_merge, merge_path, connect_speed_scale = plan_recovery_path(
        centerline=centerline,
        distance_budget_m=connect_budget_m,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )

    dense_full_path = build_full_augmented_path(
        centerline=centerline,
        merge_path=merge_path,
        s_merge=s_merge,
        total_distance_m=total_distance_m,
        dense_ds=dense_ds,
    )

    connect_mask = segment.t <= connect_time_s + 1.0e-9
    progress_profile = distance_profile.copy()
    if connect_budget_m > 1.0e-9:
        progress_profile[connect_mask] = distance_profile[connect_mask] * connect_speed_scale
    else:
        progress_profile[connect_mask] = 0.0

    post_indices = np.where(~connect_mask)[0]
    merge_path_length_m = float(merge_path.sigma[-1]) if len(merge_path.sigma) > 0 else 0.0
    if len(post_indices) > 0:
        merge_time_on_gt = float(np.interp(s_merge, progress_time_samples, progress_time_lookup))
        shifted_gt_time = merge_time_on_gt + (segment.t[post_indices] - connect_time_s)
        shifted_gt_time = np.clip(shifted_gt_time, segment.t[0], segment.t[-1])
        centerline_progress = np.interp(shifted_gt_time, segment.t, distance_profile)
        progress_profile[post_indices] = merge_path_length_m + (centerline_progress - s_merge)

    progress_profile = np.clip(progress_profile, 0.0, dense_full_path.sigma[-1])
    query_x, query_y, directional_yaw = sample_dense_path(dense_full_path, progress_profile)
    exact_speed_profile = speed_from_progress(progress_profile, segment.t)
    augmented_segment = Trajectory2D(
        t=segment.t.copy(),
        x=query_x,
        y=query_y,
        yaw=directional_yaw,
    )

    return SegmentAugmentationResult(
        original_segment=segment,
        augmented_segment=augmented_segment,
        centerline=centerline,
        dense_augmented_path=dense_full_path,
        distance_profile=distance_profile,
        progress_profile=progress_profile,
        exact_speed_profile=exact_speed_profile,
        connect_time_s=connect_time_s,
        merge_centerline_s=s_merge,
        merge_path_length_m=merge_path_length_m,
        connect_distance_budget_m=connect_budget_m,
        connect_speed_scale=connect_speed_scale,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
    )


def augment_future_trajectory(
    gt: Trajectory2D,
    current_index: int,
    lateral_offset_m: float,
    heading_offset_rad: float,
    recover_time_s: float,
    dense_ds: float = 0.05,
) -> SegmentAugmentationResult:
    future_segment = extract_future_segment(gt, current_index)
    return augment_directed_segment(
        segment=future_segment,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        connect_time_s=recover_time_s,
        dense_ds=dense_ds,
    )


def extract_time_window(
    traj: Trajectory2D,
    start_time_s: float,
    end_time_s: float,
) -> tuple[Trajectory2D, int, int]:
    indices = np.where((traj.t >= start_time_s - 1.0e-9) & (traj.t <= end_time_s + 1.0e-9))[0]
    if len(indices) == 0:
        raise ValueError("Requested time window is empty.")
    start_index = int(indices[0])
    end_index = int(indices[-1])
    return traj.slice(start_index, end_index + 1), start_index, end_index


def augment_trajectory_bidirectional(
    gt: Trajectory2D,
    current_index: int,
    lateral_offset_m: float,
    heading_offset_rad: float,
    future_recover_time_s: float,
    past_connect_time_s: float,
    dense_ds: float = 0.05,
    output_past_horizon_s: float = OUTPUT_PAST_HORIZON_S,
    output_future_horizon_s: float = OUTPUT_FUTURE_HORIZON_S,
    pattern_name: str = "generated",
) -> BidirectionalAugmentationResult:
    future_segment = extract_future_segment(gt, current_index)
    past_reverse_segment = extract_reversed_past_segment(gt, current_index)

    future_result = augment_directed_segment(
        segment=future_segment,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        connect_time_s=future_recover_time_s,
        dense_ds=dense_ds,
    )
    past_result = augment_directed_segment(
        segment=past_reverse_segment,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=-heading_offset_rad,
        connect_time_s=past_connect_time_s,
        dense_ds=dense_ds,
    )

    original_past = gt.slice(0, current_index + 1)
    original_future = gt.slice(current_index, None)

    augmented_past_x = past_result.augmented_segment.x[::-1]
    augmented_past_y = past_result.augmented_segment.y[::-1]

    augmented_full_x = gt.x.copy()
    augmented_full_y = gt.y.copy()
    augmented_full_x[:current_index] = augmented_past_x[:-1]
    augmented_full_y[:current_index] = augmented_past_y[:-1]
    augmented_full_x[current_index:] = future_result.augmented_segment.x
    augmented_full_y[current_index:] = future_result.augmented_segment.y

    augmented_full_yaw = compute_forward_yaw(augmented_full_x, augmented_full_y, fallback_yaw=gt.yaw)
    augmented_full = Trajectory2D(
        t=gt.t.copy(),
        x=augmented_full_x,
        y=augmented_full_y,
        yaw=augmented_full_yaw,
    )

    augmented_past = augmented_full.slice(0, current_index + 1)
    augmented_future = augmented_full.slice(current_index, None)

    original_window, window_start_index, window_end_index = extract_time_window(
        gt,
        start_time_s=-output_past_horizon_s,
        end_time_s=output_future_horizon_s,
    )
    augmented_window, _, _ = extract_time_window(
        augmented_full,
        start_time_s=-output_past_horizon_s,
        end_time_s=output_future_horizon_s,
    )

    return BidirectionalAugmentationResult(
        pattern_name=pattern_name,
        original_full=gt.slice(0, None),
        augmented_full=augmented_full,
        original_past=original_past,
        original_future=original_future,
        augmented_past=augmented_past,
        augmented_future=augmented_future,
        original_window=original_window,
        augmented_window=augmented_window,
        past_segment=past_result,
        future_segment=future_result,
        current_index=current_index,
        window_start_index=window_start_index,
        window_end_index=window_end_index,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        past_connect_time_s=past_connect_time_s,
        future_recover_time_s=future_recover_time_s,
    )


def sample_random_lateral_offset(
    rng: np.random.Generator,
    min_abs_m: float = 0.4,
    max_abs_m: float = 1.2,
) -> float:
    magnitude = rng.uniform(min_abs_m, max_abs_m)
    sign = rng.choice(np.array([-1.0, 1.0]))
    return float(sign * magnitude)


def format_time_suffix(prefix: str, value_s: float) -> str:
    return f"_{prefix}{value_s:.1f}".replace(".", "p") + "s"


def format_offset_suffix(value_m: float) -> str:
    sign = "p" if value_m >= 0.0 else "m"
    magnitude = f"{abs(value_m):.1f}".replace(".", "p")
    return f"_offset{sign}{magnitude}m"


def format_yaw_suffix(value_deg: float) -> str:
    sign = "p" if value_deg >= 0.0 else "m"
    magnitude = f"{abs(value_deg):.0f}"
    return f"_yaw{sign}{magnitude}deg"


def exact_arc_speed_in_window(result: BidirectionalAugmentationResult) -> tuple[np.ndarray, np.ndarray]:
    past_gt_speed = speed_from_progress(
        result.past_segment.distance_profile,
        result.past_segment.original_segment.t,
    )[::-1]
    past_aug_speed = result.past_segment.exact_speed_profile[::-1]
    future_gt_speed = speed_from_progress(
        result.future_segment.distance_profile,
        result.future_segment.original_segment.t,
    )
    future_aug_speed = result.future_segment.exact_speed_profile

    gt_full_speed = np.concatenate((past_gt_speed[:-1], future_gt_speed))
    aug_full_speed = np.concatenate((past_aug_speed[:-1], future_aug_speed))
    window_slice = slice(result.window_start_index, result.window_end_index + 1)
    return gt_full_speed[window_slice], aug_full_speed[window_slice]


def exact_arc_kinematics_in_window(
    result: BidirectionalAugmentationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = result.original_window.t.copy()
    gt_speed, augmented_speed = exact_arc_speed_in_window(result)
    gt_acceleration = acceleration_from_speed(gt_speed, time)
    augmented_acceleration = acceleration_from_speed(augmented_speed, time)
    gt_jerk = jerk_from_acceleration(gt_acceleration, time)
    augmented_jerk = jerk_from_acceleration(augmented_acceleration, time)
    return time, gt_speed, augmented_speed, gt_acceleration, augmented_acceleration, gt_jerk, augmented_jerk


def segment_kinematics(
    segment_result: SegmentAugmentationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_speed = speed_from_progress(segment_result.distance_profile, segment_result.original_segment.t)
    augmented_speed = segment_result.exact_speed_profile
    gt_acceleration = acceleration_from_speed(gt_speed, segment_result.original_segment.t)
    augmented_acceleration = acceleration_from_speed(augmented_speed, segment_result.augmented_segment.t)
    gt_jerk = jerk_from_acceleration(gt_acceleration, segment_result.original_segment.t)
    augmented_jerk = jerk_from_acceleration(augmented_acceleration, segment_result.augmented_segment.t)
    return gt_speed, augmented_speed, gt_jerk, augmented_jerk


def bridge_constraint_series(
    result: BidirectionalAugmentationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    past_gt_speed, past_aug_speed, past_gt_jerk, past_aug_jerk = segment_kinematics(result.past_segment)
    future_gt_speed, future_aug_speed, future_gt_jerk, future_aug_jerk = segment_kinematics(result.future_segment)

    past_time = -result.past_segment.original_segment.t[::-1]
    future_time = result.future_segment.original_segment.t

    full_time = np.concatenate((past_time[:-1], future_time))
    full_bridge_mask = (
        (full_time >= -result.past_connect_time_s - 1.0e-9)
        & (full_time <= result.future_recover_time_s + 1.0e-9)
    )
    full_speed_gap = np.concatenate((np.abs(past_aug_speed - past_gt_speed)[::-1][:-1], np.abs(future_aug_speed - future_gt_speed)))
    full_gt_jerk = np.concatenate((past_gt_jerk[::-1][:-1], future_gt_jerk))
    full_augmented_jerk = np.concatenate((past_aug_jerk[::-1][:-1], future_aug_jerk))
    window_slice = slice(result.window_start_index, result.window_end_index + 1)
    return (
        full_time[window_slice],
        full_bridge_mask[window_slice],
        full_speed_gap[window_slice],
        full_gt_jerk[window_slice],
        full_augmented_jerk[window_slice],
    )


def lateral_acceleration_from_speed_and_curvature(
    speed_mps: np.ndarray,
    curvature_inv_m: np.ndarray,
) -> np.ndarray:
    return speed_mps * speed_mps * curvature_inv_m


def evaluate_constraints_in_window(
    result: BidirectionalAugmentationResult,
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
) -> ConstraintDiagnostics:
    (
        _time,
        gt_arc_speed,
        augmented_arc_speed,
        _gt_acceleration,
        _augmented_acceleration,
        _window_gt_jerk,
        _window_augmented_jerk,
    ) = exact_arc_kinematics_in_window(result)
    gt_curvature = curvature_from_xy(
        result.original_window.x,
        result.original_window.y,
        cumulative_distance(result.original_window.x, result.original_window.y),
    )
    augmented_curvature = curvature_from_xy(
        result.augmented_window.x,
        result.augmented_window.y,
        cumulative_distance(result.augmented_window.x, result.augmented_window.y),
    )

    gt_lateral_accel = lateral_acceleration_from_speed_and_curvature(gt_arc_speed, gt_curvature)
    augmented_lateral_accel = lateral_acceleration_from_speed_and_curvature(
        augmented_arc_speed,
        augmented_curvature,
    )
    max_abs_augmented_lateral_accel = float(np.max(np.abs(augmented_lateral_accel)))
    bridge_time, bridge_mask, bridge_speed_gap, gt_jerk, augmented_jerk = bridge_constraint_series(result)
    max_speed_gap = float(np.max(bridge_speed_gap[bridge_mask])) if np.any(bridge_mask) else 0.0
    max_abs_bridge_jerk = float(np.max(np.abs(augmented_jerk[bridge_mask]))) if np.any(bridge_mask) else 0.0

    lateral_accel_passes = max_abs_augmented_lateral_accel <= float(max_lateral_accel_mps2) + 1.0e-9
    speed_gap_passes = max_speed_gap <= float(max_bridge_speed_gap_mps) + 1.0e-9
    jerk_passes = max_abs_bridge_jerk <= float(max_bridge_jerk_mps3) + 1.0e-9

    return ConstraintDiagnostics(
        time=bridge_time,
        bridge_mask=bridge_mask,
        lateral_accel_limit_mps2=float(max_lateral_accel_mps2),
        speed_gap_limit_mps=float(max_bridge_speed_gap_mps),
        jerk_limit_mps3=float(max_bridge_jerk_mps3),
        gt_lateral_accel_mps2=gt_lateral_accel,
        augmented_lateral_accel_mps2=augmented_lateral_accel,
        max_abs_augmented_lateral_accel_mps2=max_abs_augmented_lateral_accel,
        gt_arc_speed_mps=gt_arc_speed,
        augmented_arc_speed_mps=augmented_arc_speed,
        bridge_speed_gap_mps=bridge_speed_gap,
        max_bridge_speed_gap_mps=max_speed_gap,
        gt_longitudinal_jerk_mps3=gt_jerk,
        augmented_longitudinal_jerk_mps3=augmented_jerk,
        max_abs_bridge_jerk_mps3=max_abs_bridge_jerk,
        lateral_accel_passes=lateral_accel_passes,
        speed_gap_passes=speed_gap_passes,
        jerk_passes=jerk_passes,
        passes=lateral_accel_passes and speed_gap_passes and jerk_passes,
    )


def evaluate_lateral_accel_in_window(
    result: BidirectionalAugmentationResult,
    limit_mps2: float,
) -> ConstraintDiagnostics:
    return evaluate_constraints_in_window(
        result=result,
        max_lateral_accel_mps2=limit_mps2,
        max_bridge_speed_gap_mps=np.inf,
        max_bridge_jerk_mps3=np.inf,
    )


def generate_time_candidates(
    start_s: float,
    end_s: float,
    step_s: float = DEFAULT_DT,
) -> list[float]:
    start_tick = int(np.ceil((start_s - 1.0e-9) / step_s))
    end_tick = int(np.floor((end_s + 1.0e-9) / step_s))
    return [tick * step_s for tick in range(start_tick, end_tick + 1)]


def search_feasible_result(
    initial_result: BidirectionalAugmentationResult,
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
    search_step_s: float = DEFAULT_DT,
    max_future_recover_time_s: float = FULL_FUTURE_HORIZON_S,
    max_past_connect_time_s: float = FULL_PAST_HORIZON_S,
) -> FeasibilitySearchDiagnostics:
    initial = evaluate_constraints_in_window(
        initial_result,
        max_lateral_accel_mps2=max_lateral_accel_mps2,
        max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
        max_bridge_jerk_mps3=max_bridge_jerk_mps3,
    )
    if initial.passes:
        return FeasibilitySearchDiagnostics(
            initial=initial,
            adapted_result=None,
            adapted=None,
            adaptation_strategy=None,
        )

    def build_candidate(recover_time_s: float, past_connect_time_s: float) -> BidirectionalAugmentationResult:
        return augment_trajectory_bidirectional(
            gt=initial_result.original_full,
            current_index=initial_result.current_index,
            lateral_offset_m=initial_result.lateral_offset_m,
            heading_offset_rad=initial_result.heading_offset_rad,
            future_recover_time_s=recover_time_s,
            past_connect_time_s=past_connect_time_s,
            output_past_horizon_s=OUTPUT_PAST_HORIZON_S,
            output_future_horizon_s=OUTPUT_FUTURE_HORIZON_S,
            pattern_name=initial_result.pattern_name,
        )

    initial_n = initial_result.future_recover_time_s
    initial_m = initial_result.past_connect_time_s

    future_candidates = generate_time_candidates(
        start_s=initial_n + search_step_s,
        end_s=max_future_recover_time_s,
        step_s=search_step_s,
    )
    for candidate_n in future_candidates:
        candidate_result = build_candidate(candidate_n, initial_m)
        candidate_diag = evaluate_constraints_in_window(
            candidate_result,
            max_lateral_accel_mps2=max_lateral_accel_mps2,
            max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
            max_bridge_jerk_mps3=max_bridge_jerk_mps3,
        )
        if candidate_diag.passes:
            return FeasibilitySearchDiagnostics(
                initial=initial,
                adapted_result=candidate_result,
                adapted=candidate_diag,
                adaptation_strategy="extend N",
            )

    past_candidates = generate_time_candidates(
        start_s=initial_m + search_step_s,
        end_s=max_past_connect_time_s,
        step_s=search_step_s,
    )
    future_with_past_candidates = generate_time_candidates(
        start_s=initial_n,
        end_s=max_future_recover_time_s,
        step_s=search_step_s,
    )
    for candidate_m in past_candidates:
        for candidate_n in future_with_past_candidates:
            candidate_result = build_candidate(candidate_n, candidate_m)
            candidate_diag = evaluate_constraints_in_window(
                candidate_result,
                max_lateral_accel_mps2=max_lateral_accel_mps2,
                max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
                max_bridge_jerk_mps3=max_bridge_jerk_mps3,
            )
            if candidate_diag.passes:
                strategy = "extend M" if np.isclose(candidate_n, initial_n) else "extend M and N"
                return FeasibilitySearchDiagnostics(
                    initial=initial,
                    adapted_result=candidate_result,
                    adapted=candidate_diag,
                    adaptation_strategy=strategy,
                )

    return FeasibilitySearchDiagnostics(
        initial=initial,
        adapted_result=None,
        adapted=None,
        adaptation_strategy=None,
    )


def search_lateral_accel_feasible_result(
    initial_result: BidirectionalAugmentationResult,
    max_lateral_accel_mps2: float,
    search_step_s: float = DEFAULT_DT,
    max_future_recover_time_s: float = FULL_FUTURE_HORIZON_S,
    max_past_connect_time_s: float = FULL_PAST_HORIZON_S,
) -> FeasibilitySearchDiagnostics:
    return search_feasible_result(
        initial_result=initial_result,
        max_lateral_accel_mps2=max_lateral_accel_mps2,
        max_bridge_speed_gap_mps=np.inf,
        max_bridge_jerk_mps3=np.inf,
        search_step_s=search_step_s,
        max_future_recover_time_s=max_future_recover_time_s,
        max_past_connect_time_s=max_past_connect_time_s,
    )


def run_demo_case(
    pattern_name: str,
    pattern_dir: Path,
    seed: int,
    offset_m: float | None,
    yaw_offset_deg: float,
    recover_time_s: float | None,
    past_connect_time_s: float | None,
    max_lateral_accel_mps2: float,
    adaptive_bridge_search: bool = True,
    max_bridge_speed_gap_mps: float = 0.5,
    max_bridge_jerk_mps3: float = 5.0,
) -> DemoArtifacts:
    gt, current_index = load_test_pattern(pattern_name=pattern_name, pattern_dir=pattern_dir)
    rng = np.random.default_rng(seed)
    lateral_offset_m = offset_m if offset_m is not None else sample_random_lateral_offset(rng)
    initial_recover_time_s = (
        recover_time_s if recover_time_s is not None else MIN_BRIDGE_TIME_S
    )
    initial_past_connect_time_s = (
        past_connect_time_s if past_connect_time_s is not None else MIN_BRIDGE_TIME_S
    )
    initial_result = augment_trajectory_bidirectional(
        gt=gt,
        current_index=current_index,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=np.deg2rad(yaw_offset_deg),
        future_recover_time_s=initial_recover_time_s,
        past_connect_time_s=initial_past_connect_time_s,
        output_past_horizon_s=OUTPUT_PAST_HORIZON_S,
        output_future_horizon_s=OUTPUT_FUTURE_HORIZON_S,
        pattern_name=pattern_name,
    )
    feasibility = search_feasible_result(
        initial_result=initial_result,
        max_lateral_accel_mps2=max_lateral_accel_mps2,
        max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
        max_bridge_jerk_mps3=max_bridge_jerk_mps3,
    )
    result = initial_result
    if adaptive_bridge_search and feasibility.adapted_result is not None:
        result = feasibility.adapted_result
    if not adaptive_bridge_search:
        feasibility = FeasibilitySearchDiagnostics(
            initial=feasibility.initial,
            adapted_result=None,
            adapted=None,
            adaptation_strategy="search disabled",
        )
    return DemoArtifacts(
        seed_result=initial_result,
        selected_result=result,
        feasibility=feasibility,
        initial_recover_time_s=initial_recover_time_s,
        initial_past_connect_time_s=initial_past_connect_time_s,
        adaptive_bridge_search=adaptive_bridge_search,
    )
