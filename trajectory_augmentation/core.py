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
    curvature: np.ndarray


@dataclass
class DensePath:
    sigma: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


@dataclass
class PreparedDirectedSegment:
    segment: Trajectory2D
    centerline: Centerline
    distance_profile: np.ndarray
    progress_time_samples: np.ndarray
    progress_time_lookup: np.ndarray
    total_distance_m: float
    dense_ds: float
    gt_exact_speed_profile: np.ndarray
    gt_longitudinal_jerk_profile: np.ndarray


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


@dataclass
class BidirectionalAugmentationContext:
    original_full: Trajectory2D
    original_past: Trajectory2D
    original_future: Trajectory2D
    original_window: Trajectory2D
    current_index: int
    window_start_index: int
    window_end_index: int
    future_segment: PreparedDirectedSegment
    past_segment: PreparedDirectedSegment
    pattern_name: str
    past_window_size: int
    future_window_size: int
    gt_window_arc_speed: np.ndarray
    gt_window_curvature: np.ndarray
    gt_window_lateral_accel: np.ndarray
    gt_window_longitudinal_jerk: np.ndarray


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


def dense_heading_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        if len(x) == 0:
            return np.array([], dtype=float)
        if len(x) == 1:
            return np.array([0.0], dtype=float)
        yaw = np.arctan2(y[1] - y[0], x[1] - x[0])
        return np.array([yaw, yaw], dtype=float)

    yaw = np.empty_like(x, dtype=float)
    yaw[0] = np.arctan2(y[1] - y[0], x[1] - x[0])
    yaw[-1] = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])
    yaw[1:-1] = np.arctan2(y[2:] - y[:-2], x[2:] - x[:-2])
    return np.unwrap(yaw)


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
    if len(dense_s) >= 3:
        dense_curvature = np.gradient(dense_yaw, dense_s, edge_order=2)
    else:
        dense_curvature = np.zeros_like(dense_s, dtype=float)
    return Centerline(s=dense_s, x=dense_x, y=dense_y, yaw=dense_yaw, curvature=dense_curvature)


def infer_dense_ds(samples: np.ndarray, fallback: float = 0.05) -> float:
    if len(samples) < 2:
        return fallback
    return float(samples[1] - samples[0])


def build_longitudinal_gt_profiles(
    distance_profile: np.ndarray,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt_exact_speed_profile = speed_from_progress(distance_profile, time)
    gt_acceleration_profile = acceleration_from_speed(gt_exact_speed_profile, time)
    gt_longitudinal_jerk_profile = jerk_from_acceleration(gt_acceleration_profile, time)
    return gt_exact_speed_profile, gt_longitudinal_jerk_profile


def make_prepared_directed_segment(
    segment: Trajectory2D,
    centerline: Centerline,
    distance_profile: np.ndarray,
    dense_ds: float,
) -> PreparedDirectedSegment:
    progress_time_samples, progress_time_lookup = build_progress_time_lookup(distance_profile, segment.t)
    gt_exact_speed_profile, gt_longitudinal_jerk_profile = build_longitudinal_gt_profiles(
        distance_profile=distance_profile,
        time=segment.t,
    )
    return PreparedDirectedSegment(
        segment=segment,
        centerline=centerline,
        distance_profile=distance_profile,
        progress_time_samples=progress_time_samples,
        progress_time_lookup=progress_time_lookup,
        total_distance_m=float(distance_profile[-1]),
        dense_ds=dense_ds,
        gt_exact_speed_profile=gt_exact_speed_profile,
        gt_longitudinal_jerk_profile=gt_longitudinal_jerk_profile,
    )


def prepare_directed_segment(segment: Trajectory2D, dense_ds: float = 0.05) -> PreparedDirectedSegment:
    distance_profile = cumulative_distance(segment.x, segment.y)
    return make_prepared_directed_segment(
        segment=segment,
        centerline=build_centerline(segment, dense_ds=dense_ds),
        distance_profile=distance_profile,
        dense_ds=dense_ds,
    )


def prepared_segment_from_result(result: SegmentAugmentationResult) -> PreparedDirectedSegment:
    return make_prepared_directed_segment(
        segment=result.original_segment,
        centerline=result.centerline,
        distance_profile=result.distance_profile,
        dense_ds=infer_dense_ds(result.centerline.s),
    )


def build_window_gt_metrics(
    original_window: Trajectory2D,
    future_segment: PreparedDirectedSegment,
    past_segment: PreparedDirectedSegment,
    window_start_index: int,
    window_end_index: int,
    current_index: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    past_window_size = current_index - window_start_index + 1
    future_window_size = window_end_index - current_index + 1

    past_gt_speed = past_segment.gt_exact_speed_profile[::-1]
    future_gt_speed = future_segment.gt_exact_speed_profile
    gt_full_speed = np.concatenate((past_gt_speed[:-1], future_gt_speed))
    gt_window_arc_speed = gt_full_speed[window_start_index : window_end_index + 1]

    gt_window_sigma = cumulative_distance(original_window.x, original_window.y)
    gt_window_curvature = curvature_from_xy(
        original_window.x,
        original_window.y,
        gt_window_sigma,
    )
    gt_window_lateral_accel = lateral_acceleration_from_speed_and_curvature(
        gt_window_arc_speed,
        gt_window_curvature,
    )

    past_gt_jerk = past_segment.gt_longitudinal_jerk_profile[::-1]
    future_gt_jerk = future_segment.gt_longitudinal_jerk_profile
    gt_full_jerk = np.concatenate((past_gt_jerk[:-1], future_gt_jerk))
    gt_window_longitudinal_jerk = gt_full_jerk[window_start_index : window_end_index + 1]
    return (
        past_window_size,
        future_window_size,
        gt_window_arc_speed,
        gt_window_curvature,
        gt_window_lateral_accel,
        gt_window_longitudinal_jerk,
    )


def make_bidirectional_context(
    original_full: Trajectory2D,
    original_past: Trajectory2D,
    original_future: Trajectory2D,
    original_window: Trajectory2D,
    current_index: int,
    window_start_index: int,
    window_end_index: int,
    future_segment: PreparedDirectedSegment,
    past_segment: PreparedDirectedSegment,
    pattern_name: str,
) -> BidirectionalAugmentationContext:
    (
        past_window_size,
        future_window_size,
        gt_window_arc_speed,
        gt_window_curvature,
        gt_window_lateral_accel,
        gt_window_longitudinal_jerk,
    ) = build_window_gt_metrics(
        original_window=original_window,
        future_segment=future_segment,
        past_segment=past_segment,
        window_start_index=window_start_index,
        window_end_index=window_end_index,
        current_index=current_index,
    )
    return BidirectionalAugmentationContext(
        original_full=original_full,
        original_past=original_past,
        original_future=original_future,
        original_window=original_window,
        current_index=current_index,
        window_start_index=window_start_index,
        window_end_index=window_end_index,
        future_segment=future_segment,
        past_segment=past_segment,
        pattern_name=pattern_name,
        past_window_size=past_window_size,
        future_window_size=future_window_size,
        gt_window_arc_speed=gt_window_arc_speed,
        gt_window_curvature=gt_window_curvature,
        gt_window_lateral_accel=gt_window_lateral_accel,
        gt_window_longitudinal_jerk=gt_window_longitudinal_jerk,
    )


def build_bidirectional_context(
    gt: Trajectory2D,
    current_index: int,
    pattern_name: str,
    dense_ds: float = 0.05,
    output_past_horizon_s: float = OUTPUT_PAST_HORIZON_S,
    output_future_horizon_s: float = OUTPUT_FUTURE_HORIZON_S,
) -> BidirectionalAugmentationContext:
    future_segment = extract_future_segment(gt, current_index)
    past_reverse_segment = extract_reversed_past_segment(gt, current_index)
    original_past = gt.slice(0, current_index + 1)
    original_future = gt.slice(current_index, None)
    original_window, window_start_index, window_end_index = extract_time_window(
        gt,
        start_time_s=-output_past_horizon_s,
        end_time_s=output_future_horizon_s,
    )
    prepared_future_segment = prepare_directed_segment(future_segment, dense_ds=dense_ds)
    prepared_past_segment = prepare_directed_segment(past_reverse_segment, dense_ds=dense_ds)
    return make_bidirectional_context(
        original_full=gt,
        original_past=original_past,
        original_future=original_future,
        original_window=original_window,
        current_index=current_index,
        window_start_index=window_start_index,
        window_end_index=window_end_index,
        future_segment=prepared_future_segment,
        past_segment=prepared_past_segment,
        pattern_name=pattern_name,
    )


def build_bidirectional_context_from_result(result: BidirectionalAugmentationResult) -> BidirectionalAugmentationContext:
    future_segment = prepared_segment_from_result(result.future_segment)
    past_segment = prepared_segment_from_result(result.past_segment)
    return make_bidirectional_context(
        original_full=result.original_full,
        original_past=result.original_past,
        original_future=result.original_future,
        original_window=result.original_window,
        current_index=result.current_index,
        window_start_index=result.window_start_index,
        window_end_index=result.window_end_index,
        future_segment=future_segment,
        past_segment=past_segment,
        pattern_name=result.pattern_name,
    )


def sample_centerline(centerline: Centerline, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_clamped = np.clip(s_query, centerline.s[0], centerline.s[-1])
    x = np.interp(s_clamped, centerline.s, centerline.x)
    y = np.interp(s_clamped, centerline.s, centerline.y)
    yaw = np.interp(s_clamped, centerline.s, centerline.yaw)
    return x, y, yaw


def lateral_offset_profile(
    s: np.ndarray,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
) -> np.ndarray:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")

    unit_s = np.clip(s / s_merge, 0.0, 1.0)
    offset_basis = 1.0 - 10.0 * unit_s**3 + 15.0 * unit_s**4 - 6.0 * unit_s**5
    heading_basis = unit_s - 6.0 * unit_s**3 + 8.0 * unit_s**4 - 3.0 * unit_s**5
    return lateral_offset_m * offset_basis + np.tan(heading_offset_rad) * s_merge * heading_basis


def lateral_offset_profile_derivative(
    s: np.ndarray,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
) -> np.ndarray:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")

    unit_s = np.clip(s / s_merge, 0.0, 1.0)
    offset_basis_prime = (-30.0 * unit_s**2 + 60.0 * unit_s**3 - 30.0 * unit_s**4) / s_merge
    heading_basis_prime = 1.0 - 18.0 * unit_s**2 + 32.0 * unit_s**3 - 15.0 * unit_s**4
    return lateral_offset_m * offset_basis_prime + np.tan(heading_offset_rad) * heading_basis_prime


def merge_path_length(
    centerline: Centerline,
    s_merge: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
) -> float:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")

    s_segment = np.arange(0.0, s_merge, dense_ds)
    if len(s_segment) == 0 or abs(s_segment[-1] - s_merge) > 1.0e-12:
        s_segment = np.append(s_segment, s_merge)

    curvature = np.interp(s_segment, centerline.s, centerline.curvature)
    offset = lateral_offset_profile(
        s_segment,
        s_merge,
        lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
    )
    offset_prime = lateral_offset_profile_derivative(
        s_segment,
        s_merge,
        lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
    )
    integrand = np.sqrt(np.maximum((1.0 - curvature * offset) ** 2 + offset_prime * offset_prime, 1.0e-12))
    ds = np.diff(s_segment)
    return float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * ds))


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
    if len(s_segment) == 0 or abs(s_segment[-1] - s_merge) > 1.0e-12:
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
    yaw = dense_heading_from_xy(x, y)
    return DensePath(sigma=sigma, x=x, y=y, yaw=yaw)


def solve_merge_centerline_s(
    centerline: Centerline,
    distance_budget_m: float,
    lateral_offset_m: float,
    heading_offset_rad: float = 0.0,
    dense_ds: float = 0.05,
    tol_m: float = 1.0e-3,
) -> tuple[float, DensePath]:
    upper = min(distance_budget_m, centerline.s[-1])
    if abs(lateral_offset_m) < 1.0e-9 and abs(heading_offset_rad) < 1.0e-9:
        return distance_budget_m, build_merge_path(
            centerline,
            distance_budget_m,
            lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            dense_ds=dense_ds,
        )

    lower = min(max(dense_ds, abs(lateral_offset_m) * 0.25), upper * 0.5)

    def length_for(s_merge: float) -> float:
        return merge_path_length(
            centerline,
            s_merge,
            lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            dense_ds=dense_ds,
        )

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
    candidate_length = merge_path_length(
        centerline=centerline,
        s_merge=distance_budget_m,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=dense_ds,
    )
    if candidate_length <= distance_budget_m + 1.0e-3:
        candidate_path = build_merge_path(
            centerline=centerline,
            s_merge=distance_budget_m,
            lateral_offset_m=lateral_offset_m,
            heading_offset_rad=heading_offset_rad,
            dense_ds=dense_ds,
        )
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
    if len(continuation_s) == 0 or abs(continuation_s[-1] - continuation_end_s) > 1.0e-12:
        continuation_s = np.append(continuation_s, continuation_end_s)

    cont_x, cont_y, cont_yaw = sample_centerline(centerline, continuation_s)
    continuation_sigma = merge_path.sigma[-1] + (continuation_s - s_merge)

    full_sigma = np.concatenate((merge_path.sigma, continuation_sigma[1:]))
    full_x = np.concatenate((merge_path.x, cont_x[1:]))
    full_y = np.concatenate((merge_path.y, cont_y[1:]))
    full_yaw = np.concatenate((merge_path.yaw, cont_yaw[1:]))
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
    prepared = prepare_directed_segment(segment, dense_ds=dense_ds)
    return augment_directed_segment_prepared(
        prepared=prepared,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        connect_time_s=connect_time_s,
    )


def augment_directed_segment_prepared(
    prepared: PreparedDirectedSegment,
    lateral_offset_m: float,
    heading_offset_rad: float,
    connect_time_s: float,
) -> SegmentAugmentationResult:
    segment = prepared.segment
    if not 0.0 < connect_time_s <= segment.t[-1]:
        raise ValueError("connect_time_s must be within the segment horizon.")

    connect_budget_m = float(np.interp(connect_time_s, segment.t, prepared.distance_profile))
    s_merge, merge_path, connect_speed_scale = plan_recovery_path(
        centerline=prepared.centerline,
        distance_budget_m=connect_budget_m,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        dense_ds=prepared.dense_ds,
    )

    dense_full_path = build_full_augmented_path(
        centerline=prepared.centerline,
        merge_path=merge_path,
        s_merge=s_merge,
        total_distance_m=prepared.total_distance_m,
        dense_ds=prepared.dense_ds,
    )

    connect_mask = segment.t <= connect_time_s + 1.0e-9
    progress_profile = prepared.distance_profile.copy()
    if connect_budget_m > 1.0e-9:
        progress_profile[connect_mask] = prepared.distance_profile[connect_mask] * connect_speed_scale
    else:
        progress_profile[connect_mask] = 0.0

    post_indices = np.where(~connect_mask)[0]
    merge_path_length_m = float(merge_path.sigma[-1]) if len(merge_path.sigma) > 0 else 0.0
    if len(post_indices) > 0:
        merge_time_on_gt = float(np.interp(s_merge, prepared.progress_time_samples, prepared.progress_time_lookup))
        shifted_gt_time = merge_time_on_gt + (segment.t[post_indices] - connect_time_s)
        shifted_gt_time = np.clip(shifted_gt_time, segment.t[0], segment.t[-1])
        centerline_progress = np.interp(shifted_gt_time, segment.t, prepared.distance_profile)
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
        centerline=prepared.centerline,
        dense_augmented_path=dense_full_path,
        distance_profile=prepared.distance_profile,
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
    context = build_bidirectional_context(
        gt=gt,
        current_index=current_index,
        pattern_name=pattern_name,
        dense_ds=dense_ds,
        output_past_horizon_s=output_past_horizon_s,
        output_future_horizon_s=output_future_horizon_s,
    )
    return augment_trajectory_bidirectional_prepared(
        context=context,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        future_recover_time_s=future_recover_time_s,
        past_connect_time_s=past_connect_time_s,
    )


def augment_trajectory_bidirectional_prepared(
    context: BidirectionalAugmentationContext,
    lateral_offset_m: float,
    heading_offset_rad: float,
    future_recover_time_s: float,
    past_connect_time_s: float,
) -> BidirectionalAugmentationResult:
    future_result = augment_directed_segment_prepared(
        prepared=context.future_segment,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        connect_time_s=future_recover_time_s,
    )
    past_result = augment_directed_segment_prepared(
        prepared=context.past_segment,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=-heading_offset_rad,
        connect_time_s=past_connect_time_s,
    )
    return assemble_bidirectional_result(
        context=context,
        future_result=future_result,
        past_result=past_result,
        lateral_offset_m=lateral_offset_m,
        heading_offset_rad=heading_offset_rad,
        past_connect_time_s=past_connect_time_s,
        future_recover_time_s=future_recover_time_s,
    )


def assemble_bidirectional_result(
    context: BidirectionalAugmentationContext,
    future_result: SegmentAugmentationResult,
    past_result: SegmentAugmentationResult,
    lateral_offset_m: float,
    heading_offset_rad: float,
    past_connect_time_s: float,
    future_recover_time_s: float,
) -> BidirectionalAugmentationResult:
    augmented_past_x = past_result.augmented_segment.x[::-1]
    augmented_past_y = past_result.augmented_segment.y[::-1]

    augmented_full_x = context.original_full.x.copy()
    augmented_full_y = context.original_full.y.copy()
    augmented_full_x[: context.current_index] = augmented_past_x[:-1]
    augmented_full_y[: context.current_index] = augmented_past_y[:-1]
    augmented_full_x[context.current_index :] = future_result.augmented_segment.x
    augmented_full_y[context.current_index :] = future_result.augmented_segment.y

    augmented_full_yaw = compute_forward_yaw(
        augmented_full_x,
        augmented_full_y,
        fallback_yaw=context.original_full.yaw,
    )
    augmented_full = Trajectory2D(
        t=context.original_full.t.copy(),
        x=augmented_full_x,
        y=augmented_full_y,
        yaw=augmented_full_yaw,
    )

    augmented_past = augmented_full.slice(0, context.current_index + 1)
    augmented_future = augmented_full.slice(context.current_index, None)
    augmented_window = augmented_full.slice(
        context.window_start_index,
        context.window_end_index + 1,
    )

    return BidirectionalAugmentationResult(
        pattern_name=context.pattern_name,
        original_full=context.original_full,
        augmented_full=augmented_full,
        original_past=context.original_past,
        original_future=context.original_future,
        augmented_past=augmented_past,
        augmented_future=augmented_future,
        original_window=context.original_window,
        augmented_window=augmented_window,
        past_segment=past_result,
        future_segment=future_result,
        current_index=context.current_index,
        window_start_index=context.window_start_index,
        window_end_index=context.window_end_index,
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


def window_series_from_segments(
    context: BidirectionalAugmentationContext,
    past_series: np.ndarray,
    future_series: np.ndarray,
) -> np.ndarray:
    past_window = past_series[::-1][-context.past_window_size :]
    future_window = future_series[: context.future_window_size]
    return np.concatenate((past_window[:-1], future_window))


def evaluate_constraints_from_segments(
    context: BidirectionalAugmentationContext,
    past_result: SegmentAugmentationResult,
    future_result: SegmentAugmentationResult,
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
) -> ConstraintDiagnostics:
    augmented_arc_speed = window_series_from_segments(
        context,
        past_result.exact_speed_profile,
        future_result.exact_speed_profile,
    )
    augmented_window_x = window_series_from_segments(
        context,
        past_result.augmented_segment.x,
        future_result.augmented_segment.x,
    )
    augmented_window_y = window_series_from_segments(
        context,
        past_result.augmented_segment.y,
        future_result.augmented_segment.y,
    )
    augmented_window_sigma = cumulative_distance(augmented_window_x, augmented_window_y)
    augmented_curvature = curvature_from_xy(
        augmented_window_x,
        augmented_window_y,
        augmented_window_sigma,
    )
    augmented_lateral_accel = lateral_acceleration_from_speed_and_curvature(
        augmented_arc_speed,
        augmented_curvature,
    )
    max_abs_augmented_lateral_accel = float(np.max(np.abs(augmented_lateral_accel)))

    past_aug_acceleration = acceleration_from_speed(
        past_result.exact_speed_profile,
        past_result.augmented_segment.t,
    )
    future_aug_acceleration = acceleration_from_speed(
        future_result.exact_speed_profile,
        future_result.augmented_segment.t,
    )
    past_aug_jerk = jerk_from_acceleration(past_aug_acceleration, past_result.augmented_segment.t)
    future_aug_jerk = jerk_from_acceleration(future_aug_acceleration, future_result.augmented_segment.t)

    bridge_speed_gap = window_series_from_segments(
        context,
        np.abs(past_result.exact_speed_profile - context.past_segment.gt_exact_speed_profile),
        np.abs(future_result.exact_speed_profile - context.future_segment.gt_exact_speed_profile),
    )
    augmented_longitudinal_jerk = window_series_from_segments(
        context,
        past_aug_jerk,
        future_aug_jerk,
    )
    bridge_mask = (
        (context.original_window.t >= -past_result.connect_time_s - 1.0e-9)
        & (context.original_window.t <= future_result.connect_time_s + 1.0e-9)
    )
    max_speed_gap = float(np.max(bridge_speed_gap[bridge_mask])) if np.any(bridge_mask) else 0.0
    max_abs_bridge_jerk = (
        float(np.max(np.abs(augmented_longitudinal_jerk[bridge_mask])))
        if np.any(bridge_mask)
        else 0.0
    )

    lateral_accel_passes = max_abs_augmented_lateral_accel <= float(max_lateral_accel_mps2) + 1.0e-9
    speed_gap_passes = max_speed_gap <= float(max_bridge_speed_gap_mps) + 1.0e-9
    jerk_passes = max_abs_bridge_jerk <= float(max_bridge_jerk_mps3) + 1.0e-9

    return ConstraintDiagnostics(
        time=context.original_window.t.copy(),
        bridge_mask=bridge_mask,
        lateral_accel_limit_mps2=float(max_lateral_accel_mps2),
        speed_gap_limit_mps=float(max_bridge_speed_gap_mps),
        jerk_limit_mps3=float(max_bridge_jerk_mps3),
        gt_lateral_accel_mps2=context.gt_window_lateral_accel,
        augmented_lateral_accel_mps2=augmented_lateral_accel,
        max_abs_augmented_lateral_accel_mps2=max_abs_augmented_lateral_accel,
        gt_arc_speed_mps=context.gt_window_arc_speed,
        augmented_arc_speed_mps=augmented_arc_speed,
        bridge_speed_gap_mps=bridge_speed_gap,
        max_bridge_speed_gap_mps=max_speed_gap,
        gt_longitudinal_jerk_mps3=context.gt_window_longitudinal_jerk,
        augmented_longitudinal_jerk_mps3=augmented_longitudinal_jerk,
        max_abs_bridge_jerk_mps3=max_abs_bridge_jerk,
        lateral_accel_passes=lateral_accel_passes,
        speed_gap_passes=speed_gap_passes,
        jerk_passes=jerk_passes,
        passes=lateral_accel_passes and speed_gap_passes and jerk_passes,
    )


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
    context = build_bidirectional_context_from_result(result)
    return evaluate_constraints_from_segments(
        context=context,
        past_result=result.past_segment,
        future_result=result.future_segment,
        max_lateral_accel_mps2=max_lateral_accel_mps2,
        max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
        max_bridge_jerk_mps3=max_bridge_jerk_mps3,
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
    context = build_bidirectional_context_from_result(initial_result)
    initial = evaluate_constraints_from_segments(
        context=context,
        past_result=initial_result.past_segment,
        future_result=initial_result.future_segment,
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

    candidate_cache: dict[
        tuple[float, float],
        tuple[SegmentAugmentationResult, SegmentAugmentationResult, ConstraintDiagnostics],
    ] = {}
    future_segment_cache: dict[float, SegmentAugmentationResult] = {}
    past_segment_cache: dict[float, SegmentAugmentationResult] = {}

    def evaluate_candidate(
        recover_time_s: float,
        past_connect_time_s: float,
    ) -> tuple[SegmentAugmentationResult, SegmentAugmentationResult, ConstraintDiagnostics]:
        key = (round(past_connect_time_s, 6), round(recover_time_s, 6))
        cached = candidate_cache.get(key)
        if cached is not None:
            return cached

        future_key = round(recover_time_s, 6)
        future_result = future_segment_cache.get(future_key)
        if future_result is None:
            future_result = augment_directed_segment_prepared(
                prepared=context.future_segment,
                lateral_offset_m=initial_result.lateral_offset_m,
                heading_offset_rad=initial_result.heading_offset_rad,
                connect_time_s=recover_time_s,
            )
            future_segment_cache[future_key] = future_result

        past_key = round(past_connect_time_s, 6)
        past_result = past_segment_cache.get(past_key)
        if past_result is None:
            past_result = augment_directed_segment_prepared(
                prepared=context.past_segment,
                lateral_offset_m=initial_result.lateral_offset_m,
                heading_offset_rad=-initial_result.heading_offset_rad,
                connect_time_s=past_connect_time_s,
            )
            past_segment_cache[past_key] = past_result

        candidate_diag = evaluate_constraints_from_segments(
            context=context,
            future_result=future_result,
            past_result=past_result,
            max_lateral_accel_mps2=max_lateral_accel_mps2,
            max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
            max_bridge_jerk_mps3=max_bridge_jerk_mps3,
        )
        candidate_cache[key] = (future_result, past_result, candidate_diag)
        return future_result, past_result, candidate_diag

    feasible_n_cache: dict[float, tuple[BidirectionalAugmentationResult, ConstraintDiagnostics] | None] = {}

    def find_first_feasible_n(
        past_connect_time_s: float,
        future_candidates_s: list[float],
    ) -> tuple[BidirectionalAugmentationResult, ConstraintDiagnostics] | None:
        cache_key = round(past_connect_time_s, 6)
        if cache_key in feasible_n_cache:
            return feasible_n_cache[cache_key]

        for candidate_n in future_candidates_s:
            future_result, past_result, candidate_diag = evaluate_candidate(candidate_n, past_connect_time_s)
            if candidate_diag.passes:
                candidate_result = assemble_bidirectional_result(
                    context=context,
                    future_result=future_result,
                    past_result=past_result,
                    lateral_offset_m=initial_result.lateral_offset_m,
                    heading_offset_rad=initial_result.heading_offset_rad,
                    past_connect_time_s=past_connect_time_s,
                    future_recover_time_s=candidate_n,
                )
                candidate = (candidate_result, candidate_diag)
                feasible_n_cache[cache_key] = candidate
                return candidate

        feasible_n_cache[cache_key] = None
        return None

    initial_n = initial_result.future_recover_time_s
    initial_m = initial_result.past_connect_time_s

    future_candidates = generate_time_candidates(
        start_s=initial_n + search_step_s,
        end_s=max_future_recover_time_s,
        step_s=search_step_s,
    )
    future_only_candidate = find_first_feasible_n(initial_m, future_candidates)
    if future_only_candidate is not None:
        return FeasibilitySearchDiagnostics(
            initial=initial,
            adapted_result=future_only_candidate[0],
            adapted=future_only_candidate[1],
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
    if past_candidates:
        if find_first_feasible_n(past_candidates[-1], future_with_past_candidates) is not None:
            lo = 0
            hi = len(past_candidates) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if find_first_feasible_n(past_candidates[mid], future_with_past_candidates) is not None:
                    hi = mid
                else:
                    lo = mid + 1

            first_past_idx = lo
            local_start = max(0, first_past_idx - 2)
            for idx in range(local_start, first_past_idx + 1):
                candidate = find_first_feasible_n(past_candidates[idx], future_with_past_candidates)
                if candidate is not None:
                    strategy = (
                        "extend M"
                        if abs(candidate[0].future_recover_time_s - initial_n) <= 1.0e-9
                        else "extend M and N"
                    )
                    return FeasibilitySearchDiagnostics(
                        initial=initial,
                        adapted_result=candidate[0],
                        adapted=candidate[1],
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
