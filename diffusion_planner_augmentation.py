from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D", module="matplotlib.projections")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FULL_PAST_HORIZON_S = 5.0
FULL_FUTURE_HORIZON_S = 10.0
OUTPUT_PAST_HORIZON_S = 3.0
OUTPUT_FUTURE_HORIZON_S = 8.0
DEFAULT_DT = 0.1

SHAPE_NAMES = ("straight", "curve", "s_curve")
SPEED_PROFILE_NAMES = ("constant", "decelerating", "accelerating")

DEFAULT_PATTERN_NAME = "s_curve_constant"
DEFAULT_PATTERN_DIR = Path(__file__).resolve().parent / "test_patterns"


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
    connect_time_s: float
    merge_centerline_s: float
    merge_path_length_m: float
    connect_distance_budget_m: float
    connect_speed_scale: float
    lateral_offset_m: float


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
    past_connect_time_s: float
    future_recover_time_s: float


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
    else:
        raise ValueError(f"Unsupported speed profile: {speed_profile_name}")
    return np.clip(speed, 3.0, None)


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


def lateral_offset_profile(s: np.ndarray, s_merge: float, lateral_offset_m: float) -> np.ndarray:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")
    return lateral_offset_m * quintic_decay(s / s_merge)


def build_merge_path(
    centerline: Centerline,
    s_merge: float,
    lateral_offset_m: float,
    dense_ds: float = 0.05,
) -> DensePath:
    if s_merge <= 0.0:
        raise ValueError("s_merge must be positive.")

    s_segment = np.arange(0.0, s_merge, dense_ds)
    if len(s_segment) == 0 or not np.isclose(s_segment[-1], s_merge):
        s_segment = np.append(s_segment, s_merge)

    base_x, base_y, base_yaw = sample_centerline(centerline, s_segment)
    offset = lateral_offset_profile(s_segment, s_merge, lateral_offset_m)
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
    dense_ds: float = 0.05,
    tol_m: float = 1.0e-3,
) -> tuple[float, DensePath]:
    if abs(lateral_offset_m) < 1.0e-9:
        merge_path = build_merge_path(centerline, distance_budget_m, lateral_offset_m, dense_ds=dense_ds)
        return distance_budget_m, merge_path

    upper = min(distance_budget_m, centerline.s[-1])
    lower = min(max(dense_ds, abs(lateral_offset_m) * 0.25), upper * 0.5)

    def length_for(s_merge: float) -> float:
        return build_merge_path(centerline, s_merge, lateral_offset_m, dense_ds=dense_ds).sigma[-1]

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
    merge_path = build_merge_path(centerline, s_merge, lateral_offset_m, dense_ds=dense_ds)
    return s_merge, merge_path


def plan_recovery_path(
    centerline: Centerline,
    distance_budget_m: float,
    lateral_offset_m: float,
    dense_ds: float = 0.05,
) -> tuple[float, DensePath, float]:
    candidate_path = build_merge_path(
        centerline=centerline,
        s_merge=distance_budget_m,
        lateral_offset_m=lateral_offset_m,
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
        dense_ds=dense_ds,
    )
    return s_merge, merge_path, 1.0


def build_full_augmented_path(
    centerline: Centerline,
    merge_path: DensePath,
    s_merge: float,
    connect_budget_m: float,
    total_distance_m: float,
    dense_ds: float = 0.05,
) -> DensePath:
    continuation_end_s = s_merge + (total_distance_m - connect_budget_m)
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
    connect_time_s: float,
    dense_ds: float = 0.05,
) -> SegmentAugmentationResult:
    if not 0.0 < connect_time_s <= segment.t[-1]:
        raise ValueError("connect_time_s must be within the segment horizon.")

    centerline = build_centerline(segment, dense_ds=dense_ds)
    distance_profile = cumulative_distance(segment.x, segment.y)
    total_distance_m = distance_profile[-1]

    connect_budget_m = float(np.interp(connect_time_s, segment.t, distance_profile))
    s_merge, merge_path, connect_speed_scale = plan_recovery_path(
        centerline=centerline,
        distance_budget_m=connect_budget_m,
        lateral_offset_m=lateral_offset_m,
        dense_ds=dense_ds,
    )

    dense_full_path = build_full_augmented_path(
        centerline=centerline,
        merge_path=merge_path,
        s_merge=s_merge,
        connect_budget_m=connect_budget_m,
        total_distance_m=total_distance_m,
        dense_ds=dense_ds,
    )

    connect_mask = segment.t <= connect_time_s + 1.0e-9
    query_x = np.zeros_like(segment.x)
    query_y = np.zeros_like(segment.y)
    progress_profile = np.zeros_like(segment.t)

    connect_sigma = connect_speed_scale * distance_profile[connect_mask]
    query_x[connect_mask] = np.interp(connect_sigma, merge_path.sigma, merge_path.x)
    query_y[connect_mask] = np.interp(connect_sigma, merge_path.sigma, merge_path.y)
    progress_profile[connect_mask] = connect_sigma

    continue_mask = ~connect_mask
    if np.any(continue_mask):
        continue_s = s_merge + (distance_profile[continue_mask] - connect_budget_m)
        cont_x, cont_y, _ = sample_centerline(centerline, continue_s)
        query_x[continue_mask] = cont_x
        query_y[continue_mask] = cont_y
        progress_profile[continue_mask] = merge_path.sigma[-1] + (distance_profile[continue_mask] - connect_budget_m)

    directional_yaw = compute_forward_yaw(query_x, query_y, fallback_yaw=segment.yaw)
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
        connect_time_s=connect_time_s,
        merge_centerline_s=s_merge,
        merge_path_length_m=merge_path.sigma[-1],
        connect_distance_budget_m=connect_budget_m,
        connect_speed_scale=connect_speed_scale,
        lateral_offset_m=lateral_offset_m,
    )


def augment_future_trajectory(
    gt: Trajectory2D,
    current_index: int,
    lateral_offset_m: float,
    recover_time_s: float,
    dense_ds: float = 0.05,
) -> SegmentAugmentationResult:
    future_segment = extract_future_segment(gt, current_index)
    return augment_directed_segment(
        segment=future_segment,
        lateral_offset_m=lateral_offset_m,
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
        connect_time_s=future_recover_time_s,
        dense_ds=dense_ds,
    )
    past_result = augment_directed_segment(
        segment=past_reverse_segment,
        lateral_offset_m=lateral_offset_m,
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


def exact_arc_speed_in_window(result: BidirectionalAugmentationResult) -> tuple[np.ndarray, np.ndarray]:
    past_gt_speed = speed_from_progress(
        result.past_segment.distance_profile,
        result.past_segment.original_segment.t,
    )[::-1]
    past_aug_speed = speed_from_progress(
        result.past_segment.progress_profile,
        result.past_segment.augmented_segment.t,
    )[::-1]
    future_gt_speed = speed_from_progress(
        result.future_segment.distance_profile,
        result.future_segment.original_segment.t,
    )
    future_aug_speed = speed_from_progress(
        result.future_segment.progress_profile,
        result.future_segment.augmented_segment.t,
    )

    gt_full_speed = np.concatenate((past_gt_speed[:-1], future_gt_speed))
    aug_full_speed = np.concatenate((past_aug_speed[:-1], future_aug_speed))
    window_slice = slice(result.window_start_index, result.window_end_index + 1)
    return gt_full_speed[window_slice], aug_full_speed[window_slice]


def plot_pose_triangles(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    color: str,
    size: float,
    alpha: float = 0.9,
    zorder: float = 3.0,
) -> None:
    for px, py, psi in zip(x, y, yaw):
        ax.scatter(
            px,
            py,
            s=size,
            marker=(3, 0, np.degrees(psi) - 90.0),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )


def make_demo_figure(result: BidirectionalAugmentationResult, output_path: Path) -> None:
    gt_full = result.original_full
    gt = result.original_window
    augmented = result.augmented_window

    gt_arc_speed, augmented_arc_speed = exact_arc_speed_in_window(result)
    gt_chord_speed = chord_speed_from_trajectory(gt)
    augmented_chord_speed = chord_speed_from_trajectory(augmented)
    gt_curvature = curvature_from_xy(gt.x, gt.y, cumulative_distance(gt.x, gt.y))
    augmented_curvature = curvature_from_xy(augmented.x, augmented.y, cumulative_distance(augmented.x, augmented.y))

    current_index_in_window = result.current_index - result.window_start_index
    current_x = augmented.x[current_index_in_window]
    current_y = augmented.y[current_index_in_window]

    future_merge_x, future_merge_y, _ = sample_centerline(
        result.future_segment.centerline,
        np.array([result.future_segment.merge_centerline_s]),
    )
    past_merge_x, past_merge_y, _ = sample_centerline(
        result.past_segment.centerline,
        np.array([result.past_segment.merge_centerline_s]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(gt_full.x, gt_full.y, color="0.88", lw=1.5, label="GT full context")
    axes[0].plot(gt.x, gt.y, color="0.55", lw=2.0, label="GT window")
    axes[0].plot(augmented.x[: current_index_in_window + 1], augmented.y[: current_index_in_window + 1], color="#2ca02c", lw=2.5, label="Augmented past")
    axes[0].plot(augmented.x[current_index_in_window:], augmented.y[current_index_in_window:], color="#ff7f0e", lw=2.5, label="Augmented future")
    plot_pose_triangles(axes[0], gt.x, gt.y, gt.yaw, color="0.45", size=34, alpha=0.65, zorder=2.5)
    plot_pose_triangles(
        axes[0],
        augmented.x,
        augmented.y,
        augmented.yaw,
        color="#ff7f0e",
        size=38,
        alpha=0.85,
        zorder=3.2,
    )
    axes[0].scatter(current_x, current_y, color="#d62728", s=60, label="Offset at t0")
    axes[0].scatter(past_merge_x[0], past_merge_y[0], color="#1f77b4", s=45, label="Past merge")
    axes[0].scatter(future_merge_x[0], future_merge_y[0], color="#9467bd", s=45, label="Future merge")
    axes[0].set_title("Trajectory")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(gt.t, gt_arc_speed, color="0.45", lw=2.0, label="GT exact arc")
    axes[1].plot(gt.t, gt_chord_speed, color="0.55", lw=1.7, ls="--", label="GT chord")
    axes[1].plot(augmented.t, augmented_arc_speed, color="#ff7f0e", lw=2.0, label="Aug exact arc")
    axes[1].plot(augmented.t, augmented_chord_speed, color="#d95f02", lw=1.7, ls="--", label="Aug chord")
    axes[1].axvline(-result.past_connect_time_s, color="#1f77b4", ls="--", lw=1.2)
    axes[1].axvline(0.0, color="0.35", ls=":", lw=1.2)
    axes[1].axvline(result.future_recover_time_s, color="#9467bd", ls="--", lw=1.2)
    axes[1].set_xlim(gt.t[0], gt.t[-1])
    axes[1].set_title("Speed (Arc + Chord)")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("speed [m/s]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(gt.t, gt_curvature, color="0.55", lw=2.0, label="GT window")
    axes[2].plot(augmented.t, augmented_curvature, color="#ff7f0e", lw=2.0, label="Augmented window")
    axes[2].axvline(-result.past_connect_time_s, color="#1f77b4", ls="--", lw=1.2)
    axes[2].axvline(0.0, color="0.35", ls=":", lw=1.2)
    axes[2].axvline(result.future_recover_time_s, color="#9467bd", ls="--", lw=1.2)
    axes[2].set_xlim(gt.t[0], gt.t[-1])
    axes[2].set_title("Curvature")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("curvature [1/m]")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    end_delta = np.linalg.norm(
        np.array([augmented.x[-1] - gt.x[-1], augmented.y[-1] - gt.y[-1]])
    )
    fig.suptitle(
        (
            f"pattern={result.pattern_name}, "
            f"offset={result.lateral_offset_m:+.2f} m, "
            f"M={result.past_connect_time_s:.1f} s, "
            f"N={result.future_recover_time_s:.1f} s, "
            f"end_delta@8s={end_delta:.3f} m"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_demo(
    pattern_name: str,
    pattern_dir: Path,
    seed: int,
    output_path: Path,
    offset_m: float | None,
    recover_time_s: float,
    past_connect_time_s: float,
) -> BidirectionalAugmentationResult:
    gt, current_index = load_test_pattern(pattern_name=pattern_name, pattern_dir=pattern_dir)
    rng = np.random.default_rng(seed)
    lateral_offset_m = offset_m if offset_m is not None else sample_random_lateral_offset(rng)
    result = augment_trajectory_bidirectional(
        gt=gt,
        current_index=current_index,
        lateral_offset_m=lateral_offset_m,
        future_recover_time_s=recover_time_s,
        past_connect_time_s=past_connect_time_s,
        output_past_horizon_s=OUTPUT_PAST_HORIZON_S,
        output_future_horizon_s=OUTPUT_FUTURE_HORIZON_S,
        pattern_name=pattern_name,
    )
    make_demo_figure(result, output_path=output_path)
    return result


def run_sweep(
    pattern_name: str,
    pattern_dir: Path,
    output_prefix: Path,
    offsets: list[float],
    recover_times: list[float],
    past_connect_times: list[float],
) -> list[Path]:
    gt, current_index = load_test_pattern(pattern_name=pattern_name, pattern_dir=pattern_dir)
    output_paths: list[Path] = []
    pattern_suffix = f"_{pattern_name}"

    for past_connect_time_s in past_connect_times:
        for offset in offsets:
            for recover_time in recover_times:
                result = augment_trajectory_bidirectional(
                    gt=gt,
                    current_index=current_index,
                    lateral_offset_m=offset,
                    future_recover_time_s=recover_time,
                    past_connect_time_s=past_connect_time_s,
                    output_past_horizon_s=OUTPUT_PAST_HORIZON_S,
                    output_future_horizon_s=OUTPUT_FUTURE_HORIZON_S,
                    pattern_name=pattern_name,
                )
                suffix = (
                    pattern_suffix
                    + format_offset_suffix(offset)
                    + format_time_suffix("N", recover_time)
                    + format_time_suffix("M", past_connect_time_s)
                )
                output_path = output_prefix.with_name(f"{output_prefix.name}{suffix}.png")
                make_demo_figure(result, output_path=output_path)
                output_paths.append(output_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion Planner data augmentation demo.")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN_NAME)
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--recover-time", type=float, default=1.5)
    parser.add_argument("--past-connect-time", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--write-pattern-csvs", action="store_true", help="Generate the CSV test patterns and exit.")
    parser.add_argument("--list-patterns", action="store_true", help="List available pattern names and exit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("augmentation_demo.png"),
        help="Path to save the single-case visualization figure.",
    )
    parser.add_argument("--sweep", action="store_true", help="Render the requested offset/recovery sweep.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("augmentation_sweep"),
        help="Prefix used when saving sweep figures.",
    )
    args = parser.parse_args()

    if args.list_patterns:
        for pattern_name in list_pattern_names():
            print(pattern_name)
        return

    if args.write_pattern_csvs:
        output_paths = write_test_pattern_csvs(args.pattern_dir)
        print(f"Wrote {len(output_paths)} pattern CSV files to: {args.pattern_dir}")
        return

    if args.sweep:
        offsets = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        recover_times = [0.5, 1.0, 1.5, 2.0]
        past_connect_times = [0.5, 1.0, 1.5, 2.0]
        output_paths = run_sweep(
            pattern_name=args.pattern,
            pattern_dir=args.pattern_dir,
            output_prefix=args.output_prefix,
            offsets=offsets,
            recover_times=recover_times,
            past_connect_times=past_connect_times,
        )
        print(f"Saved {len(output_paths)} sweep images for pattern: {args.pattern}")
        if output_paths:
            print(f"First image: {output_paths[0]}")
            print(f"Last image: {output_paths[-1]}")
        return

    result = run_demo(
        pattern_name=args.pattern,
        pattern_dir=args.pattern_dir,
        seed=args.seed,
        output_path=args.output,
        offset_m=args.offset,
        recover_time_s=args.recover_time,
        past_connect_time_s=args.past_connect_time,
    )

    gt_arc_speed, augmented_arc_speed = exact_arc_speed_in_window(result)
    gt_chord_speed = chord_speed_from_trajectory(result.original_window)
    augmented_chord_speed = chord_speed_from_trajectory(result.augmented_window)
    arc_speed_abs_diff = float(np.max(np.abs(augmented_arc_speed - gt_arc_speed)))
    chord_speed_abs_diff = float(np.max(np.abs(augmented_chord_speed - gt_chord_speed)))
    curvature = curvature_from_xy(
        result.augmented_window.x,
        result.augmented_window.y,
        cumulative_distance(result.augmented_window.x, result.augmented_window.y),
    )
    end_delta = float(
        np.linalg.norm(
            np.array(
                [
                    result.augmented_window.x[-1] - result.original_window.x[-1],
                    result.augmented_window.y[-1] - result.original_window.y[-1],
                ]
            )
        )
    )

    print(f"Saved visualization to: {args.output}")
    print(f"Pattern: {result.pattern_name}")
    print(
        f"Full GT horizon: past {FULL_PAST_HORIZON_S:.1f}s / future {FULL_FUTURE_HORIZON_S:.1f}s, "
        f"augmented output window: past {OUTPUT_PAST_HORIZON_S:.1f}s / future {OUTPUT_FUTURE_HORIZON_S:.1f}s"
    )
    print(f"Lateral offset: {result.lateral_offset_m:+.3f} m")
    print(f"Past bridge M: {result.past_connect_time_s:.2f} s")
    print(f"Future bridge N: {result.future_recover_time_s:.2f} s")
    print(f"Past speed scale: {result.past_segment.connect_speed_scale:.3f}")
    print(f"Future speed scale: {result.future_segment.connect_speed_scale:.3f}")
    print(f"Max exact-arc speed abs diff vs GT in output window: {arc_speed_abs_diff:.3f} m/s")
    print(f"Max chord-speed abs diff vs GT in output window: {chord_speed_abs_diff:.3f} m/s")
    print(f"End delta at +8s: {end_delta:.3f} m")
    print(f"Max |curvature| in output window: {float(np.max(np.abs(curvature))):.4f} 1/m")


if __name__ == "__main__":
    main()
