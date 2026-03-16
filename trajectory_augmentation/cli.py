from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import (
    DEFAULT_PATTERN_DIR,
    DEFAULT_PATTERN_NAME,
    FULL_FUTURE_HORIZON_S,
    FULL_PAST_HORIZON_S,
    MIN_BRIDGE_TIME_S,
    OUTPUT_FUTURE_HORIZON_S,
    OUTPUT_PAST_HORIZON_S,
    DemoArtifacts,
    chord_speed_from_trajectory,
    cumulative_distance,
    curvature_from_xy,
    exact_arc_speed_in_window,
    format_offset_suffix,
    format_time_suffix,
    format_yaw_suffix,
    list_pattern_names,
    run_demo_case,
    write_test_pattern_csvs,
)
from .visualization import build_demo_plot_payload, write_plotly_demo_figure


def _selected_diag(artifacts: DemoArtifacts):
    if artifacts.feasibility.adapted_result is artifacts.selected_result and artifacts.feasibility.adapted is not None:
        return artifacts.feasibility.adapted
    return artifacts.feasibility.initial


def summarize_demo_case(
    artifacts: DemoArtifacts,
    output_path: Path | None = None,
) -> list[str]:
    result = artifacts.selected_result
    gt_arc_speed, selected_arc_speed = exact_arc_speed_in_window(result)
    gt_chord_speed = chord_speed_from_trajectory(result.original_window)
    selected_chord_speed = chord_speed_from_trajectory(result.augmented_window)
    curvature = curvature_from_xy(
        result.augmented_window.x,
        result.augmented_window.y,
        cumulative_distance(result.augmented_window.x, result.augmented_window.y),
    )
    payload = build_demo_plot_payload(artifacts)
    selected_diag = _selected_diag(artifacts)

    lines: list[str] = []
    if output_path is not None:
        lines.append(f"Saved visualization to: {output_path}")
    lines.extend(
        [
            f"Pattern: {result.pattern_name}",
            (
                f"Full GT horizon: past {FULL_PAST_HORIZON_S:.1f}s / future {FULL_FUTURE_HORIZON_S:.1f}s, "
                f"augmented output window: past {OUTPUT_PAST_HORIZON_S:.1f}s / future {OUTPUT_FUTURE_HORIZON_S:.1f}s"
            ),
            f"Lateral offset: {result.lateral_offset_m:+.3f} m",
            f"Heading offset: {np.degrees(result.heading_offset_rad):+.1f} deg",
            f"Initial past bridge M: {artifacts.initial_past_connect_time_s:.2f} s",
            f"Initial future bridge N: {artifacts.initial_recover_time_s:.2f} s",
        ]
    )

    if not np.isclose(result.past_connect_time_s, artifacts.initial_past_connect_time_s) or not np.isclose(
        result.future_recover_time_s,
        artifacts.initial_recover_time_s,
    ):
        lines.append(f"Selected past bridge M: {result.past_connect_time_s:.2f} s")
        lines.append(f"Selected future bridge N: {result.future_recover_time_s:.2f} s")
    else:
        lines.append(f"Past bridge M: {result.past_connect_time_s:.2f} s")
        lines.append(f"Future bridge N: {result.future_recover_time_s:.2f} s")

    lines.extend(
        [
            f"Past speed scale: {result.past_segment.connect_speed_scale:.3f}",
            f"Future speed scale: {result.future_segment.connect_speed_scale:.3f}",
            f"Max exact-arc speed abs diff vs GT in output window: {float(np.max(np.abs(selected_arc_speed - gt_arc_speed))):.3f} m/s",
            f"Max chord-speed abs diff vs GT in output window: {float(np.max(np.abs(selected_chord_speed - gt_chord_speed))):.3f} m/s",
            (
                f"Max |lateral acceleration| in output window: "
                f"{selected_diag.max_abs_augmented_lateral_accel_mps2:.3f} m/s^2 "
                f"(limit {selected_diag.lateral_accel_limit_mps2:.3f}, "
                f"{'PASS' if selected_diag.lateral_accel_passes else 'FAIL'})"
            ),
            (
                f"Max bridge speed gap |v_aug - v_gt|: "
                f"{selected_diag.max_bridge_speed_gap_mps:.3f} m/s "
                f"(limit {selected_diag.speed_gap_limit_mps:.3f}, "
                f"{'PASS' if selected_diag.speed_gap_passes else 'FAIL'})"
            ),
            (
                f"Max |bridge longitudinal jerk|: "
                f"{selected_diag.max_abs_bridge_jerk_mps3:.3f} m/s^3 "
                f"(limit {selected_diag.jerk_limit_mps3:.3f}, "
                f"{'PASS' if selected_diag.jerk_passes else 'FAIL'})"
            ),
        ]
    )

    if artifacts.feasibility.adapted_result is not None and artifacts.feasibility.adapted is not None:
        lines.append(
            f"Lowest passing candidate via {artifacts.feasibility.adaptation_strategy}: "
            f"M={artifacts.feasibility.adapted_result.past_connect_time_s:.2f} s, "
            f"N={artifacts.feasibility.adapted_result.future_recover_time_s:.2f} s, "
            f"max |a_lat|={artifacts.feasibility.adapted.max_abs_augmented_lateral_accel_mps2:.3f} m/s^2, "
            f"max dv={artifacts.feasibility.adapted.max_bridge_speed_gap_mps:.3f} m/s, "
            f"max |jerk|={artifacts.feasibility.adapted.max_abs_bridge_jerk_mps3:.3f} m/s^3"
        )
    elif artifacts.feasibility.adaptation_strategy == "search disabled":
        lines.append("Adaptive bridge search disabled.")
    elif not artifacts.feasibility.initial.passes:
        lines.append("No feasible M/N candidate found within the available full GT horizon.")

    lines.append(f"End delta at +8s: {payload.end_delta:.3f} m")
    lines.append(f"Max |curvature| in output window: {float(np.max(np.abs(curvature))):.4f} 1/m")
    lines.append(f"Max |selected longitudinal jerk| in output window: {float(np.max(np.abs(selected_diag.augmented_longitudinal_jerk_mps3))):.3f} m/s^3")
    return lines


def _render_case_outputs(
    artifacts: DemoArtifacts,
    output_path: Path | None,
) -> None:
    if output_path is not None:
        write_plotly_demo_figure(artifacts, output_path)


def run_sweep(
    pattern_name: str,
    pattern_dir: Path,
    output_prefix: Path,
    seed: int,
    offsets: list[float],
    yaw_offsets_deg: list[float],
    recover_times: list[float],
    past_connect_times: list[float],
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
    adaptive_bridge_search: bool,
) -> tuple[list[Path], list[Path]]:
    output_paths: list[Path] = []
    pattern_suffix = f"_{pattern_name}"

    for past_connect_time_s in past_connect_times:
        for offset in offsets:
            for yaw_offset_deg in yaw_offsets_deg:
                for recover_time_s in recover_times:
                    artifacts = run_demo_case(
                        pattern_name=pattern_name,
                        pattern_dir=pattern_dir,
                        seed=seed,
                        offset_m=offset,
                        yaw_offset_deg=yaw_offset_deg,
                        recover_time_s=recover_time_s,
                        past_connect_time_s=past_connect_time_s,
                        max_lateral_accel_mps2=max_lateral_accel_mps2,
                        max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
                        max_bridge_jerk_mps3=max_bridge_jerk_mps3,
                        adaptive_bridge_search=adaptive_bridge_search,
                    )
                    suffix = (
                        pattern_suffix
                        + format_offset_suffix(offset)
                        + format_yaw_suffix(yaw_offset_deg)
                        + format_time_suffix("N", recover_time_s)
                        + format_time_suffix("M", past_connect_time_s)
                    )
                    output_path = output_prefix.with_name(f"{output_prefix.name}{suffix}.html")
                    _render_case_outputs(
                        artifacts,
                        output_path=output_path,
                    )
                    output_paths.append(output_path)
    return output_paths, []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diffusion Planner data augmentation demo.")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN_NAME)
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--recover-time",
        type=float,
        default=None,
        help=(
            "Requested future bridge time N [s]. If omitted, start from "
            f"{MIN_BRIDGE_TIME_S:.1f}s and auto-search the minimum feasible N."
        ),
    )
    parser.add_argument(
        "--past-connect-time",
        type=float,
        default=None,
        help=(
            "Requested past bridge time M [s]. If omitted, start from "
            f"{MIN_BRIDGE_TIME_S:.1f}s and auto-search the minimum feasible M,N pair."
        ),
    )
    parser.add_argument("--offset", type=float, default=None)
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--max-lateral-accel",
        type=float,
        default=3.0,
        help="Absolute lateral acceleration limit [m/s^2] used for diagnostics and adaptive bridge search.",
    )
    parser.add_argument(
        "--max-bridge-speed-gap",
        type=float,
        default=0.5,
        help="Maximum allowed bridge speed gap |v_aug - v_gt| [m/s] during adaptive bridge search.",
    )
    parser.add_argument(
        "--max-bridge-jerk",
        type=float,
        default=5.0,
        help="Maximum allowed absolute longitudinal jerk [m/s^3] during adaptive bridge search.",
    )
    parser.add_argument(
        "--disable-adaptive-bridge-search",
        action="store_true",
        help="Only diagnose lateral-acceleration limit violations without searching for a feasible M/N pair.",
    )
    parser.add_argument("--write-pattern-csvs", action="store_true", help="Generate the CSV test patterns and exit.")
    parser.add_argument("--list-patterns", action="store_true", help="List available pattern names and exit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("augmentation_demo.html"),
        help="Path to save the interactive Plotly HTML figure.",
    )
    parser.add_argument("--sweep", action="store_true", help="Render the requested offset/recovery sweep.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("augmentation_sweep"),
        help="Prefix used when saving sweep HTML figures.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_patterns:
        for pattern_name in list_pattern_names():
            print(pattern_name)
        return

    if args.write_pattern_csvs:
        output_paths = write_test_pattern_csvs(args.pattern_dir)
        print(f"Wrote {len(output_paths)} pattern CSV files to: {args.pattern_dir}")
        return

    adaptive_bridge_search = not args.disable_adaptive_bridge_search

    if args.sweep:
        offsets = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        yaw_offsets_deg = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
        recover_times = [0.5, 1.0, 1.5, 2.0]
        past_connect_times = [0.5, 1.0, 1.5, 2.0]
        output_paths, _ = run_sweep(
            pattern_name=args.pattern,
            pattern_dir=args.pattern_dir,
            output_prefix=args.output_prefix,
            seed=args.seed,
            offsets=offsets,
            yaw_offsets_deg=yaw_offsets_deg,
            recover_times=recover_times,
            past_connect_times=past_connect_times,
            max_lateral_accel_mps2=args.max_lateral_accel,
            max_bridge_speed_gap_mps=args.max_bridge_speed_gap,
            max_bridge_jerk_mps3=args.max_bridge_jerk,
            adaptive_bridge_search=adaptive_bridge_search,
        )
        print(f"Saved {len(output_paths)} interactive sweep figures for pattern: {args.pattern}")
        if output_paths:
            print(f"First interactive figure: {output_paths[0]}")
            print(f"Last interactive figure: {output_paths[-1]}")
        return

    artifacts = run_demo_case(
        pattern_name=args.pattern,
        pattern_dir=args.pattern_dir,
        seed=args.seed,
        offset_m=args.offset,
        yaw_offset_deg=args.yaw_offset_deg,
        recover_time_s=args.recover_time,
        past_connect_time_s=args.past_connect_time,
        max_lateral_accel_mps2=args.max_lateral_accel,
        adaptive_bridge_search=adaptive_bridge_search,
        max_bridge_speed_gap_mps=args.max_bridge_speed_gap,
        max_bridge_jerk_mps3=args.max_bridge_jerk,
    )
    _render_case_outputs(artifacts, output_path=args.output)
    for line in summarize_demo_case(
        artifacts,
        output_path=args.output,
    ):
        print(line)
