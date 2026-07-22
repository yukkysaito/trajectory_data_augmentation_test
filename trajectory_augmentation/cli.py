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
    LateralBump,
    chord_speed_from_trajectory,
    cumulative_distance,
    curvature_from_xy,
    exact_arc_speed_in_window,
    format_offset_suffix,
    format_time_suffix,
    format_yaw_suffix,
    list_pattern_names,
    run_bump_demo_case,
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

    if result.past_lateral_bump is not None:
        bump = result.past_lateral_bump
        lines.append(
            f"Past history bump: t in [-{bump.start_time_s + bump.duration_s:.2f}, "
            f"-{bump.start_time_s:.2f}] s, amplitude {bump.amplitude_m:+.2f} m"
        )
    elif artifacts.requested_past_bump:
        lines.append(
            "Past history bump: requested but not applied "
            "(all sampled bumps violated the constraints)"
        )

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
    bump_options: list[bool] | None = None,
    randomize_bridge_times: bool = False,
    bridge_time_extra_range_s: float = 1.5,
) -> tuple[list[Path], list[Path]]:
    output_paths: list[Path] = []
    pattern_suffix = f"_{pattern_name}"
    if bump_options is None:
        bump_options = [False]

    for past_connect_time_s in past_connect_times:
        for offset in offsets:
            for yaw_offset_deg in yaw_offsets_deg:
                for recover_time_s in recover_times:
                    for past_bump in bump_options:
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
                            randomize_bridge_times=randomize_bridge_times,
                            bridge_time_extra_range_s=bridge_time_extra_range_s,
                            past_bump=past_bump,
                        )
                        suffix = (
                            pattern_suffix
                            + format_offset_suffix(offset)
                            + format_yaw_suffix(yaw_offset_deg)
                            + format_time_suffix("N", recover_time_s)
                            + format_time_suffix("M", past_connect_time_s)
                            + f"_bump{int(past_bump)}"
                        )
                        output_path = output_prefix.with_name(f"{output_prefix.name}{suffix}.html")
                        _render_case_outputs(
                            artifacts,
                            output_path=output_path,
                        )
                        output_paths.append(output_path)
    return output_paths, []


def format_bump_amplitude_suffix(value_m: float) -> str:
    sign = "p" if value_m >= 0.0 else "m"
    magnitude = f"{abs(value_m):.2f}".replace(".", "p")
    return f"_bumpA{sign}{magnitude}m"


def run_bump_sweep(
    pattern_name: str,
    pattern_dir: Path,
    output_prefix: Path,
    offset_m: float,
    yaw_offset_deg: float,
    recover_time_s: float,
    past_connect_time_s: float,
    amplitudes_m: list[float],
    durations_s: list[float],
    start_times_s: list[float],
    max_lateral_accel_mps2: float,
    max_bridge_speed_gap_mps: float,
    max_bridge_jerk_mps3: float,
    adaptive_bridge_search: bool,
) -> tuple[list[Path], int]:
    output_paths: list[Path] = []
    skipped = 0
    base_suffix = f"_{pattern_name}" + format_offset_suffix(offset_m) + format_yaw_suffix(yaw_offset_deg)

    for amplitude_m in amplitudes_m:
        for duration_s in durations_s:
            for start_time_s in start_times_s:
                if start_time_s + duration_s > OUTPUT_PAST_HORIZON_S + 1.0e-9:
                    skipped += 1
                    continue
                bump = LateralBump(
                    start_time_s=start_time_s,
                    duration_s=duration_s,
                    amplitude_m=amplitude_m,
                )
                artifacts = run_bump_demo_case(
                    pattern_name=pattern_name,
                    pattern_dir=pattern_dir,
                    offset_m=offset_m,
                    yaw_offset_deg=yaw_offset_deg,
                    recover_time_s=recover_time_s,
                    past_connect_time_s=past_connect_time_s,
                    bump=bump,
                    max_lateral_accel_mps2=max_lateral_accel_mps2,
                    max_bridge_speed_gap_mps=max_bridge_speed_gap_mps,
                    max_bridge_jerk_mps3=max_bridge_jerk_mps3,
                    adaptive_bridge_search=adaptive_bridge_search,
                )
                suffix = (
                    base_suffix
                    + format_bump_amplitude_suffix(amplitude_m)
                    + format_time_suffix("bumpD", duration_s)
                    + format_time_suffix("bumpS", start_time_s)
                )
                output_path = output_prefix.with_name(f"{output_prefix.name}{suffix}.html")
                _render_case_outputs(artifacts, output_path=output_path)
                output_paths.append(output_path)
    return output_paths, skipped


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
        "--adaptive-bridge-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the automatic bridge-time search (--adaptive-bridge-search) or "
            "disable it to only diagnose constraint violations "
            "(--no-adaptive-bridge-search). Required unless --list-patterns or "
            "--write-pattern-csvs is used."
        ),
    )
    parser.add_argument(
        "--randomize-bridge-times",
        action="store_true",
        help=(
            "After finding the minimal feasible M/N, sample random feasible bridge "
            "times above those minimums to decorrelate history and recovery."
        ),
    )
    parser.add_argument(
        "--bridge-time-extra-range",
        type=float,
        default=1.5,
        help="Upper range [s] added on top of the minimal feasible M/N when randomizing bridge times.",
    )
    parser.add_argument(
        "--past-bump",
        action="store_true",
        help="Inject one random lateral bump into the past history.",
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
        "--bump-sweep",
        action="store_true",
        help=(
            "Render a sweep over deterministic past-history bump parameters "
            "(amplitude x duration x start time) with fixed offset and bridge times. "
            "Each figure compares the no-bump baseline (seed) with the bumped variant (best)."
        ),
    )
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

    if args.adaptive_bridge_search is None:
        parser.error("--adaptive-bridge-search or --no-adaptive-bridge-search is required")
    adaptive_bridge_search = args.adaptive_bridge_search

    if args.bump_sweep:
        amplitudes_m = [-1.2, -0.1, 0.1, 1.2]
        durations_s = [0.8, 1.2, 1.6, 2.0]
        start_times_s = [0.3, 0.8, 1.3, 1.8]
        output_paths, skipped = run_bump_sweep(
            pattern_name=args.pattern,
            pattern_dir=args.pattern_dir,
            output_prefix=args.output_prefix,
            offset_m=args.offset if args.offset is not None else 1.0,
            yaw_offset_deg=args.yaw_offset_deg,
            recover_time_s=args.recover_time if args.recover_time is not None else 1.5,
            past_connect_time_s=args.past_connect_time if args.past_connect_time is not None else 1.0,
            amplitudes_m=amplitudes_m,
            durations_s=durations_s,
            start_times_s=start_times_s,
            max_lateral_accel_mps2=args.max_lateral_accel,
            max_bridge_speed_gap_mps=args.max_bridge_speed_gap,
            max_bridge_jerk_mps3=args.max_bridge_jerk,
            adaptive_bridge_search=adaptive_bridge_search,
        )
        print(f"Saved {len(output_paths)} bump sweep figures for pattern: {args.pattern}")
        if skipped:
            print(
                f"Skipped {skipped} combinations whose bump would extend beyond "
                f"the {OUTPUT_PAST_HORIZON_S:.0f}s past window"
            )
        if output_paths:
            print(f"First figure: {output_paths[0]}")
            print(f"Last figure: {output_paths[-1]}")
        return

    if args.sweep:
        offsets = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        yaw_offsets_deg = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
        recover_times = [0.5, 1.0, 1.5, 2.0]
        past_connect_times = [0.5, 1.0, 1.5, 2.0]
        bump_options = [False, True]
        output_paths, _ = run_sweep(
            pattern_name=args.pattern,
            pattern_dir=args.pattern_dir,
            output_prefix=args.output_prefix,
            seed=args.seed,
            offsets=offsets,
            yaw_offsets_deg=yaw_offsets_deg,
            recover_times=recover_times,
            past_connect_times=past_connect_times,
            bump_options=bump_options,
            max_lateral_accel_mps2=args.max_lateral_accel,
            max_bridge_speed_gap_mps=args.max_bridge_speed_gap,
            max_bridge_jerk_mps3=args.max_bridge_jerk,
            adaptive_bridge_search=adaptive_bridge_search,
            randomize_bridge_times=args.randomize_bridge_times,
            bridge_time_extra_range_s=args.bridge_time_extra_range,
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
        randomize_bridge_times=args.randomize_bridge_times,
        bridge_time_extra_range_s=args.bridge_time_extra_range,
        past_bump=args.past_bump,
    )
    _render_case_outputs(artifacts, output_path=args.output)
    for line in summarize_demo_case(
        artifacts,
        output_path=args.output,
    ):
        print(line)
