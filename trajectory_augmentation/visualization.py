from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import (
    BidirectionalAugmentationResult,
    DemoArtifacts,
    LateralAccelDiagnostics,
    chord_speed_from_trajectory,
    cumulative_distance,
    curvature_from_xy,
    exact_arc_speed_in_window,
    sample_centerline,
)


@dataclass
class DemoPlotPayload:
    artifacts: DemoArtifacts
    display_best_result: BidirectionalAugmentationResult
    gt_arc_speed: np.ndarray
    seed_arc_speed: np.ndarray
    best_arc_speed: np.ndarray
    gt_chord_speed: np.ndarray
    seed_chord_speed: np.ndarray
    best_chord_speed: np.ndarray
    gt_curvature: np.ndarray
    seed_curvature: np.ndarray
    best_curvature: np.ndarray
    best_offset_point: tuple[float, float]
    best_past_merge_point: tuple[float, float]
    best_future_merge_point: tuple[float, float]
    best_diag: LateralAccelDiagnostics
    title: str
    end_delta: float


def _best_diag(artifacts: DemoArtifacts) -> LateralAccelDiagnostics:
    if artifacts.feasibility.adapted_result is artifacts.selected_result and artifacts.feasibility.adapted is not None:
        return artifacts.feasibility.adapted
    return artifacts.feasibility.initial


def _display_best_result(artifacts: DemoArtifacts):
    if artifacts.feasibility.adapted_result is not None:
        return artifacts.feasibility.adapted_result
    return artifacts.selected_result


def _display_best_diag(artifacts: DemoArtifacts) -> LateralAccelDiagnostics:
    if artifacts.feasibility.adapted is not None:
        return artifacts.feasibility.adapted
    return _best_diag(artifacts)


def _build_title(artifacts: DemoArtifacts, end_delta: float) -> str:
    best = _display_best_result(artifacts)
    seed = artifacts.seed_result
    parts = [
        f"pattern={best.pattern_name}",
        f"offset={best.lateral_offset_m:+.2f} m",
        f"yaw={np.degrees(best.heading_offset_rad):+.0f} deg",
        f"Best M={best.past_connect_time_s:.1f} s",
        f"Best N={best.future_recover_time_s:.1f} s",
        f"end_delta@8s={end_delta:.3f} m",
    ]
    parts.append(f"Seed M={seed.past_connect_time_s:.1f} s")
    parts.append(f"Seed N={seed.future_recover_time_s:.1f} s")
    if artifacts.feasibility.adaptation_strategy == "search disabled":
        parts.append("adaptive search disabled")
    elif artifacts.feasibility.adaptation_strategy is not None and artifacts.feasibility.adapted_result is not None:
        parts.append(f"best via {artifacts.feasibility.adaptation_strategy}")
    elif not artifacts.feasibility.initial.passes and artifacts.feasibility.adaptation_strategy is None:
        parts.append("no feasible candidate in search range")
    return ", ".join(parts)


def build_demo_plot_payload(artifacts: DemoArtifacts) -> DemoPlotPayload:
    seed = artifacts.seed_result
    best = _display_best_result(artifacts)
    gt = best.original_window

    gt_arc_speed, best_arc_speed = exact_arc_speed_in_window(best)
    _, seed_arc_speed = exact_arc_speed_in_window(seed)

    gt_chord_speed = chord_speed_from_trajectory(gt)
    seed_chord_speed = chord_speed_from_trajectory(seed.augmented_window)
    best_chord_speed = chord_speed_from_trajectory(best.augmented_window)

    gt_curvature = curvature_from_xy(gt.x, gt.y, cumulative_distance(gt.x, gt.y))
    seed_curvature = curvature_from_xy(
        seed.augmented_window.x,
        seed.augmented_window.y,
        cumulative_distance(seed.augmented_window.x, seed.augmented_window.y),
    )
    best_curvature = curvature_from_xy(
        best.augmented_window.x,
        best.augmented_window.y,
        cumulative_distance(best.augmented_window.x, best.augmented_window.y),
    )

    current_index_in_window = best.current_index - best.window_start_index
    best_offset_point = (
        float(best.augmented_window.x[current_index_in_window]),
        float(best.augmented_window.y[current_index_in_window]),
    )
    past_merge_x, past_merge_y, _ = sample_centerline(
        best.past_segment.centerline,
        np.array([best.past_segment.merge_centerline_s]),
    )
    future_merge_x, future_merge_y, _ = sample_centerline(
        best.future_segment.centerline,
        np.array([best.future_segment.merge_centerline_s]),
    )

    end_delta = float(
        np.linalg.norm(
            np.array(
                [
                    best.augmented_window.x[-1] - gt.x[-1],
                    best.augmented_window.y[-1] - gt.y[-1],
                ]
            )
        )
    )
    return DemoPlotPayload(
        artifacts=artifacts,
        display_best_result=best,
        gt_arc_speed=gt_arc_speed,
        seed_arc_speed=seed_arc_speed,
        best_arc_speed=best_arc_speed,
        gt_chord_speed=gt_chord_speed,
        seed_chord_speed=seed_chord_speed,
        best_chord_speed=best_chord_speed,
        gt_curvature=gt_curvature,
        seed_curvature=seed_curvature,
        best_curvature=best_curvature,
        best_offset_point=best_offset_point,
        best_past_merge_point=(float(past_merge_x[0]), float(past_merge_y[0])),
        best_future_merge_point=(float(future_merge_x[0]), float(future_merge_y[0])),
        best_diag=_display_best_diag(artifacts),
        title=_build_title(artifacts, end_delta),
        end_delta=end_delta,
    )


def _pose_triangle_trace(
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    color: str,
    size_px: float,
    name: str,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name=name,
        marker={
            "symbol": "triangle-up",
            "size": size_px,
            "color": color,
            "angleref": "up",
            "angle": 90.0 - np.degrees(yaw),
            "line": {"width": 0},
        },
        hoverinfo="skip",
        showlegend=False,
    )


def _status_lines(payload: DemoPlotPayload) -> list[str]:
    feasibility = payload.artifacts.feasibility
    best = payload.best_diag
    lines = [
        (
            f"Seed Trajectory: {'PASS' if feasibility.initial.passes else 'FAIL'} "
            f"(max |a_lat|={feasibility.initial.max_abs_augmented_lateral_accel_mps2:.2f}/"
            f"{feasibility.initial.limit_mps2:.2f})"
        ),
        (
            f"Best Trajectory: {'PASS' if best.passes else 'FAIL'} "
            f"(max |a_lat|={best.max_abs_augmented_lateral_accel_mps2:.2f}/"
            f"{best.limit_mps2:.2f})"
        ),
    ]
    if feasibility.adaptation_strategy == "search disabled":
        lines.append("adaptive search disabled")
    elif feasibility.adaptation_strategy is not None:
        lines.append(
            f"best via {feasibility.adaptation_strategy}: "
            f"M={payload.display_best_result.past_connect_time_s:.1f}s, "
            f"N={payload.display_best_result.future_recover_time_s:.1f}s"
        )
    return lines


def _legend_html(entries: list[tuple[str, str]]) -> str:
    return "<br>".join(
        f"<span style='color:{color};font-weight:600'>■</span> {label}" for label, color in entries
    )


def _add_panel_annotation(fig: go.Figure, axis_index: int, text: str, x: float = 0.02, y: float = 0.98) -> None:
    suffix = "" if axis_index == 1 else str(axis_index)
    fig.add_annotation(
        xref=f"x{suffix} domain",
        yref=f"y{suffix} domain",
        x=x,
        y=y,
        text=text,
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(120,120,120,0.45)",
        borderwidth=1,
        font={"size": 11},
    )


def make_plotly_demo_figure(artifacts: DemoArtifacts) -> go.Figure:
    payload = build_demo_plot_payload(artifacts)
    seed = payload.artifacts.seed_result
    best = payload.display_best_result
    gt_full = best.original_full
    gt = best.original_window
    seed_window = seed.augmented_window
    best_window = best.augmented_window

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Trajectory", "Speed", "Curvature", "Lateral Acceleration"),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=gt_full.x,
            y=gt_full.y,
            mode="lines",
            name="GT Full Context",
            line={"color": "#d9d9d9", "width": 2},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=gt.x,
            y=gt.y,
            mode="lines",
            name="GT Window",
            line={"color": "#808080", "width": 3},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=best_window.x,
            y=best_window.y,
            mode="lines",
            name="Best Trajectory",
            line={"color": "#ff7f0e", "width": 6},
            opacity=0.60,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=seed_window.x,
            y=seed_window.y,
            mode="lines",
            name="Seed Trajectory",
            line={"color": "#1f77b4", "width": 3, "dash": "dash"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=best_window.x,
            y=best_window.y,
            mode="lines",
            name="Best Trajectory",
            line={"color": "#ff7f0e", "width": 2.4},
            opacity=0.45,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        _pose_triangle_trace(
            x=gt.x,
            y=gt.y,
            yaw=gt.yaw,
            color="#666666",
            size_px=8,
            name="GT Pose Triangles",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _pose_triangle_trace(
            x=best_window.x,
            y=best_window.y,
            yaw=best_window.yaw,
            color="#ff7f0e",
            size_px=9,
            name="Best Pose Triangles",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[payload.best_offset_point[0]],
            y=[payload.best_offset_point[1]],
            mode="markers",
            name="Best Offset Point",
            marker={"color": "#d62728", "size": 11},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[payload.best_past_merge_point[0], payload.best_future_merge_point[0]],
            y=[payload.best_past_merge_point[1], payload.best_future_merge_point[1]],
            mode="markers",
            name="Best Merge Points",
            marker={"color": "#9467bd", "size": 10, "symbol": "diamond"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=gt.t, y=payload.gt_arc_speed, mode="lines", name="GT Exact Arc Speed", line={"color": "#666666", "width": 3}, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=seed_window.t, y=payload.seed_arc_speed, mode="lines", name="Seed Exact Arc Speed", line={"color": "#1f77b4", "width": 2.5, "dash": "dash"}, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=best_window.t, y=payload.best_arc_speed, mode="lines", name="Best Exact Arc Speed", line={"color": "#ff7f0e", "width": 2.5}, opacity=0.70, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=gt.t, y=payload.gt_chord_speed, mode="lines", name="GT Chord Speed", line={"color": "#999999", "width": 2, "dash": "dot"}, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=seed_window.t, y=payload.seed_chord_speed, mode="lines", name="Seed Chord Speed", line={"color": "#5fa2dd", "width": 2, "dash": "dot"}, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=best_window.t, y=payload.best_chord_speed, mode="lines", name="Best Chord Speed", line={"color": "#f2a65a", "width": 2, "dash": "dot"}, showlegend=False),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(x=gt.t, y=payload.gt_curvature, mode="lines", name="GT Curvature", line={"color": "#666666", "width": 3}, showlegend=False),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=seed_window.t, y=payload.seed_curvature, mode="lines", name="Seed Curvature", line={"color": "#1f77b4", "width": 2.5, "dash": "dash"}, showlegend=False),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=best_window.t, y=payload.best_curvature, mode="lines", name="Best Curvature", line={"color": "#ff7f0e", "width": 2.5}, opacity=0.70, showlegend=False),
        row=2,
        col=1,
    )

    initial_diag = payload.artifacts.feasibility.initial
    fig.add_trace(
        go.Scatter(
            x=initial_diag.time,
            y=initial_diag.gt_lateral_accel_mps2,
            mode="lines",
            name="GT Lateral Accel",
            line={"color": "#666666", "width": 3},
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=initial_diag.time,
            y=initial_diag.augmented_lateral_accel_mps2,
            mode="lines",
            name="Seed Trajectory Lateral Accel",
            line={"color": "#1f77b4", "width": 2.5, "dash": "dash"},
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=payload.best_diag.time,
            y=payload.best_diag.augmented_lateral_accel_mps2,
            mode="lines",
            name="Best Trajectory Lateral Accel",
            line={"color": "#ff7f0e", "width": 2.5},
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    violation_mask = np.abs(payload.best_diag.augmented_lateral_accel_mps2) > payload.best_diag.limit_mps2 + 1.0e-9
    if np.any(violation_mask):
        fig.add_trace(
            go.Scatter(
                x=payload.best_diag.time[violation_mask],
                y=payload.best_diag.augmented_lateral_accel_mps2[violation_mask],
                mode="markers",
                name="Best Trajectory Limit Exceeded",
                marker={"color": "#d62728", "size": 8},
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    for row, col in ((1, 2), (2, 1), (2, 2)):
        fig.add_vline(x=0.0, line_width=1.2, line_dash="dot", line_color="#666666", row=row, col=col)
        fig.add_vline(x=-best.past_connect_time_s, line_width=1.0, line_dash="dash", line_color="#1f77b4", row=row, col=col)
        fig.add_vline(x=best.future_recover_time_s, line_width=1.0, line_dash="dash", line_color="#9467bd", row=row, col=col)
    fig.add_hline(y=payload.best_diag.limit_mps2, line_width=1.0, line_dash="dash", line_color="#d62728", row=2, col=2)
    fig.add_hline(y=-payload.best_diag.limit_mps2, line_width=1.0, line_dash="dash", line_color="#d62728", row=2, col=2)

    fig.update_xaxes(title_text="x [m]", row=1, col=1)
    fig.update_yaxes(title_text="y [m]", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="time [s]", row=1, col=2)
    fig.update_yaxes(title_text="speed [m/s]", row=1, col=2)
    fig.update_xaxes(title_text="time [s]", row=2, col=1)
    fig.update_yaxes(title_text="curvature [1/m]", row=2, col=1)
    fig.update_xaxes(title_text="time [s]", row=2, col=2)
    fig.update_yaxes(
        title_text="a_lat [m/s^2]",
        row=2,
        col=2,
        range=[-2.0 * payload.best_diag.limit_mps2, 2.0 * payload.best_diag.limit_mps2],
    )

    _add_panel_annotation(
        fig,
        1,
        _legend_html(
            [
                ("GT Full Context", "#d9d9d9"),
                ("GT Window", "#808080"),
                ("Seed Trajectory", "#1f77b4"),
                ("Best Trajectory", "#ff7f0e"),
                ("Best Offset Point", "#d62728"),
                ("Best Past Merge", "#9467bd"),
                ("Best Future Merge", "#9467bd"),
            ]
        ),
    )
    _add_panel_annotation(
        fig,
        2,
        _legend_html(
            [
                ("GT Exact Arc", "#666666"),
                ("Seed Exact Arc", "#1f77b4"),
                ("Best Exact Arc", "#ff7f0e"),
                ("GT Chord", "#999999"),
                ("Seed Chord", "#5fa2dd"),
                ("Best Chord", "#f2a65a"),
            ]
        ),
    )
    _add_panel_annotation(
        fig,
        3,
        _legend_html(
            [
                ("GT Curvature", "#666666"),
                ("Seed Curvature", "#1f77b4"),
                ("Best Curvature", "#ff7f0e"),
            ]
        ),
    )
    _add_panel_annotation(
        fig,
        4,
        _legend_html(
            [
                ("GT Lateral Accel", "#666666"),
                ("Seed Trajectory", "#1f77b4"),
                ("Best Trajectory", "#ff7f0e"),
                ("Limit Exceeded", "#d62728"),
            ]
        )
        + "<br><br>"
        + "<br>".join(_status_lines(payload)),
    )
    fig.update_layout(
        title=payload.title,
        template="plotly_white",
        hovermode="closest",
        showlegend=False,
        height=920,
    )
    return fig


def write_plotly_demo_figure(artifacts: DemoArtifacts, output_path: Path) -> None:
    fig = make_plotly_demo_figure(artifacts)
    fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
