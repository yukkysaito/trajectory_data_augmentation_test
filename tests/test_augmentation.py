import unittest

import numpy as np

from diffusion_planner_augmentation import (
    DEFAULT_PATTERN_DIR,
    FULL_FUTURE_HORIZON_S,
    FULL_PAST_HORIZON_S,
    OUTPUT_FUTURE_HORIZON_S,
    OUTPUT_PAST_HORIZON_S,
    augment_future_trajectory,
    augment_trajectory_bidirectional,
    build_progress_speed_lookup,
    cumulative_distance,
    chord_speed_from_trajectory,
    curvature_from_xy,
    exact_arc_speed_in_window,
    list_pattern_names,
    load_test_pattern,
    sample_centerline,
    sample_speed_by_progress,
    search_feasible_result,
    speed_from_progress,
    speed_from_trajectory,
    write_test_pattern_csvs,
)


class DiffusionPlannerAugmentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        write_test_pattern_csvs(DEFAULT_PATTERN_DIR)
        cls.pattern_names = list_pattern_names()

    def _augment_pattern(self, pattern_name: str, offset_m: float = 1.0):
        gt, current_index = load_test_pattern(pattern_name, DEFAULT_PATTERN_DIR)
        future_result = augment_future_trajectory(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=offset_m,
            heading_offset_rad=0.0,
            recover_time_s=1.5,
        )
        bidirectional_result = augment_trajectory_bidirectional(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=offset_m,
            heading_offset_rad=0.0,
            future_recover_time_s=1.5,
            past_connect_time_s=1.0,
            pattern_name=pattern_name,
        )
        return gt, current_index, future_result, bidirectional_result

    def test_pattern_csvs_cover_full_horizon(self) -> None:
        self.assertEqual(len(self.pattern_names), 15)
        expected_num_samples = int(round((FULL_PAST_HORIZON_S + FULL_FUTURE_HORIZON_S) / 0.1)) + 1
        for pattern_name in self.pattern_names:
            with self.subTest(pattern=pattern_name):
                gt, current_index = load_test_pattern(pattern_name, DEFAULT_PATTERN_DIR)
                self.assertEqual(len(gt.t), expected_num_samples)
                self.assertAlmostEqual(float(gt.t[0]), -FULL_PAST_HORIZON_S, places=6)
                self.assertAlmostEqual(float(gt.t[-1]), FULL_FUTURE_HORIZON_S, places=6)
                self.assertAlmostEqual(float(gt.t[current_index]), 0.0, places=9)

    def test_stop8s_pattern_stops_at_output_horizon(self) -> None:
        gt, _ = load_test_pattern("straight_stop8s", DEFAULT_PATTERN_DIR)
        gt_speed = speed_from_trajectory(gt)

        terminal_index = int(np.argmin(np.abs(gt.t - OUTPUT_FUTURE_HORIZON_S)))
        pre_stop_index = int(np.argmin(np.abs(gt.t - 7.0)))
        self.assertAlmostEqual(float(gt.t[terminal_index]), OUTPUT_FUTURE_HORIZON_S, places=9)
        self.assertLess(float(gt_speed[terminal_index]), 0.05)
        self.assertGreater(float(gt_speed[pre_stop_index]), 0.1)

    def test_output_window_matches_requested_training_horizon(self) -> None:
        expected_num_samples = int(round((OUTPUT_PAST_HORIZON_S + OUTPUT_FUTURE_HORIZON_S) / 0.1)) + 1
        for pattern_name in self.pattern_names:
            with self.subTest(pattern=pattern_name):
                _, _, _, result = self._augment_pattern(pattern_name)
                self.assertEqual(len(result.original_window.t), expected_num_samples)
                self.assertEqual(len(result.augmented_window.t), expected_num_samples)
                self.assertAlmostEqual(float(result.original_window.t[0]), -OUTPUT_PAST_HORIZON_S, places=6)
                self.assertAlmostEqual(float(result.original_window.t[-1]), OUTPUT_FUTURE_HORIZON_S, places=6)
                self.assertAlmostEqual(float(result.augmented_window.t[0]), -OUTPUT_PAST_HORIZON_S, places=6)
                self.assertAlmostEqual(float(result.augmented_window.t[-1]), OUTPUT_FUTURE_HORIZON_S, places=6)

    def test_future_speed_matches_gt_speed_at_corresponding_progress_after_recovery_time(self) -> None:
        for pattern_name in self.pattern_names:
            with self.subTest(pattern=pattern_name):
                _, _, future_result, _ = self._augment_pattern(pattern_name)
                post_merge_mask = future_result.original_segment.t >= future_result.connect_time_s + 1.0e-9
                self.assertTrue(np.any(post_merge_mask))
                gt_speed = np.clip(
                    speed_from_progress(future_result.distance_profile, future_result.original_segment.t),
                    0.0,
                    None,
                )
                progress_samples, speed_samples = build_progress_speed_lookup(
                    future_result.distance_profile,
                    gt_speed,
                )
                centerline_progress = (
                    future_result.progress_profile[post_merge_mask]
                    - future_result.merge_path_length_m
                    + future_result.merge_centerline_s
                )
                expected_speed = sample_speed_by_progress(progress_samples, speed_samples, centerline_progress)
                tolerance = 1.0e-1 if pattern_name.endswith("_stop8s") else 2.0e-2
                self.assertLess(
                    float(np.max(np.abs(future_result.exact_speed_profile[post_merge_mask] - expected_speed))),
                    tolerance,
                )

    def test_future_recovery_pose_matches_centerline_pose(self) -> None:
        for pattern_name in self.pattern_names:
            with self.subTest(pattern=pattern_name):
                _, _, future_result, _ = self._augment_pattern(pattern_name)
                recover_index = int(np.argmin(np.abs(future_result.original_segment.t - future_result.connect_time_s)))
                merge_x, merge_y, merge_yaw = sample_centerline(
                    future_result.centerline,
                    np.array([future_result.merge_centerline_s]),
                )
                self.assertAlmostEqual(future_result.augmented_segment.x[recover_index], merge_x[0], places=3)
                self.assertAlmostEqual(future_result.augmented_segment.y[recover_index], merge_y[0], places=3)
                self.assertAlmostEqual(future_result.augmented_segment.yaw[recover_index], merge_yaw[0], places=2)

    def test_speed_matches_gt_speed_at_corresponding_progress_after_bridge_windows(self) -> None:
        for pattern_name in self.pattern_names:
            with self.subTest(pattern=pattern_name):
                _, _, _, result = self._augment_pattern(pattern_name, offset_m=2.0)
                future_mask = (
                    result.future_segment.original_segment.t
                    >= result.future_segment.connect_time_s + 0.2 - 1.0e-9
                )
                self.assertTrue(np.any(future_mask))
                future_gt_speed = np.clip(
                    speed_from_progress(result.future_segment.distance_profile, result.future_segment.original_segment.t),
                    0.0,
                    None,
                )
                future_progress_samples, future_speed_samples = build_progress_speed_lookup(
                    result.future_segment.distance_profile,
                    future_gt_speed,
                )
                future_centerline_progress = (
                    result.future_segment.progress_profile[future_mask]
                    - result.future_segment.merge_path_length_m
                    + result.future_segment.merge_centerline_s
                )
                future_expected_speed = sample_speed_by_progress(
                    future_progress_samples,
                    future_speed_samples,
                    future_centerline_progress,
                )
                tolerance = 1.0e-1 if pattern_name.endswith("_stop8s") else 2.0e-2
                self.assertLess(
                    float(
                        np.max(
                            np.abs(result.future_segment.exact_speed_profile[future_mask] - future_expected_speed)
                        )
                    ),
                    tolerance,
                )

                past_mask = (
                    result.past_segment.original_segment.t
                    >= result.past_segment.connect_time_s + 0.2 - 1.0e-9
                )
                self.assertTrue(np.any(past_mask))
                past_gt_speed = np.clip(
                    speed_from_progress(result.past_segment.distance_profile, result.past_segment.original_segment.t),
                    0.0,
                    None,
                )
                past_progress_samples, past_speed_samples = build_progress_speed_lookup(
                    result.past_segment.distance_profile,
                    past_gt_speed,
                )
                past_centerline_progress = (
                    result.past_segment.progress_profile[past_mask]
                    - result.past_segment.merge_path_length_m
                    + result.past_segment.merge_centerline_s
                )
                past_expected_speed = sample_speed_by_progress(
                    past_progress_samples,
                    past_speed_samples,
                    past_centerline_progress,
                )
                self.assertLess(
                    float(np.max(np.abs(result.past_segment.exact_speed_profile[past_mask] - past_expected_speed))),
                    2.0e-2,
                )

    def test_current_pose_is_continuous_and_offset_is_correct(self) -> None:
        for pattern_name in self.pattern_names:
            with self.subTest(pattern=pattern_name):
                _, _, _, result = self._augment_pattern(pattern_name, offset_m=1.0)
                current_idx = result.current_index
                self.assertAlmostEqual(result.augmented_full.x[current_idx], result.augmented_past.x[-1], places=6)
                self.assertAlmostEqual(result.augmented_full.y[current_idx], result.augmented_past.y[-1], places=6)
                self.assertAlmostEqual(result.augmented_full.x[current_idx], result.augmented_future.x[0], places=6)
                self.assertAlmostEqual(result.augmented_full.y[current_idx], result.augmented_future.y[0], places=6)

                offset_vector = np.array(
                    [
                        result.augmented_full.x[current_idx] - result.original_full.x[current_idx],
                        result.augmented_full.y[current_idx] - result.original_full.y[current_idx],
                    ]
                )
                self.assertAlmostEqual(float(np.linalg.norm(offset_vector)), 1.0, places=3)

    def test_output_window_curvature_is_bounded(self) -> None:
        for pattern_name in self.pattern_names:
            with self.subTest(pattern=pattern_name):
                if pattern_name.endswith("_stopping") or pattern_name.endswith("_stop8s"):
                    continue
                _, _, _, result = self._augment_pattern(pattern_name, offset_m=2.0)
                sigma = cumulative_distance(result.augmented_window.x, result.augmented_window.y)
                curvature = curvature_from_xy(
                    result.augmented_window.x,
                    result.augmented_window.y,
                    sigma,
                )
                curvature_step = np.diff(curvature)
                self.assertLess(float(np.max(np.abs(curvature))), 0.24)
                self.assertLess(float(np.max(np.abs(curvature_step))), 0.14)

    def test_endpoint_lags_gt_after_merge_when_path_is_longer(self) -> None:
        gt, current_index = load_test_pattern("curve_decelerating", DEFAULT_PATTERN_DIR)
        result = augment_trajectory_bidirectional(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=-3.0,
            heading_offset_rad=0.0,
            future_recover_time_s=1.5,
            past_connect_time_s=1.0,
            pattern_name="curve_decelerating",
        )
        end_delta = np.linalg.norm(
            np.array(
                [
                    result.augmented_window.x[-1] - result.original_window.x[-1],
                    result.augmented_window.y[-1] - result.original_window.y[-1],
                ]
            )
        )
        self.assertGreater(float(end_delta), 0.3)

    def test_exact_arc_speed_shifts_in_time_for_stop8s_profile(self) -> None:
        gt, current_index = load_test_pattern("straight_stop8s", DEFAULT_PATTERN_DIR)
        result = augment_trajectory_bidirectional(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=4.0,
            heading_offset_rad=0.0,
            future_recover_time_s=0.5,
            past_connect_time_s=1.0,
            pattern_name="straight_stop8s",
        )
        gt_arc_speed, augmented_arc_speed = exact_arc_speed_in_window(result)
        terminal_index = int(np.argmin(np.abs(result.original_window.t - OUTPUT_FUTURE_HORIZON_S)))
        self.assertGreater(float(augmented_arc_speed[terminal_index]), float(gt_arc_speed[terminal_index]) + 5.0e-3)

    def test_heading_offset_is_reflected_at_current_pose(self) -> None:
        gt, current_index = load_test_pattern("straight_constant", DEFAULT_PATTERN_DIR)
        for yaw_offset_deg in (-10.0, 10.0):
            with self.subTest(yaw_offset_deg=yaw_offset_deg):
                result = augment_trajectory_bidirectional(
                    gt=gt,
                    current_index=current_index,
                    lateral_offset_m=1.0,
                    heading_offset_rad=np.deg2rad(yaw_offset_deg),
                    future_recover_time_s=1.5,
                    past_connect_time_s=1.0,
                    pattern_name="straight_constant",
                )
                self.assertAlmostEqual(
                    float(np.degrees(result.augmented_full.yaw[current_index])),
                    yaw_offset_deg,
                    delta=1.0,
                )

    def test_feasibility_search_finds_lowest_n_before_extending_m(self) -> None:
        gt, current_index = load_test_pattern("curve_decelerating", DEFAULT_PATTERN_DIR)
        result = augment_trajectory_bidirectional(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=2.5,
            heading_offset_rad=np.deg2rad(10.0),
            future_recover_time_s=0.5,
            past_connect_time_s=2.0,
            pattern_name="curve_decelerating",
        )
        diagnostics = search_feasible_result(
            result,
            max_lateral_accel_mps2=4.0,
            max_bridge_speed_gap_mps=0.5,
            max_bridge_jerk_mps3=5.0,
        )

        self.assertFalse(diagnostics.initial.passes)
        self.assertIsNotNone(diagnostics.adapted_result)
        self.assertIsNotNone(diagnostics.adapted)
        self.assertEqual(diagnostics.adaptation_strategy, "extend N")
        self.assertAlmostEqual(diagnostics.adapted_result.past_connect_time_s, 2.0, places=6)
        self.assertGreater(diagnostics.adapted_result.future_recover_time_s, 0.5)
        self.assertTrue(diagnostics.adapted.passes)
        self.assertTrue(diagnostics.adapted.lateral_accel_passes)
        self.assertTrue(diagnostics.adapted.speed_gap_passes)
        self.assertTrue(diagnostics.adapted.jerk_passes)

    def test_feasibility_search_extends_m_when_n_only_is_insufficient(self) -> None:
        gt, current_index = load_test_pattern("curve_decelerating", DEFAULT_PATTERN_DIR)
        result = augment_trajectory_bidirectional(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=3.0,
            heading_offset_rad=np.deg2rad(10.0),
            future_recover_time_s=0.5,
            past_connect_time_s=0.5,
            pattern_name="curve_decelerating",
        )
        diagnostics = search_feasible_result(
            result,
            max_lateral_accel_mps2=3.0,
            max_bridge_speed_gap_mps=0.5,
            max_bridge_jerk_mps3=5.0,
        )

        self.assertFalse(diagnostics.initial.passes)
        self.assertIsNotNone(diagnostics.adapted_result)
        self.assertIsNotNone(diagnostics.adapted)
        self.assertEqual(diagnostics.adaptation_strategy, "extend M and N")
        self.assertGreater(diagnostics.adapted_result.past_connect_time_s, 0.5)
        self.assertGreater(diagnostics.adapted_result.future_recover_time_s, 0.5)
        self.assertTrue(diagnostics.adapted.passes)
        self.assertTrue(diagnostics.adapted.lateral_accel_passes)
        self.assertTrue(diagnostics.adapted.speed_gap_passes)
        self.assertTrue(diagnostics.adapted.jerk_passes)

    def test_stopping_pattern_is_still_moving_at_the_gt_stop_time_for_short_recovery(self) -> None:
        gt, current_index = load_test_pattern("straight_stopping", DEFAULT_PATTERN_DIR)
        result = augment_trajectory_bidirectional(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=3.0,
            heading_offset_rad=np.deg2rad(10.0),
            future_recover_time_s=0.5,
            past_connect_time_s=1.0,
            pattern_name="straight_stopping",
        )

        gt_speed, augmented_speed = exact_arc_speed_in_window(result)
        stop_index = int(np.argmin(np.abs(result.original_window.t - 5.0)))
        self.assertLess(float(result.augmented_window.x[stop_index]), float(result.original_window.x[stop_index]) - 1.0e-3)
        self.assertGreater(float(augmented_speed[stop_index]), float(gt_speed[stop_index]) + 1.0e-2)

    def test_stop8s_pattern_stops_short_and_keeps_terminal_speed(self) -> None:
        gt, current_index = load_test_pattern("straight_stop8s", DEFAULT_PATTERN_DIR)
        result = augment_trajectory_bidirectional(
            gt=gt,
            current_index=current_index,
            lateral_offset_m=4.0,
            heading_offset_rad=0.0,
            future_recover_time_s=0.5,
            past_connect_time_s=1.0,
            pattern_name="straight_stop8s",
        )

        gt_speed, augmented_speed = exact_arc_speed_in_window(result)
        terminal_index = int(np.argmin(np.abs(result.original_window.t - OUTPUT_FUTURE_HORIZON_S)))
        self.assertGreater(float(augmented_speed[terminal_index]), float(gt_speed[terminal_index]) + 5.0e-3)
        self.assertLess(float(result.augmented_window.x[terminal_index]), float(result.original_window.x[terminal_index]) - 0.2)


if __name__ == "__main__":
    unittest.main()
