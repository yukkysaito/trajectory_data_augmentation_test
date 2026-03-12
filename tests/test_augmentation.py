import unittest

import numpy as np

from diffusion_planner_augmentation import (
    augment_future_trajectory,
    augment_trajectory_bidirectional,
    cumulative_distance,
    curvature_from_xy,
    generate_synthetic_gt,
    sample_centerline,
)


class DiffusionPlannerAugmentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gt, self.current_index = generate_synthetic_gt()
        self.future_result = augment_future_trajectory(
            gt=self.gt,
            current_index=self.current_index,
            lateral_offset_m=1.0,
            recover_time_s=1.5,
        )
        self.bidirectional_result = augment_trajectory_bidirectional(
            gt=self.gt,
            current_index=self.current_index,
            lateral_offset_m=1.0,
            future_recover_time_s=1.5,
            past_connect_time_s=1.0,
        )

    def test_future_merge_path_respects_distance_budget(self) -> None:
        budget = self.future_result.connect_distance_budget_m
        self.assertLessEqual(self.future_result.merge_path_length_m, budget + 2.0e-2)

    def test_future_recovery_pose_matches_centerline_pose(self) -> None:
        recover_index = int(np.argmin(np.abs(self.future_result.original_segment.t - self.future_result.connect_time_s)))
        merge_x, merge_y, merge_yaw = sample_centerline(
            self.future_result.centerline,
            np.array([self.future_result.merge_centerline_s]),
        )
        self.assertAlmostEqual(self.future_result.augmented_segment.x[recover_index], merge_x[0], places=3)
        self.assertAlmostEqual(self.future_result.augmented_segment.y[recover_index], merge_y[0], places=3)
        self.assertAlmostEqual(self.future_result.augmented_segment.yaw[recover_index], merge_yaw[0], places=2)

    def test_future_speed_does_not_exceed_gt_during_recovery(self) -> None:
        dt = float(np.mean(np.diff(self.future_result.original_segment.t)))
        augmented_distance = cumulative_distance(
            self.future_result.augmented_segment.x,
            self.future_result.augmented_segment.y,
        )
        gt_speed = np.diff(self.future_result.distance_profile, prepend=0.0) / dt
        augmented_speed = np.diff(augmented_distance, prepend=0.0) / dt
        recover_mask = self.future_result.original_segment.t <= self.future_result.connect_time_s + 1.0e-9
        speed_margin = augmented_speed[recover_mask] - gt_speed[recover_mask]
        self.assertLess(float(np.max(speed_margin)), 0.15)

    def test_bidirectional_current_pose_is_continuous(self) -> None:
        result = self.bidirectional_result
        current_idx = result.current_index
        self.assertAlmostEqual(result.augmented_full.x[current_idx], result.augmented_past.x[-1], places=6)
        self.assertAlmostEqual(result.augmented_full.y[current_idx], result.augmented_past.y[-1], places=6)
        self.assertAlmostEqual(result.augmented_full.x[current_idx], result.augmented_future.x[0], places=6)
        self.assertAlmostEqual(result.augmented_full.y[current_idx], result.augmented_future.y[0], places=6)

    def test_bidirectional_current_pose_has_requested_offset(self) -> None:
        result = self.bidirectional_result
        current_idx = result.current_index
        offset_vector = np.array(
            [
                result.augmented_full.x[current_idx] - result.original_full.x[current_idx],
                result.augmented_full.y[current_idx] - result.original_full.y[current_idx],
            ]
        )
        self.assertAlmostEqual(float(np.linalg.norm(offset_vector)), 1.0, places=3)

    def test_past_speed_does_not_exceed_gt_during_connect(self) -> None:
        past_result = self.bidirectional_result.past_segment
        dt = float(np.mean(np.diff(past_result.original_segment.t)))
        augmented_distance = cumulative_distance(
            past_result.augmented_segment.x,
            past_result.augmented_segment.y,
        )
        gt_speed = np.diff(past_result.distance_profile, prepend=0.0) / dt
        augmented_speed = np.diff(augmented_distance, prepend=0.0) / dt
        connect_mask = past_result.original_segment.t <= past_result.connect_time_s + 1.0e-9
        speed_margin = augmented_speed[connect_mask] - gt_speed[connect_mask]
        self.assertLess(float(np.max(speed_margin)), 0.15)

    def test_full_trajectory_curvature_is_bounded(self) -> None:
        sigma = cumulative_distance(
            self.bidirectional_result.augmented_full.x,
            self.bidirectional_result.augmented_full.y,
        )
        curvature = curvature_from_xy(
            self.bidirectional_result.augmented_full.x,
            self.bidirectional_result.augmented_full.y,
            sigma,
        )
        curvature_step = np.diff(curvature)
        self.assertLess(float(np.max(np.abs(curvature))), 0.10)
        self.assertLess(float(np.max(np.abs(curvature_step))), 0.05)

    def test_past_and_future_progress_are_feasible(self) -> None:
        self.assertLessEqual(
            self.bidirectional_result.future_segment.connect_speed_scale,
            1.0 + 1.0e-6,
        )
        self.assertLessEqual(
            self.bidirectional_result.past_segment.connect_speed_scale,
            1.0 + 1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
