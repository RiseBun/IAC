from __future__ import annotations

import unittest

import numpy as np

from iac_new.visual_consistency import (
    score_structural_grounding,
    score_trajectory_conditioned_likelihood,
    score_trajectory_visual_consistency,
    score_twin_differential_consistency,
    trajectory_conditioned_flow,
)


class VisualConsistencyTest(unittest.TestCase):
    def test_structural_grounding_is_calibrated_and_fail_closed(self) -> None:
        rows = [
            {
                "interval_index": i,
                "input_available": True,
                "median_flow_magnitude_px": 10.0,
                "horizontal_flow_center": 0.1,
                "vertical_flow_center": 0.2,
                "divergence": 0.3,
                "curl": 0.4,
            }
            for i in range(3)
        ]
        scales = {key: 1.0 for key in (
            "median_flow_magnitude_px",
            "horizontal_flow_center",
            "vertical_flow_center",
            "divergence",
            "curl",
        )}
        result = score_structural_grounding(rows, rows, descriptor_scales=scales)
        self.assertEqual(result["metric_id"], "GS")
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["score"], 1.0)
        unavailable = score_structural_grounding(rows[:2], rows[:2], descriptor_scales=scales)
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIsNone(unavailable["score"])

    def test_zero_trajectory_has_zero_ground_plane_flow(self) -> None:
        expected, valid = trajectory_conditioned_flow(
            np.zeros((2, 3)),
            intrinsics=np.asarray([[40.0, 0.0, 8.0], [0.0, 40.0, 6.0], [0.0, 0.0, 1.0]]),
            camera_to_ego=np.asarray([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.5],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            frame_shape=(12, 16),
        )
        self.assertGreater(float(np.mean(valid)), 0.0)
        self.assertTrue(np.allclose(expected[valid], 0.0))

    def test_forward_score_does_not_zero_fill_unavailable_intervals(self) -> None:
        observed = np.zeros((2, 4, 4, 2), dtype=np.float64)
        expected = np.zeros_like(observed)
        expected[0, ..., 0] = 1.0
        mask = np.ones((2, 4, 4), dtype=bool)
        mask[1] = False
        report = score_trajectory_visual_consistency(
            observed,
            expected,
            valid_mask=mask,
            max_median_residual_px=0.5,
        )
        self.assertEqual(report["status_counts"], {"scored": 0, "weak": 1, "unavailable": 1})
        self.assertEqual(report["interval_coverage"], 0.5)
        self.assertEqual(report["reliable_interval_fraction"], 0.0)

    def test_forward_score_accepts_matching_motion(self) -> None:
        expected = np.zeros((1, 4, 4, 2), dtype=np.float64)
        expected[..., 0] = 0.75
        report = score_trajectory_visual_consistency(
            expected,
            expected.copy(),
            max_median_residual_px=0.1,
            min_direction_cosine=0.9,
        )
        self.assertEqual(report["status_counts"]["scored"], 1)
        self.assertAlmostEqual(report["median_residual_px"], 0.0)
        self.assertAlmostEqual(report["median_direction_cosine"], 1.0)
        self.assertEqual(report["metric_id"], "MAS")

    def test_fixed_support_keeps_candidate_projection_out_of_denominator(self) -> None:
        observed = np.zeros((1, 4, 4, 2), dtype=np.float64)
        expected = np.zeros_like(observed)
        observed[..., 0] = 1.0
        expected[..., 0] = 1.0
        fixed = np.ones((1, 4, 4), dtype=bool)
        candidate_valid = fixed.copy()
        candidate_valid[:, :2] = False
        report = score_trajectory_conditioned_likelihood(
            observed,
            expected,
            fixed_support_mask=fixed,
            expected_valid_mask=candidate_valid,
            min_projection_valid_fraction=0.2,
        )
        self.assertEqual(report["status_counts"]["scored"], 1)
        self.assertAlmostEqual(report["fixed_support_fraction"], 1.0)
        self.assertAlmostEqual(report["projection_valid_fraction"], 0.5)
        self.assertAlmostEqual(report["score"], 1.0)

    def test_fixed_support_fails_closed_when_candidate_projection_is_too_small(self) -> None:
        observed = np.zeros((1, 4, 4, 2), dtype=np.float64)
        expected = np.zeros_like(observed)
        fixed = np.ones((1, 4, 4), dtype=bool)
        candidate_valid = np.zeros_like(fixed)
        report = score_trajectory_conditioned_likelihood(
            observed,
            expected,
            fixed_support_mask=fixed,
            expected_valid_mask=candidate_valid,
            min_projection_valid_fraction=0.2,
        )
        self.assertEqual(report["status_counts"], {"scored": 0, "weak": 0, "unavailable": 1})
        self.assertIsNone(report["score"])
        self.assertEqual(report["intervals"][0]["reason"], "insufficient_candidate_projection")

    def test_weak_control_retains_continuous_score(self) -> None:
        observed = np.zeros((1, 4, 4, 2), dtype=np.float64)
        expected = np.zeros_like(observed)
        observed[..., 0] = 1.0
        expected[..., 0] = -1.0
        fixed = np.ones((1, 4, 4), dtype=bool)
        report = score_trajectory_conditioned_likelihood(
            observed,
            expected,
            fixed_support_mask=fixed,
            min_direction_cosine=0.0,
        )
        self.assertEqual(report["status_counts"]["weak"], 1)
        self.assertIsNotNone(report["score"])
        self.assertIsNone(report["reliable_score"])

    def test_quality_weight_is_candidate_blind_and_reported_separately(self) -> None:
        observed = np.zeros((2, 4, 4, 2), dtype=np.float64)
        expected = observed.copy()
        expected[1, ..., 0] = 1.0
        fixed = np.ones((2, 4, 4), dtype=bool)
        report = score_trajectory_conditioned_likelihood(
            observed,
            expected,
            fixed_support_mask=fixed,
            interval_quality_weights=np.asarray([1.0, 0.1]),
        )
        self.assertEqual(report["status_counts"]["scored"], 2)
        self.assertEqual(report["quality_weighting"], "candidate_blind_interval_structure_only")
        self.assertAlmostEqual(report["intervals"][1]["interval_quality_weight"], 0.1)
        self.assertIsNotNone(report["quality_weighted_score"])

    def test_twin_score_reports_signed_temporal_persistence(self) -> None:
        observed_left = np.zeros((2, 4, 4, 2), dtype=np.float64)
        observed_right = np.zeros_like(observed_left)
        expected_left = np.zeros_like(observed_left)
        expected_right = np.zeros_like(observed_left)
        observed_left[..., 0] = 1.0
        expected_left[..., 0] = 1.0
        report = score_twin_differential_consistency(
            observed_left, observed_right, expected_left, expected_right
        )
        self.assertAlmostEqual(report["temporal_persistence"], 1.0)
        reversed_report = score_twin_differential_consistency(
            observed_left, observed_right, expected_right, expected_left
        )
        self.assertAlmostEqual(reversed_report["temporal_persistence"], 0.0)
        self.assertEqual(report["metric_id"], "RCS")


if __name__ == "__main__":
    unittest.main()
