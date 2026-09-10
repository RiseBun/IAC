import unittest
from unittest.mock import patch

import numpy as np

from iac_new.trajectory_decode import (
    _longitudinal_residual_penalty,
    _objective,
    _projection_support,
    _sample_pixels,
    _road_prior_penalty,
    _kinematic_smoothness_penalty,
    compare_continuous_trajectory,
    decode_continuous_trajectory,
    estimate_temporal_flow_scale_state,
    integrate_piecewise_controls,
)


class TrajectoryDecodeTest(unittest.TestCase):
    def test_spatial_sampler_balances_vertical_and_horizontal_strata(self) -> None:
        observed = np.ones((2, 60, 80, 2), dtype=np.float32)
        weights = np.ones((2, 60, 80), dtype=np.float32)
        coords, sampled, sampled_weights = _sample_pixels(
            observed, weights, max_points=60,
        )
        self.assertEqual(coords.shape, (60, 2))
        self.assertEqual(sampled.shape, (2, 60, 2))
        self.assertEqual(sampled_weights.shape, (2, 60))
        v = coords[:, 1] / 59.0
        u = coords[:, 0] / 79.0
        self.assertGreaterEqual(float(v.min()), 0.55)
        self.assertLess(float(v.max()), 0.85)
        for lower, upper in ((0.55, 0.65), (0.65, 0.75), (0.75, 0.85)):
            self.assertGreaterEqual(int(np.sum((v >= lower) & (v < upper))), 15)
        for lower, upper in zip(np.linspace(0.0, 1.0, 7)[:-1], np.linspace(0.0, 1.0, 7)[1:]):
            self.assertGreaterEqual(int(np.sum((u >= lower) & (u < upper))), 1)

    def test_coarse_initializer_uses_best_candidate_blind_grid_seed(self) -> None:
        observed = np.ones((1, 8, 8, 2), dtype=np.float32)
        support = {
            "projection_supported": True,
            "source_points": 16,
            "projected_points": 16,
            "source_weight": 16.0,
            "projected_weight": 16.0,
            "projected_weight_fraction": 1.0,
            "by_interval": [{
                "interval_index": 0,
                "source_points": 16,
                "projected_points": 16,
                "source_weight": 16.0,
                "projected_weight": 16.0,
                "projected_weight_fraction": 1.0,
                "projection_supported": True,
                "reasons": [],
            }],
            "policy": {
                "minimum_source_points": 1,
                "minimum_projected_points": 1,
                "minimum_projected_weight_fraction": 0.5,
            },
        }
        starts: list[tuple[float, float]] = []

        def objective(trajectory, observed_flow, *args, **kwargs):
            energy = abs(float(trajectory[-1, 0]) - 2.0) + abs(float(trajectory[-1, 2]))
            predicted = np.zeros_like(observed_flow)
            valid = np.ones(observed_flow.shape[:-1], dtype=bool)
            return energy, predicted, valid, support

        def fit_once(**kwargs):
            speed = float(kwargs["initial_speed_mps"])
            curvature = float(kwargs["initial_curvatures_1pm"][0])
            starts.append((speed, curvature))
            trajectory = integrate_piecewise_controls(
                kwargs["future_times_s"],
                speeds_mps=np.asarray([speed]),
                curvatures_1pm=np.asarray([curvature]),
            )
            return trajectory, objective(trajectory, kwargs["observed"])[0]

        with (
            patch("iac_new.trajectory_decode._objective", side_effect=objective),
            patch("iac_new.trajectory_decode._fit_once", side_effect=fit_once),
        ):
            result = decode_continuous_trajectory(
                observed_flows=observed,
                camera_to_ego=np.eye(4),
                intrinsics=np.eye(3),
                future_times_s=np.asarray([1.0]),
                roi_mask=np.ones((8, 8), dtype=bool),
                max_points=16,
                minimum_projection_points=1,
                coarse_initializer_enabled=True,
                coarse_speed_grid_mps=(1.0, 2.0, 3.0),
                coarse_curvature_grid_1pm=(-0.1, 0.0, 0.1),
                coarse_initializer_top_k=1,
            )
        self.assertEqual(starts, [(2.0, 0.0)])
        self.assertEqual(
            result["decoder_parameters"]["coarse_initializer_best"]["speed_mps"],
            2.0,
        )

    def test_objective_charges_invalid_projection_against_fixed_source_denominator(self) -> None:
        observed = np.ones((1, 4, 2), dtype=np.float64)
        predicted = observed.copy()
        valid = np.asarray([[True, True, False, False]])
        with patch(
            "iac_new.trajectory_decode._sparse_predicted_flows",
            return_value=(predicted, valid),
        ):
            energy, _, _, support = _objective(
                np.asarray([[1.0, 0.0, 0.0]]),
                observed,
                np.ones((1, 4)),
                np.asarray([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]),
                np.eye(4),
                np.eye(3),
                (8, 8),
                None,
                1.0,
                minimum_projection_points=1,
            )
        self.assertAlmostEqual(energy, 2.0)
        self.assertAlmostEqual(support["projected_weight_fraction"], 0.5)

    def test_projection_support_fails_closed_when_projection_is_empty(self) -> None:
        observed = np.ones((2, 5, 2), dtype=np.float64)
        predicted = np.full_like(observed, np.nan)
        valid = np.zeros((2, 5), dtype=bool)
        support = _projection_support(
            observed, predicted, valid, np.ones((2, 5)),
            minimum_points=1, minimum_weight_fraction=0.5,
        )
        self.assertFalse(support["projection_supported"])
        self.assertEqual(support["projected_points"], 0)
        self.assertEqual(support["projected_weight_fraction"], 0.0)
        self.assertTrue(all(not row["projection_supported"] for row in support["by_interval"]))

    def test_objective_exposes_projection_support_without_changing_energy(self) -> None:
        observed = np.ones((1, 3, 2), dtype=np.float64)
        predicted = observed.copy()
        valid = np.ones((1, 3), dtype=bool)
        with patch(
            "iac_new.trajectory_decode._sparse_predicted_flows",
            return_value=(predicted, valid),
        ):
            energy, _, _, support = _objective(
                np.asarray([[1.0, 0.0, 0.0]]),
                observed,
                np.ones((1, 3)),
                np.asarray([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
                np.eye(4),
                np.eye(3),
                (8, 8),
                None,
                1.0,
                minimum_projection_points=1,
            )
        self.assertEqual(energy, 0.0)
        self.assertTrue(support["projection_supported"])
        self.assertEqual(support["projected_points"], 3)
        self.assertEqual(support["by_interval"][0]["flow_energy"], 0.0)

    def test_decoder_marks_empty_projection_invalid(self) -> None:
        observed = np.ones((1, 8, 8, 2), dtype=np.float32)

        def empty_projection(trajectory, *args, **kwargs):
            count = len(trajectory)
            points = len(args[2])
            return (
                np.full((count, points, 2), np.nan, dtype=np.float64),
                np.zeros((count, points), dtype=bool),
            )

        with patch(
            "iac_new.trajectory_decode._sparse_predicted_flows",
            side_effect=empty_projection,
        ):
            result = decode_continuous_trajectory(
                observed_flows=observed,
                camera_to_ego=np.eye(4),
                intrinsics=np.eye(3),
                future_times_s=np.asarray([1.0]),
                roi_mask=np.ones((8, 8), dtype=bool),
                max_points=16,
                max_iterations=1,
                minimum_projection_points=1,
            )
        self.assertFalse(result["valid"])
        self.assertFalse(result["projection_supported"])
        self.assertFalse(result["measurement_available"])
        self.assertEqual(result["motion_explanation_status"], "abstain")
        self.assertEqual(result["geometry_fit_status"], "abstain")
        self.assertEqual(result["projected_points"], 0)
        self.assertEqual(result["projected_weight_fraction"], 0.0)
        self.assertEqual(result["energy"], 4.0)
        self.assertEqual(
            result["motion_explanation_by_interval"][0]["motion_explanation_status"],
            "abstain",
        )
        self.assertEqual(result["protocol"], "candidate-blind-continuous-trajectory-v1")

    def test_longitudinal_residual_penalty_is_zero_at_history_null(self) -> None:
        history = np.asarray([4.0, 4.2, 4.4, 4.6])
        self.assertAlmostEqual(
            _longitudinal_residual_penalty(
                history,
                history,
                maximum_residual_mps=3.0,
                residual_weight=0.02,
                residual_smoothness_weight=0.05,
            ),
            0.0,
        )
        changed = _longitudinal_residual_penalty(
            history + np.asarray([0.0, 0.2, 0.6, 1.0]),
            history,
            maximum_residual_mps=3.0,
            residual_weight=0.02,
            residual_smoothness_weight=0.05,
        )
        self.assertGreater(changed, 0.0)

    def test_history_anchored_decoder_bounds_speed_residual(self) -> None:
        times = np.asarray([0.5, 1.0], dtype=np.float64)
        history_speeds = np.asarray([4.0, 4.5], dtype=np.float64)
        observed = np.zeros((2, 24, 32, 2), dtype=np.float32)
        observed[..., 0] = -2.0
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        camera_to_ego[:3, 3] = [0.0, 0.0, 1.5]
        intrinsics = np.asarray(
            [[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]]
        )
        result = decode_continuous_trajectory(
            observed_flows=observed,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            future_times_s=times,
            roi_mask=np.ones((24, 32), dtype=bool),
            max_points=64,
            max_iterations=2,
            history_speeds_mps=history_speeds,
            history_initial_speed_mps=4.0,
            maximum_speed_residual_mps=0.5,
        )
        predicted = np.asarray([item["q50"] for item in result["speed_support"]])
        self.assertTrue(np.all(np.abs(predicted - history_speeds) <= 0.5 + 1e-6))
        self.assertTrue(result["decoder_parameters"]["history_anchored_speed_residual"])

    def test_kinematic_smoothness_penalty_prefers_constant_controls(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        smooth = integrate_piecewise_controls(
            times, speeds_mps=np.full(4, 4.0), curvatures_1pm=np.full(4, 0.02)
        )
        jagged = integrate_piecewise_controls(
            times, speeds_mps=np.asarray([2.0, 7.0, 2.0, 7.0]),
            curvatures_1pm=np.asarray([-0.12, 0.12, -0.12, 0.12]),
        )
        smooth_penalty = _kinematic_smoothness_penalty(
            smooth, times, speed_weight=1.0, curvature_weight=1.0,
            lateral_acceleration_weight=1.0,
        )
        jagged_penalty = _kinematic_smoothness_penalty(
            jagged, times, speed_weight=1.0, curvature_weight=1.0,
            lateral_acceleration_weight=1.0,
        )
        self.assertLess(smooth_penalty, jagged_penalty)

    def test_road_prior_penalty_distinguishes_road_and_offroad(self) -> None:
        trajectory = np.asarray([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        camera_to_ego[:3, 3] = [0.0, 0.0, 1.5]
        intrinsics = np.asarray(
            [[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]]
        )
        road = np.ones((2, 24, 32), dtype=np.float32)
        empty = np.zeros_like(road)
        road_penalty = _road_prior_penalty(
            trajectory, road, camera_to_ego=camera_to_ego, intrinsics=intrinsics,
            image_size=(32, 24), half_width_m=0.5, lateral_samples=3,
            longitudinal_step_m=0.5,
        )
        offroad_penalty = _road_prior_penalty(
            trajectory, empty, camera_to_ego=camera_to_ego, intrinsics=intrinsics,
            image_size=(32, 24), half_width_m=0.5, lateral_samples=3,
            longitudinal_step_m=0.5,
        )
        self.assertAlmostEqual(road_penalty, 0.0)
        self.assertGreater(offroad_penalty, 0.9)

    def test_road_prior_uses_anchor_mask_for_all_knots(self) -> None:
        trajectory = np.asarray([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        camera_to_ego[:3, 3] = [0.0, 0.0, 1.5]
        intrinsics = np.asarray(
            [[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]]
        )
        masks = np.ones((2, 24, 32), dtype=np.float32)
        masks[1] = 0.0
        penalty = _road_prior_penalty(
            trajectory, masks, camera_to_ego=camera_to_ego, intrinsics=intrinsics,
            image_size=(32, 24), half_width_m=0.5, lateral_samples=3,
            longitudinal_step_m=0.5,
        )
        self.assertAlmostEqual(penalty, 0.0)

    def test_fixed_speed_shape_refinement_preserves_interval_speeds(self) -> None:
        times = np.asarray([0.5, 1.0], dtype=np.float64)
        fixed_speeds = np.asarray([4.0, 6.0], dtype=np.float64)
        observed = np.zeros((2, 24, 32, 2), dtype=np.float32)
        observed[..., 0] = -1.0
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        camera_to_ego[:3, 3] = [0.0, 0.0, 1.5]
        intrinsics = np.asarray(
            [[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]]
        )
        result = decode_continuous_trajectory(
            observed_flows=observed,
            camera_to_ego=camera_to_ego,
            intrinsics=intrinsics,
            future_times_s=times,
            roi_mask=np.ones((24, 32), dtype=bool),
            max_points=64,
            max_iterations=1,
            fixed_speeds_mps=fixed_speeds,
            initial_curvatures_1pm=np.zeros(2),
        )
        trajectory = np.asarray(result["trajectory"], dtype=np.float64)
        distances = np.linalg.norm(
            np.diff(np.vstack([np.zeros((1, 2)), trajectory[:, :2]]), axis=0), axis=1
        )
        np.testing.assert_allclose(distances / np.diff(np.r_[0.0, times]), fixed_speeds)
        self.assertTrue(result["decoder_parameters"]["fixed_speed_shape_refinement"])

    def test_integrates_forward_constant_motion(self) -> None:
        trajectory = integrate_piecewise_controls(
            np.asarray([0.5, 1.0, 1.5, 2.0]),
            speeds_mps=np.full(4, 4.0),
            curvatures_1pm=np.zeros(4),
        )
        self.assertTrue(np.allclose(trajectory[:, 0], [2.0, 4.0, 6.0, 8.0]))
        self.assertTrue(np.allclose(trajectory[:, 1:], 0.0))

    def test_comparison_is_tolerant_and_directional(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        reference = integrate_piecewise_controls(times, speeds_mps=np.full(4, 4.0), curvatures_1pm=np.zeros(4))
        close = reference.copy()
        close[:, 1] += np.linspace(0.0, 0.2, 4)
        result = compare_continuous_trajectory(close, reference, times)
        self.assertGreater(result["soft_compatibility"], 0.5)
        self.assertGreater(result["joint_coverage"], 0.5)
        self.assertGreater(result["mean_heading_cosine"], 0.99)
        self.assertIn("mean_speed_relative_error", result)
        self.assertEqual(result["score_components"], ["lateral", "yaw", "curvature"])
        self.assertFalse(result["speed_scored"])

    def test_speed_can_be_reported_without_changing_primary_error(self) -> None:
        times = np.asarray([0.5, 1.0, 1.5, 2.0])
        reference = integrate_piecewise_controls(times, speeds_mps=np.full(4, 4.0), curvatures_1pm=np.zeros(4))
        slower = integrate_piecewise_controls(times, speeds_mps=np.full(4, 2.0), curvatures_1pm=np.zeros(4))
        directional = compare_continuous_trajectory(slower, reference, times, score_speed=False)
        joint = compare_continuous_trajectory(slower, reference, times, score_speed=True)
        self.assertEqual(directional["score_components"], ["lateral", "yaw", "curvature"])
        self.assertEqual(joint["score_components"][-1], "speed")
        self.assertLess(directional["weighted_mean_joint_error"], joint["weighted_mean_joint_error"])

    def test_temporal_scale_state_is_smooth_and_action_blind(self) -> None:
        times = np.asarray([0.5, 1.0], dtype=np.float64)
        trajectory = integrate_piecewise_controls(
            times, speeds_mps=np.asarray([8.0, 8.0]), curvatures_1pm=np.zeros(2)
        )
        camera_to_ego = np.eye(4, dtype=np.float64)
        camera_to_ego[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        camera_to_ego[:3, 3] = [0.0, 0.0, 1.5]
        intrinsics = np.asarray(
            [[70.0, 0.0, 40.0], [0.0, 70.0, 32.0], [0.0, 0.0, 1.0]]
        )
        from iac_new.trajectory_decode import _sample_pixels, _sparse_predicted_flows

        yy, xx = np.indices((64, 80), dtype=np.float64)
        points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
        predicted, valid = _sparse_predicted_flows(
            trajectory, camera_to_ego, intrinsics, points,
            image_size=(80, 64), depths_m=None,
        )
        observed = (predicted * 1.2).reshape(2, 64, 80, 2).astype(np.float32)
        weights = np.where(valid.reshape(2, 64, 80), 1.0, 0.0).astype(np.float32)
        result = estimate_temporal_flow_scale_state(
            trajectory, observed, weights,
            camera_to_ego=camera_to_ego, intrinsics=intrinsics,
            image_size=(80, 64), max_points=200,
        )
        self.assertTrue(result["available"])
        estimates = [row["scale_posterior"]["q50"] for row in result["rows"]]
        self.assertTrue(all(value > 1.0 for value in estimates))
        self.assertLess(abs(estimates[1] - estimates[0]), 0.25 + 1e-9)
        self.assertFalse(result["action_waypoint_used"])
        rejected = estimate_temporal_flow_scale_state(
            trajectory, observed, weights,
            camera_to_ego=camera_to_ego, intrinsics=intrinsics,
            image_size=(80, 64), max_points=200, max_log_innovation=0.05,
        )
        self.assertFalse(rejected["available"])
        self.assertTrue(all(row["innovation_accepted"] is False for row in rejected["rows"]))


if __name__ == "__main__":
    unittest.main()
