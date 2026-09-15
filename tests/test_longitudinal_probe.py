import unittest

import numpy as np

from iac_new.longitudinal_probe import (
    agreement_gate,
    cumulative_progress_from_intervals,
    estimate_homography_from_correspondences,
    estimate_planar_translation,
    estimate_translation_with_known_rotation,
)


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))].T
    projected = homography @ homogeneous
    return (projected[:2] / projected[2:3]).T


class LongitudinalProbeTest(unittest.TestCase):
    def test_planar_homography_recovers_metric_camera_displacement(self) -> None:
        K = np.asarray([[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]])
        R = np.eye(3, dtype=np.float64)
        normal = np.asarray([0.0, 0.0, 1.0])
        distance = 1.5
        # This follows the module's H = K(R - t n^T / d)K^-1 convention.
        t_next = np.asarray([-0.40, 0.0, 0.0])
        H = K @ (R - np.outer(t_next, normal) / distance) @ np.linalg.inv(K)
        H /= H[2, 2]
        points = np.asarray(
            [[20.0, 20.0], [60.0, 25.0], [100.0, 30.0], [30.0, 70.0], [90.0, 75.0], [140.0, 80.0]],
            dtype=np.float64,
        )
        next_points = _project(H, points)
        homography = estimate_homography_from_correspondences(points, next_points)
        self.assertTrue(homography["available"])
        estimate = estimate_planar_translation(
            np.asarray(homography["homography"]), K, R, normal, distance,
        )
        self.assertTrue(estimate["available"])
        # Camera-centre displacement is -R.T @ t_next, hence +0.40 m.
        self.assertAlmostEqual(estimate["longitudinal_m"], 0.40, places=3)

    def test_agreement_gate_abstains_on_conflicting_scales(self) -> None:
        planar = {"available": True, "longitudinal_m": 1.0}
        pnp = {"available": True, "longitudinal_m": 2.0}
        result = agreement_gate(planar, pnp, max_relative_disagreement=0.35)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["reason"], "scale_disagreement")

    def test_progress_uses_only_observed_frame_intervals(self) -> None:
        result = cumulative_progress_from_intervals(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, 2.0, 3.0, 4.0]),
        )
        self.assertEqual(result["endpoint_times_s"], [2.0, 3.0, 4.0])
        self.assertEqual(result["cumulative_progress_m"], [1.0, 3.0, 6.0])
        self.assertNotIn(0.0, result["endpoint_times_s"])

    def test_known_rotation_recovers_translation_with_outliers(self) -> None:
        rng = np.random.default_rng(7)
        K = np.asarray([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        points = np.c_[rng.uniform(-3.0, 3.0, 80), rng.uniform(-1.5, 1.5, 80), rng.uniform(8.0, 30.0, 80)]
        angle = 0.03
        rotation = np.asarray([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
        translation = np.asarray([-0.15, 0.02, -2.0])
        moved = (rotation @ points.T).T + translation
        projected = (K @ moved.T).T
        image_points = projected[:, :2] / projected[:, 2:3]
        image_points[:15] += rng.uniform(30.0, 80.0, (15, 2))
        result = estimate_translation_with_known_rotation(points, image_points, K, rotation)
        self.assertTrue(result["available"])
        np.testing.assert_allclose(result["translation_next_camera_m"], translation, atol=1e-5)
        self.assertGreaterEqual(result["num_inliers"], 65)


if __name__ == "__main__":
    unittest.main()
