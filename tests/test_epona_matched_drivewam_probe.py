from __future__ import annotations

import unittest

import numpy as np

from scripts.run_epona_matched_drivewam_probe import (
    _cumulative_to_epona_controls,
    _trajectory_matrix,
)
from tools.build_epona_flow_structure_manifest import _integrate_epona_controls


class EponaMatchedDriveWAMProbeTest(unittest.TestCase):
    def test_accepts_drivewam_channel_first_action_tensor(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
        actual = _trajectory_matrix(source)
        self.assertEqual(actual.shape, (8, 3))
        np.testing.assert_array_equal(actual, source[0].T)

    def test_epona_control_adapter_round_trips_standard_se2(self) -> None:
        trajectory = np.asarray(
            [
                [0.5, 0.02, 0.01],
                [1.1, 0.08, 0.03],
                [1.8, 0.18, 0.07],
                [2.6, 0.35, 0.12],
                [3.5, 0.60, 0.18],
                [4.5, 0.95, 0.25],
                [5.6, 1.40, 0.33],
                [6.8, 1.95, 0.42],
            ],
            dtype=np.float64,
        )
        xy, yaw_deg = _cumulative_to_epona_controls(trajectory)
        reconstructed = _integrate_epona_controls(xy, yaw_deg)
        np.testing.assert_allclose(reconstructed, trajectory, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
