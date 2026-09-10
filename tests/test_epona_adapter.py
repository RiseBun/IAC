import unittest

import numpy as np

from tools.build_epona_flow_structure_manifest import (
    _integrate_epona_controls,
    _source_start,
)


class EponaAdapterTest(unittest.TestCase):
    def test_right_positive_relative_controls_become_left_positive_se2(self) -> None:
        controls = np.asarray([[1.0, -0.25]] * 4, dtype=np.float64)
        trajectory = _integrate_epona_controls(controls, np.zeros((4, 1)))
        np.testing.assert_allclose(trajectory[-1], [4.0, 1.0, 0.0])

    def test_yaw_is_accumulated_in_radians(self) -> None:
        controls = np.zeros((4, 2), dtype=np.float64)
        trajectory = _integrate_epona_controls(controls, np.full((4, 1), 2.0))
        self.assertAlmostEqual(trajectory[-1, 2], np.deg2rad(8.0))

    def test_source_start_uses_persisted_generator_window(self) -> None:
        self.assertEqual(
            _source_start({"sample_index": 3, "source_start_index": 54}),
            54,
        )
        self.assertEqual(
            _source_start({"sample_index": 3, "source_key": "epona_common_random:54"}),
            54,
        )
        self.assertEqual(_source_start({"sample_index": 3}), 15)


if __name__ == "__main__":
    unittest.main()
