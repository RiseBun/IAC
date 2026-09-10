from __future__ import annotations

import unittest

import numpy as np

from tools.build_radial_progress_control import _warp_radial


class RadialProgressControlTest(unittest.TestCase):
    def test_identity_scale_is_exact(self) -> None:
        frame = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
        np.testing.assert_array_equal(_warp_radial(frame, 1.0), frame)

    def test_expansion_keeps_center_fixed(self) -> None:
        frame = np.zeros((9, 9, 3), dtype=np.uint8)
        frame[4, 4] = 255
        warped = _warp_radial(frame, 1.2)
        np.testing.assert_array_equal(warped[4, 4], np.asarray([255, 255, 255]))


if __name__ == "__main__":
    unittest.main()
