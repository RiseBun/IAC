from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.build_pair_contrast_attenuation import _blend, build_attenuated_rows


class PairContrastAttenuationTest(unittest.TestCase):
    def test_blend_endpoints(self) -> None:
        left = np.full((2, 3, 3), 20, dtype=np.uint8)
        right = np.full((2, 3, 3), 100, dtype=np.uint8)

        left_zero, right_zero = _blend(left, right, 0.0)
        np.testing.assert_array_equal(left_zero, left)
        np.testing.assert_array_equal(right_zero, right)

        left_full, right_full = _blend(left, right, 1.0)
        np.testing.assert_array_equal(left_full, right_full)
        np.testing.assert_array_equal(left_full, np.full_like(left, 60))

    def test_full_attenuation_uses_byte_identical_future_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left_path = root / "left.png"
            right_path = root / "right.png"
            cv2.imwrite(str(left_path), np.full((4, 5, 3), 10, dtype=np.uint8))
            cv2.imwrite(str(right_path), np.full((4, 5, 3), 30, dtype=np.uint8))
            rows = [
                {
                    "source_key": "sample",
                    "branch_role": role,
                    "future_frame_paths": [str(path)],
                }
                for role, path in (("left", left_path), ("right", right_path))
            ]

            output = build_attenuated_rows(
                rows,
                reference_source_keys={"sample"},
                strength=1.0,
                image_root=root / "output",
            )

            self.assertEqual(len(output), 2)
            self.assertEqual(
                output[0]["future_frame_paths"],
                output[1]["future_frame_paths"],
            )
            self.assertTrue(Path(output[0]["future_frame_paths"][0]).is_file())
            self.assertTrue(
                output[0]["metadata"]["controlled_degradation"]["validation_only"]
            )


if __name__ == "__main__":
    unittest.main()
