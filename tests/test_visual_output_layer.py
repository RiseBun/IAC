import unittest

from iac_new.visual_output_layer import classify_distance_bin, classify_yaw, score_longitudinal


class VisualOutputLayerTest(unittest.TestCase):
    def test_yaw_labels_are_frozen(self) -> None:
        self.assertEqual(classify_yaw(0.0), "straight")
        self.assertEqual(classify_yaw(0.02), "right")
        self.assertEqual(classify_yaw(-0.02), "left")

    def test_distance_is_coarse_and_nonnegative(self) -> None:
        self.assertEqual(classify_distance_bin(0.2), "stop")
        self.assertEqual(classify_distance_bin(4.0), "short")
        self.assertEqual(classify_distance_bin(-1.0), "uncertain")

    def test_large_but_explicit_tolerance_is_accepted(self) -> None:
        result = score_longitudinal(
            4.0,
            6.0,
            geometry={"matches": 100, "inlier_fraction": 0.25, "median_reprojection_error_px": 1.5},
        )
        self.assertTrue(result["acceptable"])
        self.assertTrue(result["within_tolerance"])
        self.assertEqual(result["status"], "scored")

    def test_weak_geometry_abstains_even_with_tolerance(self) -> None:
        result = score_longitudinal(
            4.0,
            5.0,
            geometry={"matches": 8, "inlier_fraction": 0.02, "median_reprojection_error_px": 1.0},
        )
        self.assertFalse(result["acceptable"])
        self.assertEqual(result["status"], "uncertain")


if __name__ == "__main__":
    unittest.main()
