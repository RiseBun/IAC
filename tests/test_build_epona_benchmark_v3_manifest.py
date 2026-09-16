import math
import unittest

from tools.build_epona_benchmark_v3_manifest import adapt_row, compose_relative_trajectory


class EponaBenchmarkAdapterTest(unittest.TestCase):
    def test_relative_yaw_is_composed_and_sampled_at_one_hz(self):
        row = {
            "source_key": "source",
            "counterfactual_group_id": "source",
            "branch_mode": "left",
            "history_images": [f"h{i}.png" for i in range(10)],
            "future_images": [f"f{i}.png" for i in range(8)],
            "action_trajectory": [[1.0, 0.0, 1.0] for _ in range(8)],
            "randomness_contract": {"common_random_numbers": True, "seed": 9},
            "camera_intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "camera_to_ego": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        }
        adapted = adapt_row(row)
        self.assertIsNotNone(adapted)
        self.assertEqual(adapted["history_frame_paths"], ["h6.png", "h7.png", "h8.png", "h9.png"])
        self.assertEqual(adapted["future_frame_paths"], ["f1.png", "f3.png", "f5.png", "f7.png"])
        self.assertAlmostEqual(adapted["action_trajectory"][-1][2], math.radians(8.0))

    def test_logged_branch_is_not_part_of_counterfactual_pair(self):
        self.assertIsNone(adapt_row({"branch_mode": "logged"}))

    def test_composition_rotates_later_translation(self):
        poses = compose_relative_trajectory([[1.0, 0.0, 90.0], [1.0, 0.0, 0.0]])
        self.assertAlmostEqual(poses[-1][0], 1.0)
        self.assertAlmostEqual(poses[-1][1], 1.0)


if __name__ == "__main__":
    unittest.main()
