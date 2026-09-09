import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.road_structure import estimate_focus_of_expansion, flow_structure_profile


class FlowStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 60
        self.width = 80
        self.intrinsics = np.asarray(
            [[100.0, 0.0, 40.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.roi = np.ones((self.height, self.width), dtype=bool)
        self.weights = np.ones((1, self.height, self.width), dtype=np.float32)

    def normalized_grid(self) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.indices((self.height, self.width), dtype=np.float64)
        return (xx - 40.0) / 100.0, (yy - 30.0) / 100.0

    def test_foe_recovers_known_radial_center_with_outliers(self) -> None:
        x, y = self.normalized_grid()
        foe = np.asarray([0.08, -0.04])
        flow = np.zeros((1, self.height, self.width, 2), dtype=np.float32)
        flow[0, ..., 0] = 12.0 * (x - foe[0])
        flow[0, ..., 1] = 12.0 * (y - foe[1])
        flow[0, ::8, ::8] = np.asarray([25.0, -18.0])
        result = estimate_focus_of_expansion(
            flow,
            self.weights,
            self.roi,
            intrinsics=self.intrinsics,
            min_flow_px=0.1,
        )
        self.assertTrue(result["valid"])
        np.testing.assert_allclose(result["foe_normalized_xy"], foe, atol=0.015)
        self.assertGreater(result["confidence"], 0.2)

    def test_profile_recovers_affine_divergence_and_curl(self) -> None:
        x, y = self.normalized_grid()
        expansion = 0.06
        rotation = -0.04
        du = expansion * x - rotation * y + 0.02
        dv = rotation * x + expansion * y - 0.01
        flow = np.zeros((1, self.height, self.width, 2), dtype=np.float32)
        flow[0, ..., 0] = du * 100.0
        flow[0, ..., 1] = dv * 100.0
        result = flow_structure_profile(
            flow,
            self.weights,
            self.roi,
            self.intrinsics,
            min_flow_px=0.1,
            min_points=50,
        )
        row = result["rows"][0]
        self.assertTrue(row["input_available"])
        self.assertTrue(row["affine_valid"])
        self.assertAlmostEqual(row["divergence"], 2.0 * expansion, places=4)
        self.assertAlmostEqual(row["curl"], 2.0 * rotation, places=4)
        self.assertGreater(row["affine_explained_fraction"], 0.99)
        self.assertFalse(result["metric_reconstruction_used"])
        self.assertFalse(result["candidate_bank_used"])

    def test_profile_abstains_when_input_support_is_sparse(self) -> None:
        flow = np.zeros((2, self.height, self.width, 2), dtype=np.float32)
        flow[:, :2, :2, 0] = 1.0
        result = flow_structure_profile(
            flow,
            np.ones((2, self.height, self.width), dtype=np.float32),
            self.roi,
            self.intrinsics,
            min_points=20,
        )
        self.assertEqual(result["status"], "abstain")
        self.assertFalse(result["measurement_available"])
        self.assertTrue(all(not row["input_available"] for row in result["rows"]))

    def test_pre_resized_frames_keep_calibration_coordinate_system(self) -> None:
        intrinsics = np.asarray(
            [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            self.assertTrue(
                cv2.imwrite(str(image_path), np.zeros((256, 448, 3), dtype=np.uint8))
            )
            _, scaled, source_size = RaftFlowExtractor._read_images(
                [str(image_path)],
                intrinsics,
                np.empty(0, dtype=np.float64),
                (448, 256),
                intrinsics_source_size=(1920, 1080),
            )
        self.assertEqual(source_size, (448, 256))
        np.testing.assert_allclose(scaled[0], [360.5, 0.0, 224.0])
        np.testing.assert_allclose(scaled[1], [0.0, 366.2222222222, 132.7407407407])


if __name__ == "__main__":
    unittest.main()
