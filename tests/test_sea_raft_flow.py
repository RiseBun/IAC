import unittest

import numpy as np

from iac_new.sea_raft_flow import SeaRaftFlowExtractor


class SeaRaftFlowContractTest(unittest.TestCase):
    def test_observe_applies_explicit_calibration_size(self) -> None:
        extractor = SeaRaftFlowExtractor.__new__(SeaRaftFlowExtractor)
        extractor.use_forward_backward = True
        extractor.fb_abs_threshold_px = 1.5
        extractor.fb_relative_threshold = 0.05

        images = [np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(3)]
        extractor._read_images = lambda *args, **kwargs: (images, np.eye(3), (448, 256))
        extractor._infer_pairs = lambda first, second, **kwargs: (
            np.zeros((len(first), 8, 12, 2), dtype=np.float32), None
        )
        intrinsics = np.asarray([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
        observation = extractor.observe(
            ["a", "b", "c"], intrinsics, np.asarray([]), (448, 256),
            inference_size=(12, 8), intrinsics_source_size=(1920, 1080),
        )

        self.assertEqual(observation.forward.shape, (2, 256, 448, 2))
        self.assertAlmostEqual(observation.intrinsics[0, 2], 224.0)
        self.assertAlmostEqual(observation.intrinsics[1, 2], 560.0 * 256.0 / 1080.0)


if __name__ == "__main__":
    unittest.main()
