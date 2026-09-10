import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from iac_new.flow import RaftFlowExtractor
from iac_new.image_geometry import validate_image_geometry_adapter


def _adapter(frame_sizes):
    return {
        "schema": "iac-image-geometry-v1",
        "adapter_id": "test-direct-resize-v1",
        "calibration_source_size": [192, 108],
        "frame_groups": [
            {
                "indices": [index],
                "operation": "identity" if size == (192, 108) else "direct_resize",
                "frame_size": list(size),
            }
            for index, size in enumerate(frame_sizes)
        ],
    }


class ImageGeometryAdapterTest(unittest.TestCase):
    def test_mixed_direct_resize_maps_to_one_canonical_camera(self) -> None:
        raw = _adapter([(192, 108), (102, 51)])
        adapter = validate_image_geometry_adapter(
            raw, frame_count=2, intrinsics_source_size=(192, 108)
        )
        intrinsics = np.asarray(
            [[150.0, 0.0, 96.0], [0.0, 150.0, 56.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        with TemporaryDirectory() as directory:
            paths = []
            for index, (width, height) in enumerate(((192, 108), (102, 51))):
                path = Path(directory) / f"frame_{index}.png"
                self.assertTrue(cv2.imwrite(str(path), np.zeros((height, width, 3), np.uint8)))
                paths.append(str(path))
            _, scaled, source_size = RaftFlowExtractor._read_images(
                paths,
                intrinsics,
                np.empty(0, dtype=np.float64),
                (96, 54),
                image_geometry_adapter=adapter,
            )
        self.assertEqual(source_size, (192, 108))
        np.testing.assert_allclose(
            scaled,
            [[75.0, 0.0, 48.0], [0.0, 75.0, 28.0], [0.0, 0.0, 1.0]],
        )

    def test_declared_frame_size_is_checked_against_file(self) -> None:
        adapter = validate_image_geometry_adapter(
            _adapter([(192, 108)]), frame_count=1, intrinsics_source_size=(192, 108)
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.png"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((54, 96, 3), np.uint8)))
            with self.assertRaisesRegex(ValueError, "does not match image geometry adapter"):
                RaftFlowExtractor._read_images(
                    [str(path)],
                    np.eye(3),
                    np.empty(0),
                    (96, 54),
                    image_geometry_adapter=adapter,
                )

    def test_adapter_must_cover_every_frame_exactly_once(self) -> None:
        raw = _adapter([(192, 108)])
        with self.assertRaisesRegex(ValueError, "does not cover frame indices"):
            validate_image_geometry_adapter(
                raw, frame_count=2, intrinsics_source_size=(192, 108)
            )

    def test_adapter_calibration_size_must_match_intrinsics(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match intrinsics_source_size"):
            validate_image_geometry_adapter(
                _adapter([(192, 108)]),
                frame_count=1,
                intrinsics_source_size=(384, 216),
            )


if __name__ == "__main__":
    unittest.main()
