from __future__ import annotations

import unittest
import pickle
import tempfile
from pathlib import Path

import numpy as np

from reproduction.drivewam.build_inputs import _model_image_paths, _model_image_times
from reproduction.drivewam.build_ccfc_manifest import _intrinsics_source_size
from reproduction.drivewam.build_ccfc_manifest import _temporal_contract
from reproduction.drivewam.repair_temporal_input_contract import repair_sample


class DriveWamInputContractTest(unittest.TestCase):
    def test_builder_keeps_current_and_future_images_only(self) -> None:
        history = ["h0", "h1", "h2", "current"]
        future = [f"f{i}" for i in range(8)]
        self.assertEqual(_model_image_paths(history, future), ["current", *future])
        self.assertEqual(
            _model_image_times([0.5 * (i + 1) for i in range(8)]),
            [0.0, *[0.5 * (i + 1) for i in range(8)]],
        )

    def test_builder_rejects_a_shifted_future_horizon(self) -> None:
        with self.assertRaises(ValueError):
            _model_image_times([1.0 + 0.5 * i for i in range(8)])

    def test_geometry_size_requires_manifest_or_explicit_adapter_value(self) -> None:
        row = {"source_key": "sample"}
        with self.assertRaises(ValueError):
            _intrinsics_source_size(row)
        self.assertEqual(_intrinsics_source_size(row, [1920, 1080]), [1920, 1080])

    def test_manifest_builder_rejects_legacy_shifted_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.pkl"
            with source.open("wb") as handle:
                pickle.dump(
                    {
                        "images": np.zeros((12, 2, 3, 3), dtype=np.uint8),
                        "metadata": {"image_paths": [str(i) for i in range(12)]},
                    },
                    handle,
                )
            with self.assertRaises(ValueError):
                _temporal_contract({"source_sample": str(source)})

    def test_legacy_pickle_is_repaired_without_mutating_source(self) -> None:
        images = np.arange(12 * 2 * 3 * 3, dtype=np.uint8).reshape(12, 2, 3, 3)
        sample = {
            "images": images,
            "history_poses": np.zeros((4, 3), dtype=np.float32),
            "future_trajectory": [{"pose": [float(i), 0.0, 0.0]} for i in range(8)],
            "metadata": {
                "image_paths": [f"frame-{i}" for i in range(12)],
                "future_times_s": [0.5 * (i + 1) for i in range(8)],
            },
        }
        repaired = repair_sample(sample, source="legacy.pkl")
        np.testing.assert_array_equal(repaired["images"], images[3:])
        np.testing.assert_array_equal(sample["images"], images)
        self.assertEqual(
            repaired["metadata"]["image_paths"],
            [f"frame-{i}" for i in range(3, 12)],
        )
        self.assertEqual(
            repaired["metadata"]["input_image_times_s"],
            [0.0, *[0.5 * (i + 1) for i in range(8)]],
        )


if __name__ == "__main__":
    unittest.main()
