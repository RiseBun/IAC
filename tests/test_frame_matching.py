import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from iac_new.frame_matching import load_manifest, scaled_intrinsics


class FrameMatchingTest(unittest.TestCase):
    def test_load_manifest_supports_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps({"sample_id": "a"}) + "\n", encoding="utf-8")
            self.assertEqual(load_manifest(path), [{"sample_id": "a"}])

    def test_intrinsics_are_scaled_explicitly(self):
        record = {
            "intrinsics": [[1000, 0, 500], [0, 1000, 250], [0, 0, 1]],
            "intrinsics_source_size": [1000, 500],
        }
        image = np.zeros((250, 500, 3), dtype=np.uint8)
        result = scaled_intrinsics(record, image)
        np.testing.assert_allclose(result, [[500, 0, 250], [0, 500, 125], [0, 0, 1]])

    def test_missing_intrinsics_fail_closed(self):
        with self.assertRaises(ValueError):
            scaled_intrinsics({}, np.zeros((10, 10, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
