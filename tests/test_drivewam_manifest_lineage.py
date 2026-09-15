import pickle
import tempfile
import unittest
from pathlib import Path

from reproduction.drivewam.build_ccfc_manifest import _source_key_from_sample


class DriveWamManifestLineageTest(unittest.TestCase):
    def test_source_identity_comes_from_consumed_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pkl"
            with path.open("wb") as handle:
                pickle.dump({"metadata": {"source_key": "immutable-source"}}, handle)
            self.assertEqual(_source_key_from_sample(path), "immutable-source")

    def test_missing_source_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pkl"
            with path.open("wb") as handle:
                pickle.dump({"metadata": {}}, handle)
            with self.assertRaisesRegex(ValueError, "missing immutable metadata.source_key"):
                _source_key_from_sample(path)


if __name__ == "__main__":
    unittest.main()
