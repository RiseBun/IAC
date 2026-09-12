import unittest

from tools.export_gs_reference_release import export


class ExportGsReferenceReleaseTest(unittest.TestCase):
    def test_pseudonymizes_and_drops_private_fields(self) -> None:
        rows, manifest = export(
            [{
                "source_key": "private-scene-1",
                "interval_index": 0,
                "stratum": "lateral_turn",
                "median_flow_magnitude_px": 2.0,
                "horizontal_flow_center": 0.1,
                "vertical_flow_center": 0.2,
                "divergence": 0.3,
                "curl": 0.4,
                "future_image_path": "/private/secret.png",
                "ego_state": [1, 2, 3],
            }],
            salt="operator-secret",
            release_id="gs-public-v1",
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["source_key"].startswith("gs:"))
        self.assertNotIn("private-scene-1", rows[0]["source_key"])
        self.assertNotIn("future_image_path", rows[0])
        self.assertNotIn("ego_state", rows[0])
        self.assertEqual(manifest["source_key_scheme"], "hmac_sha256_truncated_128bit")

    def test_duplicate_and_nonfinite_values_fail_closed(self) -> None:
        base = {"source_key": "s", "interval_index": 0, "horizontal_flow_center": 0.1}
        with self.assertRaises(ValueError):
            export([base, dict(base)], salt="s", release_id="r")
        with self.assertRaises(ValueError):
            export([{**base, "curl": float("nan")}], salt="s", release_id="r")


if __name__ == "__main__":
    unittest.main()
