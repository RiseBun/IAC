import unittest

from tools.export_gs_reference_release import export
from tools.verify_gs_reference_release import verify


class VerifyGsReferenceReleaseTest(unittest.TestCase):
    def test_verifies_descriptor_only_release(self) -> None:
        rows, manifest = export(
            [
                {"source_key": "s0", "interval_index": 0, "horizontal_flow_center": 0.1},
                {"source_key": "s0", "interval_index": 1, "horizontal_flow_center": 0.2},
            ],
            salt="secret",
            release_id="r1",
        )
        result = verify(rows, manifest)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["source_count"], 1)

    def test_rejects_private_fields_and_raw_keys(self) -> None:
        rows, manifest = export(
            [{"source_key": "s0", "interval_index": 0, "curl": 0.1}],
            salt="secret",
            release_id="r1",
        )
        rows[0]["future_image_path"] = "/private/image.png"
        with self.assertRaisesRegex(ValueError, "private_or_unknown_fields_present"):
            verify(rows, manifest)
        rows, manifest = export(
            [{"source_key": "s0", "interval_index": 0, "curl": 0.1}],
            salt="secret",
            release_id="r1",
        )
        rows[0]["source_key"] = "raw-source"
        with self.assertRaisesRegex(ValueError, "source_key_is_not_hmac_pseudonym"):
            verify(rows, manifest)


if __name__ == "__main__":
    unittest.main()
