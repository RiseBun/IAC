import unittest

from scripts.audit_benchmark_manifest import audit_public


class PublicBenchmarkAuditTest(unittest.TestCase):
    def test_redacted_public_row_passes(self) -> None:
        row = {
            "benchmark_id": "benchmark-00000",
            "sample_id": "sample",
            "source_key": "source",
            "scene_group": "scene",
            "stratum": "lateral_turn",
            "history_frame_ids": ["h0.jpg", "h1.jpg", "h2.jpg", "h3.jpg"],
            "future_times_s": [1.0, 2.0, 3.0, 4.0],
        }
        self.assertTrue(audit_public([row])["public_manifest_ready"])

    def test_private_ground_truth_is_rejected(self) -> None:
        row = {
            "benchmark_id": "benchmark-00000",
            "sample_id": "sample",
            "source_key": "source",
            "scene_group": "scene",
            "stratum": "lateral_turn",
            "history_frame_ids": ["h0.jpg", "h1.jpg", "h2.jpg", "h3.jpg"],
            "future_times_s": [1.0, 2.0, 3.0, 4.0],
            "trajectory": [[0.0, 0.0, 0.0]],
        }
        self.assertFalse(audit_public([row])["public_manifest_ready"])


if __name__ == "__main__":
    unittest.main()
