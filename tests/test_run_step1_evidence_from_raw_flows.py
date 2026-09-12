import unittest

from tools.run_step1_evidence_from_raw_flows import run


class RawEvidenceRunnerTest(unittest.TestCase):
    def test_empty_archive_is_fail_closed(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            report = run(Path(directory))
            self.assertEqual(report["branch_count"], 0)
            self.assertEqual(report["pair_count"], 0)
            self.assertIsNone(report["temporal_scored_fraction"])
            self.assertEqual(report["grounding_status"], "unavailable_without_external_reference_profile")


if __name__ == "__main__":
    unittest.main()
