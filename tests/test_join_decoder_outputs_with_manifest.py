import unittest

from scripts.join_decoder_outputs_with_manifest import join_rows


class JoinDecoderOutputsWithManifestTest(unittest.TestCase):
    def test_joins_by_sample_id_and_prefers_decoder_measurements(self) -> None:
        rows = join_rows(
            [{"sample_id": "b", "branch_role": "right", "valid": False},
             {"sample_id": "a", "branch_role": "left", "valid": False}],
            [{"sample_id": "a", "valid": True},
             {"sample_id": "b", "valid": True}],
        )
        self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])
        self.assertEqual([row["branch_role"] for row in rows], ["left", "right"])
        self.assertTrue(all(row["valid"] for row in rows))

    def test_rejects_incomplete_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing 1"):
            join_rows(
                [{"sample_id": "a"}, {"sample_id": "b"}],
                [{"sample_id": "a"}],
            )


if __name__ == "__main__":
    unittest.main()
