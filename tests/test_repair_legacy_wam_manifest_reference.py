import unittest

from scripts.repair_legacy_wam_manifest_reference import repair_row


class RepairLegacyWamManifestReferenceTest(unittest.TestCase):
    def test_removes_false_gt_and_labels_action_source(self) -> None:
        repaired = repair_row({
            "sample_id": "sample",
            "future_images_source": "wam_generated",
            "action_trajectory_source": "wam_action_head",
            "gt_candidate_id": "wam_action_head",
            "candidates": [
                {"candidate_id": "wam_action_head", "trajectory": [[1.0, 0.0, 0.0]]},
                {"candidate_id": "zero_null", "trajectory": [[0.0, 0.0, 0.0]]},
            ],
        })
        self.assertIsNone(repaired["gt_candidate_id"])
        self.assertEqual(
            repaired["candidates"][0]["trajectory_source"], "wam_action_head"
        )
        self.assertEqual(
            repaired["candidates"][0]["support_label"],
            "independent_action_head_reference",
        )

    def test_rejects_rows_without_the_known_legacy_defect(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected legacy"):
            repair_row({
                "sample_id": "sample",
                "future_images_source": "wam_generated",
                "action_trajectory_source": "wam_action_head",
                "gt_candidate_id": None,
                "candidates": [],
            })

    def test_accepts_the_logged_real_counterpart_format(self) -> None:
        repaired = repair_row({
            "sample_id": "real",
            "future_images_source": "logged_real",
            "action_trajectory_source": "wam_action_head",
            "gt_candidate_id": "wam_action_head",
            "candidates": [
                {"candidate_id": "wam_action_head", "trajectory": [[1.0, 0.0, 0.0]]},
                {"candidate_id": "zero_null", "trajectory": [[0.0, 0.0, 0.0]]},
            ],
        })
        self.assertIsNone(repaired["gt_candidate_id"])

    def test_attaches_explicit_intrinsics_source_size(self) -> None:
        repaired = repair_row({
            "sample_id": "real",
            "future_images_source": "logged_real",
            "action_trajectory_source": "wam_action_head",
            "gt_candidate_id": "wam_action_head",
            "candidates": [
                {"candidate_id": "wam_action_head", "trajectory": [[1.0, 0.0, 0.0]]},
            ],
        }, intrinsics_source_size=(1920, 1080))
        self.assertEqual(repaired["intrinsics_source_size"], [1920, 1080])

    def test_rejects_conflicting_intrinsics_source_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            repair_row({
                "sample_id": "real",
                "future_images_source": "logged_real",
                "action_trajectory_source": "wam_action_head",
                "gt_candidate_id": "wam_action_head",
                "intrinsics_source_size": [448, 256],
                "candidates": [
                    {"candidate_id": "wam_action_head", "trajectory": [[1.0, 0.0, 0.0]]},
                ],
            }, intrinsics_source_size=(1920, 1080))


if __name__ == "__main__":
    unittest.main()
