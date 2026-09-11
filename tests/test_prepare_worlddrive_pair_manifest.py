import unittest

from scripts.prepare_worlddrive_pair_manifest import adapt_rows


class PrepareWorldDrivePairManifestTest(unittest.TestCase):
    def test_native_action_is_never_gt(self) -> None:
        base = {
            "source_key": "source-1",
            "sample_id": "source-1::command_0",
            "command_index": 0,
            "history_frame_paths": ["h0", "h1"],
            "future_frame_paths": ["f0", "f1"],
            "future_times_s": [1.0, 2.0],
            "gt_candidate_id": "worlddrive_native_action",
            "action_trajectory_source": "wam_action_head",
            "future_images_source": "wam_generated",
        }
        right = dict(base)
        right.update({"sample_id": "source-1::command_2", "command_index": 2})
        rows = adapt_rows([base, right])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["gt_candidate_id"] is None for row in rows))
        self.assertTrue(all(row["action_trajectory_source"] == "wam_action_head" for row in rows))
        self.assertTrue(all(row["future_images_source"] == "wam_generated" for row in rows))


if __name__ == "__main__":
    unittest.main()
