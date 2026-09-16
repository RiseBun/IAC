import unittest

from scripts.prepare_worlddrive_pair_manifest import adapt_rows


class PrepareWorldDrivePairManifestTest(unittest.TestCase):
    def test_native_action_is_never_gt(self) -> None:
        base = {
            "source_key": "source-1",
            "sample_id": "source-1::command_0",
            "command_index": 0,
            "history_frame_paths": ["h0", "h1", "h2", "h3"],
            "future_frame_paths": ["f0", "f1", "f2", "f3"],
            "future_times_s": [1.0, 2.0, 3.0, 4.0],
            "action_trajectory": [[float(index), 0.0, 0.0] for index in range(4)],
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

    def test_selects_exact_seconds_from_native_half_second_outputs(self) -> None:
        base = {
            "source_key": "source-1",
            "sample_id": "source-1::command_0",
            "command_index": 0,
            "history_frame_paths": ["h0", "h1", "h2", "h3"],
            "future_frame_paths": [f"f{index}" for index in range(8)],
            "future_times_s": [0.5 * (index + 1) for index in range(8)],
            "action_trajectory": [[float(index), 0.0, 0.0] for index in range(8)],
        }
        right = dict(base, sample_id="source-1::command_2", command_index=2)
        rows = adapt_rows([base, right])
        self.assertEqual(rows[0]["future_frame_paths"], ["f1", "f3", "f5", "f7"])
        self.assertEqual([item[0] for item in rows[0]["action_trajectory"]], [1.0, 3.0, 5.0, 7.0])
        self.assertEqual(rows[0]["future_times_s"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(rows[0]["future_count"], 4)

    def test_native_single_requires_explicit_opt_in(self) -> None:
        row = {
            "source_key": "source-1",
            "sample_id": "source-1::native",
            "branch_id": "native",
            "history_frame_paths": ["h0", "h1", "h2", "h3"],
            "future_frame_paths": [f"f{index}" for index in range(8)],
            "future_times_s": [0.5 * (index + 1) for index in range(8)],
            "action_trajectory": [[float(index), 0.0, 0.0] for index in range(8)],
        }
        self.assertEqual(adapt_rows([row]), [])
        selected = adapt_rows([row], allow_native_single=True)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["branch_role"], "native")


if __name__ == "__main__":
    unittest.main()
