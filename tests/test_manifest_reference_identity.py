import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_wam_level1_continuous_manifest import build_manifest
from scripts.evaluate_continuous_motion_alignment import _reference
from tools.audit_decoder_calibration import logged_gt_from_source_pickle


class ManifestReferenceIdentityTest(unittest.TestCase):
    def test_wam_action_head_is_not_labeled_as_gt(self) -> None:
        trajectory = [[float(index), 0.0, 0.0] for index in range(1, 5)]
        base = {
            "sample_id": "sample",
            "source_key": "source",
            "history_frame_paths": [f"h{index}.png" for index in range(4)],
            "future_times_s": [1.0, 2.0, 3.0, 4.0],
            "intrinsics_source_size": [1920, 1080],
            "candidates": [
                {"candidate_id": "a", "trajectory": trajectory},
                {"candidate_id": "b", "trajectory": trajectory},
            ],
        }
        generated = {
            "branch_id": "left",
            "source_key": "source",
            "future_images_source": "wam_generated",
            "wam_model_id": "model",
            "future_images": [f"f{index}.png" for index in range(4)],
            "future_times_s": [1.0, 2.0, 3.0, 4.0],
            "action_trajectory": trajectory,
        }

        row = build_manifest([base], [generated])[0]

        self.assertIsNone(row["gt_candidate_id"])
        action = next(
            item for item in row["candidates"] if item["candidate_id"] == "wam_action_head"
        )
        self.assertEqual(action["trajectory_source"], "wam_action_head")
        self.assertEqual(row["intrinsics_source_size"], [1920, 1080])

    def test_logged_gt_loader_uses_realized_pickle_trajectory(self) -> None:
        poses = [[float(index), 0.1 * index, 0.01 * index] for index in range(1, 5)]
        realized = [pose + [2.0, 0.0] for pose in poses]
        payload = {
            "future_trajectory": [{"pose": pose} for pose in poses],
            "metadata": {
                "future_times_s": [1.0, 2.0, 3.0, 4.0],
                "realized_future_ego_state": realized,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.pkl"
            with source.open("wb") as handle:
                pickle.dump(payload, handle)
            record = {
                "sample_id": "sample",
                "future_times_s": [1.0, 2.0, 3.0, 4.0],
                "lineage": {"source_sample": str(source)},
                "_manifest_root": directory,
            }
            actual = logged_gt_from_source_pickle(record)

        np.testing.assert_allclose(actual, np.asarray(poses))

    def test_alignment_rejects_action_head_as_logged_gt(self) -> None:
        row = {
            "sample_id": "sample",
            "gt_candidate_id": "wam_action_head",
            "candidates": [{
                "candidate_id": "wam_action_head",
                "trajectory_source": "wam_action_head",
                "trajectory": [[1.0, 0.0, 0.0]],
            }],
        }
        with self.assertRaisesRegex(ValueError, "is not logged GT"):
            _reference(row, "logged_gt")


if __name__ == "__main__":
    unittest.main()
