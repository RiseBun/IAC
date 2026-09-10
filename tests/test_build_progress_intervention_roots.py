from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_progress_intervention_roots import build_roots


class BuildProgressInterventionRootsTest(unittest.TestCase):
    def test_scales_translation_while_preserving_shared_yaw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_sample = root / "sample.pkl"
            source_sample.write_bytes(
                pickle.dumps({"metadata": {"stratum": "acceleration"}})
            )
            source = "sample-a"
            shard = root / "input" / "shard_0"
            shard.mkdir(parents=True)
            base = np.column_stack(
                [np.arange(1.0, 9.0), np.zeros(8), np.linspace(0.0, 0.2, 8)]
            )
            rows = [
                {
                    "source_key": source,
                    "source_sample": str(source_sample),
                    "branch_mode": role,
                    "action_trajectory": (base + offset).tolist(),
                }
                for role, offset in (
                    ("left", np.zeros_like(base)),
                    ("right", np.zeros_like(base)),
                )
            ]
            (shard / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")

            report = build_roots(
                input_root=root / "input",
                output_root=root / "output",
                fast_scale=1.25,
                slow_scale=0.75,
                quotas={"acceleration": 1},
                selection_seed="test",
            )
            fast = json.loads(
                (root / "output" / "fast" / "shard_0" / "manifest.json").read_text()
            )
            slow = json.loads(
                (root / "output" / "slow" / "shard_0" / "manifest.json").read_text()
            )
            fast_action = np.asarray(fast[0]["predicted_action_trajectory"])
            slow_action = np.asarray(slow[0]["predicted_action_trajectory"])
            np.testing.assert_allclose(fast_action[:, :2], base[:, :2] * 1.25)
            np.testing.assert_allclose(slow_action[:, :2], base[:, :2] * 0.75)
            np.testing.assert_allclose(fast_action[:, 2], slow_action[:, 2])
            self.assertTrue(report["yaw_identical_by_construction"])


if __name__ == "__main__":
    unittest.main()
