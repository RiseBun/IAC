from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.prepare_pure_speed_confirmation import prepare


class PreparePureSpeedConfirmationTest(unittest.TestCase):
    def test_creates_scene_disjoint_fast_slow_twins_with_shared_yaw(self) -> None:
        rows = [
            {
                "source_key": "scene-a:sample-1",
                "scene_group": "scene-a",
                "stratum": "acceleration",
                "action_trajectory": np.column_stack(
                    [np.arange(1.0, 5.0), np.zeros(4), np.linspace(0.0, 0.3, 4)]
                ).tolist(),
            },
            {
                "source_key": "scene-b:sample-1",
                "scene_group": "scene-b",
                "stratum": "braking",
                "action_trajectory": np.column_stack(
                    [np.arange(1.0, 5.0), np.ones(4), np.linspace(0.0, -0.2, 4)]
                ).tolist(),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = prepare(
                rows,
                output_root=Path(directory),
                fast_scale=1.25,
                slow_scale=0.75,
                confirmation_fraction=0.5,
                selection_seed="test",
                model_ids=["drivewam", "epona"],
            )
            self.assertEqual(report["twin_count"], 2)
            self.assertTrue(report["yaw_identical_by_construction"])
            calibration = [
                __import__("json").loads(line)
                for line in (Path(directory) / "calibration" / "manifest.jsonl").read_text().splitlines()
            ]
            confirmation = [
                __import__("json").loads(line)
                for line in (Path(directory) / "confirmation" / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(calibration) + len(confirmation), 4)
            for values in (calibration, confirmation):
                for row in values:
                    self.assertEqual(row["intervention_type"], "pure_speed")
                    self.assertEqual(row["action_trajectory"][0][2], row["source_action_trajectory"][0][2])


if __name__ == "__main__":
    unittest.main()
