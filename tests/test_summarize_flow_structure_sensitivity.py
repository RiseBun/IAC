from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_flow_structure_sensitivity import summarize


class SummarizeFlowStructureSensitivityTest(unittest.TestCase):
    def test_passes_monotonic_full_attenuation_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for label, response in (("0", 2.0), ("50", 1.0), ("100", 0.0)):
                run = root / "model" / f"s_{label}"
                run.mkdir(parents=True)
                left_image = run / "left.bin"
                right_image = run / "right.bin"
                left_image.write_bytes(b"same" if label == "100" else b"left")
                right_image.write_bytes(b"same" if label == "100" else b"right")
                rows = [
                    {
                        "source_key": "sample",
                        "branch_role": role,
                        "action": [1, -1] if role == "left" else [1, 1],
                        "future_frame_paths": [str(image)],
                        "metadata": {"controlled_degradation": {"strength": label}},
                    }
                    for role, image in (("left", left_image), ("right", right_image))
                ]
                (run / "manifest.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                (run / "score.json").write_text(
                    json.dumps(
                        {
                            "coverage": 1.0,
                            "action_response_spearman": None if label == "100" else 0.8,
                            "action_direction_accuracy": None if label == "100" else 0.9,
                            "action_direction_pairs": 0 if label == "100" else 1,
                            "pairs": [
                                {
                                    "status": "scored",
                                    "flow_structure_delta": response,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            report = summarize(
                preregistration={
                    "protocol": "test",
                    "status": "preregistered",
                    "source_contract": {"expected_source_count": 1},
                    "degradation": {"strengths": [0.0, 0.5, 1.0]},
                    "acceptance_criteria": {
                        "coverage_at_every_strength_min": 0.9,
                        "full_attenuation_response_ratio_max": 0.1,
                    },
                    "interpretation_boundary": "sensitivity only",
                },
                run_root=root,
                model_directories={"model": "model"},
            )

            self.assertTrue(report["overall_pass"])
            self.assertEqual(
                [run["median_absolute_response"] for run in report["models"]["model"]["runs"]],
                [2.0, 1.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
