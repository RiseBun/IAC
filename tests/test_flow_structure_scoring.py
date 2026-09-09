from __future__ import annotations

import unittest

from iac_new.flow_structure_scoring import score_flow_structure_pairs


def _measurement(source: str, role: str, values: list[float | None]) -> dict:
    return {
        "source_key": source,
        "branch_role": role,
        "stratum": "lateral_turn",
        "candidate_bank_used_by_measurement": False,
        "flow_structure": {
            "rows": [
                {
                    "interval_index": index,
                    "input_available": value is not None,
                    "horizontal_flow_center": value,
                }
                for index, value in enumerate(values)
            ]
        },
    }


def _manifest(source: str, role: str, lateral: float) -> dict:
    return {
        "source_key": source,
        "branch_role": role,
        "action_trajectory": [[1.0, 0.0, 0.0], [2.0, lateral, 0.0]],
    }


class FlowStructureScoringTest(unittest.TestCase):
    def test_scores_direction_and_order_without_zero_filling(self) -> None:
        measurements = [
            _measurement("a", "left", [0.3, 0.4, None, None]),
            _measurement("a", "right", [-0.1, 0.0, None, None]),
            _measurement("b", "left", [0.8, 0.9, 1.0, None]),
            _measurement("b", "right", [0.0, 0.1, 0.2, None]),
            _measurement("c", "left", [0.2, None, None, None]),
            _measurement("c", "right", [0.0, None, None, None]),
            _measurement("d", "left", [1.2, 1.3, None, None]),
            _measurement("d", "right", [0.0, 0.1, None, None]),
        ]
        manifests = [
            _manifest("a", "left", 0.4), _manifest("a", "right", 0.0),
            _manifest("b", "left", 0.8), _manifest("b", "right", 0.0),
            _manifest("c", "left", 0.2), _manifest("c", "right", 0.0),
            _manifest("d", "left", 1.2), _manifest("d", "right", 0.0),
        ]
        report = score_flow_structure_pairs(
            measurements,
            manifests,
            minimum_common_intervals=2,
        )
        self.assertEqual(report["status_counts"], {"scored": 3, "unavailable": 1})
        self.assertEqual(report["coverage"], 0.75)
        self.assertEqual(report["action_direction_accuracy"], 1.0)
        self.assertEqual(report["command_label_direction_accuracy"], 1.0)
        self.assertEqual(report["action_response_spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
