from __future__ import annotations

import unittest

from iac_new.flow_structure_scoring import (
    score_counterfactual_structure_pairs,
    score_flow_structure_pairs,
)


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


def _progress_measurement(source: str, role: str, values: list[float | None]) -> dict:
    return {
        "source_key": source,
        "branch_role": role,
        "stratum": "acceleration",
        "candidate_bank_used_by_measurement": False,
        "flow_structure": {
            "rows": [
                {
                    "interval_index": index,
                    "input_available": value is not None,
                    "median_flow_magnitude_px": value,
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

    def test_nominally_constant_actions_do_not_create_float_noise_ranking(self) -> None:
        measurements = []
        manifests = []
        for index, flow_delta in enumerate((0.2, 0.4, 0.8)):
            source = f"constant-{index}"
            measurements.extend([
                _measurement(source, "left", [flow_delta, flow_delta]),
                _measurement(source, "right", [0.0, 0.0]),
            ])
            manifests.extend([
                _manifest(source, "left", 0.4 + index * 1e-8),
                _manifest(source, "right", 0.0),
            ])
        report = score_flow_structure_pairs(
            measurements,
            manifests,
            minimum_common_intervals=2,
        )
        self.assertIsNone(report["action_response_spearman"])
        self.assertIsNone(report["action_response_spearman_ci95"])

    def test_path_length_reference_uses_travel_not_endpoint_axis(self) -> None:
        measurements = []
        manifests = []
        for index, distance in enumerate((2.0, 3.0, 4.0)):
            source = f"progress-{index}"
            measurements.extend([
                _measurement(source, "left", [distance, distance]),
                _measurement(source, "right", [0.0, 0.0]),
            ])
            manifests.extend([
                {
                    "source_key": source,
                    "branch_role": "left",
                    "action_trajectory": [[distance, 0.0, 0.0]],
                },
                {
                    "source_key": source,
                    "branch_role": "right",
                    "action_trajectory": [[1.0, 0.0, 0.0]],
                },
            ])
        report = score_flow_structure_pairs(
            measurements,
            manifests,
            action_reference="trajectory_path_length",
            minimum_action_delta=0.5,
            minimum_common_intervals=2,
        )
        self.assertEqual(report["action_reference"], "trajectory_path_length")
        self.assertEqual(report["action_direction_accuracy"], 1.0)
        self.assertEqual(report["action_response_spearman"], 1.0)

    def test_counterfactual_delta_reports_normalization_and_persistence(self) -> None:
        measurements = [
            _progress_measurement("a", "left", [4.0, 6.0, 8.0]),
            _progress_measurement("a", "right", [2.0, 3.0, 4.0]),
            _progress_measurement("b", "left", [2.0, 2.0, None]),
            _progress_measurement("b", "right", [3.0, 3.0, None]),
        ]
        manifests = [
            {"source_key": "a", "branch_role": "left", "action_trajectory": [[6.0, 0.0, 0.0]]},
            {"source_key": "a", "branch_role": "right", "action_trajectory": [[3.0, 0.0, 0.0]]},
            {"source_key": "b", "branch_role": "left", "action_trajectory": [[2.0, 0.0, 0.0]]},
            {"source_key": "b", "branch_role": "right", "action_trajectory": [[1.0, 0.0, 0.0]]},
        ]
        report = score_counterfactual_structure_pairs(
            measurements,
            manifests,
            minimum_common_intervals=2,
            minimum_action_delta=0.5,
        )
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["action_direction_accuracy"], 0.5)
        self.assertEqual(report["pairs"][0]["temporal_persistence"], 1.0)
        self.assertAlmostEqual(report["pairs"][0]["normalized_flow_delta"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
