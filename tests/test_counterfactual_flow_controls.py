from __future__ import annotations

import unittest

from scripts.validate_counterfactual_flow_delta import run_controls


def _measurement(source: str, role: str, values: list[float]) -> dict:
    return {
        "source_key": source,
        "branch_role": role,
        "candidate_bank_used_by_measurement": False,
        "flow_structure": {
            "rows": [
                {"interval_index": i, "input_available": True, "horizontal_flow_center": value}
                for i, value in enumerate(values)
            ]
        },
    }


def _manifest(source: str, role: str, endpoint: float) -> dict:
    return {
        "source_key": source,
        "branch_role": role,
        "action_trajectory": [[0.0, 0.0, endpoint]],
    }


class CounterfactualFlowControlsTest(unittest.TestCase):
    def test_controls_have_expected_directional_behavior(self) -> None:
        measurements = [
            _measurement("a", "left", [3.0, 4.0]),
            _measurement("a", "right", [1.0, 2.0]),
            _measurement("b", "left", [2.0, 3.0]),
            _measurement("b", "right", [1.0, 1.5]),
        ]
        manifests = [
            _manifest("a", "left", 2.0),
            _manifest("a", "right", 0.0),
            _manifest("b", "left", 1.0),
            _manifest("b", "right", 0.0),
        ]
        reports = run_controls(
            measurements,
            manifests,
            descriptor="horizontal_flow_center",
            action_reference="endpoint_column",
            action_column=2,
            minimum_common_intervals=2,
            minimum_action_delta=0.01,
        )
        self.assertEqual(reports["normal_order"]["action_direction_accuracy"], 1.0)
        self.assertEqual(reports["reversed_order"]["action_direction_accuracy"], 0.0)
        self.assertEqual(reports["identity_swap"]["action_direction_accuracy"], 0.0)
        self.assertIsNone(reports["zero_contrast"]["action_response_spearman"])


if __name__ == "__main__":
    unittest.main()
