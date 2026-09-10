from __future__ import annotations

import unittest

from tools.compare_flow_structure_models import compare_reports


def _pair(
    source_key: str,
    action_delta: float | None,
    response: float | None,
    hit: bool | None,
) -> dict:
    row = {
        "source_key": source_key,
        "status": "scored" if response is not None else "unavailable",
    }
    if action_delta is not None:
        row["action_delta"] = action_delta
    if response is not None:
        row["flow_structure_delta"] = response
        row["action_direction_match"] = hit
    return row


class CompareFlowStructureModelsTest(unittest.TestCase):
    def test_compares_only_common_scored_actions_without_zero_filling(self) -> None:
        first = {
            "pairs": [
                _pair("a", 0.02, 1.0, True),
                _pair("b", 0.03, 2.0, True),
                _pair("c", 0.04, 3.0, True),
                _pair("d", 0.05, 4.0, True),
            ]
        }
        second = {
            "pairs": [
                _pair("a", 0.02, 3.0, False),
                _pair("b", 0.03, 2.0, True),
                _pair("c", 0.04, 1.0, False),
                _pair("d", None, None, None),
            ]
        }
        report = compare_reports(
            first,
            second,
            first_name="first",
            second_name="second",
            bootstrap_draws=100,
        )
        self.assertEqual(report["common_source_count"], 4)
        self.assertEqual(report["exact_action_match_count"], 3)
        self.assertEqual(report["common_scored_pairs"], 3)
        self.assertEqual(report["common_material_pairs"], 3)
        self.assertEqual(report["coverage"]["first"], 1.0)
        self.assertEqual(report["coverage"]["second"], 0.75)
        self.assertEqual(report["status"], "descriptive_posthoc_not_promotion")

    def test_rejects_mismatched_actions(self) -> None:
        first = {"pairs": [_pair("a", 0.02, 1.0, True), _pair("b", 0.03, 2.0, True), _pair("c", 0.04, 3.0, True)]}
        second = {"pairs": [_pair("a", 0.02, 1.0, True), _pair("b", 0.05, 2.0, True), _pair("c", 0.04, 3.0, True)]}
        with self.assertRaisesRegex(ValueError, "actions differ"):
            compare_reports(
                first,
                second,
                first_name="first",
                second_name="second",
                bootstrap_draws=10,
            )


if __name__ == "__main__":
    unittest.main()
