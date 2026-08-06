from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "ordered_motion") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ordered_motion"))

from ordered_motion_support import (  # noqa: E402
    INSUFFICIENT_EVIDENCE,
    SUPPORTED,
    UNSUPPORTED,
    SupportDecisionConfig,
    aggregate_segment_evidence,
    calibrate_energy_thresholds,
    classify_support,
    score_row,
    wilson_lower_bound,
)


def _row(energies: list[list[float]]) -> dict:
    return {
        "group_id": "g0",
        "sample_id": "s0",
        "source_type": "must_not_be_read",
        "ordered_motion_segment_ledger": [
            {
                "segment_index": segment_index,
                "components": [
                    {
                        "family": f"f{component_index}",
                        "energy_contribution": energy,
                        "normalized_uncertainty": 0.5,
                    }
                    for component_index, energy in enumerate(segment)
                ],
            }
            for segment_index, segment in enumerate(energies)
        ],
    }


class OrderedMotionSupportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SupportDecisionConfig(
            support_energy_max=1.0,
            unsupported_energy_min=3.0,
            min_evidence_coverage=0.6,
            max_mean_normalized_uncertainty=1.5,
            require_uncertainty=True,
        )

    def test_full_visibility_preserves_original_energy_scale(self) -> None:
        row = _row([[0.1, 0.2], [0.3, 0.4]])
        evidence = aggregate_segment_evidence(row)
        # Existing ordered-motion energy is the sum of family means.
        self.assertAlmostEqual(evidence["visibility_aware_energy"], 0.5)
        self.assertAlmostEqual(evidence["evidence_coverage"], 1.0)

    def test_visibility_masks_unobserved_segments_without_rewarding_them(self) -> None:
        row = _row([[0.2, 0.4], [100.0, 100.0]])
        row["ordered_motion_segment_visibility"] = [1.0, 0.0]
        evidence = aggregate_segment_evidence(row)
        self.assertAlmostEqual(evidence["visibility_aware_energy"], 0.6)
        self.assertAlmostEqual(evidence["evidence_coverage"], 0.5)
        state, reason = classify_support(evidence, self.config)
        self.assertEqual(state, INSUFFICIENT_EVIDENCE)
        self.assertEqual(reason, "low_evidence_coverage")

    def test_three_states_are_separated_by_frozen_thresholds(self) -> None:
        supported = score_row(_row([[0.1, 0.1], [0.1, 0.1]]), self.config)
        unsupported = score_row(_row([[2.0, 2.0], [2.0, 2.0]]), self.config)
        margin = score_row(_row([[1.0, 1.0], [1.0, 1.0]]), self.config)
        self.assertEqual(supported["ordered_motion_support_state"], SUPPORTED)
        self.assertEqual(unsupported["ordered_motion_support_state"], UNSUPPORTED)
        self.assertEqual(
            margin["ordered_motion_support_state"], INSUFFICIENT_EVIDENCE
        )
        self.assertEqual(
            margin["ordered_motion_support_reason"], "decision_margin"
        )

    def test_high_uncertainty_abstains(self) -> None:
        row = _row([[0.1, 0.1], [0.1, 0.1]])
        for segment in row["ordered_motion_segment_ledger"]:
            for component in segment["components"]:
                component["normalized_uncertainty"] = 4.0
        scored = score_row(row, self.config)
        self.assertEqual(
            scored["ordered_motion_support_reason"],
            "high_predictive_uncertainty",
        )

    def test_source_label_does_not_change_inference(self) -> None:
        left = _row([[0.1, 0.1], [0.1, 0.1]])
        right = dict(left)
        right["source_type"] = "completely_different"
        left_scored = score_row(left, self.config)
        right_scored = score_row(right, self.config)
        for key in (
            "ordered_motion_support_state",
            "ordered_motion_support_reason",
            "ordered_motion_visibility_aware_energy",
        ):
            self.assertEqual(left_scored[key], right_scored[key])

    def test_calibration_maximizes_precise_disjoint_tails(self) -> None:
        records = [
            (0.1, True),
            (0.2, True),
            (0.8, True),
            (1.0, False),
            (1.8, True),
            (2.0, False),
            (2.2, False),
        ]
        result = calibrate_energy_thresholds(
            records,
            min_supported_precision=1.0,
            min_unsupported_precision=1.0,
        )
        self.assertEqual(result["support_energy_max"], 0.8)
        self.assertEqual(result["unsupported_energy_min"], 2.0)
        self.assertEqual(result["classified_rows"], 5)

    def test_wilson_bound_penalizes_small_perfect_tails(self) -> None:
        self.assertLess(wilson_lower_bound(6, 6), 0.95)
        self.assertGreater(wilson_lower_bound(100, 100), 0.95)


if __name__ == "__main__":
    unittest.main()
