import unittest

import numpy as np

from iac_new.step1_evidence import (
    assemble_step1_evidence,
    counterfactual_response_evidence,
    response_channel_evidence,
    structure_quality_evidence,
    temporal_motion_evidence,
)


def _profile(values, available=True):
    return {
        "rows": [
            {
                "interval_index": i,
                "input_available": available,
                "horizontal_flow_center": float(value),
                "median_flow_magnitude_px": abs(float(value)),
            }
            for i, value in enumerate(values)
        ]
    }


class Step1EvidenceTest(unittest.TestCase):
    def test_temporal_evidence_is_ordinal_and_candidate_blind(self) -> None:
        result = temporal_motion_evidence(_profile([0.2, 0.3, 0.4]))
        self.assertIn(result["status"], {"scored", "weak"})
        self.assertEqual(result["temporal_persistence"], 1.0)
        self.assertFalse(result["metric_reconstruction_used"])
        self.assertFalse(result["candidate_selection_used"])

    def test_zero_action_contrast_abstains(self) -> None:
        result = counterfactual_response_evidence(_profile([0.2, 0.3]), _profile([-0.2, -0.3]), 0.0)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "zero_or_missing_action_contrast")

    def test_assembled_layer_keeps_depth_optional(self) -> None:
        result = assemble_step1_evidence(
            branch_profile=_profile([0.2, 0.3]),
            left_profile=_profile([0.2, 0.3]),
            right_profile=_profile([-0.2, -0.3]),
            action_delta=0.4,
            flow_support_fraction=0.8,
            depth_valid_mask=np.ones((2, 3), dtype=bool),
        )
        self.assertEqual(result["protocol"], "iac-step1-evidence-layer-v1")
        self.assertIn(result["channels"]["counterfactual_response"]["status"], {"scored", "weak"})
        self.assertFalse(result["channels"]["grounding"]["status"] == "scored")
        self.assertFalse(result["metric_reconstruction_used"])

    def test_action_channels_are_independent(self) -> None:
        left = _profile([0.2, 0.3])
        right = _profile([-0.2, -0.3])
        for row in left["rows"]:
            row["left_right_horizontal_contrast"] = 0.2
        for row in right["rows"]:
            row["left_right_horizontal_contrast"] = -0.2
        result = response_channel_evidence(
            left,
            right,
            {"yaw_direction": 0.4, "lateral_direction": 0.4, "longitudinal_order": 0.0},
        )
        self.assertIn(result["yaw_direction"]["status"], {"scored", "weak"})
        self.assertIn(result["lateral_direction"]["status"], {"scored", "weak"})
        self.assertEqual(result["longitudinal_order"]["status"], "unavailable")

    def test_structure_quality_is_not_action_alignment(self) -> None:
        profile = _profile([0.2, 0.3])
        for row in profile["rows"]:
            row.update({
                "affine_explained_fraction": 0.9,
                "affine_robust_weight_fraction": 0.8,
                "structure_confidence": 0.7,
                "spatial_coverage": 0.9,
                "foe": {"valid": True, "confidence": 0.8},
            })
        result = structure_quality_evidence(profile)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["affine_explained_fraction"], 0.9)
        self.assertIn("not action alignment", result["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
