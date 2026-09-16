import unittest

from iac_new.response_consistency import delta_state, score_response_consistency


CONFIG = {
    "protocol": "test-rcs",
    "intervention": {
        "allowed_types": ["command"],
        "required_roles": ["left", "right"],
        "allowed_future_images_sources": ["wam_generated"],
        "validation_only_derivatives_allowed": False,
    },
    "yaw": {"action_delta_threshold_rad": 0.01, "visual_delta_threshold_rad": 0.01},
    "progress": {
        "status": "diagnostic_only",
        "action_delta_threshold_m": 0.5,
        "visual_delta_threshold_m": 0.5,
        "required_common_intervals": 4,
    },
    "statistics": {
        "bootstrap_draws": 100,
        "bootstrap_seed": 7,
        "minimum_declared_pairs_for_formal_report": 1,
    },
    "claim_boundary": "test",
}


def manifest(role):
    return {
        "sample_id": role,
        "counterfactual_group_id": "source",
        "branch_role": role,
        "history_fingerprint": "history",
        "nuisance_seed": 3,
        "intervention_type": "command",
        "wam_model_id": "model",
        "future_images_source": "wam_generated",
        "future_frame_paths": [f"{role}-{index}.png" for index in range(4)],
    }


def as_row(role, action_yaw, visual_yaw):
    intervals = [
        {"status": "scored", "distance_m_raw": 1.0}
        for _ in range(4)
    ]
    return {
        "sample_id": role,
        "yaw": {"action_yaw_rad": action_yaw, "visual_yaw_rad": visual_yaw},
        "progress": {"action_interval_distance_m": [1.0] * 4},
        "intervals": intervals,
    }


class ResponseConsistencyTest(unittest.TestCase):
    def test_delta_state_has_frozen_deadband(self):
        self.assertEqual(delta_state(0.0, 0.1), "zero")
        self.assertEqual(delta_state(0.2, 0.1), "positive")
        self.assertEqual(delta_state(-0.2, 0.1), "negative")

    def test_matching_material_response_scores_one(self):
        result = score_response_consistency(
            [manifest("left"), manifest("right")],
            [as_row("left", 0.1, 0.2), as_row("right", -0.1, -0.2)],
            CONFIG,
        )
        self.assertEqual(result["RCS_yaw"], 1.0)
        self.assertEqual(result["material_response_effective_score"], 1.0)
        self.assertEqual(result["controls"]["visual_branch_swap_material_score"], 0.0)
        self.assertEqual(result["intervention_strength"]["bins"]["material_ge_0.12"]["pairs"], 1)

    def test_missing_visual_response_gets_no_credit(self):
        result = score_response_consistency(
            [manifest("left"), manifest("right")],
            [as_row("left", 0.1, None), as_row("right", -0.1, None)],
            CONFIG,
        )
        self.assertEqual(result["RCS_yaw"], 0.0)
        self.assertEqual(result["visual_response_coverage_on_material_pairs"], 0.0)

    def test_zero_action_and_zero_visual_is_diagnostic_only(self):
        result = score_response_consistency(
            [manifest("left"), manifest("right")],
            [as_row("left", 0.0, 0.0), as_row("right", 0.0, 0.0)],
            CONFIG,
        )
        self.assertEqual(result["RCS_yaw"], 0.0)
        self.assertEqual(result["ternary_state_agreement_diagnostic"], 1.0)
        self.assertEqual(result["zero_response_specificity"], 1.0)

    def test_validation_only_derivative_is_rejected(self):
        left = manifest("left")
        left["metadata"] = {"controlled_degradation": {"validation_only": True}}
        result = score_response_consistency(
            [left, manifest("right")],
            [as_row("left", 0.1, 0.2), as_row("right", -0.1, -0.2)],
            CONFIG,
        )
        self.assertFalse(result["formal_metric_eligible"])
        self.assertEqual(result["invalid_contract_reasons"], {"validation_only_derivative": 1})

    def test_history_mismatch_blocks_formal_metric(self):
        right = manifest("right")
        right["history_fingerprint"] = "other"
        result = score_response_consistency(
            [manifest("left"), right],
            [as_row("left", 0.1, 0.2), as_row("right", -0.1, -0.2)],
            CONFIG,
        )
        self.assertFalse(result["formal_metric_eligible"])
        self.assertEqual(result["contract_valid_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
