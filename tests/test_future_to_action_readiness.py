import unittest

from tools.audit_future_to_action_readiness import audit


class FutureToActionReadinessTest(unittest.TestCase):
    def test_two_branch_pilot_is_not_mediation_ready(self):
        rows = [
            {
                "counterfactual_group_id": "g0",
                "source_key": "s0",
                "branch_role": "future_native",
                "native_action_head_recorded": True,
                "formal_foresight_mediation_input": True,
                "native_unintervened": True,
                "action_origin": "native_action_head",
                "intervention_type": "internal_future_latent_permutation",
            },
            {
                "counterfactual_group_id": "g0",
                "source_key": "s0",
                "branch_role": "future_reverse",
                "native_action_head_recorded": True,
                "formal_foresight_mediation_input": True,
                "native_unintervened": False,
                "action_origin": "native_action_head",
                "intervention_type": "internal_future_latent_permutation",
            },
        ]
        result = audit(rows, [], [])
        self.assertEqual(result["status"], "blocked_missing_controls")
        self.assertFalse(result["claim_enabled"])
        self.assertEqual(result["inputs"]["counterfactual_groups"], 1)
        self.assertIn("future_perturbed_pathway_blocked", result["missing_controls"])


if __name__ == "__main__":
    unittest.main()
