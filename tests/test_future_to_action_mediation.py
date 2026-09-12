import unittest

from tools.score_future_to_action_mediation import score


def _rows():
    rows = []
    vectors = {
        "baseline": [0.0, 1.0],
        "future_perturbed": [1.0, 1.0],
        "future_perturbed_pathway_blocked": [0.2, 1.0],
        "future_fixed_action_pathway_control": [0.05, 1.0],
    }
    for condition, action in vectors.items():
        rows.append({
            "source_key": "s0",
            "condition": condition,
            "native_action": action,
            "history_fingerprint": "h",
            "command_fingerprint": "c",
            "nuisance_seed": 1,
            "model_revision": "m",
            "future_fingerprint": "base" if condition in {"baseline", "future_fixed_action_pathway_control"} else "perturbed",
            "pathway_state": {
                "baseline": "normal",
                "future_perturbed": "normal",
                "future_perturbed_pathway_blocked": "blocked",
                "future_fixed_action_pathway_control": "fixed_action",
            }[condition],
        })
    return rows


class FutureToActionMediationTest(unittest.TestCase):
    def test_pathway_effect_is_scored(self):
        result = score(_rows(), draws=200, seed=7)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["scored_source_count"], 1)
        self.assertAlmostEqual(result["future_effect_on_action"]["median"], 1.0)
        self.assertAlmostEqual(result["blocked_effect_on_action"]["median"], 0.2)
        self.assertAlmostEqual(result["pathway_suppression"]["median"], 0.8)

    def test_invariance_mismatch_is_unavailable(self):
        rows = _rows()
        rows[-1]["command_fingerprint"] = "different"
        result = score(rows, draws=20, seed=7)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["scored_source_count"], 0)

    def test_missing_intervention_identity_is_unavailable(self):
        rows = _rows()
        rows[1].pop("future_fingerprint")
        result = score(rows, draws=20, seed=7)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["pairs"][0]["reason"], "future_fingerprint_required")

    def test_wrong_pathway_label_is_unavailable(self):
        rows = _rows()
        rows[2]["pathway_state"] = "normal"
        result = score(rows, draws=20, seed=7)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["pairs"][0]["reason"], "pathway_state_contract_mismatch")


if __name__ == "__main__":
    unittest.main()
