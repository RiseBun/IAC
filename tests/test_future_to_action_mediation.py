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
            "wam_model_id": "toy_wam",
            "action_source": "native_action_head",
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
            "action_normalization_fingerprint": "cal-v1",
            "action_normalization_scale": [1.0, 2.0],
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

    def test_single_source_is_scored_but_not_promoted(self):
        result = score(_rows(), draws=200, seed=7)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["promotion"]["status"], "insufficient_evidence")
        self.assertFalse(result["promotion"]["claim_enabled"])

    def test_missing_normalization_is_unavailable(self):
        rows = _rows()
        rows[0].pop("action_normalization_scale")
        result = score(rows, draws=20, seed=7)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["pairs"][0]["reason"], "action_normalization_scale_invalid")

    def test_normalized_distance_is_used(self):
        result = score(_rows(), draws=200, seed=7)
        # The first action coordinate is unchanged by scale 1; the second is
        # unchanged in all conditions.  This also proves the scale is part of
        # the scored contract rather than metadata that is ignored.
        self.assertAlmostEqual(result["future_effect_on_action"]["median"], 1.0)

    def test_thirty_sources_can_pass_promotion(self):
        rows = []
        for index in range(30):
            for row in _rows():
                copied = dict(row)
                copied["source_key"] = f"s{index}"
                rows.append(copied)
        result = score(rows, draws=500, seed=7, calibration_source_keys=set())
        self.assertEqual(result["promotion"]["status"], "passed")
        self.assertTrue(result["promotion"]["claim_enabled"])

    def test_promotion_requires_explicit_source_disjoint_evidence(self):
        rows = []
        for index in range(30):
            for row in _rows():
                copied = dict(row)
                copied["source_key"] = f"s{index}"
                rows.append(copied)
        result = score(rows, draws=100, seed=7)
        self.assertEqual(result["promotion"]["status"], "insufficient_evidence")
        self.assertFalse(result["promotion"]["checks"]["source_disjoint_confirmation"])

    def test_calibration_overlap_is_excluded(self):
        result = score(_rows(), draws=20, seed=7, calibration_source_keys={"s0"})
        self.assertEqual(result["scored_source_count"], 0)
        self.assertEqual(result["pairs"][0]["reason"], "source_in_calibration_split")

    def test_model_identity_is_required(self):
        rows = _rows()
        rows[0].pop("wam_model_id")
        result = score(rows, draws=20, seed=7)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["pairs"][0]["reason"], "wam_model_id_required")

    def test_mixed_model_identity_is_unavailable(self):
        rows = _rows()
        rows[2]["wam_model_id"] = "other_wam"
        result = score(rows, draws=20, seed=7)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["pairs"][0]["reason"], "wam_model_id_mismatch")

    def test_non_native_action_provenance_is_unavailable(self):
        rows = _rows()
        rows[1]["action_source"] = "staging_candidate"
        result = score(rows, draws=20, seed=7)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["pairs"][0]["reason"], "action_source_is_not_native")


if __name__ == "__main__":
    unittest.main()
