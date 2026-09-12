import unittest

from iac_new.scorecard import (
    build_model_scorecard,
    claimed_cells,
    summarize_conditional_units,
    validate_submission,
)


def _public() -> dict:
    return {
        "sample_id": "navsim:demo:scene:1",
        "source_key": "navsim:demo:scene:1",
        "split": "benchmark",
    }


def _row(**extra):
    times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    traj = [[float(i), 0.0, 0.0] for i in range(8)]
    row = {
        "sample_id": "navsim:demo:scene:1",
        "wam_model_id": "toy_wam",
        "capability": "native_action_conditioned",
        "future_images_source": "wam_generated",
        "future_images": [f"f{i}.png" for i in range(8)],
        "future_times_s": times,
        "action_trajectory": traj,
        "action_source": "native_action_head",
    }
    row.update(extra)
    return row


class ScorecardTest(unittest.TestCase):
    def test_canonical_metric_cells_are_present(self) -> None:
        card = build_model_scorecard(model_id="x", capability="video_only")
        for cell in ("mas", "rcs", "gs", "future_to_action_mediation"):
            self.assertEqual(card["cells"][cell]["status"], "unavailable")

    def test_scorecard_is_explicitly_conditional(self) -> None:
        card = build_model_scorecard(model_id="x", capability="action_only")
        policy = card["scoring_policy"]
        self.assertFalse(policy["future_driven_assumption"])
        self.assertFalse(policy["zero_fill_unavailable"])
        self.assertEqual(policy["aggregate_score"], "not_defined")
        self.assertEqual(card["ranking_policy"]["natural_model_quality_ranking"], "not_supported")
        self.assertEqual(card["ranking_policy"]["aggregate_across_capabilities"], "not_defined")
        self.assertEqual(policy["mode"], "conditional_evaluation")
        self.assertEqual(policy["score_denominator"], "units_with_evidence_and_quality_gate_passed")
        self.assertTrue(policy["abstention_is_not_failure"])

    def test_canonical_cells_expose_claim_boundaries(self) -> None:
        card = build_model_scorecard(model_id="x", capability="native_action_conditioned")
        self.assertEqual(card["cells"]["mas"]["causal_status"], "not_mediation")
        self.assertEqual(card["cells"]["rcs"]["causal_status"], "not_mediation")
        self.assertEqual(card["cells"]["gs"]["causal_status"], "not_mediation")
        self.assertEqual(
            card["cells"]["future_to_action_mediation"]["causal_status"],
            "causal_only_if_promotion_passed",
        )
        self.assertEqual(
            card["cells"]["mas"]["conditional_evaluation"]["score_scope"],
            "units_with_evidence_and_quality_gate_passed",
        )

    def test_motion_dimension_boundary_is_explicit(self) -> None:
        card = build_model_scorecard(
            model_id="demo",
            capability="externally_controlled_video",
            measurements={"rcs": {"status": "pass", "coverage": 1.0, "score": 0.8}},
        )
        dimensions = card["cells"]["rcs"]["measurement_dimensions"]
        self.assertEqual(dimensions["yaw_direction"], "validated")
        self.assertEqual(dimensions["speed"], "diagnostic_only")
        self.assertEqual(dimensions["longitudinal_distance"], "diagnostic_only")
        self.assertEqual(dimensions["lateral_displacement"], "diagnostic_only")
        self.assertEqual(dimensions["curvature"], "diagnostic_only")
        self.assertEqual(dimensions["metric_trajectory"], "unavailable")

    def test_conditional_summary_excludes_abstention_from_score_but_not_coverage(self) -> None:
        report = summarize_conditional_units([
            {"status": "scored", "score": 1.0},
            {"status": "scored", "score": 0.5},
            {"status": "abstain", "reason": "insufficient_flow"},
            {"status": "unavailable", "reason": "reference_private"},
        ])
        self.assertAlmostEqual(report["conditional_score"], 0.75)
        self.assertAlmostEqual(report["score_coverage"], 0.5)
        self.assertEqual(report["status_counts"]["abstain"], 1)
        self.assertEqual(report["status_counts"]["unavailable"], 1)
        self.assertFalse(report["zero_fill_unavailable"])
        self.assertEqual(report["abstention_reasons"], {"insufficient_flow": 1})

    def test_canonical_metric_coverage_aliases_are_normalized(self) -> None:
        card = build_model_scorecard(
            model_id="x",
            capability="video_only",
            measurements={
                "mas": {"status": "pass", "pair_coverage": 0.94, "score": 0.84},
                "rcs": {"status": "pass", "pair_coverage": 1.0, "score": 0.86},
            },
        )
        self.assertAlmostEqual(card["cells"]["mas"]["coverage"], 0.94)
        self.assertAlmostEqual(card["cells"]["rcs"]["coverage"], 1.0)

    def test_epona_does_not_claim_ccfc(self) -> None:
        self.assertEqual(claimed_cells("externally_controlled_video"), ("a2f",))
        card = build_model_scorecard(model_id="epona", capability="externally_controlled_video")
        self.assertEqual(card["cells"]["ccfc"]["status"], "unavailable")
        self.assertEqual(card["cells"]["a2f"]["status"], "missing")

    def test_native_missing_cells_are_missing_not_zero(self) -> None:
        card = build_model_scorecard(model_id="x", capability="native_action_conditioned")
        self.assertEqual(card["cells"]["ccfc"]["status"], "unavailable")
        self.assertNotIn("score", card["cells"]["ccfc"])

    def test_submission_must_match_public_ids(self) -> None:
        report = validate_submission([_row()], [_public()])
        self.assertTrue(report["ready"])
        bad = validate_submission([_row(sample_id="unknown")], [_public()])
        self.assertFalse(bad["ready"])
        self.assertIn("sample_id_not_in_public_split", bad["issues"][0]["issues"])

    def test_submission_rejects_duplicate_sample_ids(self) -> None:
        report = validate_submission([_row(), _row()], [_public()])
        self.assertFalse(report["ready"])
        self.assertTrue(any("duplicate_sample_id" in issue for item in report["issues"] for issue in item["issues"]))

    def test_submission_rejects_mixed_model_or_capability(self) -> None:
        mixed_model = validate_submission([_row(), _row(wam_model_id="other")], [_public(), {**_public(), "sample_id": "navsim:demo:scene:2"}])
        self.assertFalse(mixed_model["ready"])
        self.assertTrue(any("submission_mixes_multiple_wam_model_ids" in issue for item in mixed_model["issues"] for issue in item["issues"]))
        mixed_capability = validate_submission([_row(), _row(capability="action_only", sample_id="navsim:demo:scene:2")], [_public(), {**_public(), "sample_id": "navsim:demo:scene:2"}])
        self.assertFalse(mixed_capability["ready"])
        self.assertTrue(any("submission_mixes_multiple_capabilities" in issue for item in mixed_capability["issues"] for issue in item["issues"]))

    def test_logged_action_is_rejected_for_native_models(self) -> None:
        report = validate_submission([_row(action_source="logged")], [_public()])
        self.assertFalse(report["ready"])
        self.assertIn("action_source_is_not_native", report["issues"][0]["issues"])

    def test_future_leak_is_rejected(self) -> None:
        report = validate_submission(
            [_row(realized_future_ego_state=[[0, 0, 0, 0, 0]] * 8)],
            [_public()],
        )
        self.assertIn("realized_future_state_leakage", report["issues"][0]["issues"])

    def test_projection_gated_coverage_is_recomputed_from_counts(self) -> None:
        card = build_model_scorecard(
            model_id="x",
            capability="native_action_conditioned",
            measurements={
                "cfac": {
                    "status": "pilot", "score": 0.7, "n": 3, "total": 10,
                    "coverage": 0.9,
                }
            },
        )
        self.assertEqual(card["cells"]["cfac"]["coverage"], 0.3)
        self.assertEqual(
            card["cells"]["cfac"]["coverage_basis"], "post_projection_abstention"
        )

    def test_projection_gated_score_requires_coverage_counts(self) -> None:
        card = build_model_scorecard(
            model_id="x",
            capability="native_action_conditioned",
            measurements={"cfac": {"status": "pilot", "score": 0.7}},
        )
        self.assertEqual(card["cells"]["cfac"]["status"], "missing")
        self.assertEqual(
            card["cells"]["cfac"]["reason"],
            "post_projection_coverage_counts_required",
        )

if __name__ == "__main__":
    unittest.main()
