import unittest

from iac_new.fcs import assess_cross_model_fcs, score_fcs_rollout


class FcsTests(unittest.TestCase):
    def test_scores_explicit_independent_outcomes(self) -> None:
        report = score_fcs_rollout([
            {"source_key": "a", "task_success": True, "task_success_source": "simulator_score", "stratum": "turn", "wam_model_id": "m", "action_trajectory_source": "m_native", "independent_realized_state": True, "action_injection_verified": True},
            {"source_key": "b", "task_success": False, "task_success_source": "simulator_score", "stratum": "straight", "wam_model_id": "m", "action_trajectory_source": "m_native", "independent_realized_state": True, "action_injection_verified": True},
        ])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["successes"], 1)
        self.assertEqual(report["scored_rows"], 2)
        self.assertEqual(report["success_rate"], 0.5)

    def test_missing_or_non_independent_rows_are_unavailable(self) -> None:
        report = score_fcs_rollout([
            {"source_key": "a", "task_success": True, "independent_realized_state": False},
            {"source_key": "b", "independent_realized_state": True, "action_injection_verified": True},
        ])
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["scored_rows"], 0)
        self.assertEqual(report["unavailable_rows"], 2)

    def test_missing_positive_evidence_is_fail_closed(self) -> None:
        report = score_fcs_rollout([{"source_key": "a", "task_success": True}])
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["unavailable_rows"], 1)

    def test_navsim_compatibility_evidence_form_is_accepted(self) -> None:
        report = score_fcs_rollout([{
            "source_key": "a",
            "task_success": True,
            "task_success_source": "simulator_score",
            "wam_model_id": "m",
            "action_trajectory_source": "m_native",
            "realized_state_available": True,
            "state_reference_source": "navsim_pdm_kinematic_bicycle_closed_loop",
            "action_injection_verified": True,
        }])
        self.assertEqual(report["status"], "pass")

    def test_untrusted_state_source_is_unavailable(self) -> None:
        report = score_fcs_rollout([{
            "source_key": "a",
            "task_success": True,
            "task_success_source": "simulator_score",
            "wam_model_id": "m",
            "action_trajectory_source": "m_native",
            "realized_state_available": True,
            "state_reference_source": "annotated_guess",
            "action_injection_verified": True,
        }])
        self.assertEqual(report["status"], "unavailable")

    def test_duplicate_source_branch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_fcs_rollout([
                {"source_key": "a", "task_success": True},
                {"source_key": "a", "task_success": False},
            ])

    def test_image_derived_success_is_unavailable(self) -> None:
        report = score_fcs_rollout([{
            "source_key": "a",
            "task_success": True,
            "task_success_source": "image",
            "wam_model_id": "m",
            "action_trajectory_source": "m_native",
            "independent_realized_state": True,
            "action_injection_verified": True,
        }])
        self.assertEqual(report["status"], "unavailable")

    def test_embedded_forbidden_provenance_is_unavailable(self) -> None:
        report = score_fcs_rollout([{
            "source_key": "a",
            "task_success": True,
            "task_success_source": "image_derived_posthoc",
            "wam_model_id": "m",
            "action_trajectory_source": "native_action_head",
            "independent_realized_state": True,
            "action_injection_verified": True,
        }])
        self.assertEqual(report["status"], "unavailable")

    def test_cross_model_assessment_requires_two_independent_reports(self) -> None:
        one = {
            "model_id": "model_a",
            "status": "pass",
            "scored_rows": 30,
            "action_sources": ["model_a_native"],
        }
        pending = assess_cross_model_fcs([one])
        self.assertEqual(pending["status"], "pending")
        self.assertIn("fewer_than_2_distinct_models", pending["errors"])
        validated = assess_cross_model_fcs([
            one,
            {"model_id": "model_b", "status": "pass", "scored_rows": 30, "action_sources": ["model_b_native"]},
        ])
        self.assertEqual(validated["status"], "validated")
        self.assertTrue(validated["claim_enabled"])

    def test_cross_model_assessment_rejects_staging_like_report(self) -> None:
        report = assess_cross_model_fcs([
            {"model_id": "model_a", "status": "pass", "scored_rows": 30, "action_sources": ["staging_candidate"]},
            {"model_id": "model_b", "status": "pass", "scored_rows": 30, "action_sources": ["model_b_native"]},
        ])
        self.assertEqual(report["status"], "pending")
        self.assertIn("report_0:native_action_source_missing", report["errors"])
