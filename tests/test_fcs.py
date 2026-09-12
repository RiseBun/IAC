import unittest

from iac_new.fcs import score_fcs_rollout


class FcsTests(unittest.TestCase):
    def test_scores_explicit_independent_outcomes(self) -> None:
        report = score_fcs_rollout([
            {"source_key": "a", "task_success": True, "stratum": "turn", "independent_realized_state": True, "action_injection_verified": True},
            {"source_key": "b", "task_success": False, "stratum": "straight", "independent_realized_state": True, "action_injection_verified": True},
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
            "realized_state_available": True,
            "state_reference_source": "navsim_pdm_kinematic_bicycle_closed_loop",
            "action_injection_verified": True,
        }])
        self.assertEqual(report["status"], "pass")

    def test_duplicate_source_branch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_fcs_rollout([
                {"source_key": "a", "task_success": True},
                {"source_key": "a", "task_success": False},
            ])
