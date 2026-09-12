from __future__ import annotations

import unittest

from iac_new.metric_evidence_contract import (
    validate_metric_evidence_packet,
    validate_metric_evidence_table,
)


class MetricEvidenceContractTest(unittest.TestCase):
    def test_mas_requires_native_action_and_visual_likelihood(self) -> None:
        packet = {
            "source_key": "s0",
            "evidence_status": "scored",
            "coverage": 0.9,
            "branch_id": "left",
            "action_source": "wam_native_action",
            "visual_evidence": {"likelihood": 0.7, "support_fraction": 0.8},
        }
        report = validate_metric_evidence_packet("MAS", packet)
        self.assertEqual(report["status"], "valid")
        self.assertTrue(report["score_allowed"])

    def test_rcs_does_not_accept_single_branch(self) -> None:
        packet = {
            "source_key": "s0",
            "evidence_status": "scored",
            "coverage": 1.0,
            "counterfactual_group_id": "g0",
            "branches": ["left"],
            "action_delta": {"yaw": 0.2},
            "visual_delta": {"likelihood": 0.8},
            "controls": {"reversed": {}, "zero": {}},
        }
        report = validate_metric_evidence_packet("RCS", packet)
        self.assertEqual(report["status"], "invalid")
        self.assertIn("branches>=2", report["missing"])

    def test_unavailable_is_valid_but_not_a_score(self) -> None:
        packet = {
            "source_key": "s0",
            "evidence_status": "unavailable",
            "failure_reason": "reference_missing",
            "reference_source": "logged_future",
            "generated_representation": {"flow": True},
            "reference_representation": None,
            "comparison": None,
        }
        report = validate_metric_evidence_packet("GS", packet)
        self.assertEqual(report["status"], "unavailable")
        self.assertFalse(report["score_allowed"])

    def test_fcs_requires_independent_paired_intervention(self) -> None:
        packet = {
            "source_key": "s0",
            "evidence_status": "scored",
            "coverage": 1.0,
            "native_action_source": "policy_native_action",
            "independent_rollout": {
                "realized_state_available": True,
                "action_injection_verified": True,
                "simulator_id": "navsim",
            },
            "intervention": {"paired": True, "type": "future_ablation"},
            "task_success": True,
        }
        report = validate_metric_evidence_packet("FCS", packet)
        self.assertEqual(report["status"], "valid")

    def test_table_preserves_unavailable_without_zero_fill(self) -> None:
        rows = [{"source_key": "s0", "evidence_status": "unavailable", "failure_reason": "no_video"}]
        report = validate_metric_evidence_table(rows, "MAS")
        self.assertEqual(report["status_counts"]["unavailable"], 1)
        self.assertEqual(report["status_counts"]["invalid"], 0)


if __name__ == "__main__":
    unittest.main()
