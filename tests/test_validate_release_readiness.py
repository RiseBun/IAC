import json
import unittest
from pathlib import Path

from tools.validate_release_readiness import validate


ROOT = Path(__file__).parents[1]


class ReleaseReadinessTest(unittest.TestCase):
    def test_current_release_is_conditionally_ready_but_not_causal_complete(self):
        readiness = json.loads((ROOT / "reports" / "release_readiness_20260912.json").read_text())
        protocol = json.loads((ROOT / "configs" / "wam_joint_evaluation_v1.json").read_text())
        report = validate(readiness, protocol)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["conditional_consistency_grounding_release"]["ready"])
        self.assertFalse(report["complete_causal_future_driven_benchmark"]["ready"])
        self.assertIn("fcs:cross_model_evidence_pending", report["warnings"])
        self.assertIn("future_to_action_mediation:confirmation_pending", report["warnings"])
        self.assertFalse(report["evidence_boundary"]["gs_reference_publicly_recomputable"])

    def test_missing_validated_metric_fails(self):
        readiness = {
            "claims": {
                "mas_yaw": {"status": "pilot", "models": ["a", "b"]},
                "rcs_yaw": {"status": "validated", "models": ["a", "b"]},
                "gs": {"status": "validated", "models": ["a", "b"]},
                "fcs": {},
                "future_to_action_mediation": {},
            }
        }
        protocol = {"status": "conditional_framework_validated_causal_and_cross_model_evidence_pending"}
        report = validate(readiness, protocol)
        self.assertEqual(report["status"], "fail")
        self.assertIn("mas_yaw:status_not_validated", report["errors"])


if __name__ == "__main__":
    unittest.main()
