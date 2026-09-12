import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_step1_evidence_channels import audit


class Step1EvidenceChannelAuditTest(unittest.TestCase):
    def test_pair_level_aggregation_is_source_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            row = {
                "source_key": "s0",
                "counterfactual_pair_evidence": {
                    "yaw_direction": {
                        "status": "weak",
                        "usable_intervals": 4,
                        "direction_accuracy": 0.75,
                    }
                },
            }
            path.write_text(json.dumps({"rows": [row, {**row, "branch_role": "right"}]}), encoding="utf-8")
            report = audit(path, bootstrap_draws=50)
            yaw = report["channels"]["yaw_direction"]
            self.assertEqual(yaw["pairs_total"], 1)
            self.assertEqual(yaw["pairs_evaluable"], 1)
            self.assertEqual(yaw["direction_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
