import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_raw_universal_response_controls import audit


class RawUniversalResponseAuditTest(unittest.TestCase):
    def test_all_gates_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.json"
            path.write_text(json.dumps({
                "pair_count": 10,
                "normal": {"coverage": 1.0, "direction_ci95": [0.8, 0.95]},
                "controls": {
                    "reversed_action": {"positive_false_positive_rate": 0.0},
                    "identity_swap": {"positive_false_positive_rate": 0.0},
                    "zero_contrast": {"directional_estimand": "unavailable", "median_observed_delta_px": 0.0},
                },
            }), encoding="utf-8")
            report = audit({"toy": path})
            self.assertTrue(report["models"][0]["promotion"])
            self.assertEqual(report["promoted_model_count"], 1)

    def test_negative_control_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.json"
            path.write_text(json.dumps({
                "pair_count": 10,
                "normal": {"coverage": 1.0, "direction_ci95": [0.8, 0.95]},
                "controls": {
                    "reversed_action": {"positive_false_positive_rate": 0.5},
                    "identity_swap": {"positive_false_positive_rate": 0.0},
                    "zero_contrast": {"directional_estimand": "unavailable", "median_observed_delta_px": 0.0},
                },
            }), encoding="utf-8")
            report = audit({"toy": path})
            self.assertFalse(report["models"][0]["promotion"])


if __name__ == "__main__":
    unittest.main()
