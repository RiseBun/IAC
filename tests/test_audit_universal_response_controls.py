import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_universal_response_controls import audit


class UniversalResponseControlAuditTest(unittest.TestCase):
    def test_missing_identity_control_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.json"
            rows = []
            for control, cosine in (("normal", 0.9), ("reversed", -0.9), ("zero", None)):
                rows.append({
                    "control": control,
                    "interval_coverage": 1.0,
                    "median_direction_cosine": cosine,
                    "temporal_persistence": 1.0 if cosine is not None else None,
                })
            path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            report = audit({"toy": path}, bootstrap_draws=100)
            model = report["models"][0]
            self.assertFalse(model["promotion"])
            self.assertFalse(model["required_controls_present"]["identity_swap"])
            self.assertFalse(report["universal_channel_promotion"])


if __name__ == "__main__":
    unittest.main()
