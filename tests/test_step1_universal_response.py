import copy
import json
import unittest
from pathlib import Path

from iac_new.step1_universal_response import validate_universal_response_config


class Step1UniversalResponseTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).parents[1]
        self.config = json.loads(
            (root / "configs" / "step1_universal_response_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_experimental_contract_is_valid(self) -> None:
        report = validate_universal_response_config(self.config)
        self.assertEqual(report["status"], "valid_experimental_contract")
        self.assertEqual(report["promotion_gates"]["minimum_architectures"], 3)

    def test_metric_reconstruction_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["representation"]["metric_reconstruction_used"] = True
        with self.assertRaises(ValueError):
            validate_universal_response_config(changed)

    def test_missing_control_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["controls"].remove("identity_swap")
        with self.assertRaises(ValueError):
            validate_universal_response_config(changed)

    def test_model_specific_thresholds_are_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["adapter_policy"]["model_specific_fields_allowed"].append("threshold")
        with self.assertRaises(ValueError):
            validate_universal_response_config(changed)

    def test_universal_gate_cannot_be_lowered(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["promotion_gates"]["minimum_architectures"] = 2
        with self.assertRaises(ValueError):
            validate_universal_response_config(changed)


if __name__ == "__main__":
    unittest.main()
