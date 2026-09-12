import copy
import json
import unittest
from pathlib import Path

from iac_new.metric_contract import validate_directional_yaw_config


class MetricContractTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).parents[1]
        self.mas = json.loads((root / "configs" / "mas_yaw_v1.json").read_text(encoding="utf-8"))
        self.rcs = json.loads((root / "configs" / "rcs_yaw_v1.json").read_text(encoding="utf-8"))

    def test_frozen_mas_and_rcs_contracts_are_valid(self) -> None:
        self.assertEqual(validate_directional_yaw_config(self.mas)["status"], "valid")
        self.assertEqual(validate_directional_yaw_config(self.rcs)["status"], "valid")

    def test_model_specific_threshold_change_is_rejected(self) -> None:
        changed = copy.deepcopy(self.mas)
        changed["comparability_contract"]["model_specific_fields_allowed"] = ["orientation", "deadband"]
        with self.assertRaises(ValueError):
            validate_directional_yaw_config(changed)

    def test_non_candidate_blind_config_is_rejected(self) -> None:
        changed = copy.deepcopy(self.rcs)
        changed["representation"]["candidate_blind"] = False
        with self.assertRaises(ValueError):
            validate_directional_yaw_config(changed)
