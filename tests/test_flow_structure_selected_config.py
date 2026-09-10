from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SelectedFlowStructureConfigTest(unittest.TestCase):
    def test_selected_s1_3_keeps_the_frozen_pilot_algorithm(self) -> None:
        pilot = json.loads(
            (ROOT / "configs" / "flow_structure_yaw_v1_3_pilot.json").read_text(
                encoding="utf-8"
            )
        )
        selected = json.loads(
            (ROOT / "configs" / "flow_structure_yaw_v1_3.json").read_text(
                encoding="utf-8"
            )
        )

        for config in (pilot, selected):
            config.pop("protocol")
            config.pop("protocol_role")
            config.pop("selection_status", None)
            config["promotion_criteria"].pop("status")

        self.assertEqual(selected, pilot)

    def test_selected_s1_3_does_not_claim_cross_model_promotion(self) -> None:
        selected = json.loads(
            (ROOT / "configs" / "flow_structure_yaw_v1_3.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(selected["protocol_role"], "selected_step1")
        self.assertEqual(
            selected["selection_status"],
            "frozen_pending_cross_model_separation",
        )
        self.assertEqual(
            selected["promotion_criteria"]["status"],
            "pending_same_distribution_two_model_separation",
        )
        self.assertEqual(
            selected["promotion_criteria"]["minimum_distinguishable_models"], 2
        )


if __name__ == "__main__":
    unittest.main()
