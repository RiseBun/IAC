from __future__ import annotations

import unittest

from scripts.evaluate_flow_structure import _apply_refinement_uncertainty_gate


class FlowStructureReliabilityTest(unittest.TestCase):
    def test_refinement_gate_recomputes_profile_availability(self) -> None:
        profile = {
            "rows": [
                {
                    "input_available": True,
                    "refinement_uncertainty": {"median": 0.02},
                },
                {
                    "input_available": True,
                    "refinement_uncertainty": {"median": 0.08},
                },
            ],
            "measurement_available": True,
            "available_interval_fraction": 1.0,
            "status": "usable",
            "parameters": {},
        }
        _apply_refinement_uncertainty_gate(
            profile,
            {"interval_gate": {"statistic": "median", "max_value": 0.05}},
        )
        self.assertTrue(profile["rows"][0]["input_available"])
        self.assertFalse(profile["rows"][1]["input_available"])
        self.assertFalse(profile["measurement_available"])
        self.assertEqual(profile["available_interval_fraction"], 0.5)
        self.assertEqual(profile["status"], "partial")


if __name__ == "__main__":
    unittest.main()
