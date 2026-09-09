import unittest

from scripts.evaluate_continuous_motion_alignment import _shape_eligibility


class Step1YawGateTest(unittest.TestCase):
    def test_yaw_is_not_gated_by_diagnostic_curvature(self) -> None:
        status, observability, reasons = _shape_eligibility(
            [{
                "projection_supported": True,
                "direction_observable": True,
                "curvature_status": "abstain",
                "composite_observability": 0.7,
            }],
            [],
            motion_explanation_status="explained",
        )
        self.assertEqual(status, ["usable"])
        self.assertEqual(observability, [0.7])
        self.assertEqual(reasons, ["direct_flow_geometry"])

    def test_non_explained_fit_abstains(self) -> None:
        status, observability, reasons = _shape_eligibility(
            [{
                "projection_supported": True,
                "direction_observable": True,
                "composite_observability": 1.0,
            }],
            [],
            motion_explanation_status="weak",
        )
        self.assertEqual(status, ["abstain"])
        self.assertEqual(observability, [0.0])
        self.assertEqual(reasons, ["motion_not_explained:weak"])


if __name__ == "__main__":
    unittest.main()
