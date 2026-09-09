import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.evaluate_cfac_fau import shape_gate as cfac_shape_gate
from scripts.evaluate_continuous_decoder import (
    apply_projection_support_contract,
    validate_manifest_records,
)
from scripts.evaluate_counterfactual_alignment import shape_gate as ccfc_shape_gate


def _quality(**extra):
    row = {
        "effective_static_pixel_fraction": 1.0,
        "direction_observable": True,
        "curvature_observable": True,
        "curvature_status": "usable",
        "status": "good",
    }
    row.update(extra)
    return row


class ProjectionSupportContractTest(unittest.TestCase):
    def test_evaluator_does_not_use_config_size_as_manifest_fallback(self) -> None:
        row = {
            "sample_id": "s",
            "frame_paths": ["a.jpg", "b.jpg"],
            "frame_times_s": [0.0, 1.0],
            "intrinsics": [[100.0, 0.0, 20.0], [0.0, 100.0, 15.0], [0.0, 0.0, 1.0]],
            "camera_to_ego": np.eye(4).tolist(),
            "candidates": [
                {"candidate_id": "a", "trajectory": [[1.0, 0.0, 0.0]]},
                {"candidate_id": "b", "trajectory": [[1.0, 0.1, 0.0]]},
            ],
        }
        config = {
            "intrinsics_source_size": [1920, 1080],
            "calibration_contract": {
                "require_explicit_intrinsics_source_size": True,
                "expected_intrinsics_source_size": [1920, 1080],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "required by the calibration contract"):
                validate_manifest_records([row], config, Path(directory))

    def test_level1_zero_projection_cannot_report_confidence_or_validity(self) -> None:
        decoded = {
            "valid": True,
            "projection_supported": False,
            "projected_weight_fraction": 0.0,
            "projection_support_by_interval": [{
                "source_points": 900,
                "projected_points": 0,
                "projected_weight_fraction": 0.0,
                "projection_supported": False,
                "reasons": ["insufficient_projected_points"],
            }],
            "speed_support": [{"observability": 1.0, "status": "usable"}],
        }
        quality = [_quality()]
        composite = apply_projection_support_contract(decoded, quality)
        self.assertEqual(composite.tolist(), [0.0])
        self.assertFalse(decoded["valid"])
        self.assertEqual(decoded["observability"], 0.0)
        self.assertEqual(decoded["speed_support"][0]["status"], "abstain")
        self.assertEqual(quality[0]["curvature_status"], "abstain")
        self.assertFalse(quality[0]["direction_observable"])

    def test_cfac_and_ccfc_abstain_without_explicit_projection_support(self) -> None:
        score = {
            "motion_explanation_status": "explained",
            "observability_by_future_interval": [_quality()],
        }
        for gate in (cfac_shape_gate, ccfc_shape_gate):
            statuses, observability, reasons = gate(score, 1)
            self.assertEqual(statuses, ["abstain"])
            self.assertEqual(observability, [0.0])
            self.assertEqual(reasons, ["projection_support_failed"])

    def test_cfac_and_ccfc_use_composite_observability(self) -> None:
        score = {
            "motion_explanation_status": "explained",
            "observability_by_future_interval": [_quality(
                projection_supported=True,
                projected_weight_fraction=0.75,
                composite_observability=0.6,
            )],
        }
        for gate in (cfac_shape_gate, ccfc_shape_gate):
            statuses, observability, reasons = gate(score, 1)
            self.assertEqual(statuses, ["usable"])
            self.assertEqual(observability, [0.6])
            self.assertEqual(reasons, ["direct_flow_geometry"])

    def test_cfac_and_ccfc_abstain_when_fit_is_weak(self) -> None:
        score = {
            "motion_explanation_status": "weak",
            "observability_by_future_interval": [_quality(
                projection_supported=True,
                composite_observability=0.8,
            )],
        }
        for gate in (cfac_shape_gate, ccfc_shape_gate):
            statuses, observability, reasons = gate(score, 1)
            self.assertEqual(statuses, ["abstain"])
            self.assertEqual(observability, [0.0])
            self.assertEqual(reasons, ["motion_not_explained:weak"])


if __name__ == "__main__":
    unittest.main()
