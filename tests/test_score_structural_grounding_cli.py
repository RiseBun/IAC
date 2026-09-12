import json
import unittest
from pathlib import Path

from tools.score_structural_grounding import score


class StructuralGroundingCliTest(unittest.TestCase):
    def test_public_scales_use_log1p_domain_for_flow_magnitude(self) -> None:
        scales = json.loads((Path(__file__).parents[1] / "configs" / "gs_descriptor_scales.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(scales["median_flow_magnitude_px"], 0.6025444157006848)

    def test_scores_and_bootstraps_user_supplied_reference(self) -> None:
        scales = {
            "median_flow_magnitude_px": 1.0,
            "horizontal_flow_center": 1.0,
            "vertical_flow_center": 1.0,
            "divergence": 1.0,
            "curl": 1.0,
        }
        rows = []
        for interval in range(3):
            base = {
                "source_key": "s0",
                "interval_index": interval,
                "input_available": True,
                "median_flow_magnitude_px": 2.0,
                "horizontal_flow_center": 0.1,
                "vertical_flow_center": 0.2,
                "divergence": 0.3,
                "curl": 0.4,
            }
            rows.append(base)
        report = score(rows, rows, scales=scales, bootstrap_draws=20)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["source_coverage"], 1.0)
        self.assertAlmostEqual(report["conditional_score"], 1.0)
        self.assertEqual(report["score_coverage"], 1.0)
        self.assertEqual(report["status_counts"], {"scored": 1, "abstain": 0, "unavailable": 0})
        self.assertAlmostEqual(report["score_median"], 1.0)

    def test_missing_reference_interval_is_not_zero_filled(self) -> None:
        scales = {key: 1.0 for key in ("median_flow_magnitude_px", "horizontal_flow_center", "vertical_flow_center", "divergence", "curl")}
        generated = [{"source_key": "s0", "interval_index": i, "input_available": True, **{key: 1.0 for key in scales}} for i in range(3)]
        reference = [dict(generated[0]), dict(generated[1])]
        report = score(generated, reference, scales=scales, bootstrap_draws=20)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["scored_source_count"], 0)
        self.assertIsNone(report["conditional_score"])
        self.assertEqual(report["status_counts"]["unavailable"], 1)

    def test_reference_only_sources_do_not_dilute_generated_coverage(self) -> None:
        scales = {key: 1.0 for key in ("median_flow_magnitude_px", "horizontal_flow_center", "vertical_flow_center", "divergence", "curl")}
        generated = [
            {"source_key": "evaluated", "interval_index": i, "input_available": True, **{key: 1.0 for key in scales}}
            for i in range(3)
        ]
        reference = list(generated) + [
            {"source_key": "extra_private_reference", "interval_index": i, "input_available": True, **{key: 1.0 for key in scales}}
            for i in range(3)
        ]
        report = score(generated, reference, scales=scales, bootstrap_draws=20)
        self.assertEqual(report["source_coverage"], 1.0)
        self.assertEqual(report["reference_only_source_count"], 1)
