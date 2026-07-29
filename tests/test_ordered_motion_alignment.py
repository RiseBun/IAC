from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
for value in (PROJECT_ROOT, TOOLS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from iac_extensions.ordered_motion_alignment import (  # noqa: E402
    MOTION_FAMILIES,
    OrderedMotionAlignment,
    OrderedMotionConfig,
    trajectory_segment_target,
)
from ordered_motion_common import sattolo_indices  # noqa: E402
from tune_fuse_ordered_motion import _fused_scores  # noqa: E402


class TrajectorySegmentTargetTest(unittest.TestCase):
    def test_speed_scaling_changes_longitudinal_segments(self) -> None:
        slow = trajectory_segment_target(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            segment_count=3,
        )
        fast = trajectory_segment_target(
            [[1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
            segment_count=3,
        )
        self.assertEqual(tuple(slow.shape), (3, len(MOTION_FAMILIES)))
        self.assertTrue(torch.all(fast[:, 0] > slow[:, 0]))
        torch.testing.assert_close(slow[:, 1:], torch.zeros_like(slow[:, 1:]))

    def test_heading_and_curvature_are_explicit(self) -> None:
        target = trajectory_segment_target(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.5, 0.2],
                [3.0, 1.5, 0.5],
            ],
            segment_count=3,
        )
        self.assertGreater(float(target[:, 2].abs().sum()), 0.0)
        self.assertGreater(float(target[:, 3].abs().sum()), 0.0)


class OrderedAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = OrderedMotionAlignment(
            OrderedMotionConfig(
                visual_dim=8,
                hidden_dim=16,
                segment_count=4,
                bandwidth=0.24,
                dropout=0.0,
            )
        ).eval()
        base = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
        self.visual = base.repeat(2, 1, 8)

    def test_visual_estimator_is_candidate_blind(self) -> None:
        output_a = self.model(self.visual)
        output_b = self.model(self.visual)
        torch.testing.assert_close(
            output_a["visual_motion_mean_normalized"],
            output_b["visual_motion_mean_normalized"],
        )

        target_a = torch.zeros(2, 4, 4)
        target_b = torch.ones(2, 4, 4)
        evidence_a = self.model.evidence(output_a, target_a)
        evidence_b = self.model.evidence(output_b, target_b)
        self.assertFalse(
            torch.allclose(
                evidence_a["ordered_motion_energy"],
                evidence_b["ordered_motion_energy"],
            )
        )

    def test_time_reversal_changes_downstream_prediction(self) -> None:
        normal = self.model(self.visual)["visual_motion_mean_normalized"]
        reversed_output = self.model(
            self.visual.flip(1)
        )["visual_motion_mean_normalized"]
        self.assertGreater(
            float((normal - reversed_output).abs().mean().detach()),
            1e-6,
        )

    def test_evidence_is_non_negative_and_exactly_additive(self) -> None:
        output = self.model(self.visual)
        evidence = self.model.evidence(output, torch.zeros(2, 4, 4))
        self.assertTrue(
            torch.all(evidence["segment_component_contribution"] >= 0.0)
        )
        torch.testing.assert_close(
            evidence["family_contribution"].sum(dim=-1),
            evidence["ordered_motion_energy"],
        )


class ControlAndFusionTest(unittest.TestCase):
    def test_sattolo_has_no_fixed_point(self) -> None:
        order = sattolo_indices(12, 20260728)
        self.assertEqual(sorted(order), list(range(12)))
        self.assertTrue(all(index != value for index, value in enumerate(order)))

    def test_inference_fusion_does_not_read_source_labels(self) -> None:
        rows = [
            {
                "group_id": "g0",
                "sample_id": "a",
                "source_type": "gt_pos",
                "base": 0.6,
                "ordered_motion_energy": 0.1,
            },
            {
                "group_id": "g0",
                "sample_id": "b",
                "source_type": "image_swap",
                "base": 0.7,
                "ordered_motion_energy": 1.0,
            },
        ]
        relabeled = copy.deepcopy(rows)
        relabeled[0]["source_type"] = "hidden_a"
        relabeled[1]["source_type"] = "hidden_b"
        original, original_penalty = _fused_scores(
            rows,
            primary_key="base",
            energy_key="ordered_motion_energy",
            beta=0.2,
            threshold=0.0,
        )
        changed, changed_penalty = _fused_scores(
            relabeled,
            primary_key="base",
            energy_key="ordered_motion_energy",
            beta=0.2,
            threshold=0.0,
        )
        self.assertEqual(original, changed)
        self.assertEqual(original_penalty, changed_penalty)


if __name__ == "__main__":
    unittest.main()
