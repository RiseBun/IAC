from __future__ import annotations

import copy
import unittest

import torch

from iac_extensions.interpretable_motion_head import (
    InterpretableTrajectoryComparator,
)
from iac_extensions.motion_attribute_layout import (
    MOTION_FAMILY_NAMES,
    aggregate_motion_family_contributions,
    build_motion_attribute_layout,
)
from tools.apply_longitudinal_motion_residual import (
    apply_longitudinal_residual,
    trajectory_geometry,
)


class MotionAttributeLayoutTest(unittest.TestCase):
    def test_names_and_families_cover_the_36d_layout(self) -> None:
        layout = build_motion_attribute_layout(segment_count=3)
        self.assertEqual(layout.attribute_dim, 36)
        self.assertEqual(set(layout.attribute_families), set(MOTION_FAMILY_NAMES))
        indices = layout.family_indices()
        flattened = sorted(index for values in indices.values() for index in values)
        self.assertEqual(flattened, list(range(36)))

    def test_family_contributions_reconstruct_total_energy(self) -> None:
        components = torch.arange(1, 73, dtype=torch.float32).reshape(2, 36)
        family = aggregate_motion_family_contributions(
            components,
            segment_count=3,
        )
        self.assertEqual(family.shape, (2, 4))
        torch.testing.assert_close(family.sum(dim=-1), components.mean(dim=-1))


class InterpretableComparatorTest(unittest.TestCase):
    def test_ledger_is_non_negative_additive_and_differentiable(self) -> None:
        comparator = InterpretableTrajectoryComparator(
            36,
            segment_count=3,
        )
        visual = torch.zeros(2, 36, requires_grad=True)
        log_variance = torch.zeros_like(visual)
        target = torch.zeros_like(visual)
        target[1, 0] = 0.8
        result = comparator(visual, log_variance, target)

        self.assertEqual(result["family_contribution"].shape, (2, 4))
        self.assertTrue(torch.all(result["component_energy"] >= 0.0))
        self.assertTrue(
            torch.all(result["weighted_component_contribution"] >= 0.0)
        )
        torch.testing.assert_close(
            result["family_contribution"].sum(dim=-1),
            result["energy"],
        )
        self.assertLess(float(result["energy"][0]), float(result["energy"][1]))
        result["energy"].sum().backward()
        self.assertIsNotNone(visual.grad)


class LongitudinalResidualTest(unittest.TestCase):
    @staticmethod
    def _primary_rows() -> list[dict]:
        return [
            {
                "group_id": "g0",
                "sample_id": "gt",
                "source_type": "gt_pos",
                "consistency_label": 1,
                "iac_consistency": 0.60,
                "candidate_traj": [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
            },
            {
                "group_id": "g0",
                "sample_id": "speed",
                "source_type": "perturb_speed",
                "consistency_label": 0,
                "iac_consistency": 0.55,
                "candidate_traj": [
                    [1.5, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.5, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                ],
            },
            {
                "group_id": "g0",
                "sample_id": "lateral",
                "source_type": "perturb_lateral",
                "consistency_label": 0,
                "iac_consistency": 0.54,
                "candidate_traj": [
                    [1.0, 2.0, 0.0],
                    [2.0, 4.0, 0.0],
                    [3.0, 6.0, 0.0],
                    [4.0, 8.0, 0.0],
                ],
            },
        ]

    @staticmethod
    def _evidence_rows() -> list[dict]:
        return [
            {
                "group_id": "g0",
                "sample_id": "gt",
                "flow_speed_energy": 0.0,
            },
            {
                "group_id": "g0",
                "sample_id": "speed",
                "flow_speed_energy": 2.0,
            },
            {
                "group_id": "g0",
                "sample_id": "lateral",
                "flow_speed_energy": 2.0,
            },
        ]

    def test_geometry_separates_longitudinal_and_lateral_candidates(self) -> None:
        rows = self._primary_rows()
        speed = trajectory_geometry(
            rows[0]["candidate_traj"],
            rows[1]["candidate_traj"],
        )
        lateral = trajectory_geometry(
            rows[0]["candidate_traj"],
            rows[2]["candidate_traj"],
        )
        self.assertGreater(speed["longitudinal_share"], 0.9)
        self.assertLess(lateral["longitudinal_share"], 0.5)

    def test_only_longitudinal_candidate_receives_the_residual(self) -> None:
        rows = apply_longitudinal_residual(
            self._primary_rows(),
            self._evidence_rows(),
            weight=1.0,
        )
        by_id = {row["sample_id"]: row for row in rows}
        self.assertTrue(
            by_id["speed"]["interpretable_longitudinal_residual"]["active"]
        )
        self.assertLess(
            by_id["speed"]["iac_consistency_interpretable"],
            by_id["speed"]["iac_consistency"],
        )
        self.assertFalse(
            by_id["lateral"]["interpretable_longitudinal_residual"]["active"]
        )
        self.assertEqual(
            by_id["lateral"]["iac_consistency_interpretable"],
            by_id["lateral"]["iac_consistency"],
        )

    def test_source_labels_do_not_change_the_transformation(self) -> None:
        primary = self._primary_rows()
        relabeled = copy.deepcopy(primary)
        for index, row in enumerate(relabeled):
            row["source_type"] = f"hidden_{index}"
        original = apply_longitudinal_residual(
            primary,
            self._evidence_rows(),
            weight=0.7,
        )
        changed = apply_longitudinal_residual(
            relabeled,
            self._evidence_rows(),
            weight=0.7,
        )
        for left, right in zip(original, changed):
            self.assertEqual(
                left["iac_consistency_interpretable"],
                right["iac_consistency_interpretable"],
            )
            self.assertEqual(
                left["interpretable_longitudinal_residual"],
                right["interpretable_longitudinal_residual"],
            )


if __name__ == "__main__":
    unittest.main()
