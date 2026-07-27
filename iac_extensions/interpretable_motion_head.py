"""Interpretable extension of the candidate-blind SCOPE motion comparator.

The existing comparator already produces a non-negative energy for every
motion attribute.  This module makes the internal accounting explicit:

* visual estimate;
* deterministic candidate target;
* visual uncertainty;
* normalized residual;
* raw per-attribute energy;
* learned non-negative weight;
* weighted per-attribute contribution;
* four named family contributions that sum exactly to total motion energy.

No new black-box fusion layer is introduced.
"""

from __future__ import annotations

from typing import Dict

import torch

from .dino_motion_head import UncertaintyAwareTrajectoryComparator
from .motion_attribute_layout import aggregate_motion_family_contributions


class InterpretableTrajectoryComparator(UncertaintyAwareTrajectoryComparator):
    """Drop-in comparator that returns a complete motion evidence ledger."""

    def __init__(
        self,
        attribute_dim: int,
        *,
        segment_count: int,
        eps: float = 1e-4,
    ) -> None:
        super().__init__(attribute_dim=attribute_dim, eps=eps)
        self.segment_count = max(0, int(segment_count))

    def forward(
        self,
        visual_mean: torch.Tensor,
        visual_log_variance: torch.Tensor,
        candidate_attributes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        result = super().forward(
            visual_mean,
            visual_log_variance,
            candidate_attributes,
        )
        variance = visual_log_variance.exp().clamp_min(self.eps)
        residual = visual_mean - candidate_attributes
        raw_component_energy = 0.5 * (
            residual.square() / variance
            + torch.nn.functional.softplus(visual_log_variance)
        )
        family_contribution = aggregate_motion_family_contributions(
            result["component_energy"],
            segment_count=self.segment_count,
        )
        result.update(
            {
                "visual_mean": visual_mean,
                "candidate_target": candidate_attributes,
                "visual_log_variance": visual_log_variance,
                "visual_standard_deviation": variance.sqrt(),
                "residual": residual,
                "normalized_residual": residual / variance.sqrt(),
                "raw_component_energy": raw_component_energy,
                "weighted_component_contribution": result["component_energy"]
                / float(self.attribute_dim),
                "family_contribution": family_contribution,
            }
        )
        return result
