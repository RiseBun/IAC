#!/usr/bin/env python3
"""Train/evaluate SCOPE with an explicit, additive motion evidence ledger.

This entry point is checkpoint-compatible with ``train_scope_motion_head.py``:
it keeps the same candidate-blind visual head and the same comparator
parameters.  The only change is that the forward output exposes named evidence
families and the values needed to diagnose every family.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

import train_dinov2_v5_minimal as iac_dino
from iac_extensions.interpretable_motion_head import (
    InterpretableTrajectoryComparator,
)
from iac_extensions.motion_attribute_layout import MOTION_FAMILY_NAMES
from train_scope_motion_head import ScopeDinoMotionCritic


class InterpretableScopeDinoMotionCritic(ScopeDinoMotionCritic):
    """SCOPE critic whose motion score is auditable family by family."""

    motion_family_names = MOTION_FAMILY_NAMES

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        self.scope_motion_comparator = InterpretableTrajectoryComparator(
            self.motion_rule_attr_dim,
            segment_count=self.motion_rule_segment_count,
        )

    def extract_probe_features(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats = super().extract_probe_features(
            history_images=history_images,
            future_images=future_images,
            ego_state=ego_state,
            candidate_traj=candidate_traj,
        )
        comparison = self.scope_motion_comparator(
            feats["visual_motion_rule_pred"],
            feats["visual_motion_rule_logvar"],
            feats["traj_motion_rule_target"],
        )
        feats["motion_rule_match_logit"] = comparison["logit"]
        feats["scope_motion_energy"] = comparison["energy"]
        feats["scope_motion_component_energy"] = comparison["component_energy"]
        feats["scope_motion_family_contribution"] = comparison[
            "family_contribution"
        ]
        self._scope_forward_aux.update(
            {
                "scope_motion_energy": comparison["energy"],
                "scope_motion_component_energy": comparison["component_energy"],
                "scope_motion_raw_component_energy": comparison[
                    "raw_component_energy"
                ],
                "scope_motion_weighted_component_contribution": comparison[
                    "weighted_component_contribution"
                ],
                "scope_motion_family_contribution": comparison[
                    "family_contribution"
                ],
                "scope_motion_normalized_residual": comparison[
                    "normalized_residual"
                ],
                "scope_motion_visual_standard_deviation": comparison[
                    "visual_standard_deviation"
                ],
            }
        )
        return feats


ConsistencyCriticModel = InterpretableScopeDinoMotionCritic


def main() -> None:
    iac_dino.DINOv2ConsistencyCritic = InterpretableScopeDinoMotionCritic
    iac_dino.ConsistencyCriticModel = InterpretableScopeDinoMotionCritic
    iac_dino.main()


if __name__ == "__main__":
    main()
