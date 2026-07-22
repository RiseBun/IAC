#!/usr/bin/env python3
"""Train IAC with a candidate-blind temporal head on frozen DINO features.

This entry point intentionally reuses IAC's existing dataset, losses, DDP,
checkpointing and evaluation-facing output contract.  The only model change is
an opt-in visual motion head and an uncertainty-aware trajectory comparator.

Usage::

    python train_scope_motion_head.py \
      --config configs/train_navsim_future_dinov2_scope_motion_head.py \
      --max-train-steps 20 --max-val-steps 10
"""

from __future__ import annotations

from typing import Any, Dict

import torch

import train_dinov2_v5_minimal as iac_dino
from iac_extensions.dino_motion_head import (
    CandidateBlindDinoMotionHead,
    UncertaintyAwareTrajectoryComparator,
)


_BaseDinoCritic = iac_dino.DINOv2ConsistencyCritic


class ScopeDinoMotionCritic(_BaseDinoCritic):
    """IAC DINO critic with visual-first structured motion evidence."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__(cfg)
        dcfg = cfg.get("dinov2", {})
        if not bool(dcfg.get("use_scope_motion_head", False)):
            raise ValueError(
                "ScopeDinoMotionCritic requires dinov2.use_scope_motion_head=True"
            )
        hidden = int(dcfg.get("scope_motion_hidden_dim", cfg["model"]["hidden_dim"]))
        self.scope_motion_head = CandidateBlindDinoMotionHead(
            feature_dim=self.image_feature_dim,
            attribute_dim=self.motion_rule_attr_dim,
            hidden_dim=hidden,
            num_layers=int(dcfg.get("scope_motion_num_layers", 2)),
            num_heads=int(dcfg.get("scope_motion_num_heads", 4)),
            dropout=float(
                dcfg.get("scope_motion_dropout", cfg["model"].get("dropout", 0.1))
            ),
            segment_count=self.motion_rule_segment_count,
            max_frames=int(dcfg.get("scope_motion_max_frames", 32)),
        )
        self.scope_motion_comparator = UncertaintyAwareTrajectoryComparator(
            self.motion_rule_attr_dim
        )
        self._scope_forward_aux: Dict[str, torch.Tensor] = {}

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
        # Deliberately compute the visual estimate before candidate attributes
        # are introduced.  This is the anti-shortcut boundary of the module.
        visual = self.scope_motion_head(feats["hist_seq"], feats["fut_seq"])
        candidate_attributes = self._traj_rule_attributes(candidate_traj)
        if self.baseline_mode in {"no_traj", "ego_only"}:
            candidate_attributes = torch.zeros_like(candidate_attributes)
        comparison = self.scope_motion_comparator(
            visual["mean"],
            visual["log_variance"],
            candidate_attributes,
        )

        # Reuse IAC's existing learned-motion-rule training losses and scorer
        # fusion.  This keeps the extension directly comparable to their head.
        feats["visual_motion_rule_pred"] = visual["mean"]
        feats["visual_motion_rule_logvar"] = visual["log_variance"]
        feats["traj_motion_rule_target"] = candidate_attributes
        feats["motion_rule_match_logit"] = comparison["logit"]
        feats["scope_motion_energy"] = comparison["energy"]
        feats["scope_motion_component_energy"] = comparison["component_energy"]
        self._scope_forward_aux = {
            "visual_motion_rule_logvar": visual["log_variance"],
            "scope_motion_energy": comparison["energy"],
            "scope_motion_component_energy": comparison["component_energy"],
        }
        return feats

    def forward(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = super().forward(
            history_images,
            future_images,
            ego_state,
            candidate_traj,
        )
        outputs.update(self._scope_forward_aux)
        return outputs


# Public alias used by probe/evaluation code that imports the model module.
ConsistencyCriticModel = ScopeDinoMotionCritic


def main() -> None:
    # The upstream main function performs all CLI/config/DDP/checkpoint work and
    # resolves this global at model-construction time.
    iac_dino.DINOv2ConsistencyCritic = ScopeDinoMotionCritic
    iac_dino.ConsistencyCriticModel = ScopeDinoMotionCritic
    iac_dino.main()


if __name__ == "__main__":
    main()
