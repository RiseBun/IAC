from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from configs.train_navsim_future_dinov2_scope_interpretable_motion import cfg
from train_scope_interpretable_motion_head import (
    InterpretableScopeDinoMotionCritic,
)


class _DummyFrameEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (1, 1))
        scalar = pooled.flatten(1).mean(dim=1, keepdim=True)
        return scalar.repeat(1, self.output_dim)


class InterpretableScopeIntegrationTest(unittest.TestCase):
    def test_full_wrapper_exports_an_additive_family_ledger(self) -> None:
        test_cfg = copy.deepcopy(cfg)
        test_cfg["dinov2"]["enabled"] = False
        model = InterpretableScopeDinoMotionCritic(test_cfg).eval()
        model.image_encoder = _DummyFrameEncoder(
            int(test_cfg["model"]["image_feature_dim"])
        )
        model.use_dinov2 = True

        batch = 2
        history = torch.randn(batch, test_cfg["history_num_frames"], 3, 32, 48)
        future = torch.randn(batch, test_cfg["future_num_frames"], 3, 32, 48)
        ego = torch.randn(batch, test_cfg["ego_state_dim"])
        trajectory = torch.randn(
            batch,
            test_cfg["candidate_traj_steps"],
            test_cfg["traj_dim"],
        )
        with torch.inference_mode():
            output = model(history, future, ego, trajectory)

        self.assertEqual(output["consistency_logit"].shape, (batch,))
        self.assertEqual(
            output["scope_motion_family_contribution"].shape,
            (batch, 4),
        )
        self.assertEqual(
            output["scope_motion_normalized_residual"].shape,
            (batch, 36),
        )
        torch.testing.assert_close(
            output["scope_motion_family_contribution"].sum(dim=-1),
            output["scope_motion_energy"],
        )


if __name__ == "__main__":
    unittest.main()
