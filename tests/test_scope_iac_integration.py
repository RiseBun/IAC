from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from configs.train_navsim_future_dinov2_scope_motion_head import cfg
from train_scope_motion_head import ScopeDinoMotionCritic


class _DummyFrameEncoder(nn.Module):
    """Avoid network/model downloads while exercising the complete wrapper."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (1, 1))
        scalar = pooled.flatten(1).mean(dim=1, keepdim=True)
        return scalar.repeat(1, self.output_dim)


class ScopeIacIntegrationTest(unittest.TestCase):
    def test_wrapper_preserves_iac_output_contract(self) -> None:
        test_cfg = copy.deepcopy(cfg)
        # Build without contacting torch.hub, then replace the encoder with a
        # deterministic stand-in.  All IAC fusion and extension code still run.
        test_cfg["dinov2"]["enabled"] = False
        model = ScopeDinoMotionCritic(test_cfg).eval()
        model.image_encoder = _DummyFrameEncoder(
            int(test_cfg["model"]["image_feature_dim"])
        )
        model.use_dinov2 = True

        batch = 2
        history = torch.randn(batch, test_cfg["history_num_frames"], 3, 32, 48)
        future = torch.randn(batch, test_cfg["future_num_frames"], 3, 32, 48)
        ego = torch.randn(batch, test_cfg["ego_state_dim"])
        trajectory = torch.randn(
            batch, test_cfg["candidate_traj_steps"], test_cfg["traj_dim"]
        )
        with torch.inference_mode():
            output = model(history, future, ego, trajectory)

        self.assertEqual(output["consistency_logit"].shape, (batch,))
        self.assertEqual(output["visual_motion_rule_pred"].shape, (batch, 36))
        self.assertEqual(output["visual_motion_rule_logvar"].shape, (batch, 36))
        self.assertEqual(output["traj_motion_rule_target"].shape, (batch, 36))
        self.assertEqual(output["scope_motion_energy"].shape, (batch,))
        self.assertEqual(
            output["scope_motion_component_energy"].shape, (batch, 36)
        )

    def test_rgb_diff_motion_head_preserves_output_contract(self) -> None:
        test_cfg = copy.deepcopy(cfg)
        test_cfg["dinov2"]["enabled"] = False
        test_cfg["dinov2"]["use_scope_rgb_diff_motion_head"] = True
        test_cfg["dinov2"]["scope_rgb_diff_spatial_size"] = 32
        model = ScopeDinoMotionCritic(test_cfg).eval()
        model.image_encoder = _DummyFrameEncoder(
            int(test_cfg["model"]["image_feature_dim"])
        )
        model.use_dinov2 = True

        batch = 2
        history = torch.randn(batch, test_cfg["history_num_frames"], 3, 32, 48)
        future = torch.randn(batch, test_cfg["future_num_frames"], 3, 32, 48)
        ego = torch.randn(batch, test_cfg["ego_state_dim"])
        trajectory = torch.randn(
            batch, test_cfg["candidate_traj_steps"], test_cfg["traj_dim"]
        )
        with torch.inference_mode():
            output = model(history, future, ego, trajectory)

        self.assertEqual(output["consistency_logit"].shape, (batch,))
        self.assertEqual(output["visual_motion_rule_pred"].shape, (batch, 36))
        self.assertEqual(output["scope_motion_energy"].shape, (batch,))


if __name__ == "__main__":
    unittest.main()
