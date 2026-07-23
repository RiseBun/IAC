#!/usr/bin/env python3
"""DINOv2-backboned Consistency Critic (minimal variant).

This is the *minimal* DINOv2 integration for IAC, deliberately scoped:

  ✔ 1️⃣ DINOv2-vits14 backbone replaces the 4-layer CNN
  ✔ 3️⃣ Explicit `diff / l2_norm / cos_sim` features concatenated into
     the fusion head (zero-cost, high-yield shortcut signal)
  ✘ 2️⃣ Multi-layer fusion (start with single layer [11] — multi-layer
     adds 6×proj params and overfits on the 357k anchor set)
  ✘ 4️⃣ AvgPool(k=2) (no ablation evidence, default off)
  ✘ 5️⃣ Ridge pretrain of layer weights (nuPlan data is too small to
     justify; PDF-cited "nuScenes SROCC 0.9275" has no source)
  ✘ 6️⃣ Geometric margin-ranking reg (likely hurts the consistency
     critic; one DINOv2 forward is wasted per training step)

Inherits everything else from train.py:
  - Same ConsistencyDataset
  - Same DDP / checkpoint / SIGTERM handling
  - Same eval flow (ConsistencyCriticModel alias)
  - Same difficulty-stratified sampler (D1..D4) when config enables it

Usage::

  # Smoke
  python train_dinov2_v5_minimal.py \
    --config configs/train_dinov2_v5_minimal.py \
    --work-dir work_dirs/iac_dinov2_v5_smoke \
    --epochs 1 --batch-size 32 --max-train-steps 20

  # Full
  python train_dinov2_v5_minimal.py \
    --config configs/train_dinov2_v5_minimal.py \
    --work-dir work_dirs/iac_dinov2_v5 \
    --epochs 5 --batch-size 32
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# ─────────────────────────── re-use train.py primitives ───────────────────────────
# We import after defining parse_args so that "python train_dinov2_v5_minimal.py
# --help" still works without the heavy torch.hub load.

_DINOV2_MODEL_NAMES = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14")
_DINOV2_SPECS: Dict[str, Dict[str, int]] = {
    "dinov2_vits14": {"feat_dim": 384, "n_blocks": 12},
    "dinov2_vitb14": {"feat_dim": 768, "n_blocks": 12},
    "dinov2_vitl14": {"feat_dim": 1024, "n_blocks": 24},
}


class ResidualLayerAttention(nn.Module):
    """Residual gating over per-layer DINO features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, layer_feats: torch.Tensor) -> torch.Tensor:
        # layer_feats: (B, L, D)
        summary = layer_feats.mean(dim=1, keepdim=True).expand_as(layer_feats)
        gate_in = torch.cat([layer_feats, summary], dim=-1)
        logits = self.score(gate_in).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        base = layer_feats[:, -1, :]
        delta = torch.sum(
            weights.unsqueeze(-1) * (layer_feats - base.unsqueeze(1)),
            dim=1,
        )
        return base + delta


def _import_train():
    """Late import so that --help works before DINOv2 is loaded."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train  # type: ignore
    return train


# ─────────────────────────── CLI ───────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINOv2 Consistency Critic (minimal v5 variant)"
    )
    p.add_argument("--config", required=True, help="Python config path")
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--baseline-mode",
        choices=["full", "no_image", "ego_only", "no_traj", "traj_only"],
        default=None,
    )
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--preflight-samples", type=int, default=128)
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume model/optimizer state from a checkpoint.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable CUDA autocast mixed precision.",
    )
    p.add_argument(
        "--dinov2-model", type=str, default=None,
        choices=list(_DINOV2_MODEL_NAMES),
    )
    p.add_argument(
        "--dinov2-freeze",
        dest="dinov2_freeze",
        action="store_true",
        default=None,
        help="Freeze DINOv2 backbone (default: use config value).",
    )
    p.add_argument(
        "--dinov2-trainable",
        dest="dinov2_trainable",
        action="store_true",
        default=False,
        help="Unfreeze DINOv2 backbone for fine-tuning.",
    )
    p.add_argument(
        "--no-dinov2",
        dest="no_dinov2",
        action="store_true",
        default=False,
        help="Disable DINOv2 backbone and use the 4-layer CNN from train.py instead. "
        "Lets you A/B the D1-D4 sampling effect with the original backbone.",
    )
    return p.parse_args()


# ─────────────────────────── DINOv2 encoder (single-layer, no AvgPool) ───────────────────────────


class DINOv2Encoder(nn.Module):
    """DINOv2 single-layer encoder. Returns a (B, out_dim) vector per image batch.

    Differences from the PDF-supplied v3 script:
      * Single layer only (default [11]).
      * No AvgPool — DINOv2 patch tokens are used as-is.
      * No Ridge pretrain — layer weights are uniform 1.0.
      * mean() pool over patch tokens (excluding the CLS token) for a
        dense single-vector representation; cheaper than concatenating
        6 layers and avoids overfitting on 357k samples.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        layer_index: int = 11,
        layer_indices: Sequence[int] | None = None,
        layer_mode: str = "single",
        layer_gate_hidden: int = 128,
        layer_residual_scale: float = 1.0,
        out_dim: int = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in _DINOV2_SPECS:
            raise ValueError(
                f"Unsupported DINOv2 model '{model_name}'. "
                f"Choose from {list(_DINOV2_MODEL_NAMES)}."
            )
        spec = _DINOV2_SPECS[model_name]
        n_blocks = spec["n_blocks"]
        if layer_indices is None:
            layer_indices = [layer_index]
        self.layer_indices = [int(idx) for idx in layer_indices]
        if not self.layer_indices:
            raise ValueError("layer_indices must contain at least one DINOv2 layer.")
        for idx in self.layer_indices:
            if idx < 0 or idx >= n_blocks:
                raise ValueError(
                    f"layer index {idx} out of range [0, {n_blocks - 1}]"
                )
        if layer_index < 0 or layer_index >= n_blocks:
            raise ValueError(
                f"layer_index {layer_index} out of range [0, {n_blocks - 1}]"
            )
        self.model_name = model_name
        self.layer_index = self.layer_indices[-1]
        self.max_layer_index = max(self.layer_indices)
        self.layer_mode = str(layer_mode)
        if self.layer_mode not in {"single", "multi", "gated"}:
            raise ValueError("layer_mode must be one of: single, multi, gated")
        self.feat_dim = spec["feat_dim"]
        self.freeze = freeze
        self.layer_residual_scale = float(layer_residual_scale)

        # Prefer the already-cached torch hub checkout on servers. This avoids
        # occasional torch.hub network/trust-list stalls during evaluation.
        hub_dir = os.environ.get("DINOV2_HUB_DIR")
        if not hub_dir:
            torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
            cached_hub = torch_home / "hub" / "facebookresearch_dinov2_main"
            if cached_hub.exists():
                hub_dir = str(cached_hub)
        if hub_dir and Path(hub_dir).exists():
            self.model = torch.hub.load(hub_dir, model_name, source="local")
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feat_dim, out_dim),
                    nn.LayerNorm(out_dim),
                )
                for _ in self.layer_indices
            ]
        )
        self.layer_selector = (
            ResidualLayerAttention(out_dim, hidden_dim=layer_gate_hidden)
            if self.layer_mode == "gated" and len(self.layer_indices) > 1
            else None
        )
        self.proj = self.projs[-1]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # Keep DINOv2 in eval mode (BN stats frozen) even if the
            # surrounding module is in train mode.
            self.model.eval()
        return self

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 expects 14*N input; 224 is fine.
        if x.shape[-2] != 224 or x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B*T, 3, H, W) → (B*T, out_dim)."""
        x = self._resize(images)
        ctx = torch.no_grad if self.freeze else torch.enable_grad
        with ctx():
            outputs = self.model.get_intermediate_layers(
                x, n=self.max_layer_index + 1, return_class_token=True,
            )
        # outputs: tuple of (patch_tokens, cls_token) per layer
        # patch_tokens: (B*T, n_patches, feat_dim); cls_token: (B*T, feat_dim)
        layer_feats: List[torch.Tensor] = []
        for proj, layer_idx in zip(self.projs, self.layer_indices):
            patch_tokens, cls_token = outputs[layer_idx]
            feat = patch_tokens.mean(dim=1)
            layer_feats.append(proj(feat))
        if len(layer_feats) == 1 or self.layer_mode == "single":
            return layer_feats[-1]
        stacked = torch.stack(layer_feats, dim=1)  # (B*T, L, out_dim)
        if self.layer_mode == "multi":
            return stacked.mean(dim=1)
        assert self.layer_selector is not None
        fused = self.layer_selector(stacked)
        if self.layer_residual_scale != 1.0:
            base = stacked[:, -1, :]
            fused = base + self.layer_residual_scale * (fused - base)
        return fused

    def forward_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return projected last-layer patch tokens: (B*T, P, out_dim)."""
        x = self._resize(images)
        ctx = torch.no_grad if self.freeze else torch.enable_grad
        with ctx():
            outputs = self.model.get_intermediate_layers(
                x, n=self.max_layer_index + 1, return_class_token=True,
            )
        patch_tokens, _cls_token = outputs[self.layer_index]
        return self.proj(patch_tokens)


# ─────────────────────────── Critic model ───────────────────────────


class DINOv2ConsistencyCritic(nn.Module):
    """Minimal DINOv2 critic: single-layer backbone + explicit distance fusion.

    When ``cfg['dinov2']['enabled']`` is False the model falls back to
    the original 4-layer CNN backbone from train.py, with the same
    fusion head and explicit-distance option. This makes the trainer
    a strict superset of train.py so that running it with dinov2
    disabled is a clean A/B against train.py itself.

    Shape contract identical to train.ConsistencyCriticModel so that
    eval_dinov2_critic.py and benchmark_wam.py work without modification.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        mcfg = cfg["model"]
        dcfg = cfg.get("dinov2", {})

        img_dim = int(mcfg["image_feature_dim"])
        act_dim = int(mcfg["action_feature_dim"])
        hidden = int(mcfg["hidden_dim"])
        fusion_dim = int(mcfg.get("fusion_dim", 256))
        dropout = float(mcfg.get("dropout", 0.0))
        ego_dim = int(cfg["ego_state_dim"])
        traj_steps = int(cfg["candidate_traj_steps"])
        consistency_traj_steps = int(
            cfg.get(
                "consistency_traj_steps",
                min(int(cfg.get("future_num_frames", traj_steps)), traj_steps),
            )
        )
        traj_d = int(cfg["traj_dim"])
        self.image_feature_dim = img_dim
        self.baseline_mode = str(cfg.get("baseline_mode", "full"))
        self.consistency_traj_steps = consistency_traj_steps
        ds_cfg = cfg.get("dataset", {})
        self.traj_normalize_mode = str(ds_cfg.get("normalize_mode", "tanh"))
        traj_scale = ds_cfg.get("traj_scale", None)
        if traj_scale is not None:
            self.register_buffer(
                "traj_scale",
                torch.tensor(traj_scale, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.traj_scale = None
        self.use_dinov2 = bool(dcfg.get("enabled", True))
        self.use_explicit_distance = bool(dcfg.get("use_explicit_distance", True))
        self.use_motion_features = bool(dcfg.get("use_motion_features", False))
        self.use_traj_geometry_features = bool(
            dcfg.get("use_traj_geometry_features", False)
        )
        self.use_future_traj_geometry_prediction = bool(
            dcfg.get("use_future_traj_geometry_prediction", False)
        )
        self.use_future_consistency_evidence = bool(
            dcfg.get("use_future_consistency_evidence", False)
        )
        self.future_consistency_mix = float(dcfg.get("future_consistency_mix", 0.5))
        self.use_hierarchical_consistency = bool(
            dcfg.get("use_hierarchical_consistency", False)
        )
        self.use_action_visual_interaction = bool(
            mcfg.get("use_action_visual_interaction", False)
        )
        self.use_path_conditioned_evidence = bool(
            dcfg.get("use_path_conditioned_evidence", False)
        )
        self.use_path_residual_score = bool(
            dcfg.get("use_path_residual_score", False)
        )
        self.path_residual_mix = float(dcfg.get("path_residual_mix", 0.5))
        self.use_path_evidence_head = bool(
            dcfg.get("use_path_evidence_head", False)
        )
        self.mix_path_evidence_into_consistency = bool(
            dcfg.get("mix_path_evidence_into_consistency", False)
        )
        self.path_evidence_mix = float(dcfg.get("path_evidence_mix", 0.0))
        self.use_path_evidence_gate = bool(
            dcfg.get("use_path_evidence_gate", False)
        )
        self.path_evidence_use_transition_context = bool(
            dcfg.get("path_evidence_use_transition_context", False)
        )
        self.use_learned_motion_rules = bool(
            dcfg.get("use_learned_motion_rules", False)
        )
        self.use_motion_latent_alignment = bool(
            dcfg.get("use_motion_latent_alignment", False)
        )
        self.motion_latent_dim = int(dcfg.get("motion_latent_dim", 64))
        self.use_video_action_cross_attention = bool(
            dcfg.get("use_video_action_cross_attention", False)
        )
        self.video_action_add_to_shared = bool(
            dcfg.get("video_action_add_to_shared", True)
        )
        self.video_action_num_heads = int(dcfg.get("video_action_num_heads", 4))
        self.video_action_mix = float(dcfg.get("video_action_mix", 0.0))
        self.use_future_latent_prediction = bool(
            dcfg.get("use_future_latent_prediction", False)
        )
        self.future_latent_mix = float(dcfg.get("future_latent_mix", 0.0))
        self.use_trajectory_reasonableness_head = bool(
            dcfg.get("use_trajectory_reasonableness_head", False)
        )
        self.use_learned_consistency_fusion = bool(
            dcfg.get("use_learned_consistency_fusion", False)
        )
        self.consistency_fusion_reasonableness_mix = float(
            dcfg.get("consistency_fusion_reasonableness_mix", 0.25)
        )
        self.consistency_fusion_use_repr = bool(
            dcfg.get("consistency_fusion_use_repr", True)
        )
        self.motion_rule_global_attr_dim = 12
        self.motion_rule_segment_attr_dim = 8
        self.motion_rule_segment_count = max(
            0, int(dcfg.get("motion_rule_segment_count", 0))
        )
        self.motion_rule_attr_dim = (
            self.motion_rule_global_attr_dim
            + self.motion_rule_segment_count * self.motion_rule_segment_attr_dim
        )
        configured_rule_attr_dim = int(
            dcfg.get("motion_rule_attr_dim", self.motion_rule_attr_dim)
        )
        if configured_rule_attr_dim != self.motion_rule_attr_dim:
            raise ValueError(
                "dinov2.motion_rule_attr_dim must match the configured "
                "global+segment rule layout; expected "
                f"{self.motion_rule_attr_dim}; got {configured_rule_attr_dim}"
            )
        self.motion_rule_mix = float(dcfg.get("motion_rule_mix", 0.0))
        self.use_progress_alignment_head = bool(
            dcfg.get(
                "use_progress_alignment_head",
                float(cfg.get("lambda_progress_alignment", 0.0)) > 0.0,
            )
        )
        self.path_conditioned_forward_m = float(
            dcfg.get("path_conditioned_forward_m", 40.0)
        )
        self.path_conditioned_lateral_m = float(
            dcfg.get("path_conditioned_lateral_m", 10.0)
        )
        self.path_conditioned_width = float(
            dcfg.get("path_conditioned_width", 0.10)
        )
        self.path_conditioned_segment_count = max(
            1, int(dcfg.get("path_conditioned_segment_count", 1))
        )
        self.temporal_encoder_type = str(mcfg.get("temporal_encoder", "mean"))
        if self.temporal_encoder_type not in {"mean", "gru"}:
            raise ValueError("model.temporal_encoder must be one of: mean, gru")

        if self.temporal_encoder_type == "gru":
            self.history_temporal_encoder = nn.GRU(
                input_size=img_dim,
                hidden_size=img_dim,
                batch_first=True,
            )
            self.future_temporal_encoder = nn.GRU(
                input_size=img_dim,
                hidden_size=img_dim,
                batch_first=True,
            )
        else:
            self.history_temporal_encoder = None
            self.future_temporal_encoder = None

        if self.use_dinov2:
            model_name = str(dcfg.get("model_name", "dinov2_vits14"))
            layer_index = int(dcfg.get("layer_index", 11))
            layer_indices = dcfg.get("layer_indices")
            layer_mode = str(dcfg.get("layer_mode", "single"))
            layer_gate_hidden = int(dcfg.get("layer_gate_hidden", 128))
            layer_residual_scale = float(dcfg.get("layer_residual_scale", 1.0))
            freeze = bool(dcfg.get("freeze", True))
            self.image_encoder = DINOv2Encoder(
                model_name=model_name,
                layer_index=layer_index,
                layer_indices=layer_indices,
                layer_mode=layer_mode,
                layer_gate_hidden=layer_gate_hidden,
                layer_residual_scale=layer_residual_scale,
                out_dim=img_dim,
                freeze=freeze,
            )
            self.history_proj = self.image_encoder.proj
            self.future_proj = self.image_encoder.proj
        else:
            # Fall back to the original 4-layer CNN backbone. We
            # import lazily so that a pure DINOv2 run never has to
            # load train.py's ConsistencyCriticModel class.
            from train import ConsistencyCriticModel as _CNNCritic  # type: ignore
            cnn = _CNNCritic(cfg)
            self.image_encoder = cnn  # for state-dict symmetry
            self.history_proj = cnn.history_proj
            self.future_proj = cnn.future_proj
            self._cnn_shared_backbone = cnn.shared_backbone

        if self.use_action_visual_interaction:
            self.action_to_visual_delta = nn.Sequential(
                nn.Linear(act_dim * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, img_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.action_to_visual_delta = None

        self.consistency_traj_encoder = nn.Sequential(
            nn.Linear(consistency_traj_steps * traj_d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, act_dim),
            nn.ReLU(inplace=True),
        )
        self.validity_traj_encoder = nn.Sequential(
            nn.Linear(traj_steps * traj_d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, act_dim),
            nn.ReLU(inplace=True),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, act_dim),
            nn.ReLU(inplace=True),
        )
        if self.use_future_traj_geometry_prediction:
            self.future_traj_geometry_head = nn.Sequential(
                nn.Linear(img_dim * 3, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 8),
            )
        else:
            self.future_traj_geometry_head = None

        if self.use_future_consistency_evidence:
            self.future_consistency_evidence_head = nn.Sequential(
                nn.Linear(img_dim * 3, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, img_dim),
            )
            self.future_consistency_evidence_logit_head = nn.Sequential(
                nn.Linear(img_dim, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.future_consistency_evidence_head = None
            self.future_consistency_evidence_logit_head = None

        if self.use_hierarchical_consistency:
            self.physics_consistency_head = nn.Sequential(
                nn.Linear(act_dim + 8, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
            self.action_support_head = nn.Sequential(
                nn.Linear(act_dim * 2, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
            self.future_support_head = nn.Sequential(
                nn.Linear(img_dim, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
            self.consistency_fuse_head = nn.Sequential(
                nn.Linear(fusion_dim * 2 + 3, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_dim, 1),
            )
        else:
            self.physics_consistency_head = None
            self.action_support_head = None
            self.future_support_head = None
            self.consistency_fuse_head = None

        # consistency_dim = hist + fut (+ diff + l2 + cos if explicit)
        # + optional action-visual interaction, motion, and geometry features
        consistency_dim = img_dim * 2 + act_dim * 2
        if self.use_explicit_distance:
            consistency_dim += img_dim + 2
        if self.use_action_visual_interaction:
            consistency_dim += img_dim * 5
        if self.use_motion_features:
            consistency_dim += img_dim * 2 + 4 + 6
        if self.use_traj_geometry_features:
            consistency_dim += 8
        if self.use_future_traj_geometry_prediction:
            consistency_dim += 8 * 3
        if self.use_future_consistency_evidence:
            consistency_dim += img_dim * 2
        if self.use_path_conditioned_evidence and not self.use_path_residual_score:
            consistency_dim += img_dim * 5
        if self.use_video_action_cross_attention and self.video_action_add_to_shared:
            consistency_dim += img_dim + 1
        if self.use_future_latent_prediction:
            consistency_dim += img_dim * 3 + 1
        if self.use_path_conditioned_evidence:
            self.path_conditioned_traj_proj = nn.Sequential(
                nn.Linear(act_dim, img_dim),
                nn.ReLU(inplace=True),
                nn.Linear(img_dim, img_dim),
            )
            if self.path_conditioned_segment_count > 1:
                self.path_conditioned_temporal_head = nn.Sequential(
                    nn.Linear(
                        img_dim * (1 + self.path_conditioned_segment_count),
                        img_dim,
                    ),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(img_dim),
                )
            else:
                self.path_conditioned_temporal_head = None
            self.path_conditioned_fusion = nn.Sequential(
                nn.Linear(img_dim * 4, img_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(img_dim),
            )
            path_head_dim = img_dim * 3
            if self.path_evidence_use_transition_context:
                path_head_dim += img_dim * 2
            self.path_residual_head = nn.Sequential(
                nn.Linear(path_head_dim, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
            if self.use_path_evidence_head:
                self.path_evidence_head = nn.Sequential(
                    nn.Linear(path_head_dim, hidden // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden // 2, 1),
                )
            else:
                self.path_evidence_head = None
            if self.use_path_evidence_gate:
                self.path_evidence_gate_head = nn.Sequential(
                    nn.Linear(fusion_dim + img_dim + act_dim, hidden // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden // 2, 1),
                )
            else:
                self.path_evidence_gate_head = None
        else:
            self.path_conditioned_traj_proj = None
            self.path_conditioned_temporal_head = None
            self.path_conditioned_fusion = None
            self.path_residual_head = None
            self.path_evidence_head = None
            self.path_evidence_gate_head = None

        if self.use_video_action_cross_attention:
            if img_dim % self.video_action_num_heads != 0:
                raise ValueError(
                    "dinov2.video_action_num_heads must divide image_feature_dim"
                )
            self.traj_token_encoder = nn.Sequential(
                nn.Linear(traj_d, img_dim),
                nn.ReLU(inplace=True),
                nn.Linear(img_dim, img_dim),
            )
            self.video_to_traj_attn = nn.MultiheadAttention(
                embed_dim=img_dim,
                num_heads=self.video_action_num_heads,
                batch_first=True,
            )
            self.traj_to_video_attn = nn.MultiheadAttention(
                embed_dim=img_dim,
                num_heads=self.video_action_num_heads,
                batch_first=True,
            )
            self.video_action_fusion = nn.Sequential(
                nn.Linear(img_dim * 6, hidden),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, img_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(img_dim),
            )
            self.video_action_match_head = nn.Sequential(
                nn.Linear(img_dim, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.traj_token_encoder = None
            self.video_to_traj_attn = None
            self.traj_to_video_attn = None
            self.video_action_fusion = None
            self.video_action_match_head = None

        if self.use_future_latent_prediction:
            self.future_latent_predictor = nn.Sequential(
                nn.Linear(img_dim + act_dim * 2, hidden),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, img_dim),
            )
            self.future_latent_match_head = nn.Sequential(
                nn.Linear(img_dim * 4, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.future_latent_predictor = None
            self.future_latent_match_head = None

        self.shared_fusion = nn.Sequential(
            nn.Linear(consistency_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.validity_fusion = nn.Sequential(
            nn.Linear(act_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        if self.use_progress_alignment_head:
            self.progress_alignment_head = nn.Sequential(
                nn.Linear(img_dim * 3, 1),
                nn.Softplus(),
            )
        else:
            self.progress_alignment_head = None

        if self.use_learned_motion_rules:
            self.motion_rule_visual_head = nn.Sequential(
                nn.Linear(
                    img_dim * (3 + self.motion_rule_segment_count),
                    hidden,
                ),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.motion_rule_attr_dim),
            )
            self.motion_rule_match_head = nn.Sequential(
                nn.Linear(self.motion_rule_attr_dim * 4, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.motion_rule_visual_head = None
            self.motion_rule_match_head = None
        if self.use_motion_latent_alignment:
            self.visual_motion_latent_head = nn.Sequential(
                nn.Linear(
                    img_dim * (3 + self.motion_rule_segment_count),
                    hidden,
                ),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.motion_latent_dim),
            )
            self.traj_motion_latent_head = nn.Sequential(
                nn.Linear(self.motion_rule_attr_dim + act_dim, hidden),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.motion_latent_dim),
            )
            self.motion_latent_match_head = nn.Sequential(
                nn.Linear(self.motion_latent_dim * 4, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.visual_motion_latent_head = None
            self.traj_motion_latent_head = None
            self.motion_latent_match_head = None

        # Same 6 heads as train.py. In separated-head configs,
        # consistency_head is the image-trajectory correspondence head.
        self.consistency_head = nn.Linear(fusion_dim, 1)
        if self.use_trajectory_reasonableness_head:
            self.trajectory_reasonableness_head = nn.Sequential(
                nn.Linear(fusion_dim, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.trajectory_reasonableness_head = None
        if self.use_learned_consistency_fusion:
            gate_dim = 2 + (fusion_dim if self.consistency_fusion_use_repr else 0)
            self.consistency_fusion_gate_head = nn.Sequential(
                nn.Linear(gate_dim, hidden // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.consistency_fusion_gate_head = None
        self.speed_consistency_head = nn.Linear(fusion_dim, 1)
        self.steering_consistency_head = nn.Linear(fusion_dim, 1)
        self.progress_consistency_head = nn.Linear(fusion_dim, 1)
        self.temporal_coherence_head = nn.Linear(fusion_dim, 1)
        self.validity_head = nn.Linear(fusion_dim, 1)

    def _encode_image_sequence(
        self, images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (B, T, 3, H, W) → (B, out_dim)."""
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        if self.use_dinov2:
            feat = self.image_encoder(flat)
        else:
            feat = self._cnn_shared_backbone(flat).flatten(1)
            feat = self.history_proj(feat)
        return feat.reshape(b, t, -1)

    def _encode_images(
        self, images: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_image_sequence(images).mean(dim=1)

    def _encode_sequence(
        self,
        sequence: torch.Tensor,
        temporal_encoder: nn.GRU | None,
    ) -> torch.Tensor:
        if temporal_encoder is None:
            return sequence.mean(dim=1)
        _, hidden = temporal_encoder(sequence)
        return hidden[-1]

    def _encode_image_sequence_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Encode (B, T, 3, H, W) -> (B, T, out_dim) without pooling."""
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        feat = self.image_encoder(flat)
        return feat.reshape(b, t, -1)

    def _metric_traj(self, traj: torch.Tensor) -> torch.Tensor:
        if self.traj_normalize_mode == "linear" and self.traj_scale is not None:
            scale = self.traj_scale.to(device=traj.device, dtype=traj.dtype)
            return traj * scale
        return traj

    def _path_patch_mask(
        self,
        traj: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        metric = self._metric_traj(traj[:, : self.consistency_traj_steps, :])
        xy = metric[..., :2]
        forward = xy[..., 0].clamp_min(0.0)
        lateral = xy[..., 1]
        max_forward = max(self.path_conditioned_forward_m, 1.0)
        max_lateral = max(self.path_conditioned_lateral_m, 1.0)
        v = (patch_h - 1) - (forward / max_forward).clamp(0.0, 1.0) * patch_h * 0.62
        u = (patch_w / 2.0) - (lateral / max_lateral).clamp(-1.0, 1.0) * patch_w * 0.32

        yy = torch.arange(patch_h, device=traj.device, dtype=traj.dtype).view(1, 1, patch_h, 1)
        xx = torch.arange(patch_w, device=traj.device, dtype=traj.dtype).view(1, 1, 1, patch_w)
        uu = u.view(u.shape[0], u.shape[1], 1, 1)
        vv = v.view(v.shape[0], v.shape[1], 1, 1)
        radius = max(1.0, float(max(patch_h, patch_w)) * self.path_conditioned_width)
        dist2 = (xx - uu).pow(2) + (yy - vv).pow(2)
        mask = (dist2 <= radius * radius).any(dim=1).float()
        return mask.reshape(traj.shape[0], patch_h * patch_w)

    def _encode_path_conditioned_future(
        self,
        future_images: torch.Tensor,
        candidate_traj: torch.Tensor,
        z_traj_cons: torch.Tensor,
    ) -> torch.Tensor:
        b, t, c, h, w = future_images.shape
        flat = future_images.reshape(b * t, c, h, w)
        if self.use_dinov2 and hasattr(self.image_encoder, "forward_patch_tokens"):
            patch_tokens = self.image_encoder.forward_patch_tokens(flat)
        else:
            return torch.zeros(
                (b, self.image_feature_dim),
                device=future_images.device,
                dtype=future_images.dtype,
            )
        patch_count = patch_tokens.shape[1]
        patch_side = int(round(math.sqrt(patch_count)))
        if patch_side * patch_side != patch_count:
            frame_pooled = patch_tokens.mean(dim=1).reshape(b, t, -1)
        else:
            mask = self._path_patch_mask(candidate_traj, patch_side, patch_side)
            mask = mask[:, None, :, None].expand(b, t, patch_count, 1).reshape(
                b * t, patch_count, 1
            )
            denom = mask.sum(dim=1).clamp_min(1.0)
            frame_pooled = ((patch_tokens * mask).sum(dim=1) / denom).reshape(b, t, -1)
        if self.path_conditioned_segment_count > 1 and t > 1:
            segment_features = [frame_pooled.mean(dim=1)]
            boundaries = [
                int(round(i * t / self.path_conditioned_segment_count))
                for i in range(self.path_conditioned_segment_count + 1)
            ]
            for seg_idx in range(self.path_conditioned_segment_count):
                start = max(0, min(t - 1, boundaries[seg_idx]))
                end = max(start + 1, min(t, boundaries[seg_idx + 1]))
                segment_features.append(frame_pooled[:, start:end, :].mean(dim=1))
            assert self.path_conditioned_temporal_head is not None
            pooled = self.path_conditioned_temporal_head(
                torch.cat(segment_features, dim=-1)
            )
        else:
            pooled = frame_pooled.mean(dim=1)
        assert self.path_conditioned_traj_proj is not None
        assert self.path_conditioned_fusion is not None
        traj_path = self.path_conditioned_traj_proj(z_traj_cons)
        return self.path_conditioned_fusion(
            torch.cat(
                [
                    pooled,
                    traj_path,
                    pooled * traj_path,
                    (pooled - traj_path).abs(),
                ],
                dim=-1,
            )
        )

    def _traj_motion_features(self, traj: torch.Tensor) -> torch.Tensor:
        xy = traj[:, : self.consistency_traj_steps, :2]
        origin = torch.zeros_like(xy[:, :1, :])
        prev = torch.cat([origin, xy[:, :-1, :]], dim=1)
        step = xy - prev
        step_dist = torch.norm(step, p=2, dim=-1)
        path_len = step_dist.sum(dim=1, keepdim=True)
        final_xy = xy[:, -1, :]
        final_disp = torch.norm(final_xy, p=2, dim=-1, keepdim=True)
        progress_x = final_xy[:, :1]
        lateral_abs = final_xy[:, 1:2].abs()
        mean_step = step_dist.mean(dim=1, keepdim=True)
        max_step = step_dist.max(dim=1, keepdim=True).values
        return torch.cat(
            [path_len, final_disp, progress_x, lateral_abs, mean_step, max_step],
            dim=-1,
        )

    def _traj_geometry_features(self, traj: torch.Tensor) -> torch.Tensor:
        xy = traj[:, : self.consistency_traj_steps, :2]
        yaw = (
            traj[:, : self.consistency_traj_steps, 2:3]
            if traj.shape[-1] > 2
            else torch.zeros_like(xy[:, :, :1])
        )
        origin = torch.zeros_like(xy[:, :1, :])
        prev = torch.cat([origin, xy[:, :-1, :]], dim=1)
        step = xy - prev
        step_dist = torch.norm(step, p=2, dim=-1)
        path_len = step_dist.sum(dim=1, keepdim=True)
        final_xy = xy[:, -1, :]
        final_disp = torch.norm(final_xy, p=2, dim=-1, keepdim=True)
        progress_x = final_xy[:, :1]
        lateral_abs = final_xy[:, 1:2].abs()
        mean_step = step_dist.mean(dim=1, keepdim=True)
        max_step = step_dist.max(dim=1, keepdim=True).values
        yaw_delta = yaw[:, -1, :] - yaw[:, 0, :]
        yaw_abs_delta = yaw_delta.abs()
        return torch.cat(
            [
                path_len,
                final_disp,
                progress_x,
                lateral_abs,
                mean_step,
                max_step,
                yaw_delta,
                yaw_abs_delta,
            ],
            dim=-1,
        )

    def _sequence_segments(
        self,
        sequence: torch.Tensor,
        segment_count: int,
    ) -> List[torch.Tensor]:
        if segment_count <= 0:
            return []
        t = sequence.shape[1]
        boundaries = [
            int(round(i * t / segment_count))
            for i in range(segment_count + 1)
        ]
        segments: List[torch.Tensor] = []
        for seg_idx in range(segment_count):
            start = max(0, min(t - 1, boundaries[seg_idx]))
            end = max(start + 1, min(t, boundaries[seg_idx + 1]))
            segments.append(sequence[:, start:end, :].mean(dim=1))
        return segments

    def _motion_rule_visual_context(
        self,
        z_hist: torch.Tensor,
        z_fut: torch.Tensor,
        fut_seq: torch.Tensor,
    ) -> torch.Tensor:
        parts = [z_hist, z_fut, z_fut - z_hist]
        parts.extend(self._sequence_segments(fut_seq, self.motion_rule_segment_count))
        return torch.cat(parts, dim=-1)

    def _traj_rule_segment_attributes(self, metric: torch.Tensor) -> torch.Tensor:
        if self.motion_rule_segment_count <= 0:
            return metric.new_zeros((metric.shape[0], 0))
        xy = metric[..., :2]
        b, t, _ = xy.shape
        origin = torch.zeros_like(xy[:, :1, :])
        prev = torch.cat([origin, xy[:, :-1, :]], dim=1)
        step = xy - prev
        heading = torch.atan2(step[..., 1], step[..., 0])
        attrs: List[torch.Tensor] = []
        boundaries = [
            int(round(i * t / self.motion_rule_segment_count))
            for i in range(self.motion_rule_segment_count + 1)
        ]
        for seg_idx in range(self.motion_rule_segment_count):
            start = max(0, min(t - 1, boundaries[seg_idx]))
            end = max(start + 1, min(t, boundaries[seg_idx + 1]))
            seg_xy = xy[:, start:end, :]
            if start == 0:
                start_xy = torch.zeros_like(seg_xy[:, 0, :])
            else:
                start_xy = xy[:, start - 1, :]
            end_xy = seg_xy[:, -1, :]
            seg_delta = end_xy - start_xy
            seg_step = step[:, start:end, :]
            seg_step_dist = torch.norm(seg_step, p=2, dim=-1)
            seg_path_len = seg_step_dist.sum(dim=1)
            seg_direct = torch.norm(seg_delta, p=2, dim=-1) / seg_path_len.clamp_min(1e-4)
            if end - start > 1:
                seg_heading_delta = heading[:, start + 1:end] - heading[:, start:end - 1]
                seg_heading_delta = torch.atan2(
                    torch.sin(seg_heading_delta),
                    torch.cos(seg_heading_delta),
                )
                seg_yaw = seg_heading_delta.sum(dim=1)
                seg_curvature = seg_heading_delta.abs().mean(dim=1)
            else:
                seg_yaw = torch.zeros(b, device=metric.device, dtype=metric.dtype)
                seg_curvature = torch.zeros(b, device=metric.device, dtype=metric.dtype)
            attrs.extend(
                [
                    (seg_delta[:, 0] / 15.0).clamp(-1.0, 1.0),
                    (seg_delta[:, 1] / 5.0).clamp(-1.0, 1.0),
                    (end_xy[:, 1] / 10.0).clamp(-1.0, 1.0),
                    (seg_path_len / 15.0).clamp(0.0, 1.0),
                    (seg_step_dist.mean(dim=1) / 5.0).clamp(0.0, 1.0),
                    (seg_yaw / math.pi).clamp(-1.0, 1.0),
                    (seg_yaw.abs() / math.pi).clamp(0.0, 1.0),
                    seg_curvature.clamp(0.0, 1.0),
                ]
            )
        return torch.stack(attrs, dim=-1)

    def _traj_rule_attributes(self, traj: torch.Tensor) -> torch.Tensor:
        metric = self._metric_traj(traj[:, : self.consistency_traj_steps, :])
        xy = metric[..., :2]
        b, t, _ = xy.shape
        origin = torch.zeros_like(xy[:, :1, :])
        prev = torch.cat([origin, xy[:, :-1, :]], dim=1)
        step = xy - prev
        step_dist = torch.norm(step, p=2, dim=-1)
        eps = 1e-4

        final_xy = xy[:, -1, :]
        final_disp = torch.norm(final_xy, p=2, dim=-1)
        path_len = step_dist.sum(dim=1)
        directness = final_disp / path_len.clamp_min(eps)
        span = max(1, t // 3)
        early_speed = step_dist[:, :span].mean(dim=1)
        late_speed = step_dist[:, -span:].mean(dim=1)
        speed_delta = late_speed - early_speed

        heading = torch.atan2(step[..., 1], step[..., 0])
        if metric.shape[-1] > 2:
            yaw_delta = metric[:, -1, 2] - metric[:, 0, 2]
        else:
            yaw_delta = heading[:, -1] - heading[:, 0]
        yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))

        if t > 1:
            heading_step_delta = heading[:, 1:] - heading[:, :-1]
            heading_step_delta = torch.atan2(
                torch.sin(heading_step_delta),
                torch.cos(heading_step_delta),
            )
            curvature_proxy = heading_step_delta.abs().mean(dim=1)
            turn_energy = heading_step_delta.abs()
            time_grid = torch.linspace(
                0.0,
                1.0,
                steps=t - 1,
                device=traj.device,
                dtype=traj.dtype,
            ).unsqueeze(0)
            turn_timing = (
                turn_energy
                / turn_energy.sum(dim=1, keepdim=True).clamp_min(eps)
                * time_grid
            ).sum(dim=1)
        else:
            curvature_proxy = torch.zeros(b, device=traj.device, dtype=traj.dtype)
            turn_timing = torch.zeros(b, device=traj.device, dtype=traj.dtype)

        max_abs_lateral = xy[..., 1].abs().max(dim=1).values
        global_attrs = torch.stack(
            [
                (final_xy[:, 0] / 40.0).clamp(-1.0, 1.0),
                (final_xy[:, 1] / 10.0).clamp(-1.0, 1.0),
                (final_xy[:, 1].abs() / 10.0).clamp(0.0, 1.0),
                (path_len / 40.0).clamp(0.0, 1.0),
                directness.clamp(0.0, 1.0),
                (step_dist.mean(dim=1) / 5.0).clamp(0.0, 1.0),
                (speed_delta / 5.0).clamp(-1.0, 1.0),
                (yaw_delta / math.pi).clamp(-1.0, 1.0),
                (yaw_delta.abs() / math.pi).clamp(0.0, 1.0),
                (max_abs_lateral / 10.0).clamp(0.0, 1.0),
                curvature_proxy.clamp(0.0, 1.0),
                (turn_timing * 2.0 - 1.0).clamp(-1.0, 1.0),
            ],
            dim=-1,
        )
        segment_attrs = self._traj_rule_segment_attributes(metric)
        if segment_attrs.numel() == 0:
            return global_attrs
        return torch.cat([global_attrs, segment_attrs], dim=-1)

    def forward(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats = self.extract_probe_features(
            history_images=history_images,
            future_images=future_images,
            ego_state=ego_state,
            candidate_traj=candidate_traj,
        )
        image_trajectory_consistency_logit = self.consistency_head(
            feats["z_shared"]
        ).squeeze(-1)
        consistency_logit = image_trajectory_consistency_logit
        if "future_consistency_evidence_logit" in feats:
            consistency_logit = consistency_logit + self.future_consistency_mix * feats[
                "future_consistency_evidence_logit"
            ]
        if "path_residual_logit" in feats:
            consistency_logit = consistency_logit + self.path_residual_mix * torch.tanh(feats[
                "path_residual_logit"
            ])
        if (
            self.mix_path_evidence_into_consistency
            and "path_evidence_logit" in feats
        ):
            evidence_term = torch.tanh(feats["path_evidence_logit"])
            if self.use_path_evidence_gate and "path_conditioned_evidence" in feats:
                assert self.path_evidence_gate_head is not None
                gate_in = torch.cat(
                    [
                        feats["z_shared"],
                        feats["path_conditioned_evidence"],
                        feats["z_traj_cons"],
                    ],
                    dim=-1,
                )
                gate = torch.sigmoid(self.path_evidence_gate_head(gate_in).squeeze(-1))
                evidence_term = gate * evidence_term
                feats["path_evidence_gate"] = gate
            consistency_logit = consistency_logit + self.path_evidence_mix * evidence_term
        if "motion_rule_match_logit" in feats:
            consistency_logit = (
                consistency_logit
                + self.motion_rule_mix * torch.tanh(feats["motion_rule_match_logit"])
            )
        if "video_action_match_logit" in feats:
            consistency_logit = (
                consistency_logit
                + self.video_action_mix * torch.tanh(feats["video_action_match_logit"])
            )
        if "future_latent_match_logit" in feats:
            consistency_logit = (
                consistency_logit
                + self.future_latent_mix * torch.tanh(feats["future_latent_match_logit"])
            )
        if self.use_hierarchical_consistency and "consistency_fuse_logit" in feats:
            consistency_logit = feats["consistency_fuse_logit"]
        trajectory_reasonableness_logit = None
        if self.trajectory_reasonableness_head is not None:
            trajectory_reasonableness_logit = self.trajectory_reasonableness_head(
                feats["z_validity"]
            ).squeeze(-1)
        if (
            self.consistency_fusion_gate_head is not None
            and trajectory_reasonableness_logit is not None
        ):
            gate_parts = [
                image_trajectory_consistency_logit.unsqueeze(-1),
                trajectory_reasonableness_logit.unsqueeze(-1),
            ]
            if self.consistency_fusion_use_repr:
                gate_parts.insert(0, feats["z_shared"])
            gate = torch.sigmoid(
                self.consistency_fusion_gate_head(
                    torch.cat(gate_parts, dim=-1)
                ).squeeze(-1)
            )
            feats["consistency_fusion_gate"] = gate
            consistency_logit = (
                consistency_logit
                + self.consistency_fusion_reasonableness_mix
                * gate
                * torch.tanh(trajectory_reasonableness_logit)
            )
        outputs = {
            "consistency_logit": consistency_logit,
            "image_trajectory_consistency_logit": image_trajectory_consistency_logit,
            "speed_consistency_logit": self.speed_consistency_head(feats["z_shared"]).squeeze(-1),
            "steering_consistency_logit": self.steering_consistency_head(feats["z_shared"]).squeeze(-1),
            "progress_consistency_logit": self.progress_consistency_head(feats["z_shared"]).squeeze(-1),
            "temporal_coherence_logit": self.temporal_coherence_head(feats["z_shared"]).squeeze(-1),
            "validity_logit": self.validity_head(feats["z_validity"]).squeeze(-1),
        }
        if trajectory_reasonableness_logit is not None:
            outputs["trajectory_reasonableness_logit"] = trajectory_reasonableness_logit
        for key in (
            "future_consistency_evidence_logit",
            "physics_support_logit",
            "action_support_logit",
            "future_support_logit",
            "consistency_fuse_logit",
            "path_residual_logit",
            "path_evidence_logit",
            "path_evidence_gate",
            "motion_rule_match_logit",
            "motion_latent_match_logit",
            "video_action_match_logit",
            "future_latent_match_logit",
            "consistency_fusion_gate",
        ):
            if key in feats:
                outputs[key] = feats[key]
        if "visual_motion_rule_pred" in feats:
            outputs["visual_motion_rule_pred"] = feats["visual_motion_rule_pred"]
            outputs["traj_motion_rule_target"] = feats["traj_motion_rule_target"]
        if "visual_motion_latent" in feats:
            outputs["visual_motion_latent"] = feats["visual_motion_latent"]
            outputs["traj_motion_latent"] = feats["traj_motion_latent"]
        if "pred_future_latent" in feats:
            outputs["pred_future_latent"] = feats["pred_future_latent"]
            outputs["future_latent_target"] = feats["future_latent_target"]
        if feats.get("future_traj_geometry_pred") is not None:
            outputs["future_traj_geometry_pred"] = feats["future_traj_geometry_pred"]
            outputs["future_traj_geometry_target"] = feats["future_traj_geometry_target"]
        if "progress_alignment_value" in feats:
            outputs["progress_alignment_value"] = feats["progress_alignment_value"]
        return outputs

    def extract_probe_features(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hist_seq = self._encode_image_sequence_raw(history_images)
        fut_seq = self._encode_image_sequence_raw(future_images)
        z_hist = self._encode_sequence(hist_seq, self.history_temporal_encoder)
        z_fut = self._encode_sequence(fut_seq, self.future_temporal_encoder)
        consistency_traj = candidate_traj[:, : self.consistency_traj_steps, :]
        z_traj_cons = self.consistency_traj_encoder(consistency_traj.flatten(1))
        z_traj_val = self.validity_traj_encoder(candidate_traj.flatten(1))
        z_ego = self.ego_encoder(ego_state)

        mode = self.baseline_mode
        if mode in {"no_image", "ego_only", "traj_only"}:
            hist_seq = torch.zeros_like(hist_seq)
            fut_seq = torch.zeros_like(fut_seq)
            z_hist = torch.zeros_like(z_hist)
            z_fut = torch.zeros_like(z_fut)
        if mode in {"no_traj", "ego_only"}:
            z_traj_cons = torch.zeros_like(z_traj_cons)
            z_traj_val = torch.zeros_like(z_traj_val)
        if mode == "traj_only":
            z_ego = torch.zeros_like(z_ego)

        traj_geometry = None
        future_traj_geometry_pred = None
        parts: List[torch.Tensor] = [z_hist, z_fut]
        if self.use_explicit_distance:
            diff = z_hist - z_fut
            l2_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
            cos_sim = F.cosine_similarity(z_hist, z_fut, dim=-1).unsqueeze(-1)
            parts.extend([diff, l2_norm, cos_sim])
        if self.use_action_visual_interaction:
            visual_delta = z_fut - z_hist
            visual_abs_delta = visual_delta.abs()
            assert self.action_to_visual_delta is not None
            action_delta = self.action_to_visual_delta(
                torch.cat([z_traj_cons, z_ego], dim=-1)
            )
            action_visual_product = action_delta * visual_delta
            action_visual_gap = (action_delta - visual_delta).abs()
            parts.extend(
                [
                    visual_delta,
                    visual_abs_delta,
                    action_delta,
                    action_visual_product,
                    action_visual_gap,
                ]
            )
        if self.use_motion_features:
            hist_last = hist_seq[:, -1, :]
            fut_first = fut_seq[:, 0, :]
            fut_last = fut_seq[:, -1, :]
            bridge_delta = fut_last - hist_last
            future_delta = fut_last - fut_first
            image_motion_scalars = torch.cat(
                [
                    torch.norm(bridge_delta, p=2, dim=-1, keepdim=True),
                    F.cosine_similarity(hist_last, fut_last, dim=-1).unsqueeze(-1),
                    torch.norm(future_delta, p=2, dim=-1, keepdim=True),
                    F.cosine_similarity(fut_first, fut_last, dim=-1).unsqueeze(-1),
                ],
                dim=-1,
            )
            traj_motion = self._traj_motion_features(consistency_traj)
            if mode in {"no_traj", "ego_only"}:
                traj_motion = torch.zeros_like(traj_motion)
            parts.extend(
                [bridge_delta, future_delta, image_motion_scalars, traj_motion],
            )
        if self.use_traj_geometry_features or self.use_future_traj_geometry_prediction:
            traj_geometry = self._traj_geometry_features(consistency_traj)
        if self.use_traj_geometry_features:
            assert traj_geometry is not None
            if mode in {"no_traj", "ego_only"}:
                parts.append(torch.zeros_like(traj_geometry))
            else:
                parts.append(traj_geometry)
        if self.use_future_traj_geometry_prediction:
            assert traj_geometry is not None
            assert self.future_traj_geometry_head is not None
            visual_context = torch.cat([z_hist, z_fut, z_fut - z_hist], dim=-1)
            future_traj_geometry_pred = self.future_traj_geometry_head(visual_context)
            traj_geometry_for_cmp = traj_geometry
            if mode in {"no_traj", "ego_only"}:
                traj_geometry_for_cmp = torch.zeros_like(traj_geometry_for_cmp)
            geom_delta = traj_geometry_for_cmp - future_traj_geometry_pred
            parts.extend([future_traj_geometry_pred, geom_delta, geom_delta.abs()])
        if self.use_future_consistency_evidence:
            assert self.future_consistency_evidence_head is not None
            assert self.future_consistency_evidence_logit_head is not None
            future_evidence_in = torch.cat(
                [
                    z_hist,
                    z_fut,
                    z_fut - z_hist,
                ],
                dim=-1,
            )
            future_consistency_evidence = self.future_consistency_evidence_head(
                future_evidence_in
            )
            future_consistency_evidence_logit = self.future_consistency_evidence_logit_head(
                future_consistency_evidence
            ).squeeze(-1)
            parts.extend([future_consistency_evidence, (future_consistency_evidence - z_fut).abs()])
        if self.use_path_conditioned_evidence:
            path_evidence = self._encode_path_conditioned_future(
                future_images,
                consistency_traj,
                z_traj_cons,
            )
            assert self.path_conditioned_traj_proj is not None
            traj_path_for_parts = self.path_conditioned_traj_proj(z_traj_cons)
            if not self.use_path_residual_score:
                parts.extend(
                    [
                        path_evidence,
                        path_evidence - z_fut,
                        (path_evidence - z_fut).abs(),
                        path_evidence * traj_path_for_parts,
                        (path_evidence - traj_path_for_parts).abs(),
                    ]
                )
        else:
            path_evidence = None
            traj_path_for_parts = None
        video_action_feature = None
        video_action_match_logit = None
        if self.use_video_action_cross_attention:
            assert self.traj_token_encoder is not None
            assert self.video_to_traj_attn is not None
            assert self.traj_to_video_attn is not None
            assert self.video_action_fusion is not None
            assert self.video_action_match_head is not None
            video_tokens = torch.cat([hist_seq, fut_seq], dim=1)
            traj_tokens = self.traj_token_encoder(consistency_traj)
            if mode in {"no_image", "ego_only", "traj_only"}:
                video_tokens = torch.zeros_like(video_tokens)
            if mode in {"no_traj", "ego_only"}:
                traj_tokens = torch.zeros_like(traj_tokens)
            video_ctx, _ = self.video_to_traj_attn(
                video_tokens,
                traj_tokens,
                traj_tokens,
                need_weights=False,
            )
            traj_ctx, _ = self.traj_to_video_attn(
                traj_tokens,
                video_tokens,
                video_tokens,
                need_weights=False,
            )
            video_ctx_pool = video_ctx.mean(dim=1)
            traj_ctx_pool = traj_ctx.mean(dim=1)
            video_pool = video_tokens.mean(dim=1)
            traj_pool = traj_tokens.mean(dim=1)
            cross_delta = video_ctx_pool - traj_ctx_pool
            video_action_feature = self.video_action_fusion(
                torch.cat(
                    [
                        video_ctx_pool,
                        traj_ctx_pool,
                        video_pool,
                        traj_pool,
                        cross_delta.abs(),
                        video_ctx_pool * traj_ctx_pool,
                    ],
                    dim=-1,
                )
            )
            video_action_match_logit = self.video_action_match_head(
                video_action_feature
            ).squeeze(-1)
            if self.video_action_add_to_shared:
                parts.extend(
                    [video_action_feature, video_action_match_logit.unsqueeze(-1)]
                )
        pred_future_latent = None
        future_latent_match_logit = None
        if self.use_future_latent_prediction:
            assert self.future_latent_predictor is not None
            assert self.future_latent_match_head is not None
            pred_future_latent = self.future_latent_predictor(
                torch.cat([z_hist, z_traj_cons, z_ego], dim=-1)
            )
            future_latent_delta = pred_future_latent - z_fut
            future_latent_match_in = torch.cat(
                [
                    pred_future_latent,
                    z_fut,
                    future_latent_delta.abs(),
                    pred_future_latent * z_fut,
                ],
                dim=-1,
            )
            future_latent_match_logit = self.future_latent_match_head(
                future_latent_match_in
            ).squeeze(-1)
            parts.extend(
                [
                    pred_future_latent,
                    future_latent_delta,
                    future_latent_delta.abs(),
                    future_latent_match_logit.unsqueeze(-1),
                ]
            )
        parts.extend([z_traj_cons, z_ego])
        z_all = torch.cat(parts, dim=-1)
        z_shared = self.shared_fusion(z_all)
        z_validity = self.validity_fusion(torch.cat([z_traj_val, z_ego], dim=-1))
        visual_motion_rule_pred = None
        traj_motion_rule_target = None
        motion_rule_match_logit = None
        visual_motion_latent = None
        traj_motion_latent = None
        motion_latent_match_logit = None
        if self.use_learned_motion_rules or self.use_motion_latent_alignment:
            visual_context = self._motion_rule_visual_context(
                z_hist,
                z_fut,
                fut_seq,
            )
            traj_motion_rule_target = self._traj_rule_attributes(consistency_traj)
            if mode in {"no_traj", "ego_only"}:
                traj_motion_rule_target = torch.zeros_like(traj_motion_rule_target)
            if self.use_learned_motion_rules:
                assert self.motion_rule_visual_head is not None
                assert self.motion_rule_match_head is not None
                visual_motion_rule_pred = torch.tanh(
                    self.motion_rule_visual_head(visual_context)
                )
                rule_delta = visual_motion_rule_pred - traj_motion_rule_target
                rule_match_in = torch.cat(
                    [
                        visual_motion_rule_pred,
                        traj_motion_rule_target,
                        rule_delta.abs(),
                        visual_motion_rule_pred * traj_motion_rule_target,
                    ],
                    dim=-1,
                )
                motion_rule_match_logit = self.motion_rule_match_head(
                    rule_match_in
                ).squeeze(-1)
            if self.use_motion_latent_alignment:
                assert self.visual_motion_latent_head is not None
                assert self.traj_motion_latent_head is not None
                assert self.motion_latent_match_head is not None
                visual_motion_latent = F.normalize(
                    self.visual_motion_latent_head(visual_context),
                    dim=-1,
                )
                traj_motion_latent = F.normalize(
                    self.traj_motion_latent_head(
                        torch.cat([traj_motion_rule_target, z_traj_cons], dim=-1)
                    ),
                    dim=-1,
                )
                latent_delta = visual_motion_latent - traj_motion_latent
                latent_match_in = torch.cat(
                    [
                        visual_motion_latent,
                        traj_motion_latent,
                        latent_delta.abs(),
                        visual_motion_latent * traj_motion_latent,
                    ],
                    dim=-1,
                )
                motion_latent_match_logit = self.motion_latent_match_head(
                    latent_match_in
                ).squeeze(-1)
        feats: Dict[str, torch.Tensor] = {
            "hist_seq": hist_seq,
            "fut_seq": fut_seq,
            "hist_seq_mean": hist_seq.mean(dim=1),
            "fut_seq_mean": fut_seq.mean(dim=1),
            "hist_seq_last": hist_seq[:, -1, :],
            "fut_seq_last": fut_seq[:, -1, :],
            "z_hist": z_hist,
            "z_fut": z_fut,
            "z_traj_cons": z_traj_cons,
            "z_traj_val": z_traj_val,
            "z_ego": z_ego,
            "z_all": z_all,
            "z_shared": z_shared,
            "z_validity": z_validity,
        }
        if visual_motion_rule_pred is not None:
            assert traj_motion_rule_target is not None
            assert motion_rule_match_logit is not None
            feats["visual_motion_rule_pred"] = visual_motion_rule_pred
            feats["traj_motion_rule_target"] = traj_motion_rule_target
            feats["motion_rule_match_logit"] = motion_rule_match_logit
        if visual_motion_latent is not None:
            assert traj_motion_latent is not None
            assert motion_latent_match_logit is not None
            feats["visual_motion_latent"] = visual_motion_latent
            feats["traj_motion_latent"] = traj_motion_latent
            feats["motion_latent_match_logit"] = motion_latent_match_logit
        if video_action_feature is not None:
            assert video_action_match_logit is not None
            feats["video_action_feature"] = video_action_feature
            feats["video_action_match_logit"] = video_action_match_logit
        if pred_future_latent is not None:
            assert future_latent_match_logit is not None
            feats["pred_future_latent"] = pred_future_latent
            feats["future_latent_target"] = z_fut
            feats["future_latent_match_logit"] = future_latent_match_logit
        if future_traj_geometry_pred is not None:
            assert traj_geometry is not None
            feats["future_traj_geometry_pred"] = future_traj_geometry_pred
            feats["future_traj_geometry_target"] = traj_geometry
        if self.use_future_consistency_evidence:
            feats["future_consistency_evidence"] = future_consistency_evidence
            feats["future_consistency_evidence_logit"] = future_consistency_evidence_logit
        if self.progress_alignment_head is not None:
            # Visual-only progress avoids the shortcut where the auxiliary
            # head simply reads candidate trajectory geometry from z_shared.
            progress_in = torch.cat([z_hist, z_fut, z_fut - z_hist], dim=-1)
            feats["progress_alignment_value"] = self.progress_alignment_head(
                progress_in
            ).squeeze(-1)
        if path_evidence is not None:
            feats["path_conditioned_evidence"] = path_evidence
            assert traj_path_for_parts is not None
            path_head_parts = [
                path_evidence,
                path_evidence * traj_path_for_parts,
                (path_evidence - traj_path_for_parts).abs(),
            ]
            if self.path_evidence_use_transition_context:
                path_head_parts.extend([z_hist, z_fut - z_hist])
            path_head_in = torch.cat(path_head_parts, dim=-1)
            if self.use_path_evidence_head:
                assert self.path_evidence_head is not None
                feats["path_evidence_logit"] = self.path_evidence_head(
                    path_head_in
                ).squeeze(-1)
            if self.use_path_residual_score:
                assert self.path_residual_head is not None
                feats["path_residual_logit"] = self.path_residual_head(
                    path_head_in
                ).squeeze(-1)

        if self.use_hierarchical_consistency:
            assert self.physics_consistency_head is not None
            assert self.action_support_head is not None
            assert self.future_support_head is not None
            assert self.consistency_fuse_head is not None
            physics_support = self.physics_consistency_head(
                torch.cat(
                    [
                        z_traj_cons,
                        traj_geometry if traj_geometry is not None else torch.zeros_like(z_traj_cons[:, :8]),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            action_support = self.action_support_head(
                torch.cat([z_traj_cons, z_ego], dim=-1)
            ).squeeze(-1)
            if self.use_future_consistency_evidence and "future_consistency_evidence_logit" in feats:
                future_support = self.future_support_head(future_consistency_evidence).squeeze(-1)
            else:
                future_support = self.future_support_head(z_fut).squeeze(-1)
            fuse_in = torch.cat(
                [
                    z_shared,
                    z_validity,
                    physics_support.unsqueeze(-1),
                    action_support.unsqueeze(-1),
                    future_support.unsqueeze(-1),
                ],
                dim=-1,
            )
            feats["physics_support_logit"] = physics_support
            feats["action_support_logit"] = action_support
            feats["future_support_logit"] = future_support
            feats["consistency_fuse_logit"] = self.consistency_fuse_head(fuse_in).squeeze(-1)
        return feats


# Public alias so eval_critic.py / benchmark_wam.py can use either trainer
# interchangeably. train.py's eval_critic does `from train import
# ConsistencyCriticModel`, so we set the same attribute on this module too —
# but only after train.py is imported.
ConsistencyCriticModel = DINOv2ConsistencyCritic  # late rebound below


# ─────────────────────────── main ───────────────────────────


def main() -> None:
    args = parse_args()
    train = _import_train()

    # Late-bound alias: some downstream scripts do
    # `from train import ConsistencyCriticModel` and won't see DINOv2.
    # We keep the local name consistent but do NOT mutate train.
    globals()["ConsistencyDataset"] = train.ConsistencyDataset
    globals()["load_config"] = train.load_config
    globals()["run_consistency_epoch"] = train.run_consistency_epoch
    globals()["build_dataloader"] = train.build_dataloader
    globals()["validate_index_image_paths"] = train.validate_index_image_paths
    globals()["save_checkpoint"] = train.save_checkpoint
    globals()["setup_distributed"] = train.setup_distributed
    globals()["cleanup_distributed"] = train.cleanup_distributed
    globals()["set_seed"] = train.set_seed
    globals()["is_main_process"] = train.is_main_process
    globals()["sigterm_received"] = train.sigterm_received
    globals()["_sigterm_handler"] = train._sigterm_handler

    cfg = load_config(args.config)
    if args.work_dir is not None:
        cfg["work_dir"] = args.work_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.baseline_mode is not None:
        cfg["baseline_mode"] = args.baseline_mode
    if cfg.get("model_type") != "consistency":
        raise ValueError("model_type must be 'consistency'.")
    # Apply CLI overrides for DINOv2 settings
    dcfg = cfg.setdefault("dinov2", {})
    if args.dinov2_model is not None:
        dcfg["model_name"] = args.dinov2_model
    if args.dinov2_freeze:
        dcfg["freeze"] = True
    if args.dinov2_trainable:
        dcfg["freeze"] = False
    if args.no_dinov2:
        dcfg["enabled"] = False
    if args.amp:
        cfg["amp"] = True

    signal.signal(signal.SIGTERM, _sigterm_handler)
    dist_info = setup_distributed()
    set_seed(int(cfg["seed"]) + dist_info["rank"])

    device = torch.device(
        f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu"
    )
    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    from train import ensure_parent  # type: ignore
    ensure_parent(work_dir / "config_snapshot.json")
    if is_main_process():
        with (work_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    if is_main_process() and int(args.preflight_samples) > 0:
        validate_index_image_paths(
            cfg, [cfg["train_index"], cfg["val_index"]], int(args.preflight_samples),
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    train_loader = build_dataloader(cfg, cfg["train_index"], training=True, epoch=0)
    val_loader = build_dataloader(cfg, cfg["val_index"], training=False)

    model = DINOv2ConsistencyCritic(cfg).to(device)
    trainable_prefixes = [
        str(prefix)
        for prefix in cfg.get("trainable_parameter_prefixes", [])
        if str(prefix)
    ]
    if trainable_prefixes:
        for name, param in model.named_parameters():
            param.requires_grad = any(
                name.startswith(prefix) for prefix in trainable_prefixes
            )
        trainable_params = [
            param for param in model.parameters() if param.requires_grad
        ]
        if not trainable_params:
            raise ValueError(
                "trainable_parameter_prefixes matched no parameters: "
                f"{trainable_prefixes}"
            )
    else:
        trainable_params = list(model.parameters())
    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]] if torch.cuda.is_available() else None,
            output_device=dist_info["local_rank"] if torch.cuda.is_available() else None,
            find_unused_parameters=bool(dcfg.get("find_unused_parameters", False)),
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )

    def _checkpoint_score(metrics: Dict[str, float]) -> float:
        mode = str(cfg.get("checkpoint_metric", "val_loss"))
        if mode == "val_loss":
            return -float(metrics["loss"])
        if mode == "val_c_score_gap":
            return float(metrics.get("c_score_gap", 0.0))
        if mode == "val_c_balanced_acc":
            return float(metrics.get("c_balanced_acc", 0.0))
        if mode == "val_iac_consistency":
            return (
                float(metrics.get("c_score_gap", 0.0))
                + 0.25 * float(metrics.get("c_balanced_acc", 0.0))
                - 0.05 * float(metrics.get("group_rank_loss", 0.0))
            )
        if mode == "val_iac_precision":
            return (
                0.55 * float(metrics.get("c_precision", 0.0))
                + 0.35 * float(metrics.get("c_tnr", 0.0))
                + 0.20 * float(metrics.get("c_balanced_acc", 0.0))
                + 0.10 * float(metrics.get("c_score_gap", 0.0))
                - 0.03 * float(metrics.get("group_rank_loss", 0.0))
            )
        raise ValueError(f"unknown checkpoint_metric: {mode}")

    best_val_loss = math.inf
    best_metric_name = str(cfg.get("checkpoint_metric", "val_loss"))
    best_metric_value = -math.inf
    total_epochs = int(cfg["epochs"])
    start_epoch = 1
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        target_model = model.module if isinstance(model, DDP) else model
        model_state = target_model.state_dict()
        checkpoint_model = checkpoint["model"]
        skipped_shape = [
            key for key, value in checkpoint_model.items()
            if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
        ]
        if skipped_shape:
            checkpoint_model = {
                key: value for key, value in checkpoint_model.items()
                if key not in skipped_shape
            }
        missing, unexpected = target_model.load_state_dict(
            checkpoint_model,
            strict=False,
        )
        if checkpoint.get("optimizer") and not skipped_shape and not trainable_prefixes:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as exc:
                if is_main_process():
                    print(f"[Resume][WARNING] optimizer state skipped: {exc}")
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        interrupted = bool(checkpoint.get("interrupted", False))
        if interrupted:
            # The interrupted checkpoint may have been written after a
            # one-step validation pass. Reset best tracking for the resumed
            # run so best.pth reflects a full validation epoch in this workdir.
            best_val_loss = math.inf
            best_metric_value = -math.inf
        else:
            best_metric_value = float(
                checkpoint.get("best_metric_value", -best_val_loss)
            )
        ckpt_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = ckpt_epoch if interrupted else ckpt_epoch + 1
        if is_main_process():
            print(
                f"[Resume] loaded {resume_path} "
                f"epoch={ckpt_epoch} start_epoch={start_epoch} "
                f"best_val_loss={best_val_loss}"
            )
            if trainable_prefixes:
                target_model_for_count = model.module if isinstance(model, DDP) else model
                trainable_count = sum(
                    p.numel() for p in target_model_for_count.parameters()
                    if p.requires_grad
                )
                total_count = sum(
                    p.numel() for p in target_model_for_count.parameters()
                )
                print(
                    "[Trainable] prefixes="
                    f"{trainable_prefixes} params={trainable_count}/{total_count}"
                )
            if missing:
                print(f"[Resume][WARNING] missing keys: {missing[:8]}")
            if unexpected:
                print(f"[Resume][WARNING] unexpected keys: {unexpected[:8]}")
            if skipped_shape:
                print(f"[Resume][WARNING] skipped shape-mismatched keys: {skipped_shape[:8]}")
    start_time = time.time()

    if is_main_process():
        mcfg = cfg.get("model", {})
        print("=" * 60)
        print("DINOv2 Consistency Critic v5 (minimal, ablation-aware)")
        if dcfg.get("enabled", True):
            print(
                "  Backbone        : DINOv2 "
                f"{dcfg.get('model_name','dinov2_vits14')} "
                f"mode={dcfg.get('layer_mode', 'single')} "
                f"layers={dcfg.get('layer_indices', [dcfg.get('layer_index', 11)])}"
            )
        else:
            print("  Backbone        : 4-layer CNN (from train.py)")
        print(f"  DINOv2 freeze   : {dcfg.get('freeze', True) if dcfg.get('enabled', True) else 'N/A'}")
        print(f"  Explicit dist   : {dcfg.get('use_explicit_distance', True)}")
        print(f"  Temporal encoder: {mcfg.get('temporal_encoder', 'mean')}")
        print(f"  Act-vis inter.  : {mcfg.get('use_action_visual_interaction', False)}")
        print(f"  Motion features : {dcfg.get('use_motion_features', False)}")
        print(
            "  Motion rules    : "
            f"{dcfg.get('use_learned_motion_rules', False)} "
            f"segments={dcfg.get('motion_rule_segment_count', 0)}"
        )
        print(
            "  Sep. heads      : "
            f"reason={dcfg.get('use_trajectory_reasonableness_head', False)} "
            f"fusion={dcfg.get('use_learned_consistency_fusion', False)}"
        )
        print(f"  Progress align. : {cfg.get('lambda_progress_alignment', 0.0)}")
        print(f"  Path grounding  : {cfg.get('lambda_path_grounding', 0.0)}")
        print(f"  Traj-specific   : {cfg.get('lambda_trajectory_specific_grounding', 0.0)}")
        group_batches = bool(
            cfg.get("ranking", {}).get(
                "group_batches",
                float(cfg.get("lambda_group_ranking", 0.0)) > 0.0,
            )
        )
        print(f"  Group batches   : {group_batches}")
        print(f"  D1-D4 sampling  : {cfg.get('difficulty_sampling', {}).get('enabled', False)}")
        print(f"  Work dir        : {work_dir}")
        print(f"  World size      : {dist_info['world_size']}")
        if torch.cuda.is_available():
            mem_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            print(f"  GPU memory      : {mem_total:.1f} GB")
        print("=" * 60)

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            train_metrics = run_consistency_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                device=device, epoch=epoch, cfg=cfg, training=True,
                max_steps=args.max_train_steps or 0,
            )
            val_metrics = run_consistency_epoch(
                model=model, loader=val_loader, optimizer=optimizer,
                device=device, epoch=epoch, cfg=cfg, training=False,
                max_steps=args.max_val_steps or 0,
            )
            current_metric_value = _checkpoint_score(val_metrics)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
            is_best = current_metric_value > best_metric_value
            if is_best:
                best_metric_value = current_metric_value
            if is_main_process():
                print(
                    f"[Epoch {epoch}/{total_epochs}] "
                    f"loss={train_metrics['loss']:.4f} "
                    f"c_acc={train_metrics['c_acc']:.4f} "
                    f"c_gap={train_metrics.get('c_score_gap', 0.0):.4f} "
                    f"v_acc={train_metrics['v_acc']:.4f} "
                    f"rank_loss={train_metrics.get('group_rank_loss', 0.0):.4f} "
                    f"prog_align={train_metrics.get('progress_alignment_loss', 0.0):.4f} "
                    f"path_ground_loss={train_metrics.get('path_grounding_loss', 0.0):.4f} "
                    f"traj_spec_loss={train_metrics.get('trajectory_specific_grounding_loss', 0.0):.4f} "
                    f"traj_spec_pos={train_metrics.get('trajectory_specific_positive_controls', 0.0):.0f} "
                    f"traj_spec_dist={train_metrics.get('trajectory_specific_wrong_distance_mean', 0.0):.2f} "
                    f"traj_spec_excl={train_metrics.get('trajectory_specific_exclusive_fraction_mean', 0.0):.4f} "
                    f"motion_attr={train_metrics.get('motion_rule_attribute_loss', 0.0):.4f} "
                    f"motion_match={train_metrics.get('motion_rule_match_loss', 0.0):.4f} "
                    f"motion_rank={train_metrics.get('motion_rule_rank_loss', 0.0):.4f} "
                    f"scope_temp={train_metrics.get('scope_motion_temporal_contrast_loss', 0.0):.4f} "
                    f"video_action={train_metrics.get('video_action_match_loss', 0.0):.4f} "
                    f"video_rank={train_metrics.get('video_action_rank_loss', 0.0):.4f} "
                    f"future_latent={train_metrics.get('future_latent_prediction_loss', 0.0):.4f} "
                    f"future_match={train_metrics.get('future_latent_match_loss', 0.0):.4f} "
                    f"reason={train_metrics.get('trajectory_reasonableness_loss', 0.0):.4f} "
                    f"reason_mae={train_metrics.get('trajectory_reasonableness_mae', 0.0):.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_c_acc={val_metrics['c_acc']:.4f} "
                    f"val_c_bal={val_metrics.get('c_balanced_acc', 0.0):.4f} "
                    f"val_c_recall={val_metrics.get('c_recall', 0.0):.4f} "
                    f"val_c_gap={val_metrics.get('c_score_gap', 0.0):.4f} "
                    f"val_v_acc={val_metrics['v_acc']:.4f} "
                    f"val_rank_loss={val_metrics.get('group_rank_loss', 0.0):.4f} "
                    f"val_prog_align={val_metrics.get('progress_alignment_loss', 0.0):.4f} "
                    f"val_motion_attr={val_metrics.get('motion_rule_attribute_loss', 0.0):.4f} "
                    f"val_motion_match={val_metrics.get('motion_rule_match_loss', 0.0):.4f} "
                    f"val_motion_rank={val_metrics.get('motion_rule_rank_loss', 0.0):.4f} "
                    f"val_scope_temp={val_metrics.get('scope_motion_temporal_contrast_loss', 0.0):.4f} "
                    f"val_video_action={val_metrics.get('video_action_match_loss', 0.0):.4f} "
                    f"val_video_rank={val_metrics.get('video_action_rank_loss', 0.0):.4f} "
                    f"val_future_latent={val_metrics.get('future_latent_prediction_loss', 0.0):.4f} "
                    f"val_future_match={val_metrics.get('future_latent_match_loss', 0.0):.4f} "
                    f"val_reason={val_metrics.get('trajectory_reasonableness_loss', 0.0):.4f} "
                    f"val_reason_mae={val_metrics.get('trajectory_reasonableness_mae', 0.0):.4f} "
                    f"ckpt_metric={best_metric_name}:{current_metric_value:.4f}"
                )
                if epoch % int(cfg["save_interval"]) == 0:
                    save_checkpoint(
                        work_dir=work_dir, epoch=epoch, model=model,
                        optimizer=optimizer, cfg=cfg,
                        best_val_loss=best_val_loss, is_best=is_best,
                        best_metric_name=best_metric_name,
                        best_metric_value=best_metric_value,
                    )
            if sigterm_received():
                if is_main_process():
                    print(f"[WARNING] SIGTERM at epoch={epoch}, saving interrupted ckpt...")
                    save_checkpoint(
                        work_dir=work_dir, epoch=epoch, model=model,
                        optimizer=optimizer, cfg=cfg,
                        best_val_loss=best_val_loss, is_best=False,
                        best_metric_name=best_metric_name,
                        best_metric_value=best_metric_value,
                        tag=f"interrupted_epoch_{epoch}", interrupted=True,
                    )
                break
    except Exception as e:
        rank = dist_info["rank"]
        print(f"\n[ERROR][rank={rank}] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            ma = torch.cuda.memory_allocated(device) / (1024 ** 3)
            mr = torch.cuda.memory_reserved(device) / (1024 ** 3)
            print(f"[ERROR] GPU mem allocated={ma:.2f}GB reserved={mr:.2f}GB")
        cleanup_distributed()
        sys.exit(1)

    if is_main_process():
        elapsed = time.time() - start_time
        print("=" * 60)
        print("Training finished")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Best metric:   {best_metric_name}={best_metric_value:.4f}")
        print(f"Elapsed:      {elapsed:.1f}s")
        print("=" * 60)
    cleanup_distributed()


if __name__ == "__main__":
    main()
