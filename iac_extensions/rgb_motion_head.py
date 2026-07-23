"""Candidate-blind RGB temporal-difference motion head.

This module is intentionally independent from DINO.  It reads the ordered image
sequence directly, encodes signed frame differences, and predicts the same IAC
trajectory-motion attributes used by the DINO scope-motion head.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_motion_head import _segment_means


class CandidateBlindRgbDiffMotionHead(nn.Module):
    """Predict motion attributes from ordered RGB frame differences only."""

    def __init__(
        self,
        attribute_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        segment_count: int = 3,
        max_frames: int = 32,
        spatial_size: int = 96,
    ) -> None:
        super().__init__()
        if attribute_dim <= 0 or hidden_dim <= 0:
            raise ValueError("attribute_dim and hidden_dim must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if max_frames < 2:
            raise ValueError("max_frames must be at least two")
        self.attribute_dim = int(attribute_dim)
        self.segment_count = max(1, int(segment_count))
        self.max_frames = int(max_frames)
        self.spatial_size = max(16, int(spatial_size))

        self.pair_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_frames - 1, hidden_dim)
        )
        self.phase_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=max(1, int(num_layers)),
            enable_nested_tensor=False,
        )

        context_tokens = 6 + self.segment_count
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * context_tokens, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.attribute_dim * 2),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.phase_embedding, std=0.02)

    def _phase_indices(
        self,
        history_length: int,
        future_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_length = history_length + future_length
        pair_count = total_length - 1
        phases: List[int] = []
        for pair_index in range(pair_count):
            if pair_index < history_length - 1:
                phases.append(0)
            elif pair_index == history_length - 1:
                phases.append(1)
            else:
                phases.append(2)
        return torch.tensor(phases, dtype=torch.long, device=device)

    def forward(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if history_images.ndim != 5 or future_images.ndim != 5:
            raise ValueError("history/future images must have shape (B,T,C,H,W)")
        if history_images.shape[0] != future_images.shape[0]:
            raise ValueError("history and future batch sizes differ")
        if history_images.shape[2] != future_images.shape[2]:
            raise ValueError("history and future channel counts differ")
        history_length = int(history_images.shape[1])
        future_length = int(future_images.shape[1])
        total_length = history_length + future_length
        if history_length < 1 or future_length < 1:
            raise ValueError("history and future must both contain at least one frame")
        if total_length > self.max_frames:
            raise ValueError(
                f"received {total_length} frames, configured maximum is {self.max_frames}"
            )

        sequence = torch.cat([history_images, future_images], dim=1)
        signed_delta = sequence[:, 1:] - sequence[:, :-1]
        pair_input = torch.cat([signed_delta, signed_delta.abs()], dim=2)
        batch, pair_count, channels, height, width = pair_input.shape
        pair_input = pair_input.reshape(batch * pair_count, channels, height, width)
        if max(height, width) != self.spatial_size:
            pair_input = F.interpolate(
                pair_input,
                size=(self.spatial_size, self.spatial_size),
                mode="bilinear",
                align_corners=False,
            )
        encoded = self.pair_encoder(pair_input).reshape(batch, pair_count, -1)
        encoded = encoded + self.position_embedding[:, :pair_count]
        phases = self._phase_indices(
            history_length,
            future_length,
            sequence.device,
        )
        encoded = encoded + self.phase_embedding.index_select(0, phases).unsqueeze(0)
        encoded = self.temporal_encoder(encoded)

        history_pairs = encoded[:, : max(history_length - 1, 1)]
        bridge_pair = encoded[:, history_length - 1]
        future_pairs = encoded[:, history_length:]
        if future_pairs.shape[1] == 0:
            future_pairs = bridge_pair.unsqueeze(1)
        future_mean = future_pairs.mean(dim=1)
        future_last = future_pairs[:, -1]
        all_mean = encoded.mean(dim=1)
        history_mean = history_pairs.mean(dim=1)
        bridge_delta = future_last - history_mean
        context = torch.cat(
            [
                all_mean,
                history_mean,
                bridge_pair,
                future_mean,
                future_last,
                bridge_delta,
                *_segment_means(future_pairs, self.segment_count),
            ],
            dim=-1,
        )
        raw = self.output_head(context)
        raw_mean, raw_log_variance = raw.chunk(2, dim=-1)
        return {
            "mean": torch.tanh(raw_mean),
            "log_variance": raw_log_variance.clamp(-6.0, 2.0),
            "temporal_context": context,
        }
