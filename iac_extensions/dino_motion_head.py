"""Candidate-blind temporal motion head for frozen DINO frame features.

The important separation is architectural rather than cosmetic:

1. :class:`CandidateBlindDinoMotionHead` sees only history/future image
   features.  It cannot copy progress or direction from a candidate path.
2. :class:`UncertaintyAwareTrajectoryComparator` receives the candidate motion
   attributes only after the visual prediction has been made.

This file contains no IAC dataset assumptions, so the two modules can also be
used in probes or downstream evaluators.  The IAC integration lives in
``train_scope_motion_head.py``.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _segment_means(sequence: torch.Tensor, count: int) -> List[torch.Tensor]:
    """Return deterministic, non-empty temporal segment means."""

    if sequence.ndim != 3:
        raise ValueError(f"expected (B,T,D), got {tuple(sequence.shape)}")
    if sequence.shape[1] < 1:
        raise ValueError("a motion sequence must contain at least one frame")
    count = max(1, int(count))
    length = int(sequence.shape[1])
    boundaries = [int(round(index * length / count)) for index in range(count + 1)]
    outputs: List[torch.Tensor] = []
    for index in range(count):
        start = max(0, min(length - 1, boundaries[index]))
        end = max(start + 1, min(length, boundaries[index + 1]))
        outputs.append(sequence[:, start:end].mean(dim=1))
    return outputs


class CandidateBlindDinoMotionHead(nn.Module):
    """Predict structured motion attributes from DINO frame features only.

    Parameters
    ----------
    feature_dim:
        Width of each projected DINO frame feature.
    attribute_dim:
        Number of normalized trajectory attributes predicted by the head.  IAC
        currently uses 12 global attributes plus 8 per temporal segment.
    hidden_dim:
        Width of the temporal transformer and prediction MLP.
    num_layers / num_heads:
        Temporal transformer size.  Two small layers are sufficient for the
        intended frozen-backbone experiment.
    segment_count:
        Number of future-video temporal summaries retained in the final visual
        context.  This should normally match IAC's motion-rule segments.

    Notes
    -----
    The output mean is bounded to ``[-1, 1]`` because IAC's trajectory motion
    attributes use that range.  ``log_variance`` is bounded for stable mixed
    precision training.  No candidate trajectory is accepted by ``forward``.
    """

    def __init__(
        self,
        feature_dim: int,
        attribute_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        segment_count: int = 3,
        max_frames: int = 32,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or attribute_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature, attribute and hidden dimensions must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if max_frames < 2:
            raise ValueError("max_frames must be at least two")

        self.attribute_dim = int(attribute_dim)
        self.segment_count = max(1, int(segment_count))
        self.max_frames = int(max_frames)
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_frames, hidden_dim)
        )
        self.phase_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
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

        # history last, future mean, future last, signed/absolute bridge delta,
        # plus future temporal segment summaries.
        context_tokens = 5 + self.segment_count
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * context_tokens, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.attribute_dim * 2),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.phase_embedding, std=0.02)

    def forward(
        self,
        history_features: torch.Tensor,
        future_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if history_features.ndim != 3 or future_features.ndim != 3:
            raise ValueError("history/future features must have shape (B,T,D)")
        if history_features.shape[0] != future_features.shape[0]:
            raise ValueError("history and future batch sizes differ")
        if history_features.shape[2] != future_features.shape[2]:
            raise ValueError("history and future feature widths differ")
        history_length = int(history_features.shape[1])
        future_length = int(future_features.shape[1])
        total_length = history_length + future_length
        if history_length < 1 or future_length < 1:
            raise ValueError("history and future must both contain at least one frame")
        if total_length > self.max_frames:
            raise ValueError(
                f"received {total_length} frames, configured maximum is {self.max_frames}"
            )

        sequence = torch.cat([history_features, future_features], dim=1)
        encoded = self.input_projection(sequence)
        phases = torch.cat(
            [
                torch.zeros(history_length, dtype=torch.long, device=sequence.device),
                torch.ones(future_length, dtype=torch.long, device=sequence.device),
            ],
            dim=0,
        )
        encoded = (
            encoded
            + self.position_embedding[:, :total_length]
            + self.phase_embedding.index_select(0, phases).unsqueeze(0)
        )
        encoded = self.temporal_encoder(encoded)
        history_encoded = encoded[:, :history_length]
        future_encoded = encoded[:, history_length:]

        history_last = history_encoded[:, -1]
        future_mean = future_encoded.mean(dim=1)
        future_last = future_encoded[:, -1]
        bridge_delta = future_last - history_last
        context = torch.cat(
            [
                history_last,
                future_mean,
                future_last,
                bridge_delta,
                bridge_delta.abs(),
                *_segment_means(future_encoded, self.segment_count),
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


class UncertaintyAwareTrajectoryComparator(nn.Module):
    """Compare visual motion evidence with candidate attributes.

    The per-attribute weights and final scale are constrained non-negative so
    one contradictory component cannot cancel another.  Lower energy means the
    candidate is better supported by the video.
    """

    def __init__(self, attribute_dim: int, eps: float = 1e-4) -> None:
        super().__init__()
        if attribute_dim <= 0:
            raise ValueError("attribute_dim must be positive")
        self.attribute_dim = int(attribute_dim)
        self.eps = float(eps)
        self.raw_attribute_weights = nn.Parameter(torch.zeros(attribute_dim))
        self.raw_logit_scale = nn.Parameter(torch.tensor(0.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        visual_mean: torch.Tensor,
        visual_log_variance: torch.Tensor,
        candidate_attributes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        expected = visual_mean.shape
        if visual_log_variance.shape != expected or candidate_attributes.shape != expected:
            raise ValueError(
                "visual mean/log-variance and candidate attributes must share a shape"
            )
        if expected[-1] != self.attribute_dim:
            raise ValueError(
                f"expected {self.attribute_dim} attributes, received {expected[-1]}"
            )
        variance = visual_log_variance.exp().clamp_min(self.eps)
        component_energy = 0.5 * (
            (visual_mean - candidate_attributes).square() / variance
            + F.softplus(visual_log_variance)
        )
        weights = F.softplus(self.raw_attribute_weights) + self.eps
        normalized_weights = weights / weights.mean().clamp_min(self.eps)
        weighted_components = component_energy * normalized_weights
        energy = weighted_components.mean(dim=-1)
        scale = F.softplus(self.raw_logit_scale) + self.eps
        return {
            "energy": energy,
            "component_energy": weighted_components,
            "logit": self.logit_bias - scale * energy,
            "attribute_weights": normalized_weights,
        }


def uncertainty_weighted_motion_loss(
    prediction: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Stable non-negative heteroscedastic regression loss.

    ``softplus(log_variance)`` retains the useful uncertainty penalty while
    keeping the reported loss non-negative.  ``reduction='none'`` returns one
    value per sample, matching IAC's weighted auxiliary-loss interface.
    """

    if prediction.shape != log_variance.shape or prediction.shape != target.shape:
        raise ValueError("prediction, log_variance and target must share a shape")
    log_variance = log_variance.clamp(-6.0, 2.0)
    per_attribute = 0.5 * (
        torch.exp(-log_variance) * (prediction - target).square()
        + F.softplus(log_variance)
    )
    per_sample = per_attribute.mean(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"unsupported reduction: {reduction!r}")
