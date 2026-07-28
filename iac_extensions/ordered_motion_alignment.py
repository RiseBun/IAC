"""Candidate-blind, ordered video-to-trajectory motion evidence.

The trusted V-JEPA gate consumes time-ordered visual tokens and trajectory
tokens, but its final cross-attention and mean pooling are permutation
invariant.  This module keeps time explicit:

1. visual tokens are assigned normalized temporal positions;
2. candidate-blind segment queries attend only to a local temporal band;
3. the visual branch predicts per-segment longitudinal, lateral, heading and
   path-shape motion;
4. a candidate is scored by non-negative, additive residual evidence.

Source labels are never inputs to this module.  They may be used by external
training/evaluation code as supervision or report-only strata.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MOTION_FAMILIES: Tuple[str, ...] = (
    "longitudinal",
    "lateral",
    "heading",
    "path_shape",
)


@dataclass(frozen=True)
class OrderedMotionConfig:
    visual_dim: int
    hidden_dim: int = 128
    segment_count: int = 8
    bandwidth: float = 0.22
    dropout: float = 0.10
    min_log_scale: float = -3.0
    max_log_scale: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _wrap_scalar_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _unwrap_angles(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    result = [float(values[0])]
    for value in values[1:]:
        result.append(result[-1] + _wrap_scalar_angle(float(value) - result[-1]))
    return result


def _valid_trajectory_points(raw: Any) -> List[Tuple[float, float, float | None]]:
    points: List[Tuple[float, float, float | None]] = []
    if not isinstance(raw, (list, tuple)):
        return points
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x = float(item[0])
            y = float(item[1])
            heading = float(item[2]) if len(item) >= 3 else None
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if heading is not None and not math.isfinite(heading):
            heading = None
        points.append((x, y, heading))
    return points


def trajectory_segment_target(
    raw: Any,
    *,
    segment_count: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Convert future poses into fixed, ordered motion segment targets.

    The origin pose is prepended.  Trajectories with a different number of
    points are linearly resampled along their provided time index.  The four
    channels are ``dx``, ``dy``, ``delta_heading`` and curvature.
    """

    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    points = _valid_trajectory_points(raw)
    if not points:
        return torch.zeros(segment_count, len(MOTION_FAMILIES), dtype=torch.float32)

    inferred: List[float] = []
    prev_x = 0.0
    prev_y = 0.0
    prev_heading = 0.0
    for x, y, heading in points:
        if heading is None:
            dx = x - prev_x
            dy = y - prev_y
            current = math.atan2(dy, dx) if math.hypot(dx, dy) > eps else prev_heading
        else:
            current = float(heading)
        inferred.append(current)
        prev_x, prev_y, prev_heading = x, y, current

    x_values = [0.0, *[item[0] for item in points]]
    y_values = [0.0, *[item[1] for item in points]]
    heading_values = _unwrap_angles([0.0, *inferred])
    poses = torch.tensor(
        [x_values, y_values, heading_values],
        dtype=torch.float32,
    ).unsqueeze(0)
    poses = F.interpolate(
        poses,
        size=segment_count + 1,
        mode="linear",
        align_corners=True,
    ).squeeze(0).transpose(0, 1)

    delta_xy = poses[1:, :2] - poses[:-1, :2]
    delta_heading = poses[1:, 2] - poses[:-1, 2]
    step = torch.linalg.vector_norm(delta_xy, dim=-1).clamp_min(eps)
    curvature = delta_heading / step
    return torch.stack(
        (
            delta_xy[:, 0],
            delta_xy[:, 1],
            delta_heading,
            curvature,
        ),
        dim=-1,
    )


def trajectory_targets_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    segment_count: int,
) -> torch.Tensor:
    return torch.stack(
        [
            trajectory_segment_target(
                row.get("candidate_traj"),
                segment_count=segment_count,
            )
            for row in rows
        ],
        dim=0,
    )


def _time_features(positions: torch.Tensor) -> torch.Tensor:
    positions = positions.float()
    return torch.stack(
        (
            positions,
            torch.sin(2.0 * math.pi * positions),
            torch.cos(2.0 * math.pi * positions),
            torch.sin(4.0 * math.pi * positions),
            torch.cos(4.0 * math.pi * positions),
        ),
        dim=-1,
    )


class OrderedMotionAlignment(nn.Module):
    """Predict ordered motion from video tokens without seeing a candidate."""

    def __init__(self, config: OrderedMotionConfig) -> None:
        super().__init__()
        if config.visual_dim <= 0:
            raise ValueError("visual_dim must be positive")
        if config.hidden_dim <= 0 or config.segment_count <= 0:
            raise ValueError("hidden_dim and segment_count must be positive")
        if config.bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        self.config = config

        self.visual_input = nn.Sequential(
            nn.Linear(config.visual_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.visual_time = nn.Linear(5, config.hidden_dim, bias=False)
        self.segment_time = nn.Linear(5, config.hidden_dim, bias=False)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.segment_query = nn.Parameter(
            torch.zeros(config.segment_count, config.hidden_dim)
        )
        nn.init.trunc_normal_(self.segment_query, std=0.02)

        self.motion_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(MOTION_FAMILIES) * 2),
        )
        initial_weight = math.log(math.expm1(1.0))
        self.component_log_weight = nn.Parameter(
            torch.full((len(MOTION_FAMILIES),), initial_weight)
        )

    def forward(self, visual_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape (batch,time,feature), "
                f"got {tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[-1] != self.config.visual_dim:
            raise ValueError(
                f"expected visual dim {self.config.visual_dim}, "
                f"got {visual_tokens.shape[-1]}"
            )
        batch, time_count, _ = visual_tokens.shape
        if time_count < 2:
            raise ValueError("ordered alignment requires at least two visual tokens")

        delta = torch.diff(visual_tokens, dim=1, prepend=visual_tokens[:, :1])
        hidden = self.visual_input(torch.cat((visual_tokens, delta), dim=-1))

        visual_pos = torch.linspace(
            0.0,
            1.0,
            time_count,
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        segment_pos = (
            torch.arange(
                self.config.segment_count,
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            )
            + 0.5
        ) / float(self.config.segment_count)
        hidden = hidden + self.visual_time(_time_features(visual_pos)).unsqueeze(0)
        query = (
            self.segment_query
            + self.segment_time(_time_features(segment_pos))
        ).unsqueeze(0).expand(batch, -1, -1)

        logits = torch.matmul(
            query,
            self.key(hidden).transpose(1, 2),
        ) / math.sqrt(float(self.config.hidden_dim))
        temporal_distance = segment_pos[:, None] - visual_pos[None, :]
        logits = logits - 0.5 * (
            temporal_distance / float(self.config.bandwidth)
        ).square().unsqueeze(0)
        local_mask = temporal_distance.abs() > 2.0 * float(self.config.bandwidth)
        logits = logits.masked_fill(local_mask.unsqueeze(0), -1e4)
        attention = torch.softmax(logits, dim=-1)
        aligned = torch.matmul(attention, self.value(hidden))

        prediction = self.motion_head(aligned)
        mean, log_scale = prediction.chunk(2, dim=-1)
        log_scale = log_scale.clamp(
            min=float(self.config.min_log_scale),
            max=float(self.config.max_log_scale),
        )
        return {
            "visual_motion_mean_normalized": mean,
            "visual_motion_log_scale": log_scale,
            "temporal_attention": attention,
            "visual_time_position": visual_pos,
            "segment_time_position": segment_pos,
        }

    def evidence(
        self,
        output: Mapping[str, torch.Tensor],
        candidate_target_normalized: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mean = output["visual_motion_mean_normalized"]
        log_scale = output["visual_motion_log_scale"]
        if candidate_target_normalized.shape != mean.shape:
            raise ValueError(
                "candidate target shape must match visual prediction: "
                f"{tuple(candidate_target_normalized.shape)} vs {tuple(mean.shape)}"
            )
        scale = log_scale.exp().clamp_min(1e-4)
        normalized_residual = (
            mean - candidate_target_normalized
        ) / scale
        raw = 0.5 * normalized_residual.square()
        weights = F.softplus(self.component_log_weight)
        weights = weights / weights.mean().clamp_min(1e-6)
        contribution = raw * weights.view(1, 1, -1)
        family = contribution.mean(dim=1)
        total = family.sum(dim=-1)
        return {
            "normalized_residual": normalized_residual,
            "segment_component_contribution": contribution,
            "family_contribution": family,
            "ordered_motion_energy": total,
            "component_weight": weights,
        }


def standardize_targets(
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if targets.ndim != 3 or targets.shape[-1] != len(MOTION_FAMILIES):
        raise ValueError(
            "targets must have shape (batch,segment,4), "
            f"got {tuple(targets.shape)}"
        )
    mean = targets.mean(dim=(0, 1), keepdim=True)
    std = targets.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    return mean, std


def normalize_targets(
    targets: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    clip: float = 8.0,
) -> torch.Tensor:
    result = (targets - mean) / std
    if clip > 0.0:
        result = result.clamp(min=-float(clip), max=float(clip))
    return result


def gaussian_motion_loss(
    output: Mapping[str, torch.Tensor],
    target_normalized: torch.Tensor,
) -> torch.Tensor:
    mean = output["visual_motion_mean_normalized"]
    log_scale = output["visual_motion_log_scale"]
    scale = log_scale.exp().clamp_min(1e-4)
    residual = (mean - target_normalized) / scale
    return (0.5 * residual.square() + log_scale).mean()


def load_feature_cache(
    path: Path,
    *,
    key: str = "x_tokens",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if key not in cache:
        raise KeyError(f"feature cache {path} does not contain key {key!r}")
    features = cache[key].float()
    if features.ndim != 3:
        raise ValueError(
            f"{key} must have shape (row,time,feature), got {tuple(features.shape)}"
        )
    sample_ids = [str(item) for item in cache.get("sample_id", [])]
    if len(sample_ids) != features.shape[0]:
        raise ValueError(
            f"sample_id count {len(sample_ids)} != feature rows {features.shape[0]}"
        )
    return (
        {sample_id: features[index] for index, sample_id in enumerate(sample_ids)},
        dict(cache.get("metadata", {})),
    )


def match_rows_to_features(
    rows: Sequence[Mapping[str, Any]],
    feature_by_sample: Mapping[str, torch.Tensor],
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    matched_rows: List[Dict[str, Any]] = []
    matched_features: List[torch.Tensor] = []
    for raw in rows:
        sample_id = str(raw.get("sample_id", ""))
        feature = feature_by_sample.get(sample_id)
        if feature is None:
            continue
        matched_rows.append(dict(raw))
        matched_features.append(feature)
    if not matched_rows:
        raise ValueError("no input rows matched the visual feature cache")
    return matched_rows, torch.stack(matched_features, dim=0)


def save_bundle(
    path: Path,
    *,
    model: OrderedMotionAlignment,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "ordered_motion_alignment",
            "state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "target_mean": target_mean.detach().cpu(),
            "target_std": target_std.detach().cpu(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_bundle(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("kind") != "ordered_motion_alignment":
        raise ValueError(f"unexpected model kind: {bundle.get('kind')!r}")
    config = OrderedMotionConfig(**dict(bundle["config"]))
    model = OrderedMotionAlignment(config)
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    result = dict(bundle)
    result["model"] = model
    result["target_mean"] = bundle["target_mean"].to(device)
    result["target_std"] = bundle["target_std"].to(device)
    return result


@torch.no_grad()
def score_batches(
    bundle: Mapping[str, Any],
    visual: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | str,
) -> Dict[str, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model: OrderedMotionAlignment = bundle["model"]
    target_mean: torch.Tensor = bundle["target_mean"]
    target_std: torch.Tensor = bundle["target_std"]
    collected: Dict[str, List[torch.Tensor]] = {
        "visual_motion_mean_normalized": [],
        "visual_motion_log_scale": [],
        "temporal_attention": [],
        "normalized_residual": [],
        "segment_component_contribution": [],
        "family_contribution": [],
        "ordered_motion_energy": [],
        "component_weight": [],
    }
    for start in range(0, visual.shape[0], batch_size):
        end = min(start + batch_size, visual.shape[0])
        visual_batch = visual[start:end].to(device)
        target_batch = targets[start:end].to(device)
        target_normalized = normalize_targets(
            target_batch,
            target_mean,
            target_std,
        )
        output = model(visual_batch)
        evidence = model.evidence(output, target_normalized)
        values = {**output, **evidence}
        for key in collected:
            item = values[key].detach().cpu()
            if key == "component_weight":
                item = item.unsqueeze(0).expand(end - start, -1)
            collected[key].append(item)
    result = {key: torch.cat(items, dim=0) for key, items in collected.items()}
    result["visual_motion_mean"] = (
        result["visual_motion_mean_normalized"] * target_std.cpu()
        + target_mean.cpu()
    )
    result["candidate_motion_target"] = targets.cpu()
    result["visual_motion_standard_deviation"] = (
        result["visual_motion_log_scale"].exp() * target_std.cpu()
    )
    return result


def iter_batched_indices(length: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)
