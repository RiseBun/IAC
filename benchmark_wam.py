#!/usr/bin/env python3
"""Benchmark WAM outputs with an IAC critic.

Input is a JSONL or PT manifest. Each sample should contain:
  - history_images: paths or nested tensor-like arrays, shape T,H,W,C or T,C,H,W
  - future_images: paths or arrays for WAM-generated future frames
  - ego_state: list[float]
  - candidate_traj: list[list[float]]

Optional fields used for reporting:
  - wam_name / model_name
  - action_type / source_type / sample_type / perturb_type
  - consistency_label / label
  - validity_label
  - group_id / anchor_id for ranking among candidates
  - perturb_magnitude / perturb_level for graded curves
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_critic import _compute_head_metrics
from train import ConsistencyCriticModel, load_config
from iac_video_metrics import compute_all_visual_metrics, load_frames_from_paths
from iac_traj_metrics import (
    compute_trajectory_accuracy,
    estimate_trajectory_from_video,
    ego_state_to_traj,
    candidate_traj_to_traj,
)
from iac_memory_metrics import compute_memory_symmetry, compute_loop_closure_drift


def _threshold_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> Dict[str, Any]:
    preds = scores >= threshold
    positives = labels > 0.5
    negatives = ~positives
    tp = int((preds & positives).sum().item())
    fp = int((preds & negatives).sum().item())
    fn = int(((~preds) & positives).sum().item())
    tn = int(((~preds) & negatives).sum().item())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    balanced_accuracy = (
        (recall + tnr) / 2.0
        if recall is not None and tnr is not None
        else None
    )
    total = max(1, len(labels))
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total),
        "precision": precision,
        "recall": recall,
        "tnr": tnr,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _threshold_sweep(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    raw_thresholds = [float(v) for v in torch.unique(scores.detach().cpu())]
    if raw_thresholds:
        eps = 1e-6
        raw_thresholds.extend([min(raw_thresholds) - eps, max(raw_thresholds) + eps])
    thresholds = sorted(set(raw_thresholds))
    if not thresholds:
        return {"num_labeled": int(labels.numel())}
    best_balanced: Dict[str, Any] | None = None
    best_f1: Dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = _threshold_metrics(scores, labels, threshold)
        balanced = metrics.get("balanced_accuracy")
        if (
            balanced is not None
            and (
                best_balanced is None
                or balanced > float(best_balanced["balanced_accuracy"])
                or (
                    balanced == float(best_balanced["balanced_accuracy"])
                    and (metrics.get("f1") or 0.0) > (best_balanced.get("f1") or 0.0)
                )
            )
        ):
            best_balanced = metrics
        if (
            metrics.get("f1") is not None
            and (
                best_f1 is None
                or float(metrics["f1"]) > float(best_f1["f1"])
                or (
                    metrics["f1"] == best_f1["f1"]
                    and (balanced or 0.0) > (best_f1.get("balanced_accuracy") or 0.0)
                )
            )
        ):
            best_f1 = metrics
    return {
        "num_labeled": int(labels.numel()),
        "best_balanced_accuracy": best_balanced,
        "best_f1": best_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IAC benchmark on WAM outputs")
    parser.add_argument("--input", required=True, help="WAM output manifest: .jsonl/.json/.pt")
    parser.add_argument("--checkpoint", required=True, help="Trained IAC checkpoint")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument(
        "--consistency-score-key",
        default="consistency_logit",
        help=(
            "Model output logit to use as the benchmark consistency score. "
            "Use path_evidence_logit to evaluate the independent path evidence head."
        ),
    )
    parser.add_argument(
        "--progress-fusion-beta",
        type=float,
        default=0.0,
        help=(
            "If > 0, subtract beta * abs(visual_progress - trajectory_progress) "
            "from the selected consistency logit. Requires model output "
            "progress_alignment_value."
        ),
    )
    parser.add_argument(
        "--progress-fusion-mode",
        choices=["path_length", "forward", "final_displacement"],
        default="final_displacement",
        help="Raw trajectory progress definition for progress-fused scoring.",
    )
    parser.add_argument(
        "--progress-fusion-scale",
        type=float,
        default=40.0,
        help="Scale divisor for raw trajectory progress in progress-fused scoring.",
    )
    parser.add_argument("--image-root", default=None, help="Resolve relative image paths")
    parser.add_argument("--output-dir", default="work_dirs/wam_benchmark", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help=(
            "Limit to the first N candidate groups with at least two rows. "
            "Useful for trajectory-specific causal checks on shuffled JSONL files."
        ),
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help=(
            "Drop rows whose history/future image paths are missing before applying "
            "--max-groups. This keeps larger file-backed manifests reproducible when "
            "the local image mirror is incomplete."
        ),
    )
    parser.add_argument(
        "--missing-report",
        default=None,
        help="Optional JSON report listing rows dropped by --skip-missing-images.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--group-key", default="group_id", help="Group key for candidate ranking")
    parser.add_argument("--wam-key", default="wam_name", help="Field identifying the WAM/model")
    parser.add_argument(
        "--visual-metrics",
        action="store_true",
        help="Also compute iWorld-Bench style no-reference visual metrics (brightness/color/sharpness/iq).",
    )
    parser.add_argument(
        "--visual-size",
        type=int,
        default=224,
        help="Resize for visual metrics (kept small for speed).",
    )
    parser.add_argument(
        "--geometric-metrics",
        action="store_true",
        help="Also compute iWorld-Bench style geometry metrics (recover trajectory from future frames, compare to GT).",
    )
    parser.add_argument(
        "--memory-metrics",
        action="store_true",
        help="Also compute iWorld-Bench style memory symmetry / loop-closure drift.",
    )
    parser.add_argument(
        "--path-causal-metrics",
        action="store_true",
        help=(
            "Run path-grounded causal checks by masking the trajectory-projected "
            "future path ROI and a sky/background control ROI."
        ),
    )
    parser.add_argument(
        "--trajectory-specific-causal-metrics",
        action="store_true",
        help=(
            "Mask the candidate path and a same-group wrong-candidate path. "
            "This tests whether score drops are tied to the current trajectory, "
            "not merely to generic road/path pixels."
        ),
    )
    parser.add_argument(
        "--wrong-path-selection",
        choices=["trajectory_distance", "mask_iou"],
        default="trajectory_distance",
        help=(
            "Select same-group wrong path controls by max trajectory distance "
            "or by lowest projected path IoU. mask_iou creates a sharper "
            "trajectory-specific causal contrast."
        ),
    )
    parser.add_argument(
        "--path-mask-width",
        type=float,
        default=0.10,
        help="Relative image width used as the projected path corridor thickness.",
    )
    parser.add_argument(
        "--path-trajectory-mode",
        choices=["cumulative", "positions"],
        default="cumulative",
        help=(
            "Interpret candidate_traj as per-step deltas (cumulative, legacy) "
            "or ego-frame future positions (positions)."
        ),
    )
    parser.add_argument(
        "--path-projection-mode",
        choices=["relative", "fixed"],
        default="relative",
        help=(
            "relative normalizes each trajectory independently; fixed uses "
            "global meter scales and preserves speed/progress differences."
        ),
    )
    parser.add_argument(
        "--path-forward-m",
        type=float,
        default=40.0,
        help="Forward meter scale for fixed path projection.",
    )
    parser.add_argument(
        "--path-lateral-m",
        type=float,
        default=10.0,
        help="Lateral meter scale for fixed path projection.",
    )
    parser.add_argument(
        "--sky-mask-ratio",
        type=float,
        default=0.25,
        help="Top image fraction used as the sky/background control mask.",
    )
    parser.add_argument(
        "--model-kind",
        choices=["auto", "cnn", "dinov2"],
        default="auto",
        help="Checkpoint model family. auto uses checkpoint/config metadata.",
    )
    return parser.parse_args()


def _candidate_traj_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        else:
            arr = arr.reshape(-1, min(arr.size, 3))
    if arr.ndim != 2:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.shape[1] < 3:
        pad = np.zeros((arr.shape[0], 3 - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :3]


def _trajectory_image_polyline(
    candidate_traj: Any,
    height: int,
    width: int,
    *,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
) -> List[tuple[int, int]]:
    """Project ego-frame candidate trajectory to a rough image-space corridor.

    This is intentionally lightweight: it is not a camera-calibrated BEV
    projection. The diagnostic only needs a stable, path-focused ROI so that
    path masking can be compared against a sky/background control mask.
    """
    arr = _candidate_traj_array(candidate_traj)
    if arr.size == 0:
        return [(width // 2, height - 1), (width // 2, max(0, int(height * 0.45)))]

    if trajectory_mode == "positions":
        xy = arr
    else:
        xy = np.cumsum(arr, axis=0)
    forward = np.maximum(xy[:, 0], 0.0)
    lateral = xy[:, 1]
    if projection_mode == "fixed":
        max_forward = max(float(forward_m), 1.0)
        max_lateral = max(float(lateral_m), 1.0)
    else:
        max_forward = max(float(np.percentile(np.abs(forward), 90)), 1.0)
        max_lateral = max(float(np.percentile(np.abs(lateral), 90)), 2.0)

    pts: List[tuple[int, int]] = [(width // 2, height - 1)]
    for x_fwd, y_lat in zip(forward, lateral):
        # x forward moves upward; positive y_lat is vehicle-left, image-left.
        v = int((height - 1) - np.clip(x_fwd / max_forward, 0.0, 1.0) * height * 0.62)
        u = int((width / 2.0) - np.clip(y_lat / max_lateral, -1.0, 1.0) * width * 0.32)
        pts.append((max(0, min(width - 1, u)), max(0, min(height - 1, v))))
    return pts


def _draw_disk(mask: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    h, w = mask.shape
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy = torch.arange(y0, y1, device=mask.device)[:, None]
    xx = torch.arange(x0, x1, device=mask.device)[None, :]
    mask[y0:y1, x0:x1] |= (yy - cy).pow(2) + (xx - cx).pow(2) <= radius * radius


def _draw_line(mask: torch.Tensor, p0: tuple[int, int], p1: tuple[int, int], radius: int) -> None:
    x0, y0 = p0
    x1, y1 = p1
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _draw_disk(mask, x, y, radius)


def _path_mask_for_traj(
    candidate_traj: Any,
    height: int,
    width: int,
    device: torch.device,
    width_ratio: float,
    *,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    pts = _trajectory_image_polyline(
        candidate_traj,
        height,
        width,
        trajectory_mode=trajectory_mode,
        projection_mode=projection_mode,
        forward_m=forward_m,
        lateral_m=lateral_m,
    )
    radius = max(2, int(round(width * float(width_ratio))))
    for a, b in zip(pts[:-1], pts[1:]):
        _draw_line(mask, a, b, radius)
    return mask


def _path_mask_for_row(
    row: Dict[str, Any],
    height: int,
    width: int,
    device: torch.device,
    width_ratio: float,
    *,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
) -> torch.Tensor:
    return _path_mask_for_traj(
        row.get("candidate_traj", []),
        height,
        width,
        device,
        width_ratio,
        trajectory_mode=trajectory_mode,
        projection_mode=projection_mode,
        forward_m=forward_m,
        lateral_m=lateral_m,
    )


def _mask_future_images(
    future_images: torch.Tensor,
    rows: List[Dict[str, Any]],
    mode: str,
    *,
    path_width_ratio: float,
    sky_ratio: float,
    trajectory_mode: str,
    projection_mode: str,
    forward_m: float,
    lateral_m: float,
) -> tuple[torch.Tensor, List[float]]:
    """Return masked future images and per-row masked area fractions.

    Input future_images are already normalized. Filling masked pixels with 0
    means "dataset mean color", which removes evidence without injecting a
    strong black patch shortcut.
    """
    masked = future_images.clone()
    bsz, _, _, height, width = masked.shape
    fractions: List[float] = []
    for b in range(bsz):
        if mode == "path":
            mask = _path_mask_for_row(
                rows[b],
                height,
                width,
                masked.device,
                path_width_ratio,
                trajectory_mode=trajectory_mode,
                projection_mode=projection_mode,
                forward_m=forward_m,
                lateral_m=lateral_m,
            )
        elif mode == "wrong_path":
            mask = _path_mask_for_traj(
                rows[b].get("_wrong_candidate_traj", []),
                height,
                width,
                masked.device,
                path_width_ratio,
                trajectory_mode=trajectory_mode,
                projection_mode=projection_mode,
                forward_m=forward_m,
                lateral_m=lateral_m,
            )
        elif mode == "sky":
            mask = torch.zeros((height, width), dtype=torch.bool, device=masked.device)
            # Match the path-mask area as closely as possible, capped to the
            # top sky/background band. Otherwise a larger sky mask can create
            # an unfairly large score drop that is about occlusion area, not
            # semantic dependence.
            path_ref = _path_mask_for_row(
                rows[b],
                height,
                width,
                masked.device,
                path_width_ratio,
                trajectory_mode=trajectory_mode,
                projection_mode=projection_mode,
                forward_m=forward_m,
                lateral_m=lateral_m,
            )
            target_fraction = min(float(path_ref.float().mean().item()), float(sky_ratio))
            sky_h = max(1, min(height, int(round(height * target_fraction))))
            mask[:sky_h, :] = True
        else:
            raise ValueError(f"unknown mask mode: {mode}")
        fractions.append(float(mask.float().mean().item()))
        masked[b, :, :, mask] = 0.0
    return masked, fractions


def _equal_area_exclusive_masks(
    candidate_mask: torch.Tensor,
    wrong_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cand_only = candidate_mask & ~wrong_mask
    wrong_only = wrong_mask & ~candidate_mask
    cand_idx = cand_only.flatten().nonzero(as_tuple=False).flatten()
    wrong_idx = wrong_only.flatten().nonzero(as_tuple=False).flatten()
    k = int(min(cand_idx.numel(), wrong_idx.numel()))
    out_cand = torch.zeros_like(candidate_mask)
    out_wrong = torch.zeros_like(wrong_mask)
    if k <= 0:
        return out_cand, out_wrong
    # Deterministic subsampling keeps benchmark runs exactly reproducible.
    out_cand.flatten()[cand_idx[:k]] = True
    out_wrong.flatten()[wrong_idx[:k]] = True
    return out_cand, out_wrong


def _mask_future_images_exclusive_paths(
    future_images: torch.Tensor,
    rows: List[Dict[str, Any]],
    *,
    path_width_ratio: float,
    trajectory_mode: str,
    projection_mode: str,
    forward_m: float,
    lateral_m: float,
) -> tuple[torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    cand_masked = future_images.clone()
    wrong_masked = future_images.clone()
    bsz, _, _, height, width = future_images.shape
    stats: List[Dict[str, float]] = []
    for b in range(bsz):
        cand = _path_mask_for_row(
            rows[b],
            height,
            width,
            future_images.device,
            path_width_ratio,
            trajectory_mode=trajectory_mode,
            projection_mode=projection_mode,
            forward_m=forward_m,
            lateral_m=lateral_m,
        )
        wrong = _path_mask_for_traj(
            rows[b].get("_wrong_candidate_traj", []),
            height,
            width,
            future_images.device,
            path_width_ratio,
            trajectory_mode=trajectory_mode,
            projection_mode=projection_mode,
            forward_m=forward_m,
            lateral_m=lateral_m,
        )
        cand_excl, wrong_excl = _equal_area_exclusive_masks(cand, wrong)
        union = (cand | wrong).float().sum().item()
        inter = (cand & wrong).float().sum().item()
        cand_masked[b, :, :, cand_excl] = 0.0
        wrong_masked[b, :, :, wrong_excl] = 0.0
        stats.append(
            {
                "exclusive_mask_fraction": float(cand_excl.float().mean().item()),
                "path_mask_iou": float(inter / max(union, 1.0)),
                "candidate_only_fraction": float((cand & ~wrong).float().mean().item()),
                "wrong_only_fraction": float((wrong & ~cand).float().mean().item()),
            }
        )
    return cand_masked, wrong_masked, stats


def _attach_wrong_path_controls(
    rows: List[Dict[str, Any]],
    group_key: str,
    *,
    selection: str = "trajectory_distance",
    path_width_ratio: float = 0.10,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
    mask_height: int = 224,
    mask_width: int = 224,
) -> int:
    def traj_distance(a: Any, b: Any) -> float:
        aa = _candidate_traj_array(a)
        bb = _candidate_traj_array(b)
        n = min(len(aa), len(bb))
        if n <= 0:
            return 0.0
        diff = aa[:n, :2] - bb[:n, :2]
        mean_l2 = float(np.linalg.norm(diff, axis=1).mean())
        final_l2 = float(np.linalg.norm(diff[-1]))
        return mean_l2 + final_l2

    def mask_contrast(a: Any, b: Any) -> tuple[float, float]:
        device = torch.device("cpu")
        mask_a = _path_mask_for_traj(
            a,
            mask_height,
            mask_width,
            device,
            path_width_ratio,
            trajectory_mode=trajectory_mode,
            projection_mode=projection_mode,
            forward_m=forward_m,
            lateral_m=lateral_m,
        )
        mask_b = _path_mask_for_traj(
            b,
            mask_height,
            mask_width,
            device,
            path_width_ratio,
            trajectory_mode=trajectory_mode,
            projection_mode=projection_mode,
            forward_m=forward_m,
            lateral_m=lateral_m,
        )
        union = float((mask_a | mask_b).float().sum().item())
        inter = float((mask_a & mask_b).float().sum().item())
        iou = inter / max(union, 1.0)
        exclusive = float(((mask_a & ~mask_b) | (mask_b & ~mask_a)).float().mean().item())
        return iou, exclusive

    def wrong_key(idx: int, candidate: int) -> tuple[float, float, float]:
        distance = traj_distance(
            rows[idx].get("candidate_traj", []),
            rows[candidate].get("candidate_traj", []),
        )
        if selection == "mask_iou":
            iou, exclusive = mask_contrast(
                rows[idx].get("candidate_traj", []),
                rows[candidate].get("candidate_traj", []),
            )
            return -iou, exclusive, distance
        return distance, 0.0, 0.0

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        group = _candidate_group_id(row, group_key)
        if group is not None:
            groups[group].append(idx)

    attached = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        positives = [
            idx for idx in indices
            if float(rows[idx].get("consistency_label", rows[idx].get("label", 0.0))) >= 0.5
        ]
        negatives = [idx for idx in indices if idx not in positives]
        for idx in indices:
            if idx in positives and negatives:
                other_idx = max(
                    negatives,
                    key=lambda candidate: wrong_key(idx, candidate),
                )
            elif positives and idx not in positives:
                other_idx = positives[0]
            else:
                others = [candidate for candidate in indices if candidate != idx]
                other_idx = (
                    max(
                        others,
                        key=lambda candidate: wrong_key(idx, candidate),
                    )
                    if others
                    else None
                )
            if other_idx is None:
                continue
            rows[idx]["_wrong_candidate_traj"] = rows[other_idx].get("candidate_traj", [])
            rows[idx]["wrong_path_source_type"] = (
                rows[other_idx].get("source_type")
                or rows[other_idx].get("sample_type")
                or "unknown"
            )
            rows[idx]["wrong_path_sample_id"] = rows[other_idx].get("sample_id")
            attached += 1
    return attached


def _limit_rows_by_groups(
    rows: List[Dict[str, Any]],
    group_key: str,
    max_groups: int,
) -> List[Dict[str, Any]]:
    if max_groups <= 0:
        return rows
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    group_order: List[str] = []
    for row in rows:
        group = _candidate_group_id(row, group_key)
        if group is None:
            continue
        if group not in grouped:
            group_order.append(group)
        grouped[group].append(row)

    selected: List[Dict[str, Any]] = []
    used = 0
    for group in group_order:
        items = grouped[group]
        if len(items) < 2:
            continue
        selected.extend(items)
        used += 1
        if used >= max_groups:
            break
    return selected


def _causal_aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if "iac_consistency_path_masked" in row and "iac_consistency_sky_masked" in row
    ]
    if not valid:
        return {"count": 0}

    def val(row: Dict[str, Any], key: str) -> float:
        return float(row[key])

    def delta(row: Dict[str, Any], key: str) -> float:
        return val(row, "iac_consistency") - val(row, key)

    by_source: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[str(row.get("source_type") or row.get("sample_type") or "unknown")].append(row)
    for source, items in sorted(grouped.items()):
        path_d = [delta(row, "iac_consistency_path_masked") for row in items]
        sky_d = [delta(row, "iac_consistency_sky_masked") for row in items]
        by_source[source] = {
            "count": len(items),
            "mean_path_delta": _mean(path_d),
            "mean_sky_delta": _mean(sky_d),
            "mean_path_minus_sky_delta": _mean(p - s for p, s in zip(path_d, sky_d)),
            "path_delta_gt_sky_fraction": _mean(float(p > s) for p, s in zip(path_d, sky_d)),
        }

    path_d = [delta(row, "iac_consistency_path_masked") for row in valid]
    sky_d = [delta(row, "iac_consistency_sky_masked") for row in valid]
    path_minus_sky = [p - s for p, s in zip(path_d, sky_d)]
    return {
        "count": len(valid),
        "definition": (
            "Mask trajectory-projected future path ROI and compare score drop "
            "against a top-image sky/background control mask."
        ),
        "mean_path_delta": _mean(path_d),
        "mean_sky_delta": _mean(sky_d),
        "mean_path_minus_sky_delta": _mean(path_minus_sky),
        "path_delta_gt_sky_fraction": _mean(float(p > s) for p, s in zip(path_d, sky_d)),
        "mean_path_mask_fraction": _mean(row.get("path_mask_fraction", 0.0) for row in valid),
        "mean_sky_mask_fraction": _mean(row.get("sky_mask_fraction", 0.0) for row in valid),
        "is_path_grounded": bool((_mean(path_minus_sky) or 0.0) > 0.01),
        "by_source_type": by_source,
    }


def _trajectory_specific_aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if "iac_consistency_path_masked" in row
        and "iac_consistency_wrong_path_masked" in row
        and row.get("wrong_path_source_type") is not None
    ]
    if not valid:
        return {"count": 0}

    def delta(row: Dict[str, Any], key: str) -> float:
        return float(row["iac_consistency"]) - float(row[key])

    def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        cand_d = [delta(row, "iac_consistency_path_masked") for row in items]
        wrong_d = [delta(row, "iac_consistency_wrong_path_masked") for row in items]
        diff = [c - w for c, w in zip(cand_d, wrong_d)]
        out = {
            "count": len(items),
            "mean_candidate_path_delta": _mean(cand_d),
            "mean_wrong_path_delta": _mean(wrong_d),
            "mean_candidate_minus_wrong_delta": _mean(diff),
            "candidate_delta_gt_wrong_fraction": _mean(
                float(c > w) for c, w in zip(cand_d, wrong_d)
            ),
        }
        exclusive = [
            row for row in items
            if "candidate_exclusive_path_delta" in row
            and "wrong_exclusive_path_delta" in row
        ]
        if exclusive:
            cand_ex = [float(row["candidate_exclusive_path_delta"]) for row in exclusive]
            wrong_ex = [float(row["wrong_exclusive_path_delta"]) for row in exclusive]
            diff_ex = [c - w for c, w in zip(cand_ex, wrong_ex)]
            out.update(
                {
                    "exclusive_count": len(exclusive),
                    "mean_candidate_exclusive_path_delta": _mean(cand_ex),
                    "mean_wrong_exclusive_path_delta": _mean(wrong_ex),
                    "mean_candidate_minus_wrong_exclusive_delta": _mean(diff_ex),
                    "candidate_exclusive_delta_gt_wrong_fraction": _mean(
                        float(c > w) for c, w in zip(cand_ex, wrong_ex)
                    ),
                    "mean_exclusive_mask_fraction": _mean(
                        row.get("exclusive_mask_fraction", 0.0) for row in exclusive
                    ),
                    "mean_path_mask_iou": _mean(
                        row.get("path_mask_iou", 0.0) for row in exclusive
                    ),
                    "mean_candidate_only_fraction": _mean(
                        row.get("candidate_only_fraction", 0.0) for row in exclusive
                    ),
                    "mean_wrong_only_fraction": _mean(
                        row.get("wrong_only_fraction", 0.0) for row in exclusive
                    ),
                }
            )
        return out

    by_source: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[str(row.get("source_type") or row.get("sample_type") or "unknown")].append(row)
    for source, items in sorted(grouped.items()):
        by_source[source] = summarize(items)

    positives = [
        row for row in valid
        if float(row.get("consistency_label", row.get("label", 0.0))) >= 0.5
    ]
    overall = summarize(valid)
    pos_summary = summarize(positives) if positives else {"count": 0}
    return {
        **overall,
        "definition": (
            "Mask the current candidate's projected path ROI and compare it "
            "against a same-group wrong candidate path ROI. This asks whether "
            "the score depends on the trajectory-specific path, not generic road pixels."
        ),
        "positive_rows": pos_summary,
        "mean_wrong_path_mask_fraction": _mean(
            row.get("wrong_path_mask_fraction", 0.0) for row in valid
        ),
        "is_trajectory_specific_path_grounded": bool(
            (overall.get("mean_candidate_minus_wrong_delta") or 0.0) > 0.01
        ),
        "is_exclusive_trajectory_specific_path_grounded": bool(
            (overall.get("mean_candidate_minus_wrong_exclusive_delta") or 0.0) > 0.005
        ),
        "by_source_type": by_source,
    }


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported manifest format: {path}")

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("samples", "data", "rows"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError(f"Cannot find sample list in {path}")


def _image_paths_from_value(value: Any, image_root: Path) -> List[Path]:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        paths: List[Path] = []
        for item in value:
            path = Path(item)
            paths.append(path if path.is_absolute() else image_root / path)
        return paths
    return []


def _filter_rows_with_existing_images(
    rows: List[Dict[str, Any]],
    image_root: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        future_value = row.get(
            "future_images",
            row.get("generated_future_images", row.get("generated_images")),
        )
        paths = _image_paths_from_value(row.get("history_images"), image_root)
        paths.extend(_image_paths_from_value(future_value, image_root))
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            dropped.append(
                {
                    "row_index": idx,
                    "sample_id": row.get("sample_id"),
                    "group_id": row.get("group_id") or row.get("anchor_id"),
                    "source_type": row.get("source_type") or row.get("sample_type"),
                    "missing_count": len(missing),
                    "missing_paths": missing[:8],
                }
            )
        else:
            kept.append(row)
    return kept, dropped


def _as_tensor_image_sequence(value: Any, image_root: Path, size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float()
    elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        frames = []
        for item in value:
            path = Path(item)
            if not path.is_absolute():
                path = image_root / path
            with Image.open(path) as img:
                image = img.convert("RGB").resize((size, size))
            arr = np.asarray(image, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))
        tensor = torch.stack(frames, dim=0)
    else:
        tensor = torch.tensor(value, dtype=torch.float32)

    if tensor.ndim != 4:
        raise ValueError(f"Image sequence must be 4D, got shape={tuple(tensor.shape)}")
    # Accept T,H,W,C or T,C,H,W.
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.max().item() > 2.0:
        tensor = tensor / 255.0
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return (tensor - mean[:, None, None]) / std[:, None, None]


class WAMManifestDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], cfg: Dict[str, Any], image_root: str | None) -> None:
        self.rows = rows
        self.cfg = cfg
        self.image_root = Path(image_root or cfg["image_root"])
        self.image_size = int(cfg["image_size"])
        self.history_num_frames = int(cfg["history_num_frames"])
        self.future_num_frames = int(cfg["future_num_frames"])
        self.ego_state_dim = int(cfg["ego_state_dim"])
        self.candidate_traj_steps = int(cfg["candidate_traj_steps"])
        self.traj_dim = int(cfg["traj_dim"])
        ds_cfg = cfg.get("dataset", {})
        self.mean = torch.tensor(ds_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=torch.float32)
        self.std = torch.tensor(ds_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=torch.float32)
        self.normalize_ego = bool(ds_cfg.get("normalize_ego_state", True))
        self.normalize_traj = bool(ds_cfg.get("normalize_candidate_traj", True))
        self.normalize_mode = ds_cfg.get("normalize_mode", "tanh")
        traj_scale = ds_cfg.get("traj_scale")
        self.traj_scale = torch.tensor(traj_scale, dtype=torch.float32) if traj_scale is not None else None

    def __len__(self) -> int:
        return len(self.rows)

    def _prepare_vector(self, values: Any, length: int) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32).flatten()
        if tensor.numel() < length:
            tensor = F.pad(tensor, (0, length - tensor.numel()))
        return tensor[:length]

    def _prepare_traj(self, values: Any) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"candidate_traj must be 2D, got shape={tuple(tensor.shape)}")
        if tensor.shape[1] < self.traj_dim:
            tensor = F.pad(tensor, (0, self.traj_dim - tensor.shape[1]))
        tensor = tensor[:, : self.traj_dim]
        if tensor.shape[0] < self.candidate_traj_steps:
            tensor = F.pad(tensor, (0, 0, 0, self.candidate_traj_steps - tensor.shape[0]))
        return tensor[: self.candidate_traj_steps]

    def _select_frames(self, tensor: torch.Tensor, count: int) -> torch.Tensor:
        selected = tensor[-count:]
        if selected.shape[0] < count:
            pad = selected[:1].repeat(count - selected.shape[0], 1, 1, 1)
            selected = torch.cat([pad, selected], dim=0)
        return selected

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        future_value = row.get("future_images", row.get("generated_future_images", row.get("generated_images")))
        if future_value is None:
            raise KeyError("Sample must contain future_images/generated_future_images/generated_images")

        hist = _as_tensor_image_sequence(
            row["history_images"], self.image_root, self.image_size, self.mean, self.std,
        )
        fut = _as_tensor_image_sequence(
            future_value, self.image_root, self.image_size, self.mean, self.std,
        )
        ego = self._prepare_vector(row["ego_state"], self.ego_state_dim)
        traj = self._prepare_traj(row["candidate_traj"])
        if self.normalize_ego:
            ego = torch.tanh(ego)
        if self.normalize_traj:
            if self.normalize_mode == "linear" and self.traj_scale is not None:
                traj = traj / self.traj_scale
            else:
                traj = torch.tanh(traj)
        return {
            "history_images": self._select_frames(hist, self.history_num_frames),
            "future_images": self._select_frames(fut, self.future_num_frames),
            "ego_state": ego,
            "candidate_traj": traj,
        }


def _state_looks_dinov2(state: Dict[str, Any]) -> bool:
    return any(
        key.startswith("image_encoder.")
        or key.startswith("module.image_encoder.")
        or key.startswith("_cnn_shared_backbone.")
        for key in state
    )


def _resolve_model_kind(
    requested: str,
    cfg: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> str:
    if requested != "auto":
        return requested
    dcfg = cfg.get("dinov2")
    if isinstance(dcfg, dict) and bool(dcfg.get("enabled", False)):
        return "dinov2"
    state = checkpoint.get("model", {})
    if isinstance(state, dict) and _state_looks_dinov2(state):
        return "dinov2"
    return "cnn"


def _load_model(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    model_kind: str,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved_kind = _resolve_model_kind(model_kind, cfg, checkpoint)
    if resolved_kind == "dinov2":
        from train_dinov2_v5_minimal import DINOv2ConsistencyCritic

        model = DINOv2ConsistencyCritic(cfg).to(device)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model = ConsistencyCriticModel(cfg).to(device)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, {
        "kind": resolved_kind,
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _label(row: Dict[str, Any], key: str, fallback: str = "label") -> float | None:
    if key in row:
        return float(row[key])
    if fallback in row:
        return float(row[fallback])
    return None


def _sigmoid_scores_from_output(
    out: Dict[str, torch.Tensor],
    key: str,
) -> List[float]:
    if key not in out:
        available = ", ".join(sorted(out.keys()))
        raise KeyError(
            f"model output does not contain score key '{key}'. "
            f"Available output keys: {available}"
        )
    return torch.sigmoid(out[key]).cpu().tolist()


def _trajectory_progress_from_rows(
    rows: List[Dict[str, Any]],
    *,
    mode: str,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values: List[float] = []
    for row in rows:
        arr = _candidate_traj_array(row.get("candidate_traj", []))
        if arr.size == 0:
            values.append(0.0)
            continue
        xy = arr[:, :2].astype(np.float32)
        if mode == "path_length":
            origin = np.zeros((1, 2), dtype=np.float32)
            prev = np.concatenate([origin, xy[:-1]], axis=0)
            value = float(np.linalg.norm(xy - prev, axis=1).sum())
        elif mode == "forward":
            value = float(max(xy[-1, 0], 0.0))
        elif mode == "final_displacement":
            value = float(np.linalg.norm(xy[-1]))
        else:
            raise ValueError(f"unknown progress_fusion_mode: {mode}")
        values.append(value / max(float(scale), 1e-6))
    return torch.tensor(values, device=device, dtype=dtype)


def _progress_fused_scores_from_output(
    out: Dict[str, torch.Tensor],
    key: str,
    batch_rows: List[Dict[str, Any]],
    *,
    beta: float,
    mode: str,
    scale: float,
    return_extras: bool = False,
) -> Tuple[List[float], Dict[str, List[float]]]:
    if key not in out:
        available = ", ".join(sorted(out.keys()))
        raise KeyError(
            f"model output does not contain score key '{key}'. "
            f"Available output keys: {available}"
        )
    logits = out[key]
    extras: Dict[str, List[float]] = {}
    if "progress_alignment_value" in out:
        image_progress = out["progress_alignment_value"]
        traj_progress = _trajectory_progress_from_rows(
            batch_rows,
            mode=mode,
            scale=scale,
            device=image_progress.device,
            dtype=image_progress.dtype,
        )
        progress_error = (image_progress - traj_progress).abs()
        if beta > 0.0:
            logits = logits - float(beta) * progress_error.to(dtype=logits.dtype)
        if return_extras:
            extras = {
                "iac_base_consistency": torch.sigmoid(out[key]).cpu().tolist(),
                "progress_alignment_value": image_progress.cpu().tolist(),
                "trajectory_progress_value": traj_progress.cpu().tolist(),
                "progress_alignment_error": progress_error.cpu().tolist(),
            }
    elif beta > 0.0:
        available = ", ".join(sorted(out.keys()))
        raise KeyError(
            "progress-fused scoring requested, but model output does not contain "
            f"progress_alignment_value. Available output keys: {available}"
        )
    return torch.sigmoid(logits).cpu().tolist(), extras


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _ndcg(labels: List[float], scores: List[float], k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = np.array(labels)[order]
    discounts = 1.0 / np.log2(np.arange(len(gains)) + 2)
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def _candidate_group_id(row: Dict[str, Any], group_key: str) -> str | None:
    group = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if group is not None:
        return str(group)
    sample_id = row.get("sample_id")
    if sample_id is None:
        return None
    sample_id = str(sample_id)
    source = row.get("source_type") or row.get("sample_type") or row.get("action_type")
    if source is not None:
        suffix = f"__{source}"
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    if "__" in sample_id:
        return sample_id.rsplit("__", 1)[0]
    return sample_id


def _row_source(row: Dict[str, Any], wam_key: str = "wam_name") -> str:
    for key in ("source_type", "action_type", wam_key, "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _evidence_delta(row: Dict[str, Any]) -> float | None:
    for key in (
        "candidate_minus_wrong_exclusive_path_delta",
        "candidate_minus_wrong_path_delta",
    ):
        if row.get(key) is not None:
            return float(row[key])
    return None


def _ambiguity_metrics(
    scored: List[Dict[str, Any]],
    group_key: str,
    wam_key: str,
    *,
    close_margin: float = 0.02,
    near_sources: Iterable[str] = (
        "perturb_speed",
        "perturb_lateral",
        "perturb_heading",
    ),
    evidence_margin: float = 0.0,
) -> Dict[str, Any]:
    near_source_set = {str(source) for source in near_sources}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        group = _candidate_group_id(row, group_key)
        if group is not None and row.get("consistency_label") is not None:
            groups[str(group)].append(row)

    counts = {
        "hit": 0,
        "ambiguous_accept": 0,
        "evidence_supported_miss": 0,
        "likely_model_error": 0,
    }
    miss_sources: Dict[str, int] = defaultdict(int)
    close_gaps: List[float] = []
    miss_gaps: List[float] = []
    recovered_scores: List[float] = []
    per_group: List[Dict[str, Any]] = []

    for group_id, rows in groups.items():
        if len(rows) < 2:
            continue
        positives = [
            row for row in rows
            if float(row.get("consistency_label", row.get("label", 0.0))) > 0.5
        ]
        if not positives:
            continue
        positive = positives[0]
        ranked = sorted(rows, key=lambda row: float(row["iac_consistency"]), reverse=True)
        pos_rank = next(idx + 1 for idx, row in enumerate(ranked) if row is positive)
        winner = ranked[0]
        winner_source = _row_source(winner, wam_key)
        gt_score = float(positive["iac_consistency"])
        winner_score = float(winner["iac_consistency"])
        score_gap = max(0.0, winner_score - gt_score)
        path_minus_sky = positive.get("path_minus_sky_delta")
        exact_delta = _evidence_delta(positive)
        recovered = exact_delta if exact_delta is not None else path_minus_sky
        if recovered is not None:
            recovered_scores.append(float(recovered))

        if pos_rank == 1:
            category = "hit"
        else:
            miss_gaps.append(score_gap)
            miss_sources[winner_source] += 1
            path_supported = path_minus_sky is not None and float(path_minus_sky) > 0.0
            exact_supported = exact_delta is not None and float(exact_delta) > evidence_margin
            is_ambiguous = (
                winner_source in near_source_set
                and score_gap <= close_margin
                and path_supported
            )
            if is_ambiguous:
                category = "ambiguous_accept"
                close_gaps.append(score_gap)
            elif exact_supported:
                category = "evidence_supported_miss"
            else:
                category = "likely_model_error"
        counts[category] += 1
        per_group.append(
            {
                "group_id": group_id,
                "category": category,
                "positive_rank": pos_rank,
                "winning_source": winner_source,
                "score_gap": score_gap,
                "gt_score": gt_score,
                "winner_score": winner_score,
                "gt_path_minus_sky_delta": path_minus_sky,
                "gt_exact_path_delta": exact_delta,
                "recovered_path_agreement_score": recovered,
                "winner_sample_id": winner.get("sample_id"),
                "gt_sample_id": positive.get("sample_id"),
            }
        )

    num_groups = len(per_group)
    misses = num_groups - counts["hit"]
    ambiguity_adjusted_hits = counts["hit"] + counts["ambiguous_accept"]
    return {
        "definition": (
            "IAC-PathBench v2 ambiguity-aware ranking audit. Near-neighbor "
            "speed/lateral/heading winners with a small score gap and positive "
            "GT path evidence are treated as ambiguous accepts, not full errors."
        ),
        "num_groups": num_groups,
        "raw_miss_fraction": misses / num_groups if num_groups else None,
        "close_margin": close_margin,
        "near_sources": sorted(near_source_set),
        "evidence_margin": evidence_margin,
        "formal_categories": [
            "hit",
            "ambiguous_accept",
            "evidence_supported_miss",
            "likely_model_error",
        ],
        "counts": counts,
        "hard_top1": counts["hit"] / num_groups if num_groups else None,
        "ambiguity_adjusted_top1": (
            ambiguity_adjusted_hits / num_groups if num_groups else None
        ),
        "misses": misses,
        "close_miss_rate": (
            sum(gap <= close_margin for gap in miss_gaps) / misses
            if misses else None
        ),
        "ambiguous_accept_fraction_of_misses": (
            counts["ambiguous_accept"] / misses if misses else None
        ),
        "evidence_supported_fraction_of_misses": (
            counts["evidence_supported_miss"] / misses if misses else None
        ),
        "likely_model_error_fraction_of_misses": (
            counts["likely_model_error"] / misses if misses else None
        ),
        "miss_source_distribution": dict(sorted(miss_sources.items())),
        "mean_miss_gap": _mean(miss_gaps),
        "mean_close_ambiguity_gap": _mean(close_gaps),
        "recovered_path_agreement_score": _mean(recovered_scores),
        "per_group": per_group,
    }


def _iac_pathbench_v2_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    path = summary.get("path_causal_metrics", {})
    traj = summary.get("trajectory_specific_causal_metrics", {})
    positive_traj = traj.get("positive_rows", {}) if isinstance(traj, dict) else {}
    ranking = summary.get("ranking", {})
    ambiguity = summary.get("ambiguity_metrics", {})
    threshold = (
        summary.get("overall", {})
        .get("consistency_threshold_sweep", {})
        .get("best_balanced_accuracy", {})
    )
    return {
        "protocol": {
            "name": "IAC-PathBench v2 ambiguity-aware",
            "primary_scientific_metrics": [
                "exact_path_win_fraction",
                "exact_path_delta",
                "path_minus_sky_delta",
                "ambiguity_adjusted_top1",
            ],
            "secondary_ranking_metrics": [
                "hard_top1",
                "mrr",
                "ndcg@3",
                "ndcg@5",
                "best_balanced_accuracy",
            ],
            "formal_categories": [
                "hit",
                "ambiguous_accept",
                "evidence_supported_miss",
                "likely_model_error",
            ],
            "hard_top1_is_secondary": True,
        },
        "primary_scientific_metrics": {
            "exact_path_win_fraction": positive_traj.get(
                "candidate_exclusive_delta_gt_wrong_fraction",
                positive_traj.get("candidate_delta_gt_wrong_fraction"),
            ),
            "exact_path_delta": positive_traj.get(
                "mean_candidate_minus_wrong_exclusive_delta",
                positive_traj.get("mean_candidate_minus_wrong_delta"),
            ),
            "path_minus_sky_delta": path.get("mean_path_minus_sky_delta"),
            "ambiguity_adjusted_top1": ambiguity.get("ambiguity_adjusted_top1"),
        },
        "secondary_ranking_metrics": {
            "hard_top1": ranking.get("top1_hit_rate"),
            "mrr": ranking.get("mrr"),
            "ndcg@3": ranking.get("ndcg@3"),
            "ndcg@5": ranking.get("ndcg@5"),
            "best_balanced_accuracy": threshold.get("balanced_accuracy"),
        },
        "diagnostic_metrics": {
            "close_miss_rate": ambiguity.get("close_miss_rate"),
            "raw_miss_fraction": ambiguity.get("raw_miss_fraction"),
            "miss_source_distribution": ambiguity.get("miss_source_distribution"),
            "likely_model_error_fraction": ambiguity.get(
                "likely_model_error_fraction_of_misses"
            ),
            "ambiguity_supported_miss_fraction": ambiguity.get(
                "ambiguous_accept_fraction_of_misses"
            ),
            "recovered_path_agreement_score": ambiguity.get(
                "recovered_path_agreement_score"
            ),
        },
    }


def _recovered_set_metrics(
    scored: List[Dict[str, Any]],
    group_key: str,
    wam_key: str,
) -> Dict[str, Any]:
    rows_with_recovered = [
        row for row in scored
        if row.get("recovered_set_minade") is not None
    ]
    if not rows_with_recovered:
        return {}

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        group = _candidate_group_id(row, group_key)
        if group is not None and row.get("consistency_label") is not None:
            groups[str(group)].append(row)

    source_values: Dict[str, List[float]] = defaultdict(list)
    for row in rows_with_recovered:
        source_values[_row_source(row, wam_key)].append(float(row["recovered_set_minade"]))

    per_source = {
        source: {
            "count": len(values),
            "mean_minade": _mean(values),
            "median_minade": float(np.median(values)) if values else None,
            "p90_minade": float(np.percentile(values, 90)) if values else None,
        }
        for source, values in sorted(source_values.items())
    }

    per_group: List[Dict[str, Any]] = []
    current_hits: List[float] = []
    recovered_hits: List[float] = []
    gt_minades: List[float] = []
    winner_minades: List[float] = []
    gt_better_than_winner: List[float] = []
    winner_supported: List[float] = []
    gt_supported: List[float] = []
    set_sizes: List[float] = []
    categories: Dict[str, int] = defaultdict(int)
    indistinguishability_categories: Dict[str, int] = defaultdict(int)
    indistinguishable_hits: List[float] = []
    true_model_errors: List[float] = []
    clear_negative_rejections: List[float] = []
    indistinguishable_gaps: List[float] = []
    near_sources = {"perturb_speed", "perturb_lateral", "perturb_heading"}
    clear_negative_sources = {
        "image_swap",
        "time_shift_future",
        "traj_swap",
        "reverse_traj",
    }
    close_margin = 0.02

    for group_id, rows in groups.items():
        if len(rows) < 2:
            continue
        positives = [
            row for row in rows
            if float(row.get("consistency_label", row.get("label", 0.0))) > 0.5
        ]
        if not positives:
            continue
        positive = positives[0]
        rows = [row for row in rows if row.get("recovered_set_minade") is not None]
        if len(rows) < 2 or positive.get("recovered_set_minade") is None:
            continue
        current = max(rows, key=lambda row: float(row["iac_consistency"]))
        recovered = min(rows, key=lambda row: float(row["recovered_set_minade"]))
        radius = positive.get("recovered_set_conformal_radius")
        if radius is None:
            radius = positive.get("recovered_set_radius")
        radius_value = float(radius) if radius is not None else None
        supported_rows: List[Dict[str, Any]] = []
        if radius_value is not None:
            supported_rows = [
                row for row in rows
                if float(row["recovered_set_minade"]) <= radius_value
            ]
        current_hit = current is positive
        recovered_hit = recovered is positive
        gt_minade = float(positive["recovered_set_minade"])
        current_minade = float(current["recovered_set_minade"])
        source = _row_source(current, wam_key)
        score_gap = max(
            0.0,
            float(current["iac_consistency"]) - float(positive["iac_consistency"]),
        )
        current_supported = (
            current.get("recovered_set_supported")
            if current.get("recovered_set_supported") is not None
            else (
                float(current["recovered_set_minade"]) <= radius_value
                if radius_value is not None else None
            )
        )
        gt_is_supported = (
            positive.get("recovered_set_supported")
            if positive.get("recovered_set_supported") is not None
            else (
                gt_minade <= radius_value if radius_value is not None else None
            )
        )
        if current_hit:
            category = "hit"
        elif bool(current_supported) and source in near_sources:
            category = "set_ambiguous_near_miss"
        elif gt_minade < current_minade:
            category = "set_prefers_gt"
        else:
            category = "set_prefers_winner_or_error"
        gt_supported_bool = bool(gt_is_supported)
        current_supported_bool = bool(current_supported)
        is_visual_indistinguishable = (
            (not current_hit)
            and source in near_sources
            and score_gap <= close_margin
            and gt_supported_bool
            and current_supported_bool
        )
        if current_hit:
            indistinguishability_category = "hit"
        elif is_visual_indistinguishable:
            indistinguishability_category = "visually_indistinguishable_near_miss"
            indistinguishable_gaps.append(score_gap)
        elif source in clear_negative_sources:
            if current_supported_bool:
                indistinguishability_category = "clear_negative_supported_error"
            else:
                indistinguishability_category = "clear_negative_rejected_but_ranked"
        elif not gt_supported_bool:
            indistinguishability_category = "unsupported_gt_error"
        else:
            indistinguishability_category = "ambiguous_or_model_error"
        categories[category] += 1
        indistinguishability_categories[indistinguishability_category] += 1
        current_hits.append(float(current_hit))
        recovered_hits.append(float(recovered_hit))
        gt_minades.append(gt_minade)
        winner_minades.append(current_minade)
        gt_better_than_winner.append(float(gt_minade < current_minade))
        indistinguishable_hits.append(float(current_hit or is_visual_indistinguishable))
        true_model_errors.append(float(
            not current_hit
            and not is_visual_indistinguishable
            and (
                source in clear_negative_sources
                or not gt_supported_bool
                or not current_supported_bool
            )
        ))
        clear_negative_rejections.append(float(
            source in clear_negative_sources and not current_supported_bool
        ))
        if current_supported is not None:
            winner_supported.append(float(current_supported))
        if gt_is_supported is not None:
            gt_supported.append(float(gt_is_supported))
        if supported_rows:
            set_sizes.append(float(len(supported_rows)))
        per_group.append(
            {
                "group_id": group_id,
                "category": category,
                "current_top1_hit": current_hit,
                "recovered_set_top1_hit": recovered_hit,
                "current_winner_source": source,
                "recovered_set_winner_source": _row_source(recovered, wam_key),
                "visual_indistinguishability_category": indistinguishability_category,
                "score_gap": score_gap,
                "gt_minade": gt_minade,
                "current_winner_minade": current_minade,
                "recovered_set_winner_minade": float(recovered["recovered_set_minade"]),
                "gt_better_than_current_winner": gt_minade < current_minade,
                "ambiguity_radius": radius_value,
                "ambiguity_set_size": len(supported_rows) if supported_rows else None,
                "current_winner_supported": (
                    bool(current_supported)
                    if current_supported is not None else None
                ),
                "gt_supported": (
                    bool(gt_is_supported)
                    if gt_is_supported is not None else None
                ),
                "gt_sample_id": positive.get("sample_id"),
                "current_winner_sample_id": current.get("sample_id"),
            }
        )

    return {
        "definition": (
            "IAC-PathBench v3 recovered-set support audit. Candidate support "
            "is measured by minADE to a K-path set recovered from future image "
            "features; conformal support fields are used when present."
        ),
        "num_rows_with_recovered_set": len(rows_with_recovered),
        "num_groups": len(per_group),
        "current_hard_top1": _mean(current_hits),
        "recovered_set_top1": _mean(recovered_hits),
        "mean_gt_minade": _mean(gt_minades),
        "mean_current_winner_minade": _mean(winner_minades),
        "gt_minade_lt_current_winner_fraction": _mean(gt_better_than_winner),
        "current_winner_supported_fraction": _mean(winner_supported),
        "gt_supported_fraction": _mean(gt_supported),
        "mean_ambiguity_set_size": _mean(set_sizes),
        "support_categories": dict(sorted(categories.items())),
        "visual_indistinguishability": {
            "definition": (
                "A miss is visually indistinguishable only when a near "
                "speed/lateral/heading winner beats GT by <=0.02 and both GT "
                "and winner are inside the recovered-path conformal support set."
            ),
            "close_margin": close_margin,
            "near_sources": sorted(near_sources),
            "clear_negative_sources": sorted(clear_negative_sources),
            "categories": dict(sorted(indistinguishability_categories.items())),
            "visual_support_set_accuracy": _mean(indistinguishable_hits),
            "visually_indistinguishable_near_miss_fraction": (
                indistinguishability_categories[
                    "visually_indistinguishable_near_miss"
                ] / len(per_group)
                if per_group else None
            ),
            "true_model_error_fraction": _mean(true_model_errors),
            "clear_negative_rejection_fraction": _mean(clear_negative_rejections),
            "mean_indistinguishable_gap": _mean(indistinguishable_gaps),
        },
        "by_source_type": per_source,
        "per_group": per_group,
    }


def _iac_pathbench_v3_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    v2 = summary.get("iac_pathbench_v2", {})
    primary = v2.get("primary_scientific_metrics", {})
    secondary = v2.get("secondary_ranking_metrics", {})
    diagnostic = v2.get("diagnostic_metrics", {})
    recovered = summary.get("recovered_set_metrics", {})
    return {
        "primary_scientific_metrics": {
            "exact_path_win_fraction": primary.get("exact_path_win_fraction"),
            "exact_path_delta": primary.get("exact_path_delta"),
            "path_minus_sky_delta": primary.get("path_minus_sky_delta"),
            "ambiguity_adjusted_top1": primary.get("ambiguity_adjusted_top1"),
            "recovered_set_gt_supported_fraction": recovered.get("gt_supported_fraction"),
            "recovered_set_winner_supported_fraction": recovered.get("current_winner_supported_fraction"),
        },
        "secondary_ranking_metrics": {
            "hard_top1": secondary.get("hard_top1"),
            "mrr": secondary.get("mrr"),
            "recovered_set_top1": recovered.get("recovered_set_top1"),
            "recovered_set_gt_minade": recovered.get("mean_gt_minade"),
        },
        "diagnostic_metrics": {
            "likely_model_error_fraction": diagnostic.get("likely_model_error_fraction"),
            "ambiguity_supported_miss_fraction": diagnostic.get("ambiguity_supported_miss_fraction"),
            "recovered_set_gt_better_than_winner_fraction": recovered.get(
                "gt_minade_lt_current_winner_fraction"
            ),
            "recovered_set_mean_ambiguity_set_size": recovered.get(
                "mean_ambiguity_set_size"
            ),
            "recovered_set_support_categories": recovered.get("support_categories"),
            "visual_indistinguishability": recovered.get("visual_indistinguishability"),
            "recovered_set_by_source_type": recovered.get("by_source_type"),
        },
    }


def _ranking_summary(scored: List[Dict[str, Any]], group_key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        group = _candidate_group_id(row, group_key)
        if group is not None and row.get("consistency_label") is not None:
            groups[str(group)].append(row)

    top1_hits, top2_hits, top3_hits, mrrs, ndcg3, ndcg5 = [], [], [], [], [], []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        labels = [float(row["consistency_label"]) for row in rows]
        if max(labels) <= 0:
            continue
        scores = [float(row["iac_consistency"]) for row in rows]
        order = np.argsort(scores)[::-1]
        sorted_labels = np.array(labels)[order]
        top1_hits.append(float(sorted_labels[0] > 0))
        top2_hits.append(float(np.max(sorted_labels[:2]) > 0))
        top3_hits.append(float(np.max(sorted_labels[:3]) > 0))
        first_pos = np.where(sorted_labels > 0)[0]
        mrrs.append(1.0 / float(first_pos[0] + 1) if len(first_pos) else 0.0)
        ndcg3.append(_ndcg(labels, scores, 3))
        ndcg5.append(_ndcg(labels, scores, 5))

    return {
        "num_groups": len(groups),
        "num_ranked_groups": len(top1_hits),
        "top1_hit_rate": _mean(top1_hits),
        "top2_hit_rate": _mean(top2_hits),
        "top3_hit_rate": _mean(top3_hits),
        "mrr": _mean(mrrs),
        "ndcg@3": _mean(ndcg3),
        "ndcg@5": _mean(ndcg5),
    }


def _summary(
    scored: List[Dict[str, Any]],
    wam_key: str,
    group_key: str,
    consistency_score_key: str = "consistency_logit",
    visual_metrics: List[Dict[str, Any]] | None = None,
    geometric_metrics: List[Dict[str, Any]] | None = None,
    memory_metrics: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    c_labels = [row.get("consistency_label") for row in scored]
    v_labels = [row.get("validity_label") for row in scored]
    c_scores = torch.tensor([row["iac_consistency"] for row in scored], dtype=torch.float32)
    v_scores = torch.tensor([row["iac_validity"] for row in scored], dtype=torch.float32)

    summary: Dict[str, Any] = {
        "num_samples": len(scored),
        "consistency_score_key": consistency_score_key,
        "overall": {
            "mean_consistency": float(c_scores.mean().item()),
            "mean_validity": float(v_scores.mean().item()),
        },
        "by_wam": {},
        "by_action_type": {},
        "ranking": _ranking_summary(scored, group_key),
        "graded_perturbation_curve": {},
    }

    if visual_metrics:
        # Aggregate per-key mean across all rows that have a value.
        keys = set().union(*(m.keys() for m in visual_metrics if m))
        agg = {k: float(np.mean([m[k] for m in visual_metrics if k in m and m[k] is not None]))
               for k in keys}
        summary["visual_metrics"] = agg
    if geometric_metrics:
        keys = set().union(*(m.keys() for m in geometric_metrics if m))
        agg = {k: float(np.mean([m[k] for m in geometric_metrics if k in m]))
               for k in keys}
        summary["geometric_metrics"] = agg
    if memory_metrics:
        keys = set().union(*(m.keys() for m in memory_metrics if m))
        agg = {k: float(np.mean([m[k] for m in memory_metrics if k in m and isinstance(m[k], (int, float))]))
               for k in keys if k != "loop_closure"}
        summary["memory_metrics"] = agg
        # Loop-closure is a list of dicts, not a scalar
        summary["memory_metrics_loop_closure"] = [
            m for m in memory_metrics if "loop_closure" in m
        ]
    if any("iac_consistency_path_masked" in row for row in scored):
        summary["path_causal_metrics"] = _causal_aggregate(scored)
    if any("iac_consistency_wrong_path_masked" in row for row in scored):
        summary["trajectory_specific_causal_metrics"] = _trajectory_specific_aggregate(scored)
    ambiguity = _ambiguity_metrics(scored, group_key, wam_key)
    per_group = ambiguity.pop("per_group", [])
    summary["ambiguity_metrics"] = ambiguity
    if per_group:
        summary["ambiguity_metrics"]["num_per_group_records"] = len(per_group)
    recovered_set = _recovered_set_metrics(scored, group_key, wam_key)
    recovered_per_group = recovered_set.pop("per_group", []) if recovered_set else []
    if recovered_set:
        summary["recovered_set_metrics"] = recovered_set
        if recovered_per_group:
            summary["recovered_set_metrics"]["num_per_group_records"] = len(
                recovered_per_group
            )

    if all(label is not None for label in c_labels):
        labels = torch.tensor([float(label) for label in c_labels], dtype=torch.float32)
        logits = torch.logit(c_scores.clamp(1e-6, 1 - 1e-6))
        summary["overall"]["consistency_binary"] = _compute_head_metrics(logits, labels)
        summary["overall"]["consistency_threshold_sweep"] = _threshold_sweep(
            c_scores, labels,
        )
    if all(label is not None for label in v_labels):
        labels = torch.tensor([float(label) for label in v_labels], dtype=torch.float32)
        logits = torch.logit(v_scores.clamp(1e-6, 1 - 1e-6))
        summary["overall"]["validity_binary"] = _compute_head_metrics(logits, labels)
        summary["overall"]["validity_threshold_sweep"] = _threshold_sweep(
            v_scores, labels,
        )

    for key_name, output_key in ((wam_key, "by_wam"), ("action_type", "by_action_type")):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in scored:
            value = row.get(key_name) or row.get("model_name") or row.get("source_type") or row.get("sample_type") or "unknown"
            groups[str(value)].append(row)
        for value, rows in groups.items():
            summary[output_key][value] = {
                "count": len(rows),
                "mean_consistency": _mean(row["iac_consistency"] for row in rows),
                "mean_validity": _mean(row["iac_validity"] for row in rows),
            }

    graded: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        if row.get("perturb_magnitude") is None:
            continue
        ptype = row.get("perturb_type") or row.get("action_type") or "perturb"
        level = row.get("perturb_level", "unknown")
        graded[f"{ptype}:{level}"].append(row)
    for key, rows in graded.items():
        summary["graded_perturbation_curve"][key] = {
            "count": len(rows),
            "mean_consistency": _mean(row["iac_consistency"] for row in rows),
            "mean_perturb_magnitude": _mean(float(row["perturb_magnitude"]) for row in rows),
        }
    summary["iac_pathbench_v2"] = _iac_pathbench_v2_summary(summary)
    if "recovered_set_metrics" in summary:
        summary["iac_pathbench_v3"] = _iac_pathbench_v3_summary(summary)
    return summary


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(args.config) if args.config else checkpoint["config"]
    rows = _load_rows(Path(args.input))
    image_root_path = Path(args.image_root) if args.image_root else Path(cfg["image_root"])
    missing_report: Dict[str, Any] | None = None
    if args.skip_missing_images:
        original_count = len(rows)
        rows, dropped = _filter_rows_with_existing_images(rows, image_root_path)
        missing_report = {
            "input": str(args.input),
            "image_root": str(image_root_path),
            "original_count": original_count,
            "kept_count": len(rows),
            "dropped_count": len(dropped),
            "dropped_rows": dropped,
        }
        print(
            f"[INFO] skip-missing-images kept {len(rows)}/{original_count} rows "
            f"(dropped {len(dropped)})."
        )
        if args.missing_report:
            report_path = Path(args.missing_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(missing_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    if args.max_groups:
        rows = _limit_rows_by_groups(rows, args.group_key, args.max_groups)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if args.trajectory_specific_causal_metrics:
        attached = _attach_wrong_path_controls(
            rows,
            args.group_key,
            selection=args.wrong_path_selection,
            path_width_ratio=args.path_mask_width,
            trajectory_mode=args.path_trajectory_mode,
            projection_mode=args.path_projection_mode,
            forward_m=args.path_forward_m,
            lateral_m=args.path_lateral_m,
            mask_height=int(cfg.get("image_size", 224)),
            mask_width=int(cfg.get("image_size", 224)),
        )
        if attached == 0:
            print(
                "[WARN] trajectory-specific causal metrics requested, "
                "but no same-group wrong paths were found.",
                file=sys.stderr,
            )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_info = _load_model(Path(args.checkpoint), cfg, device, args.model_kind)
    dataset = WAMManifestDataset(rows, cfg, args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    scored: List[Dict[str, Any]] = []
    visual_metrics_rows: List[Dict[str, Any]] = []
    geometric_metrics_rows: List[Dict[str, Any]] = []
    memory_metrics_rows: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_rows = [rows[offset + i] for i in range(len(batch["candidate_traj"]))]
            hist_images = batch["history_images"].to(device, non_blocking=True)
            fut_images = batch["future_images"].to(device, non_blocking=True)
            ego_state = batch["ego_state"].to(device, non_blocking=True)
            candidate_traj = batch["candidate_traj"].to(device, non_blocking=True)
            out = model(
                hist_images,
                fut_images,
                ego_state,
                candidate_traj,
            )
            c_scores, score_extras = _progress_fused_scores_from_output(
                out,
                args.consistency_score_key,
                batch_rows,
                beta=float(args.progress_fusion_beta),
                mode=str(args.progress_fusion_mode),
                scale=float(args.progress_fusion_scale),
                return_extras=True,
            )
            v_scores = torch.sigmoid(out["validity_logit"]).cpu().tolist()
            path_scores: List[float] | None = None
            sky_scores: List[float] | None = None
            wrong_path_scores: List[float] | None = None
            cand_exclusive_scores: List[float] | None = None
            wrong_exclusive_scores: List[float] | None = None
            path_fracs: List[float] = []
            sky_fracs: List[float] = []
            wrong_path_fracs: List[float] = []
            exclusive_stats: List[Dict[str, float]] = []
            if args.path_causal_metrics or args.trajectory_specific_causal_metrics:
                path_fut, path_fracs = _mask_future_images(
                    fut_images,
                    batch_rows,
                    "path",
                    path_width_ratio=args.path_mask_width,
                    sky_ratio=args.sky_mask_ratio,
                    trajectory_mode=args.path_trajectory_mode,
                    projection_mode=args.path_projection_mode,
                    forward_m=args.path_forward_m,
                    lateral_m=args.path_lateral_m,
                )
                path_out = model(hist_images, path_fut, ego_state, candidate_traj)
                path_scores, _ = _progress_fused_scores_from_output(
                    path_out,
                    args.consistency_score_key,
                    batch_rows,
                    beta=float(args.progress_fusion_beta),
                    mode=str(args.progress_fusion_mode),
                    scale=float(args.progress_fusion_scale),
                )

            if args.path_causal_metrics:
                sky_fut, sky_fracs = _mask_future_images(
                    fut_images,
                    batch_rows,
                    "sky",
                    path_width_ratio=args.path_mask_width,
                    sky_ratio=args.sky_mask_ratio,
                    trajectory_mode=args.path_trajectory_mode,
                    projection_mode=args.path_projection_mode,
                    forward_m=args.path_forward_m,
                    lateral_m=args.path_lateral_m,
                )
                sky_out = model(hist_images, sky_fut, ego_state, candidate_traj)
                sky_scores, _ = _progress_fused_scores_from_output(
                    sky_out,
                    args.consistency_score_key,
                    batch_rows,
                    beta=float(args.progress_fusion_beta),
                    mode=str(args.progress_fusion_mode),
                    scale=float(args.progress_fusion_scale),
                )
            if args.trajectory_specific_causal_metrics:
                wrong_fut, wrong_path_fracs = _mask_future_images(
                    fut_images,
                    batch_rows,
                    "wrong_path",
                    path_width_ratio=args.path_mask_width,
                    sky_ratio=args.sky_mask_ratio,
                    trajectory_mode=args.path_trajectory_mode,
                    projection_mode=args.path_projection_mode,
                    forward_m=args.path_forward_m,
                    lateral_m=args.path_lateral_m,
                )
                wrong_out = model(hist_images, wrong_fut, ego_state, candidate_traj)
                wrong_path_scores, _ = _progress_fused_scores_from_output(
                    wrong_out,
                    args.consistency_score_key,
                    batch_rows,
                    beta=float(args.progress_fusion_beta),
                    mode=str(args.progress_fusion_mode),
                    scale=float(args.progress_fusion_scale),
                )
                cand_excl_fut, wrong_excl_fut, exclusive_stats = (
                    _mask_future_images_exclusive_paths(
                        fut_images,
                        batch_rows,
                        path_width_ratio=args.path_mask_width,
                        trajectory_mode=args.path_trajectory_mode,
                        projection_mode=args.path_projection_mode,
                        forward_m=args.path_forward_m,
                        lateral_m=args.path_lateral_m,
                    )
                )
                cand_excl_out = model(hist_images, cand_excl_fut, ego_state, candidate_traj)
                wrong_excl_out = model(hist_images, wrong_excl_fut, ego_state, candidate_traj)
                cand_exclusive_scores, _ = _progress_fused_scores_from_output(
                    cand_excl_out,
                    args.consistency_score_key,
                    batch_rows,
                    beta=float(args.progress_fusion_beta),
                    mode=str(args.progress_fusion_mode),
                    scale=float(args.progress_fusion_scale),
                )
                wrong_exclusive_scores, _ = _progress_fused_scores_from_output(
                    wrong_excl_out,
                    args.consistency_score_key,
                    batch_rows,
                    beta=float(args.progress_fusion_beta),
                    mode=str(args.progress_fusion_mode),
                    scale=float(args.progress_fusion_scale),
                )
            for i, (c_score, v_score) in enumerate(zip(c_scores, v_scores)):
                row = dict(rows[offset + i])
                row.pop("_wrong_candidate_traj", None)
                if row.get(args.group_key) is None and row.get("group_id") is None:
                    inferred_group = _candidate_group_id(row, args.group_key)
                    if inferred_group is not None:
                        row["group_id"] = inferred_group
                row["iac_consistency"] = float(c_score)
                row["iac_validity"] = float(v_score)
                for key, values in score_extras.items():
                    row[key] = float(values[i])
                if path_scores is not None:
                    row["iac_consistency_path_masked"] = float(path_scores[i])
                    row["path_mask_delta"] = float(c_score - path_scores[i])
                    row["path_mask_fraction"] = float(path_fracs[i])
                if sky_scores is not None:
                    row["iac_consistency_sky_masked"] = float(sky_scores[i])
                    row["sky_mask_delta"] = float(c_score - sky_scores[i])
                    row["sky_mask_fraction"] = float(sky_fracs[i])
                    if path_scores is not None:
                        row["path_minus_sky_delta"] = float(
                            (c_score - path_scores[i]) - (c_score - sky_scores[i])
                        )
                if wrong_path_scores is not None and row.get("wrong_path_source_type") is not None:
                    row["iac_consistency_wrong_path_masked"] = float(wrong_path_scores[i])
                    row["wrong_path_delta"] = float(c_score - wrong_path_scores[i])
                    row["wrong_path_mask_fraction"] = float(wrong_path_fracs[i])
                    if path_scores is not None:
                        row["candidate_minus_wrong_path_delta"] = float(
                            (c_score - path_scores[i])
                            - (c_score - wrong_path_scores[i])
                        )
                    if cand_exclusive_scores is not None and wrong_exclusive_scores is not None:
                        row["iac_consistency_candidate_exclusive_path_masked"] = float(
                            cand_exclusive_scores[i]
                        )
                        row["iac_consistency_wrong_exclusive_path_masked"] = float(
                            wrong_exclusive_scores[i]
                        )
                        row["candidate_exclusive_path_delta"] = float(
                            c_score - cand_exclusive_scores[i]
                        )
                        row["wrong_exclusive_path_delta"] = float(
                            c_score - wrong_exclusive_scores[i]
                        )
                        row["candidate_minus_wrong_exclusive_path_delta"] = float(
                            (c_score - cand_exclusive_scores[i])
                            - (c_score - wrong_exclusive_scores[i])
                        )
                        if i < len(exclusive_stats):
                            row.update(exclusive_stats[i])
                c_label = _label(row, "consistency_label")
                v_label = _label(row, "validity_label")
                if c_label is not None:
                    row["consistency_label"] = c_label
                if v_label is not None:
                    row["validity_label"] = v_label
                scored.append(row)

                # Optional: iWorld-Bench style cross-validation metrics
                if args.visual_metrics:
                    try:
                        fut_paths = row.get("future_images") or row.get("generated_future_images") or row.get("generated_images")
                        if isinstance(fut_paths, list) and fut_paths and all(isinstance(x, str) for x in fut_paths):
                            abs_paths = [
                                str(Path(x) if Path(x).is_absolute() else image_root_path / x)
                                for x in fut_paths
                            ]
                            frames = load_frames_from_paths(abs_paths, size=args.visual_size)
                            visual_metrics_rows.append(compute_all_visual_metrics(frames))
                        else:
                            visual_metrics_rows.append({})
                    except Exception as exc:  # noqa: BLE001
                        visual_metrics_rows.append({"error": str(exc)})

                if args.geometric_metrics:
                    try:
                        fut_paths = row.get("future_images") or row.get("generated_future_images") or row.get("generated_images")
                        if isinstance(fut_paths, list) and fut_paths and all(isinstance(x, str) for x in fut_paths):
                            abs_paths = [
                                str(Path(x) if Path(x).is_absolute() else image_root_path / x)
                                for x in fut_paths
                            ]
                            est = estimate_trajectory_from_video(abs_paths, size=args.visual_size)
                            gt = candidate_traj_to_traj(row["candidate_traj"])
                            geometric_metrics_rows.append({
                                "trajectory_accuracy": compute_trajectory_accuracy(est, gt),
                            })
                        else:
                            geometric_metrics_rows.append({})
                    except Exception as exc:  # noqa: BLE001
                        geometric_metrics_rows.append({"error": str(exc)})

                if args.memory_metrics:
                    row_mm: Dict[str, Any] = {}
                    try:
                        fut_paths = row.get("future_images") or row.get("generated_future_images") or row.get("generated_images")
                        if isinstance(fut_paths, list) and fut_paths and all(isinstance(x, str) for x in fut_paths):
                            abs_paths = [
                                str(Path(x) if Path(x).is_absolute() else image_root_path / x)
                                for x in fut_paths
                            ]
                            frames = load_frames_from_paths(abs_paths, size=args.visual_size)
                            row_mm["memory_symmetry"] = compute_memory_symmetry(frames)
                    except Exception as exc:  # noqa: BLE001
                        row_mm["memory_symmetry_error"] = str(exc)
                    # Loop-closure drift uses GT traj + reverse traj if supplied
                    rev = row.get("reverse_candidate_traj")
                    if rev is not None:
                        fwd = candidate_traj_to_traj(row["candidate_traj"])
                        rev_t = candidate_traj_to_traj(rev)
                        row_mm["loop_closure"] = compute_loop_closure_drift(fwd, rev_t)
                    memory_metrics_rows.append(row_mm)

            offset += len(c_scores)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "wam_iac_scores.jsonl"
    with scored_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summary(
        scored, args.wam_key, args.group_key, args.consistency_score_key,
        visual_metrics=visual_metrics_rows or None,
        geometric_metrics=geometric_metrics_rows or None,
        memory_metrics=memory_metrics_rows or None,
    )
    summary["progress_fusion"] = {
        "beta": float(args.progress_fusion_beta),
        "mode": str(args.progress_fusion_mode),
        "scale": float(args.progress_fusion_scale),
        "enabled": float(args.progress_fusion_beta) > 0.0,
        "available": any("progress_alignment_value" in row for row in scored),
    }
    summary["input"] = str(args.input)
    summary["checkpoint"] = str(args.checkpoint)
    if missing_report is not None:
        summary["missing_image_filter"] = {
            key: value for key, value in missing_report.items() if key != "dropped_rows"
        }
    summary["model"] = {
        "kind": model_info["kind"],
        "epoch": model_info["epoch"],
        "best_val_loss": model_info["best_val_loss"],
        "missing_key_count": len(model_info["missing_keys"]),
        "unexpected_key_count": len(model_info["unexpected_keys"]),
    }
    summary_path = out_dir / "wam_iac_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nWAM IAC Benchmark")
    print("=" * 60)
    print(f"samples={summary['num_samples']}")
    print(f"mean_consistency={summary['overall']['mean_consistency']:.4f}")
    print(f"mean_validity={summary['overall']['mean_validity']:.4f}")
    print(f"scores={scored_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()

