#!/usr/bin/env python3
"""Extract candidate-blind whole-clip yaw evidence from frozen backends.

The output deliberately uses the decoder-compatible envelope consumed by the
shared confidence audit.  The embedded one-point trajectory is only a carrier
for a dimensionless yaw proxy; it is not a reconstructed ego trajectory.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_manifests(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        paths = sorted(glob.glob(pattern)) or [pattern]
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def clip_paths(row: dict[str, Any]) -> list[str]:
    history = row.get("history_frame_paths") or row.get("history_images") or []
    future = row.get("future_frame_paths") or row.get("future_images") or []
    times = row.get("future_times_s") or []
    if not history or len(future) < 4:
        return []
    if len(times) == len(future) and len(future) > 4:
        indices = [int(np.argmin(np.abs(np.asarray(times, dtype=float) - target))) for target in (1.0, 2.0, 3.0, 4.0)]
        selected = [future[index] for index in indices]
    else:
        selected = future[:4]
    return [history[-1], *selected]


def load_clip(paths: list[str], row: dict[str, Any], width: int, height: int) -> np.ndarray:
    frames = []
    intrinsics = np.asarray(row.get("intrinsics") or row.get("camera_intrinsic") or [], dtype=np.float64)
    distortion = np.asarray(row.get("distortion") or row.get("camera_distortion") or [], dtype=np.float64)
    declared_size = row.get("intrinsics_source_size")
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        if intrinsics.shape == (3, 3) and distortion.size:
            source_height, source_width = image.shape[:2]
            calibration_width, calibration_height = (
                (float(declared_size[0]), float(declared_size[1]))
                if declared_size is not None else (float(source_width), float(source_height))
            )
            camera = intrinsics.copy()
            camera[0, :] *= source_width / calibration_width
            camera[1, :] *= source_height / calibration_height
            image = cv2.undistort(image, camera, distortion, None, camera)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return np.asarray(frames)


def roi_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray([
        [0.08 * width, 0.98 * height], [0.92 * width, 0.98 * height],
        [0.63 * width, 0.53 * height], [0.37 * width, 0.53 * height],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * np.sum(weights), side="left")])


def spatial_indices(valid: np.ndarray, weights: np.ndarray, max_points: int = 3000) -> np.ndarray:
    height, width = valid.shape
    yy, xx = np.indices(valid.shape)
    cell = np.minimum(yy * 3 // height, 2) * 6 + np.minimum(xx * 6 // width, 5)
    flat_valid = valid.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_cell = cell.reshape(-1)
    eligible = np.unique(flat_cell[flat_valid])
    if len(eligible) < 6:
        return np.empty(0, dtype=np.int64)
    quota, remainder = divmod(max_points, len(eligible))
    chosen = []
    for position, cell_id in enumerate(eligible):
        candidates = np.flatnonzero(flat_valid & (flat_cell == cell_id))
        order = np.lexsort((candidates, -flat_weights[candidates]))
        chosen.append(candidates[order[: quota + (1 if position < remainder else 0)]])
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)


def flow_scalar(flows: np.ndarray, validity: np.ndarray, uncertainty: np.ndarray | None, roi: np.ndarray) -> tuple[float | None, float, list[dict]]:
    values, details = [], []
    for index, flow in enumerate(flows):
        magnitude = np.linalg.norm(flow, axis=-1)
        # Direction needs non-trivial displacement, but the low-motion pixels
        # are precisely the evidence needed by the independent stop channel.
        # Measure motion before applying the direction-only 0.5 px floor.
        motion_valid = roi & validity[index] & np.isfinite(flow).all(axis=-1)
        motion_values = magnitude[motion_valid]
        motion_summary = {
            "motion_points": int(len(motion_values)),
            "motion_support_fraction": float(len(motion_values) / max(int(roi.sum()), 1)),
            "flow_magnitude_q10_px": float(np.quantile(motion_values, 0.10)) if len(motion_values) else None,
            "flow_magnitude_q25_px": float(np.quantile(motion_values, 0.25)) if len(motion_values) else None,
            "flow_magnitude_median_px": float(np.median(motion_values)) if len(motion_values) else None,
            "flow_magnitude_q75_px": float(np.quantile(motion_values, 0.75)) if len(motion_values) else None,
        }
        valid = motion_valid & (magnitude >= 0.5)
        pixel_weights = np.ones(valid.shape, dtype=np.float64) if uncertainty is None else np.exp(-np.clip(uncertainty[index], -10.0, 10.0))
        indices = spatial_indices(valid, pixel_weights)
        count = int(valid.sum())
        if len(indices) < 100:
            details.append({"interval": index, "available": False, "points": count, "sampled_points": int(len(indices)), **motion_summary})
            continue
        horizontal = flow[..., 0].reshape(-1)[indices]
        if uncertainty is None:
            value = float(np.median(horizontal))
            effective = float(len(indices))
        else:
            weights = pixel_weights.reshape(-1)[indices]
            value = weighted_median(horizontal, weights)
            effective = float(np.sum(weights) ** 2 / max(np.sum(weights * weights), 1e-12))
        values.append(value / flow.shape[1])
        details.append({"interval": index, "available": True, "points": count, "sampled_points": int(len(indices)), "effective_points": effective, **motion_summary})
    return (float(np.median(values)) if len(values) == 4 else None, len(values) / 4.0, details)


class RaftBackend:
    def __init__(self, device: str) -> None:
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.torch = torch
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights, progress=False).to(device).eval()
        self.device = device

    def __call__(self, frames: np.ndarray, roi: np.ndarray) -> tuple[float | None, float, list[dict]]:
        torch = self.torch
        first = torch.from_numpy(frames[:-1]).permute(0, 3, 1, 2).float().to(self.device).div_(127.5).sub_(1.0)
        second = torch.from_numpy(frames[1:]).permute(0, 3, 1, 2).float().to(self.device).div_(127.5).sub_(1.0)
        with torch.inference_mode():
            predictions = self.model(first, second, num_flow_updates=32)
            flows = predictions[-1].float().permute(0, 2, 3, 1).cpu().numpy()
            tail = torch.stack(predictions[-8:], dim=0)
            refinement = torch.sqrt(torch.mean(torch.sum((tail - tail[-1:]) ** 2, dim=2), dim=0))
            refinement = (refinement / (1.0 + torch.linalg.vector_norm(predictions[-1], dim=1))).cpu().numpy()
        valid = np.ones(flows.shape[:-1], dtype=bool)
        refinement_medians = []
        for index in range(len(valid)):
            median = float(np.median(refinement[index][roi]))
            refinement_medians.append(median)
            if median > 0.2653818726539612:
                valid[index] = False
        value, fraction, details = flow_scalar(flows, valid, None, roi)
        for detail, median in zip(details, refinement_medians):
            detail["refinement_uncertainty_median"] = median
        return value, fraction, details


class U2FlowBackend:
    def __init__(self, root: str, checkpoint: str, device: str) -> None:
        import torch
        sys.path[:0] = [root, str(Path(root) / "core")]
        from configs.KITTI_MV import get_cfg
        from core.Networks import build_network
        cfg = get_cfg()
        model = build_network(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self.torch = torch
        self.device = device

    def __call__(self, frames: np.ndarray, roi: np.ndarray) -> tuple[float | None, float, list[dict]]:
        torch = self.torch
        first = torch.from_numpy(frames[:-1]).permute(0, 3, 1, 2).float().div_(255.0)
        second = torch.from_numpy(frames[1:]).permute(0, 3, 1, 2).float().div_(255.0)
        pairs = torch.stack([first, second], dim=1).to(self.device)
        with torch.inference_mode():
            output = self.model(pairs, only_fw=True)
            flows = output["flows_f12"][-1].float().permute(0, 2, 3, 1).cpu().numpy()
            uncertainty = output["uncertainty_f12"][-1].float().squeeze(1).cpu().numpy()
        valid = np.ones(flows.shape[:-1], dtype=bool)
        return flow_scalar(flows, valid, uncertainty, roi)


class CoTrackerBackend:
    def __init__(self, root: str, checkpoint: str, device: str, grid_size: int, readout: str) -> None:
        import torch
        sys.path.insert(0, root)
        from cotracker.predictor import CoTrackerPredictor
        self.model = CoTrackerPredictor(checkpoint=checkpoint, offline=True).to(device).eval()
        self.torch = torch
        self.device = device
        self.grid_size = grid_size
        self.readout = readout

    def __call__(self, frames: np.ndarray, roi: np.ndarray) -> tuple[float | None, float, list[dict]]:
        torch = self.torch
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)
        mask = torch.from_numpy(roi.astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            tracks, visibility = self.model(video, grid_size=self.grid_size, grid_query_frame=0, segm_mask=mask)
        tracks = tracks[0].float().cpu().numpy()
        visibility = visibility[0].cpu().numpy().astype(bool)
        rotations, motions, details = [], [], []
        for index in range(4):
            keep = visibility[index] & visibility[index + 1]
            source, target = tracks[index, keep], tracks[index + 1, keep]
            if len(source) < 12:
                details.append({"interval": index, "available": False, "tracks": int(len(source))})
                continue
            matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.99)
            if matrix is None or inliers is None or int(inliers.sum()) < 12:
                details.append({"interval": index, "available": False, "tracks": int(len(source))})
                continue
            angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
            inlier_mask = inliers[:, 0].astype(bool)
            motion_x = float(np.median((target[inlier_mask] - source[inlier_mask])[:, 0]) / frames.shape[2])
            rotations.append(angle)
            motions.append(motion_x)
            details.append({"interval": index, "available": True, "tracks": int(len(source)), "inliers": int(inliers.sum()), "rotation": angle, "motion_x": motion_x})
        selected = rotations if self.readout == "rotation" else motions
        return (float(np.median(selected)) if len(selected) == 4 else None, len(selected) / 4.0, details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--source-filter", nargs="+", help="optional JSONL files whose source/sample IDs define the evaluated source set")
    parser.add_argument("--backend", choices=("raft", "u2flow", "cotracker"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--u2flow-root", default="/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/U2FLOW")
    parser.add_argument("--u2flow-checkpoint", default="/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/U2FLOW/checkpoints/kitti.pth")
    parser.add_argument("--cotracker-root", default="/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/co-tracker")
    parser.add_argument("--cotracker-checkpoint", default="/mnt/slurmfs-4090node3/user_data/zchen897/wam_repro/co-tracker/checkpoints/scaled_offline.pth")
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--cotracker-readout", choices=("rotation", "motion_x"), default="rotation")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    rows = read_manifests(args.manifest)
    if args.source_filter:
        allowed = set()
        for item in read_manifests(args.source_filter):
            source = item.get("source_key")
            if source is None:
                source = str(item.get("sample_id", "")).rsplit("::", 1)[0]
            allowed.add(str(source))
        rows = [row for row in rows if str(row.get("source_key")) in allowed]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.backend == "raft":
        backend = RaftBackend(args.device)
    elif args.backend == "u2flow":
        backend = U2FlowBackend(args.u2flow_root, args.u2flow_checkpoint, args.device)
    else:
        backend = CoTrackerBackend(args.cotracker_root, args.cotracker_checkpoint, args.device, args.grid_size, args.cotracker_readout)
    roi = roi_mask(args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            paths = clip_paths(row)
            # Joined WAM manifests may retain the source dataset's
            # ``record_type=native_dataset_pair``.  Future provenance and the
            # explicit branch role, not that inherited field, determine
            # whether the frames are logged-real.
            is_logged_real = (
                str(row.get("branch_role", "")) == "real"
                or str(row.get("future_images_source", "")).startswith("navsim_native_realized")
            )
            sample_id = (
                f"{row.get('source_key')}::real"
                if is_logged_real
                else str(row.get("sample_id") or f"{row.get('source_key')}::{row.get('branch_role', 'unknown')}")
            )
            try:
                frames = load_clip(paths, row, args.width, args.height)
                value, interval_fraction, details = backend(frames, roi)
                status = "explained" if value is not None else "unavailable"
                error = None
            except Exception as exc:  # Preserve a per-record fail-closed audit trail.
                value, interval_fraction, details = None, 0.0, []
                status, error = "unavailable", f"{type(exc).__name__}:{exc}"
            output = {
                "sample_id": sample_id,
                "source_key": str(row.get("source_key")),
                "backend": args.backend,
                "candidate_bank_used_by_measurement": False,
                "metric_reconstruction_used": False,
                "motion_explanation_status": status,
                "decoder": {
                    "trajectory": None if value is None else [[0.0, 0.0, value]],
                    "motion_explanation_status": status,
                },
                "available_interval_fraction": interval_fraction,
                "intervals": details,
                "error": error,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
            completed += 1
            if completed % 10 == 0:
                print(json.dumps({"completed": completed, "total": len(rows), "backend": args.backend}), flush=True)


if __name__ == "__main__":
    main()
