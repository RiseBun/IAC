#!/usr/bin/env python3
"""Candidate-blind Metric3Dv2 + fixed-rotation longitudinal probe.

The motion backend is intentionally fixed: Reloc3r supplies relative rotation
and a known-rotation translation solver estimates only camera translation.
Metric3Dv2 is used for metric 3-D points. SegFormer is an optional auxiliary
correspondence filter and never supplies motion, scale, or a required gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_metric3d(root: Path, checkpoint: Path, device: str) -> tuple[Any, Any, Any]:
    import torch
    from mmengine import Config

    sys.path.insert(0, str(root))
    from mono.model.monodepth_model import get_configured_monodepth_model

    config = Config.fromfile(str(root / "mono" / "configs" / "HourglassDecoder" / "vit.raft5.small.py"))
    model = get_configured_monodepth_model(config)
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model.to(device).eval(), torch, config


def _metric_depth(image_bgr: np.ndarray, K_original: np.ndarray, model: Any, torch: Any, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    input_size = (616, 1064)  # official ViT preprocessing size: H, W
    rgb_origin = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb_origin.shape[:2]
    scale = min(input_size[0] / height, input_size[1] / width)
    resized = cv2.resize(rgb_origin, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_LINEAR)
    K_resized = np.asarray(K_original, dtype=np.float64).copy()
    K_resized[0, :] *= scale
    K_resized[1, :] *= scale
    pad_h = input_size[0] - resized.shape[0]
    pad_w = input_size[1] - resized.shape[1]
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[123.675, 116.28, 103.53],
    )
    mean = torch.tensor([123.675, 116.28, 103.53], device=device).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375], device=device).float()[:, None, None]
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).to(device).float()
    tensor = ((tensor - mean) / std)[None]
    with torch.inference_mode():
        predicted, confidence, output = model.inference({"input": tensor})
    depth = predicted.squeeze().float()
    depth = depth[pad_top : depth.shape[0] - pad_bottom, pad_left : depth.shape[1] - pad_right]
    depth = torch.nn.functional.interpolate(depth[None, None], size=(height, width), mode="bilinear", align_corners=False).squeeze()
    # Metric3D predicts depth in a canonical camera; official de-canonical
    # transform multiplies by resized focal length / canonical focal length.
    depth = depth * float(K_resized[0, 0]) / 1000.0
    depth_np = depth.cpu().numpy().astype(np.float32)
    confidence_np = confidence.squeeze().float().cpu().numpy() if confidence is not None else None
    return depth_np, {
        "model": "Metric3Dv2-v2-S",
        "input_size": list(input_size),
        "resize_scale": float(scale),
        "canonical_focal_px": 1000.0,
        "metric_focal_px": float(K_resized[0, 0]),
        "depth_min_m": float(np.nanpercentile(depth_np, 1)),
        "depth_median_m": float(np.nanmedian(depth_np)),
        "depth_max_m": float(np.nanpercentile(depth_np, 99)),
        "confidence_median": float(np.nanmedian(confidence_np)) if confidence_np is not None else None,
    }


def _static_masks(paths: list[str], record: dict[str, Any], images: list[np.ndarray], model_id: str, device: str) -> tuple[list[np.ndarray], dict[str, Any]]:
    from iac_new.perception import SegFormerPerception
    import cv2

    perception = SegFormerPerception(
        {
            "model_id": model_id,
            "local_files_only": True,
            "confidence_threshold": 0.55,
            "traversable_labels": ["road"],
            "actor_labels": ["car", "truck", "bus", "person", "rider", "bicycle", "motorcycle"],
        },
        device=device,
    )
    observation = perception.observe(
        paths,
        target_size=(images[0].shape[1], images[0].shape[0]),
        intrinsics=np.asarray(
            record.get(
                "intrinsics",
                [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]],
            ),
            dtype=np.float64,
        ),
        distortion=np.asarray(record.get("distortion") or [], dtype=np.float64),
    )
    masks = [
        cv2.erode((np.asarray(road, dtype=bool) & ~np.asarray(actor, dtype=bool)).astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        for road, actor in zip(observation.traversable_masks, observation.actor_masks)
    ]
    return masks, {
        "backend": observation.backend,
        "model_id": observation.model_id,
        "road_fraction_mean": float(np.mean(observation.traversable_masks)),
        "road_non_actor_fraction_mean": float(np.mean(masks)),
        "road_non_actor_nonempty_fraction": float(np.mean([mask.any() for mask in masks])),
    }


def _translation_from_matches(points_2d_current: np.ndarray, points_2d_next: np.ndarray, depth: np.ndarray, K: np.ndarray, rotation: np.ndarray, camera_to_ego: np.ndarray, solver: Any) -> dict[str, Any]:
    xy = np.rint(points_2d_current).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < depth.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < depth.shape[0])
    z = np.full(len(xy), np.nan, dtype=np.float64)
    z[inside] = depth[xy[inside, 1], xy[inside, 0]]
    valid = inside & np.isfinite(z) & (z > 0.5) & (z < 150.0)
    if int(valid.sum()) < 6:
        return {"available": False, "reason": "insufficient_metric_depth_points", "num_points": int(valid.sum())}
    pixels = np.c_[points_2d_current[valid], np.ones(int(valid.sum()))].T
    points_3d = (np.linalg.inv(K) @ pixels) * z[valid].reshape(1, -1)
    candidates = []
    for convention, candidate_rotation in (("reloc", rotation), ("reloc_transpose", rotation.T)):
        result = solver(
            points_3d.T,
            points_2d_next[valid],
            K,
            candidate_rotation,
            camera_to_ego=camera_to_ego,
            reprojection_error_px=3.0,
            max_trials=400,
        )
        result["rotation_convention"] = convention
        candidates.append(result)
    chosen = dict(max(candidates, key=lambda item: (int(item.get("num_inliers", 0)), -float(item.get("median_reprojection_error_px") or 1e9))))
    chosen["num_depth_valid"] = int(valid.sum())
    chosen["candidates"] = candidates
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric3d-root", type=Path, required=True)
    parser.add_argument("--metric3d-checkpoint", type=Path, required=True)
    parser.add_argument("--reloc3r-root", type=Path, required=True)
    parser.add_argument(
        "--reloc3r-checkpoint",
        type=Path,
        help="Local Hugging Face model directory containing config.json and model.safetensors.",
    )
    parser.add_argument("--segformer-model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--unique-source", action="store_true")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse rows already present in --output and process only unseen sample_id values.",
    )
    args = parser.parse_args()

    import cv2
    sys.path.insert(0, str(args.reloc3r_root))
    from reloc3r.reloc3r_relpose import Reloc3rRelpose, inference_relpose, setup_reloc3r_relpose_model
    from reloc3r.utils.device import to_numpy
    from reloc3r.utils.image import check_images_shape_format, load_images
    from iac_new.frame_matching import load_manifest, scaled_intrinsics, sift_matches
    from iac_new.longitudinal_probe import estimate_translation_with_known_rotation

    records = load_manifest(args.manifest)
    if args.unique_source:
        unique, seen = [], set()
        for record in records:
            key = str(record.get("source_key") or record.get("sample_id") or "")
            if key not in seen:
                unique.append(record)
                seen.add(key)
        records = unique
    if args.limit > 0:
        records = records[: args.limit]
    metric3d, torch, _ = _load_metric3d(args.metric3d_root, args.metric3d_checkpoint, args.device)
    if args.reloc3r_checkpoint is not None:
        reloc = Reloc3rRelpose.from_pretrained(str(args.reloc3r_checkpoint)).to(args.device).eval()
        print(f"Reloc3r loaded from local registry: {args.reloc3r_checkpoint}", flush=True)
    else:
        reloc = setup_reloc3r_relpose_model(str(args.resolution), args.device)
    rows: list[dict[str, Any]] = []
    completed_sample_ids: set[str] = set()
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        rows = list(previous.get("rows") or [])
        completed_sample_ids = {
            str(row.get("sample_id")) for row in rows if row.get("sample_id") is not None
        }
        print(f"Resuming with {len(rows)} existing rows", flush=True)
    for record_index, record in enumerate(records, 1):
        if str(record.get("sample_id")) in completed_sample_ids:
            continue
        history_paths = list(record.get("history_frame_paths") or [])
        future_paths = list(record.get("future_frame_paths") or [])
        paths = [history_paths[-1], *future_paths] if history_paths else []
        row: dict[str, Any] = {"sample_id": record.get("sample_id"), "source_key": record.get("source_key"), "intervals": []}
        try:
            images_reloc = check_images_shape_format(load_images(paths, size=int(args.resolution), verbose=False), args.device)
            images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in paths]
            if any(image is None for image in images):
                raise FileNotFoundError("future image missing")
            masks = None
            if args.segformer_model is not None:
                try:
                    masks, seg_diag = _static_masks(
                        paths, record, images, str(args.segformer_model), args.device
                    )
                except Exception as exc:
                    seg_diag = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            else:
                seg_diag = {"status": "not_requested"}
            depths, depth_diag = [], []
            for image in images[:-1]:
                depth, diag = _metric_depth(image, scaled_intrinsics(record, image), metric3d, torch, args.device)
                depths.append(depth)
                depth_diag.append(diag)
            for index in range(len(paths) - 1):
                forward = to_numpy(inference_relpose([images_reloc[index], images_reloc[index + 1]], reloc, args.device, use_amp=True)[0])
                rotation = np.asarray(forward[:3, :3], dtype=np.float64)
                K = scaled_intrinsics(record, images[index])
                estimates: dict[str, Any] = {}
                for region in ("all", "segformer"):
                    if region == "segformer" and masks is None:
                        estimates[region] = {
                            "match": {"available": False, "reason": "segmentation_unavailable", "selected_matches": 0},
                            "motion": {"available": False, "reason": "segmentation_unavailable"},
                        }
                        continue
                    source, target, match_diag = sift_matches(
                        images[index],
                        images[index + 1],
                        args.max_width,
                        static_masks=(masks[index], masks[index + 1]) if masks is not None and region == "segformer" else None,
                    )
                    motion = _translation_from_matches(
                        source, target, depths[index], K, rotation,
                        np.asarray(record["camera_to_ego"], dtype=np.float64),
                        estimate_translation_with_known_rotation,
                    )
                    estimates[region] = {"match": match_diag, "motion": motion}
                    if region == "segformer":
                        seg_source, seg_target = source, target
                row["intervals"].append({
                    "index": index,
                    "reloc3r_rotation": rotation.tolist(),
                    "estimators": estimates,
                    "depth": depth_diag[index],
                })
            row["segmentation"] = seg_diag
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"protocol": "iac-metric3d-longitudinal-probe-v1", "incomplete": True, "rows": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"[{record_index}/{len(records)}] {record.get('sample_id')} {row['status']}", flush=True)
    args.output.write_text(json.dumps({
        "protocol": "iac-metric3d-longitudinal-probe-v1",
        "backend": {"depth": "Metric3Dv2-v2-S", "motion": "Reloc3r-512 + known-rotation translation", "road_support": "optional SegFormer road/non-actor mask", "trained_on_iac": False},
        "candidate_blind": True,
        "formal_metric_eligible": True,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
