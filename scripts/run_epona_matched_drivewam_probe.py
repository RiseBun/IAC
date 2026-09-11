#!/usr/bin/env python3
"""Run Epona on the same NAVSIM sources and action trajectories as DriveWAM."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from einops import rearrange

from scripts.run_epona_common_random_probe import (
    _camera_calibration,
    _load_config,
    _load_image,
)


def _trajectory_matrix(value: Any) -> np.ndarray:
    trajectory = np.asarray(value, dtype=np.float32).squeeze()
    if trajectory.ndim != 2:
        raise ValueError(f"DriveWAM trajectory must be rank 2 after squeeze, got {trajectory.shape}")
    if trajectory.shape[0] == 3:
        trajectory = trajectory.T
    if trajectory.shape[1] != 3:
        raise ValueError(f"DriveWAM trajectory must have x/y/yaw columns, got {trajectory.shape}")
    if len(trajectory) != 8 or not np.isfinite(trajectory).all():
        raise ValueError("matched probe requires eight finite 0.5 s action poses")
    return trajectory


def _cumulative_to_epona_controls(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert left-positive cumulative SE(2) poses to Epona relative controls."""
    previous = np.zeros(3, dtype=np.float64)
    xy_controls: list[list[float]] = []
    yaw_controls: list[list[float]] = []
    for current in np.asarray(trajectory, dtype=np.float64):
        delta = current[:2] - previous[:2]
        cosine, sine = np.cos(previous[2]), np.sin(previous[2])
        local_x = cosine * delta[0] + sine * delta[1]
        local_y_left = -sine * delta[0] + cosine * delta[1]
        delta_yaw = np.arctan2(
            np.sin(current[2] - previous[2]),
            np.cos(current[2] - previous[2]),
        )
        # Epona's translation y is right-positive; yaw remains left-positive.
        xy_controls.append([float(local_x), float(-local_y_left)])
        yaw_controls.append([float(np.rad2deg(delta_yaw))])
        previous = current.copy()
    return np.asarray(xy_controls, dtype=np.float32), np.asarray(yaw_controls, dtype=np.float32)


def _drivewam_actions(root: Path) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}
    manifest_paths = sorted(root.glob("shard_*/manifest.json"))
    if not manifest_paths:
        manifest_paths = sorted(root.glob("shards/shard_*/manifest.json"))
    if not manifest_paths:
        raise ValueError(f"no DriveWAM shard manifests found under {root}")
    for manifest_path in manifest_paths:
        for row in json.loads(manifest_path.read_text(encoding="utf-8")):
            source_sample = Path(row["source_sample"])
            with source_sample.open("rb") as handle:
                source_key = str(pickle.load(handle)["metadata"]["source_key"])
            if source_key in actions:
                raise ValueError(f"duplicate DriveWAM source key: {source_key}")
            actions[source_key] = _trajectory_matrix(row["predicted_action_trajectory"])
    return actions


def _matched_sources(
    input_root: Path,
    left_actions: dict[str, np.ndarray],
    right_actions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    log_cache: dict[str, list[dict[str, Any]]] = {}
    timestamp_cache: dict[str, dict[int, int]] = {}
    matched: list[dict[str, Any]] = []
    sample_paths = sorted(input_root.glob("shard_*/sample_*.pkl"))
    if not sample_paths:
        sample_paths = sorted(input_root.glob("shards/shard_*/sample_*.pkl"))
    if not sample_paths:
        raise ValueError(f"no temporal-fixed source samples found under {input_root}")
    for sample_path in sample_paths:
        with sample_path.open("rb") as handle:
            sample = pickle.load(handle)
        metadata = sample["metadata"]
        source_key = str(metadata["source_key"])
        if source_key not in left_actions or source_key not in right_actions:
            continue
        log_path = str(metadata["source_pkl"])
        if log_path not in log_cache:
            with open(log_path, "rb") as handle:
                frames = pickle.load(handle)
            log_cache[log_path] = frames
            timestamp_cache[log_path] = {
                int(frame["timestamp"]): index for index, frame in enumerate(frames)
            }
        frames = log_cache[log_path]
        anchor_index = timestamp_cache[log_path].get(int(metadata["timestamp_us"]))
        if anchor_index is None:
            raise ValueError(f"source timestamp is absent from NAVSIM log: {source_key}")
        begin, end = anchor_index - 9, anchor_index + 8
        if begin < 0 or end >= len(frames):
            continue
        window = frames[begin : end + 1]
        if any(str(frame["scene_token"]) != str(metadata["scene_token"]) for frame in window):
            continue
        matched.append(
            {
                "source_key": source_key,
                "sample_path": str(sample_path),
                "metadata": metadata,
                "window": window,
                "left_action": left_actions[source_key],
                "right_action": right_actions[source_key],
            }
        )
    return matched


def _flow_row(
    item: dict[str, Any],
    branch: str,
    history_paths: list[str],
    future_paths: list[str],
    action: np.ndarray,
    seed: int,
    action_trajectory_source: str = "matched_drivewam_navsim_action_head",
    future_images_source: str = "epona_generated_matched_drivewam_action",
    protocol: str = "same_source_same_physical_action_v1",
) -> dict[str, Any]:
    metadata = item["metadata"]
    anchor = item["window"][9]
    intrinsic, distortion, camera_to_ego = _camera_calibration(anchor)
    selected = [1, 3, 5, 7]
    return {
        "sample_id": f"{item['source_key']}::{branch}",
        "source_key": item["source_key"],
        "counterfactual_group_id": item["source_key"],
        "branch_role": branch,
        "scene_id": str(metadata["scene_token"]),
        "history_frame_paths": history_paths[-4:],
        "future_frame_paths": [future_paths[index] for index in selected],
        "history_times_s": [-1.5, -1.0, -0.5, 0.0],
        "future_times_s": [1.0, 2.0, 3.0, 4.0],
        "intrinsics": intrinsic,
        "distortion": distortion,
        "camera_to_ego": camera_to_ego,
        "intrinsics_source_size": [1920, 1080],
        "image_geometry_adapter": {
            "schema": "iac-image-geometry-v1",
            "adapter_id": "epona_nuplan_direct_resize_v1",
            "calibration_source_size": [1920, 1080],
            "frame_groups": [
                {
                    "indices": [0, 1, 2, 3],
                    "operation": "identity",
                    "frame_size": [1920, 1080],
                },
                {
                    "indices": [4, 5, 6, 7],
                    "operation": "direct_resize",
                    "frame_size": [1024, 512],
                },
            ],
            "provenance": {
                "producer": "run_epona_matched_drivewam_probe.py",
                "operation": "cv2.resize(source, (1024, 512), INTER_AREA)",
                "candidate_dependent": False,
            },
        },
        "action_trajectory": action[selected].tolist(),
        "action_trajectory_source": action_trajectory_source,
        "future_images_source": future_images_source,
        "wam_model_id": "epona_nuplan",
        "candidate_bank_used_by_decoder": False,
        "command_override": branch,
        "metadata": {
            "stratum": metadata["stratum"],
            "protocol": protocol,
            "source_sample": item["sample_path"],
            "epona_history_frames": 10,
            "randomness_contract": {
                "common_random_numbers": True,
                "seed": seed,
                "scope": "all autoregressive diffusion draws within counterfactual group",
            },
        },
        "candidates": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--drivewam-left-root", type=Path, required=True)
    parser.add_argument("--drivewam-right-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--epona-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9102026)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-trajectory-source", default="matched_drivewam_navsim_action_head")
    parser.add_argument("--future-images-source", default="epona_generated_matched_drivewam_action")
    parser.add_argument("--protocol", default="same_source_same_physical_action_v1")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    os.environ["EPONA_ROOT"] = str(args.epona_root)
    sys.path.insert(0, str(args.epona_root))
    from models.model import TrainTransformersDiT
    from models.modules.tokenizer import VAETokenizer
    from utils.preprocess import get_rel_pose

    left_actions = _drivewam_actions(args.drivewam_left_root)
    right_actions = _drivewam_actions(args.drivewam_right_root)
    all_items = _matched_sources(args.input_root, left_actions, right_actions)
    if not all_items:
        raise ValueError("no same-source DriveWAM/Epona records satisfy the history contract")
    for eligible_index, item in enumerate(all_items):
        item["eligible_index"] = eligible_index
    items = [
        item for index, item in enumerate(all_items)
        if index % args.num_shards == args.shard_index
    ]
    if args.num_samples is not None:
        items = items[: args.num_samples]

    config = _load_config(args.epona_root, args)
    torch.cuda.set_device(torch.device(args.device))
    model = TrainTransformersDiT(
        config,
        load_path=str(args.checkpoint),
        local_rank=0,
        condition_frames=config.condition_frames,
    ).eval()
    tokenizer = VAETokenizer(config, 0)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []

    for local_index, item in enumerate(items):
        global_index = int(item["eligible_index"])
        branch_seed = args.seed + global_index
        window = item["window"]
        poses = torch.from_numpy(
            np.asarray([frame["ego2global"] for frame in window[:10]], dtype=np.float32)
        ).unsqueeze(0).to(args.device)
        history_pose, history_yaw = get_rel_pose(poses)
        loaded = [_load_image(args.sensor_root, frame) for frame in window[:10]]
        history_paths = [path for _, path in loaded]
        images = torch.stack([image for image, _ in loaded])
        images = ((images - 0.5) * 2.0).unsqueeze(0).to(args.device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            history_latents = tokenizer.encode_to_z(images)
            for branch in ("left", "right"):
                action = item[f"{branch}_action"]
                action_xy, action_yaw = _cumulative_to_epona_controls(action)
                torch.manual_seed(branch_seed)
                torch.cuda.manual_seed_all(branch_seed)
                pose_sequence = history_pose.clone()
                yaw_sequence = history_yaw.clone()
                latents = history_latents.clone()
                future_paths: list[str] = []
                branch_dir = args.output / f"sample_{global_index:06d}" / branch
                branch_dir.mkdir(parents=True, exist_ok=True)
                for step in range(8):
                    pose_new = torch.from_numpy(action_xy[step]).to(args.device).view(1, 1, 2)
                    yaw_new = torch.from_numpy(action_yaw[step]).to(args.device).view(1, 1, 1)
                    pose_sequence = torch.cat([pose_sequence, pose_new], dim=1)
                    yaw_sequence = torch.cat([yaw_sequence, yaw_new], dim=1)
                    predicted = model.generate_gt_pose_gt_yaw(
                        latents, pose_sequence[:, -11:], yaw_sequence[:, -11:]
                    )
                    if predicted.ndim == 5:
                        predicted = predicted[:, 0]
                    token = rearrange(predicted, "b h w c -> b 1 (h w) c")
                    latents = torch.cat([latents[:, 1:], token], dim=1)
                    image = tokenizer.z_to_image(predicted).float().cpu()[0]
                    array = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    path = branch_dir / f"future_{step + 1:02d}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                    future_paths.append(str(path.resolve()))

                raw_rows.append(
                    {
                        "source_key": item["source_key"],
                        "source_sample": item["sample_path"],
                        "branch_mode": branch,
                        "future_images": future_paths,
                        "action_trajectory": action.tolist(),
                        "action_trajectory_source": args.action_trajectory_source,
                        "future_images_source": args.future_images_source,
                        "protocol": args.protocol,
                        "random_seed": branch_seed,
                    }
                )
                flow_rows.append(
                    _flow_row(
                        item,
                        branch,
                        history_paths,
                        future_paths,
                        action,
                        branch_seed,
                        args.action_trajectory_source,
                        args.future_images_source,
                        args.protocol,
                    )
                )
                del latents
                torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "completed": local_index + 1,
                    "shard_total": len(items),
                    "global_eligible": len(all_items),
                    "source_key": item["source_key"],
                }
            ),
            flush=True,
        )

    (args.output / "manifest.json").write_text(
        json.dumps(raw_rows, indent=2), encoding="utf-8"
    )
    (args.output / "flow_manifest.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in flow_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "eligible_sources": len(all_items),
                "shard_sources": len(items),
                "rows": len(flow_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
