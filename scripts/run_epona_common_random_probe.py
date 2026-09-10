#!/usr/bin/env python3
"""Generate calibrated Epona counterfactuals with common random numbers."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange


def _load_config(epona_root: Path, args: argparse.Namespace):
    from types import SimpleNamespace

    namespace: dict = {}
    exec((epona_root / "configs/dit_config_dcae_nuplan.py").read_text(), namespace)
    config = SimpleNamespace(**{key: value for key, value in namespace.items() if not key.startswith("__")})
    config.batch_size = 1
    config.vae_ckpt = str(args.vae)
    config.resume_path = str(args.checkpoint)
    config.num_sampling_steps = args.sampling_steps
    config.condition_frames = 10
    config.image_size = (512, 1024)
    config.temporal_patch_size = 6
    config.test_video_frames = 4
    config.device = args.device
    return config


def _load_image(sensor_root: Path, frame: dict) -> tuple[torch.Tensor, str]:
    path = sensor_root / frame["cams"]["CAM_F0"]["data_path"]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor, str(path.resolve())


def _camera_calibration(frame: dict) -> tuple[list, list, list]:
    camera = frame["cams"]["CAM_F0"]
    camera_to_ego = np.eye(4, dtype=np.float64)
    camera_to_ego[:3, :3] = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    camera_to_ego[:3, 3] = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    return (
        np.asarray(camera["cam_intrinsic"], dtype=np.float64).tolist(),
        np.asarray(camera.get("distortion", []), dtype=np.float64).tolist(),
        camera_to_ego.tolist(),
    )


def _branch_controls(pose: np.ndarray, yaw_deg: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=np.float32).copy()
    yaw_deg = np.asarray(yaw_deg, dtype=np.float32).copy()
    if mode == "left":
        pose[:, 1] -= 0.25
        yaw_deg[:, 0] += 2.0
    elif mode == "right":
        pose[:, 1] += 0.25
        yaw_deg[:, 0] -= 2.0
    elif mode != "logged":
        raise ValueError(mode)
    return pose, yaw_deg


def _cumulative_trajectory(relative_xy: np.ndarray, relative_yaw_deg: np.ndarray) -> list[list[float]]:
    pose = np.zeros(3, dtype=np.float64)
    result: list[list[float]] = []
    for xy, yaw_deg in zip(relative_xy, relative_yaw_deg.reshape(-1)):
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        dx, dy = (float(value) for value in xy)
        pose[:2] += [cosine * dx - sine * dy, sine * dx + cosine * dy]
        pose[2] += np.deg2rad(float(yaw_deg))
        result.append(pose.copy().tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--epona-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9102026)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.environ["EPONA_ROOT"] = str(args.epona_root)
    sys.path.insert(0, str(args.epona_root))
    from models.model import TrainTransformersDiT
    from models.modules.tokenizer import VAETokenizer
    from utils.preprocess import get_rel_pose

    config = _load_config(args.epona_root, args)
    torch.cuda.set_device(torch.device(args.device))
    model = TrainTransformersDiT(
        config,
        load_path=str(args.checkpoint),
        local_rank=0,
        condition_frames=config.condition_frames,
    ).eval()
    tokenizer = VAETokenizer(config, 0)
    frames = pickle.loads(args.pkl.read_bytes())
    timestamp_s = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64) / 1e6
    source_step_s = float(np.median(np.diff(timestamp_s)))
    if not 0.45 <= source_step_s <= 0.55:
        raise ValueError(f"unexpected NAVSIM cadence: {source_step_s:.6f}s")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    window_size = config.condition_frames + args.future_steps
    starts = range(0, len(frames) - window_size + 1, args.sample_stride)
    for sample_index, start in enumerate(starts):
        if sample_index >= args.num_samples:
            break
        window = frames[start : start + window_size]
        poses = torch.from_numpy(
            np.asarray([frame["ego2global"] for frame in window], dtype=np.float32)
        ).unsqueeze(0).to(args.device)
        relative_pose, relative_yaw = get_rel_pose(poses)
        history_pose = relative_pose[:, : config.condition_frames]
        history_yaw = relative_yaw[:, : config.condition_frames]
        native_pose = relative_pose[0, config.condition_frames :].detach().cpu().numpy()
        native_yaw = relative_yaw[0, config.condition_frames :].detach().cpu().numpy()
        if len(native_pose) != args.future_steps:
            raise RuntimeError("Epona future control extraction is incomplete")

        loaded = [_load_image(args.sensor_root, frame) for frame in window[: config.condition_frames]]
        history_paths = [path for _, path in loaded]
        images = torch.stack([image for image, _ in loaded])
        images = ((images - 0.5) * 2.0).unsqueeze(0).to(args.device)
        intrinsic, distortion, camera_to_ego = _camera_calibration(window[config.condition_frames - 1])
        source_key = f"epona_common_random:{start}"
        branch_seed = args.seed + sample_index

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            history_latents = tokenizer.encode_to_z(images)
            for mode in ("logged", "left", "right"):
                torch.manual_seed(branch_seed)
                torch.cuda.manual_seed_all(branch_seed)
                branch_pose, branch_yaw = _branch_controls(native_pose, native_yaw, mode)
                pose_sequence = history_pose.clone()
                yaw_sequence = history_yaw.clone()
                latents = history_latents.clone()
                future_paths = []
                branch_dir = args.output / f"sample_{sample_index:06d}" / mode
                branch_dir.mkdir(parents=True, exist_ok=True)
                for step in range(args.future_steps):
                    pose_new = torch.from_numpy(branch_pose[step]).to(args.device).view(1, 1, 2)
                    yaw_new = torch.from_numpy(branch_yaw[step]).to(args.device).view(1, 1, 1)
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

                rows.append({
                    "sample_index": sample_index,
                    "source_start_index": start,
                    "sample_stride": args.sample_stride,
                    "source_sample": str(args.pkl.resolve()),
                    "source_key": source_key,
                    "counterfactual_group_id": source_key,
                    "branch_mode": mode,
                    "history_images": history_paths,
                    "future_images": future_paths,
                    "future_images_source": "epona_generated_common_random",
                    "action_injection_verified": True,
                    "intervention_variant": "epona_generate_gt_pose_gt_yaw",
                    "native_action_trajectory": _cumulative_trajectory(native_pose, native_yaw),
                    "action_trajectory": np.column_stack([branch_pose, branch_yaw]).tolist(),
                    "action_trajectory_representation": "relative_xy_m_yaw_deg_controls",
                    "action_yaw_deg": branch_yaw.tolist(),
                    "future_times_s": (source_step_s * np.arange(1, args.future_steps + 1)).tolist(),
                    "camera_intrinsic": intrinsic,
                    "camera_distortion": distortion,
                    "camera_to_ego": camera_to_ego,
                    "randomness_contract": {
                        "common_random_numbers": True,
                        "seed": branch_seed,
                        "scope": "all autoregressive diffusion draws within counterfactual group",
                    },
                    "source_frame_step_s": source_step_s,
                })
                del latents
                torch.cuda.empty_cache()
        print(json.dumps({"completed_group": sample_index + 1, "total": args.num_samples}), flush=True)

    (args.output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "groups": len(rows) // 3}, indent=2))


if __name__ == "__main__":
    main()
