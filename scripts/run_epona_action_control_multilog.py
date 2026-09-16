#!/usr/bin/env python3
"""Generate one same-history Epona intervention group from each independent log."""

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

EPONA_ROOT = os.environ.get("EPONA_ROOT", str(Path.cwd() / "third_party" / "Epona"))
sys.path.insert(0, EPONA_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_epona_action_control_native import _branch_controls, _load_cfg, _load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-root", type=Path, required=True)
    parser.add_argument("--sensor-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-logs", type=int, default=40)
    parser.add_argument("--skip-logs", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=80)
    parser.add_argument("--future-steps", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9104000)
    parser.add_argument("--speed-spec", default="stop=0,normal=1.0")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    speed_scales = {}
    for item in args.speed_spec.split(","):
        name, value = item.split("=", 1)
        speed_scales[name.strip()] = float(value)
    modes = list(speed_scales)

    from models.model import TrainTransformersDiT
    from models.modules.tokenizer import VAETokenizer
    from utils.preprocess import get_rel_pose

    cfg = _load_cfg(args)
    torch.cuda.set_device(torch.device(args.device))
    model = TrainTransformersDiT(cfg, load_path=args.checkpoint, local_rank=0, condition_frames=cfg.condition_frames).eval()
    tokenizer = VAETokenizer(cfg, 0)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    group_index = 0

    for source in sorted(args.pkl_root.glob("*.pkl"))[args.skip_logs:]:
        if group_index >= args.max_logs:
            break
        frames = pickle.load(open(source, "rb"))
        start = int(args.start_index)
        if len(frames) < start + 15:
            continue
        window = frames[start : start + 15]
        history_frames = window[: cfg.condition_frames]
        pose_mats = torch.from_numpy(np.asarray([np.asarray(frame["ego2global"], dtype=np.float32) for frame in window[:11]])).unsqueeze(0).to(args.device)
        rel_pose, rel_yaw = get_rel_pose(pose_mats)
        native_pose = rel_pose[0].detach().cpu().numpy()
        native_yaw = rel_yaw[0].detach().cpu().numpy()
        history_pose = rel_pose[:, : cfg.condition_frames]
        history_yaw = rel_yaw[:, : cfg.condition_frames]
        images = torch.stack([_load_image(args.sensor_root, frame) for frame in history_frames])
        images = ((images - 0.5) * 2.0).unsqueeze(0).to(args.device)
        source_key = f"epona_log:{source.stem}"
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            history_latents = tokenizer.encode_to_z(images)
            for mode in modes:
                torch.manual_seed(args.seed + group_index)
                branch_dir = args.output / f"sample_{group_index:06d}" / mode
                branch_dir.mkdir(parents=True, exist_ok=True)
                pose_controls, yaw_controls = _branch_controls(
                    native_pose[cfg.condition_frames : cfg.condition_frames + args.future_steps],
                    native_yaw[cfg.condition_frames : cfg.condition_frames + args.future_steps],
                    mode,
                    args.future_steps,
                    speed_scales=speed_scales,
                )
                pose_seq, yaw_seq, latents = history_pose.clone(), history_yaw.clone(), history_latents.clone()
                future_paths = []
                for step in range(args.future_steps):
                    pose_seq = torch.cat([pose_seq, torch.from_numpy(pose_controls[step]).to(args.device).view(1, 1, 2)], dim=1)
                    yaw_seq = torch.cat([yaw_seq, torch.from_numpy(yaw_controls[step]).to(args.device).view(1, 1, 1)], dim=1)
                    pred_hw = model.generate_gt_pose_gt_yaw(latents, pose_seq[:, -cfg.condition_frames - 1 :], yaw_seq[:, -cfg.condition_frames - 1 :])
                    if pred_hw.ndim == 5:
                        pred_hw = pred_hw[:, 0]
                    pred_tok = rearrange(pred_hw, "b h w c -> b 1 (h w) c")
                    latents = torch.cat([latents[:, 1:], pred_tok], dim=1)
                    image = tokenizer.z_to_image(pred_hw).float().cpu()
                    if image.ndim == 4:
                        image = image[0]
                    array = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    path = branch_dir / f"future_{step + 1:02d}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                    future_paths.append(str(path))
                rows.append({
                    "sample_index": group_index,
                    "source_start_index": start,
                    "source_sample": str(source),
                    "source_key": source_key,
                    "counterfactual_group_id": source_key,
                    "nuisance_seed": args.seed + group_index,
                    "branch_mode": mode,
                    "future_images": future_paths,
                    "future_images_source": "epona_generated",
                    "action_injection_verified": True,
                    "intervention_variant": "epona_generate_gt_pose_gt_yaw",
                    "native_action_trajectory": native_pose[cfg.condition_frames : cfg.condition_frames + args.future_steps].tolist(),
                    "action_trajectory": pose_controls.tolist(),
                    "action_yaw_deg": yaw_controls.tolist(),
                    "future_times_s": [0.5 * (index + 1) for index in range(args.future_steps)],
                })
        group_index += 1
        print(json.dumps({"completed_logs": group_index, "source": str(source)}), flush=True)

    (args.output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "groups": group_index}, indent=2))


if __name__ == "__main__":
    main()
