#!/usr/bin/env python3
"""Generate Epona futures under explicit native NAVSIM pose interventions.

Unlike a post-hoc action predictor, Epona's ``generate_gt_pose_gt_yaw`` API
receives the next relative ego pose and yaw before each video frame is sampled.
This script keeps one native history fixed and generates logged/left/right
counterfactual branches from that same context.
"""
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


def _load_cfg(args):
    from types import SimpleNamespace

    ns = {}
    exec(Path(f"{EPONA_ROOT}/configs/dit_config_dcae_nuplan.py").read_text(), ns)
    cfg = SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})
    cfg.batch_size = 1
    cfg.vae_ckpt = args.vae
    cfg.resume_path = args.checkpoint
    cfg.num_sampling_steps = args.sampling_steps
    cfg.condition_frames = 10
    cfg.image_size = (512, 1024)
    cfg.temporal_patch_size = 6
    cfg.test_video_frames = 4
    cfg.device = args.device
    return cfg


def _load_image(sensor_root, frame, size=(1024, 512)):
    path = Path(sensor_root) / frame["cams"]["CAM_F0"]["data_path"]
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def _extend(pose: np.ndarray, yaw: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    if pose.ndim == 1:
        pose = pose.reshape(1, -1)
    if yaw.ndim == 1:
        yaw = yaw.reshape(1, -1)
    if len(pose) == 0:
        pose = np.zeros((steps, 2), dtype=np.float32)
        yaw = np.zeros((steps, 1), dtype=np.float32)
        return pose, yaw
    while len(pose) < steps:
        pose = np.concatenate([pose, pose[-1:]], axis=0)
        yaw = np.concatenate([yaw, yaw[-1:]], axis=0)
    return pose[:steps], yaw[:steps]


def _branch_controls(native_rel_pose, native_rel_yaw, mode, steps: int, speed_scales=None):
    pose, yaw = _extend(
        np.asarray(native_rel_pose, dtype=np.float32).copy(),
        np.asarray(native_rel_yaw, dtype=np.float32).copy(),
        steps,
    )
    if mode == "left":
        # Epona convention: y is right-positive and yaw is left-positive.
        pose[:, 1] -= 0.25
        yaw[:, 0] += 2.0
    elif mode == "right":
        pose[:, 1] += 0.25
        yaw[:, 0] -= 2.0
    elif mode == "stop":
        pose[:, :] = 0.0
        yaw[:, :] = 0.0
    elif speed_scales is not None and mode in speed_scales:
        scale = float(speed_scales[mode])
        pose[:, 0] *= scale
    elif mode != "logged":
        raise ValueError(f"unsupported branch mode: {mode}")
    return pose, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--sensor-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--future-steps", type=int, default=4)
    ap.add_argument("--sampling-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=9103000)
    ap.add_argument(
        "--speed-spec",
        default="stop=0,slow=0.5,normal=1.0,fast=1.5",
        help="comma-separated speed branch names and longitudinal scales",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

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
    model = TrainTransformersDiT(
        cfg, load_path=args.checkpoint, local_rank=0, condition_frames=cfg.condition_frames
    ).eval()
    tokenizer = VAETokenizer(cfg, 0)
    frames = pickle.load(open(args.pkl, "rb"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    starts = range(args.start_index, len(frames) - 15, 5)
    for sample_index, start in enumerate(starts):
        if sample_index >= args.num_samples:
            break
        window = frames[start : start + 15]
        # Ten observed frames, followed by four native future controls.
        history_frames = window[: cfg.condition_frames]
        pose_mats = torch.from_numpy(
            np.asarray([np.asarray(f["ego2global"], dtype=np.float32) for f in window[: 11]])
        ).unsqueeze(0).to(args.device)
        rel_pose, rel_yaw = get_rel_pose(pose_mats)
        native_pose = rel_pose[0].detach().cpu().numpy()
        native_yaw = rel_yaw[0].detach().cpu().numpy()
        history_pose = rel_pose[:, : cfg.condition_frames]
        history_yaw = rel_yaw[:, : cfg.condition_frames]
        images = torch.stack([_load_image(args.sensor_root, f) for f in history_frames])
        images = ((images - 0.5) * 2.0).unsqueeze(0).to(args.device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            history_latents = tokenizer.encode_to_z(images)
            for mode in modes:
                # Common random numbers make the speed intervention the only
                # intended branch difference within each source group.
                torch.manual_seed(args.seed + sample_index)
                branch_dir = output / f"sample_{sample_index:06d}" / mode
                branch_dir.mkdir(parents=True, exist_ok=True)
                pose_controls, yaw_controls = _branch_controls(
                    native_pose[cfg.condition_frames : cfg.condition_frames + args.future_steps],
                    native_yaw[cfg.condition_frames : cfg.condition_frames + args.future_steps],
                    mode,
                    args.future_steps,
                    speed_scales=speed_scales,
                )
                pose_seq = history_pose.clone()
                yaw_seq = history_yaw.clone()
                latents = history_latents.clone()
                future_paths = []
                for step in range(args.future_steps):
                    pose_new = torch.from_numpy(pose_controls[step]).to(args.device).view(1, 1, 2)
                    yaw_new = torch.from_numpy(yaw_controls[step]).to(args.device).view(1, 1, 1)
                    pose_seq = torch.cat([pose_seq, pose_new], dim=1)
                    yaw_seq = torch.cat([yaw_seq, yaw_new], dim=1)
                    pred_hw = model.generate_gt_pose_gt_yaw(
                        latents,
                        pose_seq[:, -cfg.condition_frames - 1 :],
                        yaw_seq[:, -cfg.condition_frames - 1 :],
                    )
                    if pred_hw.ndim == 5:
                        pred_hw = pred_hw[:, 0]
                    pred_tok = rearrange(pred_hw, "b h w c -> b 1 (h w) c")
                    if latents.ndim != 4:
                        raise RuntimeError(f"expected latents [B,T,L,C], got {tuple(latents.shape)}")
                    latents = torch.cat([latents[:, 1:], pred_tok], dim=1)
                    image = tokenizer.z_to_image(pred_hw).float().cpu()
                    if image.ndim == 4:
                        image = image[0]
                    array = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    path = branch_dir / f"future_{step + 1:02d}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                    future_paths.append(str(path))

                rows.append({
                    "sample_index": sample_index,
                    "source_sample": str(args.pkl),
                    "source_key": f"epona_native:{sample_index}",
                    "counterfactual_group_id": f"epona_native:{sample_index}",
                    "nuisance_seed": args.seed + sample_index,
                    "branch_mode": mode,
                    "future_images": future_paths,
                    "future_images_source": "epona_generated",
                    "action_injection_verified": True,
                    "intervention_variant": "epona_generate_gt_pose_gt_yaw",
                    "native_action_trajectory": native_pose[cfg.condition_frames : cfg.condition_frames + args.future_steps].tolist(),
                    "action_trajectory": pose_controls.tolist(),
                    "action_yaw_deg": yaw_controls.tolist(),
                    "future_times_s": [0.5 * (i + 1) for i in range(args.future_steps)],
                })
                del latents
                torch.cuda.empty_cache()
        print(json.dumps(rows[-1])[:1200], flush=True)

    (output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "groups": len(rows) // len(modes)}, indent=2))


if __name__ == "__main__":
    main()
