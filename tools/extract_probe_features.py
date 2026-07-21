#!/usr/bin/env python3
"""Extract layer-wise probe features from an IAC critic checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import ConsistencyDataset, load_config  # noqa: E402
from train_dinov2_v5_minimal import DINOv2ConsistencyCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract probe features from IAC")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args()


def _load_labels(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    traj = batch["candidate_traj_raw"]
    xy = traj[:, :, :2]
    step = xy[:, 1:, :] - xy[:, :-1, :]
    step_speed = torch.norm(step, p=2, dim=-1) / 0.5
    path_len = torch.sum(torch.norm(step, p=2, dim=-1), dim=1)
    mean_speed = step_speed.mean(dim=1)
    speed_std = step_speed.std(dim=1, unbiased=False)
    final_disp = torch.norm(xy[:, -1, :], p=2, dim=-1)
    progress_x = xy[:, -1, 0]
    lateral_abs = xy[:, -1, 1].abs()
    heading_change = traj[:, -1, 2] - traj[:, 0, 2]
    curvature = heading_change.abs() / (path_len.clamp_min(1e-6))
    turn_left = (heading_change > 0.05).float()
    turn_right = (heading_change < -0.05).float()
    straight = ((heading_change.abs() <= 0.05)).float()
    return {
        "path_len": path_len,
        "mean_speed": mean_speed,
        "speed_std": speed_std,
        "final_disp": final_disp,
        "progress_x": progress_x,
        "lateral_abs": lateral_abs,
        "heading_change": heading_change,
        "curvature": curvature,
        "turn_left": turn_left,
        "turn_right": turn_right,
        "straight": straight,
        "consistency_label": batch["consistency_label"],
        "validity_label": batch["validity_label"],
    }


def _strip_future_geometry_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)
    dinov2 = dict(cfg.get("dinov2", {}))
    cfg["dinov2"] = dinov2
    model = dict(cfg.get("model", {}))
    cfg["model"] = model
    return cfg


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_config(args.config) if args.config else checkpoint.get("config", {})
    cfg = _strip_future_geometry_keys(cfg)
    if args.split == "val":
        index_path = cfg["val_index"]
    else:
        index_path = cfg["train_index"]
    dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv2ConsistencyCritic(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    rows: List[Dict[str, Any]] = []
    total = 0
    with torch.no_grad():
        for batch in loader:
            if args.max_samples and total >= args.max_samples:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            feats = model.extract_probe_features(
                history_images=batch["history_images"],
                future_images=batch["future_images"],
                ego_state=batch["ego_state"],
                candidate_traj=batch["candidate_traj"],
            )
            if "hist_seq_mean" not in feats:
                feats["hist_seq_mean"] = feats["hist_seq"].mean(dim=1)
            if "fut_seq_mean" not in feats:
                feats["fut_seq_mean"] = feats["fut_seq"].mean(dim=1)
            if "hist_seq_last" not in feats:
                feats["hist_seq_last"] = feats["hist_seq"][:, -1, :]
            if "fut_seq_last" not in feats:
                feats["fut_seq_last"] = feats["fut_seq"][:, -1, :]
            labels = _load_labels(batch)
            bs = batch["consistency_label"].shape[0]
            for i in range(bs):
                if args.max_samples and total >= args.max_samples:
                    break
                row = {
                    "sample_id": batch["sample_id"][i] if isinstance(batch["sample_id"], list) else str(batch["sample_id"]),
                    "group_id": batch["group_id"][i] if isinstance(batch["group_id"], list) else str(batch["group_id"]),
                    "source_type": batch["source_type"][i] if isinstance(batch["source_type"], list) else str(batch["source_type"]),
                }
                for name, tensor in feats.items():
                    row[name] = tensor[i].detach().cpu().tolist()
                for name, tensor in labels.items():
                    row[name] = float(tensor[i].detach().cpu().item())
                rows.append(row)
                total += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples -> {out}")


if __name__ == "__main__":
    main()
