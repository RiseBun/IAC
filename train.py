#!/usr/bin/env python3
import argparse
import datetime
import importlib.util
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NuPlan critic training")
    parser.add_argument("--config", required=True, help="Python config path")
    parser.add_argument("--work-dir", type=str, default=None, help="Override work dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override workers")
    parser.add_argument(
        "--baseline-mode",
        choices=["full", "no_image", "ego_only", "no_traj", "traj_only"],
        default=None,
        help="P0 shortcut audit baseline mode for consistency critic",
    )
    parser.add_argument("--max-train-steps", type=int, default=None, help="Debug: cap train iterations per epoch")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Debug: cap val iterations per epoch")
    parser.add_argument(
        "--preflight-samples",
        type=int,
        default=128,
        help="Validate image paths from each index before training; 0 disables.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("nuplan_critic_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "cfg"):
        raise ValueError(f"Config file must define `cfg`: {path}")
    cfg = dict(module.cfg)
    cfg["_config_path"] = str(path)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_enabled() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> Dict[str, int]:
    if not is_dist_enabled():
        return {"rank": 0, "world_size": 1, "local_rank": 0}

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # 多节点训练需要较长的超时时间，避免因网络波动导致进程被误杀
    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group(backend=backend, init_method="env://", timeout=timeout)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return reduced


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# 全局标志：是否收到终止信号
_SIGTERM_RECEIVED = False


def _sigterm_handler(signum: int, frame: Any) -> None:
    """捕获 SIGTERM 信号，设置标志位让训练循环优雅退出"""
    global _SIGTERM_RECEIVED
    _SIGTERM_RECEIVED = True
    if is_main_process():
        print(
            "\n[WARNING] 收到 SIGTERM 信号，将在当前 step 结束后保存 checkpoint 并退出..."
        )


def sigterm_received() -> bool:
    """检查是否收到终止信号"""
    return _SIGTERM_RECEIVED


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        return self.net(traj.flatten(1))


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, os.PathLike))


def _sanitize_namespace(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "source"
    chars: List[str] = []
    prev_underscore = False
    for ch in text:
        keep = ch.isalnum() or ch in {"_", "-", "."}
        out_ch = ch if keep else "_"
        if out_ch == "_":
            if prev_underscore:
                continue
            prev_underscore = True
        else:
            prev_underscore = False
        chars.append(out_ch)
    cleaned = "".join(chars).strip("._-")
    return cleaned or "source"


def _coerce_index_source_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, dict):
        out = dict(spec)
        index_path = out.pop("index_path", out.pop("path", out.pop("jsonl", None)))
        if index_path is None:
            raise ValueError("index source spec must contain index_path/path/jsonl")
        out["index_path"] = str(Path(index_path).expanduser())
        image_roots = out.pop("image_roots", None)
        image_root = out.pop("image_root", None)
        roots_raw = image_roots if image_roots is not None else image_root
        if roots_raw is None:
            out["image_roots"] = []
        elif _is_nonstring_sequence(roots_raw):
            out["image_roots"] = [str(Path(root).expanduser()) for root in roots_raw]
        else:
            out["image_roots"] = [str(Path(roots_raw).expanduser())]
        if "source_name" in out and out["source_name"] is not None:
            out["source_name"] = _sanitize_namespace(out["source_name"])
        return out
    if isinstance(spec, (str, os.PathLike, Path)):
        return {
            "index_path": str(Path(spec).expanduser()),
            "image_roots": [],
        }
    raise TypeError(f"Unsupported index spec type: {type(spec)!r}")


def _normalize_index_sources(index_spec: Any) -> List[Dict[str, Any]]:
    if _is_nonstring_sequence(index_spec):
        sources: List[Dict[str, Any]] = []
        for item in index_spec:
            sources.extend(_normalize_index_sources(item))
        return sources
    return [_coerce_index_source_spec(index_spec)]


# ────────────────── Consistency Critic ──────────────────


class ConsistencyDataset(Dataset):
    """Consistency Critic 数据集，包含历史+未来图像和双标签"""

    def __init__(
        self, index_path: str, cfg: Dict[str, Any], training: bool,
    ) -> None:
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"索引文件不存在: {self.index_path}. "
                "请先运行 tools/build_consistency_index.py 生成索引。"
            )
        self.training = training
        self.image_root = Path(cfg["image_root"])
        self.image_size = int(cfg["image_size"])
        self.history_num_frames = int(cfg["history_num_frames"])
        self.future_num_frames = int(cfg["future_num_frames"])
        self.candidate_traj_steps = int(cfg["candidate_traj_steps"])
        self.consistency_traj_steps = int(
            cfg.get("consistency_traj_steps", min(self.future_num_frames, self.candidate_traj_steps)),
        )
        self.ego_state_dim = int(cfg["ego_state_dim"])
        self.traj_dim = int(cfg["traj_dim"])
        ds_cfg = cfg.get("dataset", {})
        self.normalize_ego = bool(ds_cfg.get("normalize_ego_state", True))
        self.normalize_traj = bool(
            ds_cfg.get("normalize_candidate_traj", True),
        )
        self.normalize_mode: str = ds_cfg.get("normalize_mode", "tanh")
        traj_scale_raw = ds_cfg.get("traj_scale", None)
        if self.normalize_mode == "linear" and traj_scale_raw is None:
            raise ValueError(
                "normalize_mode='linear' 时必须在 dataset 配置中提供 traj_scale"
            )
        self.traj_scale: torch.Tensor | None = (
            torch.tensor(traj_scale_raw, dtype=torch.float32)
            if traj_scale_raw is not None
            else None
        )
        self.candidate_quality_score_fields = [
            str(field)
            for field in cfg.get(
                "candidate_quality_score_fields",
                [
                    "epdms_score",
                    "pdms_score",
                    "planning_score",
                    "candidate_quality_score",
                ],
            )
        ]
        self.image_mean = torch.tensor(
            ds_cfg.get("image_mean", [0.485, 0.456, 0.406]),
            dtype=torch.float32,
        )
        self.image_std = torch.tensor(
            ds_cfg.get("image_std", [0.229, 0.224, 0.225]),
            dtype=torch.float32,
        )
        self.samples = self._load_jsonl()

    def _load_jsonl(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        required = {
            "history_images", "future_images", "ego_state",
            "candidate_traj", "consistency_label", "validity_label",
        }
        with self.index_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                missing = required - set(sample)
                if missing:
                    raise ValueError(
                        f"缺少字段 {sorted(missing)}，"
                        f"位于 {self.index_path}:{line_idx}"
                    )
                samples.append(sample)
        if not samples:
            raise ValueError(f"索引文件为空: {self.index_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, image_path: str) -> Path:
        p = Path(image_path)
        return p if p.is_absolute() else self.image_root / p

    def _load_image(self, image_path: str) -> torch.Tensor:
        path = self._resolve_path(image_path)
        with Image.open(path) as img:
            image = img.convert("RGB").resize(
                (self.image_size, self.image_size),
            )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (
            (tensor - self.image_mean[:, None, None])
            / self.image_std[:, None, None]
        )
        return tensor

    def _prepare_images(
        self, paths: List[str], num_frames: int,
    ) -> torch.Tensor:
        selected = list(paths[-num_frames:])
        if len(selected) < num_frames:
            selected = (
                [selected[0]] * (num_frames - len(selected)) + selected
            )
        return torch.stack([self._load_image(p) for p in selected], dim=0)

    def selected_image_paths(
        self, sample: Dict[str, Any], key: str, num_frames: int,
    ) -> List[Path]:
        paths = list(sample[key][-num_frames:])
        if not paths:
            raise ValueError(f"样本缺少图像路径字段: {key}")
        if len(paths) < num_frames:
            paths = [paths[0]] * (num_frames - len(paths)) + paths
        return [self._resolve_path(path) for path in paths]

    def _prepare_vector(
        self, values: List[Any], length: int,
    ) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        if tensor.numel() < length:
            tensor = F.pad(tensor, (0, length - tensor.numel()))
        elif tensor.numel() > length:
            tensor = tensor[:length]
        return tensor

    def _prepare_traj(self, traj: List[List[Any]]) -> torch.Tensor:
        tensor = torch.tensor(traj, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(
                f"candidate_traj 必须为 2D，当前 shape={tuple(tensor.shape)}"
            )
        steps, dims = tensor.shape
        if dims < self.traj_dim:
            tensor = F.pad(tensor, (0, self.traj_dim - dims))
        elif dims > self.traj_dim:
            tensor = tensor[:, : self.traj_dim]
        if steps < self.candidate_traj_steps:
            tensor = F.pad(
                tensor, (0, 0, 0, self.candidate_traj_steps - steps),
            )
        elif steps > self.candidate_traj_steps:
            tensor = tensor[: self.candidate_traj_steps]
        return tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        hist_imgs = self._prepare_images(
            sample["history_images"], self.history_num_frames,
        )
        fut_imgs = self._prepare_images(
            sample["future_images"], self.future_num_frames,
        )
        ego = self._prepare_vector(sample["ego_state"], self.ego_state_dim)
        traj = self._prepare_traj(sample["candidate_traj"])

        if self.normalize_ego:
            ego = torch.tanh(ego)
        if self.normalize_traj:
            if self.normalize_mode == "linear" and self.traj_scale is not None:
                traj = traj / self.traj_scale  # 广播 (steps, dim) / (dim,)
            else:
                traj = torch.tanh(traj)

        c_label = torch.tensor(
            float(sample["consistency_label"]), dtype=torch.float32,
        )
        v_label = torch.tensor(
            float(sample["validity_label"]), dtype=torch.float32,
        )
        candidate_quality = math.nan
        for field in self.candidate_quality_score_fields:
            if field in sample and sample[field] is not None:
                candidate_quality = float(sample[field])
                break
        return {
            "sample_index": int(index),
            "history_images": hist_imgs,
            "future_images": fut_imgs,
            "ego_state": ego,
            "ego_state_raw": torch.tensor(sample["ego_state"], dtype=torch.float32),
            "candidate_traj": traj,
            "candidate_traj_raw": torch.tensor(sample["candidate_traj"], dtype=torch.float32),
            "consistency_label": c_label,
            "validity_label": v_label,
            "candidate_quality_score": torch.tensor(
                candidate_quality,
                dtype=torch.float32,
            ),
            "sample_id": str(sample.get("sample_id", index)),
            "group_id": str(
                sample.get(
                    "group_id",
                    str(sample.get("sample_id", index)).rsplit("__", 1)[0],
                )
            ),
            "source_type": str(sample.get("source_type", "unknown")),
            "label_quality": str(
                sample.get(
                    "label_quality",
                    "positive" if float(sample["consistency_label"]) > 0.5 else "clean_negative",
                )
            ),
        }


class ConsistencyCriticModel(nn.Module):
    """P0-audited Action-Image Consistency Critic

    结构:
        HistoryImageEncoder -> z_hist (256)
        FutureImageEncoder  -> z_future (256)
        ConsistencyTrajectoryEncoder -> z_traj_consistency (128)
        ValidityTrajectoryEncoder    -> z_traj_validity (128)
        EgoEncoder          -> z_ego (128)

    P0 约束:
        Consistency 只看与 future images 对齐的前 consistency_traj_steps 步轨迹。
        Validity 只看 ego + 完整轨迹，不接图像特征，避免场景 shortcut。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        mcfg = cfg["model"]
        img_dim = int(mcfg["image_feature_dim"])
        act_dim = int(mcfg["action_feature_dim"])
        hidden = int(mcfg["hidden_dim"])
        fusion_dim = int(mcfg.get("fusion_dim", 256))
        dropout = float(mcfg.get("dropout", 0.0))
        ego_dim = int(cfg["ego_state_dim"])
        traj_steps = int(cfg["candidate_traj_steps"])
        consistency_traj_steps = int(
            cfg.get("consistency_traj_steps", min(int(cfg.get("future_num_frames", traj_steps)), traj_steps)),
        )
        traj_d = int(cfg["traj_dim"])
        self.baseline_mode = str(cfg.get("baseline_mode", "full"))
        self.consistency_traj_steps = consistency_traj_steps
        self.use_action_visual_interaction = bool(
            mcfg.get("use_action_visual_interaction", False),
        )
        self.temporal_encoder_type = str(mcfg.get("temporal_encoder", "mean"))
        if self.temporal_encoder_type not in {"mean", "gru"}:
            raise ValueError(
                "model.temporal_encoder must be one of: mean, gru",
            )

        # 共享 CNN backbone
        self.shared_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.history_proj = nn.Linear(256, img_dim)
        self.future_proj = nn.Linear(256, img_dim)
        if self.temporal_encoder_type == "gru":
            self.history_temporal_encoder = nn.GRU(
                input_size=img_dim,
                hidden_size=img_dim,
                batch_first=True,
            )
            self.future_temporal_encoder = nn.GRU(
                input_size=img_dim,
                hidden_size=img_dim,
                batch_first=True,
            )
        else:
            self.history_temporal_encoder = None
            self.future_temporal_encoder = None

        self.consistency_traj_encoder = TrajectoryEncoder(
            consistency_traj_steps * traj_d, hidden, act_dim,
        )
        self.validity_traj_encoder = TrajectoryEncoder(
            traj_steps * traj_d, hidden, act_dim,
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, act_dim),
            nn.ReLU(inplace=True),
        )

        if self.use_action_visual_interaction:
            self.action_to_visual_delta = nn.Sequential(
                nn.Linear(act_dim * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, img_dim),
                nn.ReLU(inplace=True),
            )
            consistency_dim = img_dim * 7 + act_dim * 2
        else:
            self.action_to_visual_delta = None
            consistency_dim = img_dim * 2 + act_dim * 2
        self.shared_fusion = nn.Sequential(
            nn.Linear(consistency_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        validity_dim = act_dim * 2
        self.validity_fusion = nn.Sequential(
            nn.Linear(validity_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Consistency 是主监督。细粒度 heads 仅保留为审计输出，默认 loss 权重为 0。
        self.consistency_head = nn.Linear(fusion_dim, 1)  # overall consistency
        self.speed_consistency_head = nn.Linear(fusion_dim, 1)  # speed consistency
        self.steering_consistency_head = nn.Linear(fusion_dim, 1)  # steering consistency
        self.progress_consistency_head = nn.Linear(fusion_dim, 1)  # progress consistency
        self.temporal_coherence_head = nn.Linear(fusion_dim, 1)  # temporal coherence
        self.validity_head = nn.Linear(fusion_dim, 1)  # driving validity

    def _encode_images(
        self,
        images: torch.Tensor,
        proj: nn.Linear,
        temporal_encoder: nn.GRU | None,
    ) -> torch.Tensor:
        """编码 (B, T, C, H, W) 图像序列为 (B, dim)"""
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w)
        x = self.shared_backbone(x).flatten(1)
        x = proj(x)
        x = x.reshape(b, t, -1)
        if temporal_encoder is None:
            return x.mean(dim=1)
        _, hidden = temporal_encoder(x)
        return hidden[-1]

    def _encode_image_sequence_raw(
        self,
        images: torch.Tensor,
        proj: nn.Linear,
    ) -> torch.Tensor:
        """Encode (B, T, C, H, W) -> (B, T, dim) without temporal pooling."""
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w)
        x = self.shared_backbone(x).flatten(1)
        x = proj(x)
        return x.reshape(b, t, -1)

    def forward(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats = self.extract_probe_features(
            history_images=history_images,
            future_images=future_images,
            ego_state=ego_state,
            candidate_traj=candidate_traj,
        )
        return {
            "consistency_logit": self.consistency_head(feats["z_shared"]).squeeze(-1),
            "speed_consistency_logit": self.speed_consistency_head(feats["z_shared"]).squeeze(-1),
            "steering_consistency_logit": self.steering_consistency_head(feats["z_shared"]).squeeze(-1),
            "progress_consistency_logit": self.progress_consistency_head(feats["z_shared"]).squeeze(-1),
            "temporal_coherence_logit": self.temporal_coherence_head(feats["z_shared"]).squeeze(-1),
            "validity_logit": self.validity_head(feats["z_validity"]).squeeze(-1),
        }

    def extract_probe_features(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hist_seq = self._encode_image_sequence_raw(history_images, self.history_proj)
        fut_seq = self._encode_image_sequence_raw(future_images, self.future_proj)
        z_hist = self._encode_images(
            history_images, self.history_proj, self.history_temporal_encoder,
        )
        z_fut = self._encode_images(
            future_images, self.future_proj, self.future_temporal_encoder,
        )
        consistency_traj = candidate_traj[:, : self.consistency_traj_steps, :]
        z_traj_consistency = self.consistency_traj_encoder(consistency_traj)
        z_traj_validity = self.validity_traj_encoder(candidate_traj)
        z_ego = self.ego_encoder(ego_state)

        mode = self.baseline_mode
        if mode in {"no_image", "ego_only", "traj_only"}:
            z_hist = torch.zeros_like(z_hist)
            z_fut = torch.zeros_like(z_fut)
        if mode in {"no_traj", "ego_only"}:
            z_traj_consistency = torch.zeros_like(z_traj_consistency)
            z_traj_validity = torch.zeros_like(z_traj_validity)
        if mode == "traj_only":
            z_ego = torch.zeros_like(z_ego)

        parts: List[torch.Tensor]
        if self.use_action_visual_interaction:
            visual_delta = z_fut - z_hist
            visual_abs_delta = visual_delta.abs()
            assert self.action_to_visual_delta is not None
            action_delta = self.action_to_visual_delta(
                torch.cat([z_traj_consistency, z_ego], dim=-1),
            )
            action_visual_product = action_delta * visual_delta
            action_visual_gap = (action_delta - visual_delta).abs()
            parts = [
                z_hist,
                z_fut,
                visual_delta,
                visual_abs_delta,
                action_delta,
                action_visual_product,
                action_visual_gap,
                z_traj_consistency,
                z_ego,
            ]
        else:
            parts = [z_hist, z_fut, z_traj_consistency, z_ego]

        z_all = torch.cat(parts, dim=-1)
        z_shared = self.shared_fusion(z_all)
        z_validity = self.validity_fusion(torch.cat([z_traj_validity, z_ego], dim=-1))

        return {
            "hist_seq": hist_seq,
            "fut_seq": fut_seq,
            "hist_seq_mean": hist_seq.mean(dim=1),
            "fut_seq_mean": fut_seq.mean(dim=1),
            "hist_seq_last": hist_seq[:, -1, :],
            "fut_seq_last": fut_seq[:, -1, :],
            "z_hist": z_hist,
            "z_fut": z_fut,
            "z_traj_consistency": z_traj_consistency,
            "z_traj_validity": z_traj_validity,
            "z_ego": z_ego,
            "z_all": z_all,
            "z_shared": z_shared,
            "z_validity": z_validity,
        }


class GroupRankingBatchSampler:
    """Batch full candidate groups so ranking loss gets real pos/neg pairs."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        batch_size: int,
        num_samples_per_epoch: int,
        seed: int = 0,
        world_size: int = 1,
        rank: int = 0,
        source_weights: Dict[str, float] | None = None,
        hard_negative_sources: Sequence[str] | None = None,
        max_negatives_per_group: int = 0,
        sample_difficulties: Sequence[int] | None = None,
        difficulty_mix: Sequence[float] | None = None,
    ) -> None:
        self.samples = samples
        self.batch_size = max(1, int(batch_size))
        self.num_samples = max(1, int(num_samples_per_epoch))
        self.seed = int(seed)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.source_weights = {
            str(k): float(v)
            for k, v in (source_weights or {}).items()
        }
        self.hard_negative_sources = {
            str(v) for v in (hard_negative_sources or ())
        }
        self.max_negatives_per_group = max(0, int(max_negatives_per_group))
        self.sample_difficulties = (
            [int(v) for v in sample_difficulties]
            if sample_difficulties is not None
            else []
        )
        mix = list(difficulty_mix or ())
        self.difficulty_mix = (
            [float(v) for v in mix[:4]]
            if len(mix) >= 4 and sum(float(v) for v in mix[:4]) > 0
            else []
        )
        self.epoch = 0

        groups: Dict[str, List[int]] = {}
        for i, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id", i))
            group_id = str(sample.get("group_id", sample_id.rsplit("__", 1)[0]))
            groups.setdefault(group_id, []).append(i)

        self.groups = [
            idxs for idxs in groups.values()
            if any(samples[i].get("consistency_label", 0) == 1 for i in idxs)
            and any(samples[i].get("consistency_label", 1) == 0 for i in idxs)
        ]
        if not self.groups:
            raise ValueError(
                "Group ranking sampler requires group_id groups with at least "
                "one positive and one negative sample."
            )
        self.group_weights = [self._group_weight(idxs) for idxs in self.groups]
        avg_group = sum(len(g) for g in self.groups) / max(len(self.groups), 1)
        if self.max_negatives_per_group > 0:
            avg_group = min(avg_group, float(self.max_negatives_per_group + 1))
        self.num_groups = max(1, int(math.ceil(self.num_samples / max(avg_group, 1.0))))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, int(math.ceil(self.num_samples / self.batch_size)))

    def _sample_source(self, index: int) -> str:
        return str(self.samples[index].get("source_type", "unknown"))

    def _is_positive(self, index: int) -> bool:
        return float(self.samples[index].get("consistency_label", 0)) > 0.5

    def _source_priority(self, index: int) -> float:
        source = self._sample_source(index)
        priority = self.source_weights.get(source, 1.0)
        if source in self.hard_negative_sources:
            priority += max(priority, 1.0)
        difficulty = self._sample_difficulty(index)
        if self.difficulty_mix and difficulty > 0:
            priority *= 1.0 + self.difficulty_mix[difficulty - 1]
        return float(priority)

    def _sample_difficulty(self, index: int) -> int:
        if 0 <= index < len(self.sample_difficulties):
            return max(0, min(4, int(self.sample_difficulties[index])))
        return 0

    def _group_weight(self, indices: List[int]) -> float:
        neg_priorities = [
            self._source_priority(i)
            for i in indices
            if not self._is_positive(i)
        ]
        return max(neg_priorities) if neg_priorities else 1.0

    def _trim_group(self, indices: List[int], rng: random.Random) -> List[int]:
        """Keep ranking groups useful even when a group is larger than batch_size."""
        positives = [i for i in indices if self._is_positive(i)]
        negatives = [i for i in indices if not self._is_positive(i)]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        negatives.sort(
            key=lambda i: (self._sample_difficulty(i), self._source_priority(i)),
            reverse=True,
        )

        if self.max_negatives_per_group > 0:
            negatives = self._select_group_negatives(negatives, rng)
            indices = positives[:1] + negatives
            if len(indices) <= self.batch_size:
                out = list(indices)
                rng.shuffle(out)
                return out

        if len(indices) <= self.batch_size:
            out = list(indices)
            rng.shuffle(out)
            return out

        out: List[int] = []
        if positives:
            out.append(positives[0])
        slots = self.batch_size - len(out)
        out.extend(negatives[:slots])

        if len(out) < self.batch_size:
            used = set(out)
            remainder = [i for i in positives[1:] + negatives[slots:] if i not in used]
            rng.shuffle(remainder)
            out.extend(remainder[: self.batch_size - len(out)])

        rng.shuffle(out)
        return out

    def _select_group_negatives(
        self, negatives: List[int], rng: random.Random,
    ) -> List[int]:
        if self.max_negatives_per_group <= 0:
            return negatives

        selected: List[int] = []
        used = set()

        if self.difficulty_mix:
            raw = [self.max_negatives_per_group * m for m in self.difficulty_mix]
            quotas = [int(math.floor(v)) for v in raw]
            remaining_quota = self.max_negatives_per_group - sum(quotas)
            order = sorted(
                range(4),
                key=lambda i: (raw[i] - quotas[i], i),
                reverse=True,
            )
            for i in order[:remaining_quota]:
                quotas[i] += 1

            for difficulty in (4, 3, 2, 1):
                quota = quotas[difficulty - 1]
                if quota <= 0:
                    continue
                bucket = [
                    i for i in negatives
                    if i not in used and self._sample_difficulty(i) == difficulty
                ]
                bucket.sort(key=lambda i: self._source_priority(i), reverse=True)
                for idx in bucket[:quota]:
                    selected.append(idx)
                    used.add(idx)
                    if len(selected) >= self.max_negatives_per_group:
                        rng.shuffle(selected)
                        return selected

        remainder = [i for i in negatives if i not in used]
        remainder.sort(key=lambda i: self._source_priority(i), reverse=True)
        for idx in remainder:
            selected.append(idx)
            if len(selected) >= self.max_negatives_per_group:
                break
        rng.shuffle(selected)
        return selected

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        group_pairs = list(zip(self.groups, self.group_weights))
        rng.shuffle(group_pairs)
        groups = [g for g, _ in group_pairs]
        weights = [w for _, w in group_pairs]
        groups = groups[self.rank::self.world_size]
        weights = weights[self.rank::self.world_size]
        if not groups:
            return

        if self.num_groups <= len(groups):
            selected = groups[: self.num_groups]
        else:
            selected = rng.choices(groups, weights=weights, k=self.num_groups)

        batch: List[int] = []
        for group in selected:
            indices = self._trim_group(list(group), rng)
            if batch and len(batch) + len(indices) > self.batch_size:
                yield batch
                batch = []
            batch.extend(indices)
        if batch:
            yield batch


def _metadata_list(
    batch: Dict[str, Any],
    key: str,
    default: str,
    batch_size: int,
) -> List[str]:
    value = batch.get(key)
    if value is None:
        return [default] * batch_size
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * batch_size


def _weights_from_strings(
    values: List[str],
    mapping: Dict[str, float],
    device: torch.device,
    default: float = 1.0,
) -> torch.Tensor:
    if not mapping:
        return torch.full((len(values),), float(default), device=device)
    weights = [float(mapping.get(value, default)) for value in values]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _source_mask_from_strings(
    values: List[str],
    include: set[str],
    exclude: set[str],
    device: torch.device,
) -> torch.Tensor:
    mask = [
        (not include or value in include) and value not in exclude
        for value in values
    ]
    return torch.tensor(mask, dtype=torch.bool, device=device)


def _masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    if mask is None:
        return losses.mean()
    weights = mask.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def _trajectory_image_polyline_from_tensor(
    traj: torch.Tensor,
    height: int,
    width: int,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
) -> List[tuple[int, int]]:
    """Lightweight ego-trajectory to image-path projection for ROI masking.

    This is a diagnostic/training ROI, not a calibrated camera projection.
    It maps forward motion upward from the ego vehicle and lateral motion left
    or right around the image center.
    """
    if traj.ndim != 2 or traj.numel() == 0:
        return [(width // 2, height - 1), (width // 2, max(0, int(height * 0.45)))]
    t = traj.detach().float()
    if t.size(1) < 3:
        t = F.pad(t, (0, 3 - t.size(1)))
    t = t[:, :3]
    xy = t if trajectory_mode == "positions" else torch.cumsum(t, dim=0)
    forward = xy[:, 0].clamp_min(0.0)
    lateral = xy[:, 1]
    if projection_mode == "fixed":
        max_forward = max(float(forward_m), 1.0)
        max_lateral = max(float(lateral_m), 1.0)
    else:
        max_forward = max(float(torch.quantile(forward.abs(), 0.9).item()), 1.0)
        max_lateral = max(float(torch.quantile(lateral.abs(), 0.9).item()), 2.0)

    pts: List[tuple[int, int]] = [(width // 2, height - 1)]
    for x_fwd, y_lat in zip(forward.tolist(), lateral.tolist()):
        v = int((height - 1) - max(0.0, min(float(x_fwd) / max_forward, 1.0)) * height * 0.62)
        u = int((width / 2.0) - max(-1.0, min(float(y_lat) / max_lateral, 1.0)) * width * 0.32)
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
        alpha = step / steps
        x = int(round(x0 + (x1 - x0) * alpha))
        y = int(round(y0 + (y1 - y0) * alpha))
        _draw_disk(mask, x, y, radius)


def _path_mask_from_traj(
    traj: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    width_ratio: float,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    pts = _trajectory_image_polyline_from_tensor(
        traj,
        height,
        width,
        trajectory_mode=trajectory_mode,
        projection_mode=projection_mode,
        forward_m=forward_m,
        lateral_m=lateral_m,
    )
    radius = max(2, int(round(width * float(width_ratio))))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        _draw_line(mask, p0, p1, radius)
    return mask


def _mask_future_for_path_grounding(
    future_images: torch.Tensor,
    traj_raw: torch.Tensor,
    mode: str,
    width_ratio: float,
    sky_ratio: float,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
) -> torch.Tensor:
    """Mask normalized future images with mean-color fill for causal training."""
    masked = future_images.clone()
    bsz, _, _, height, width = masked.shape
    for b in range(bsz):
        if mode == "path":
            mask = _path_mask_from_traj(
                traj_raw[b],
                height,
                width,
                masked.device,
                width_ratio,
                trajectory_mode=trajectory_mode,
                projection_mode=projection_mode,
                forward_m=forward_m,
                lateral_m=lateral_m,
            )
        elif mode == "sky":
            mask = torch.zeros((height, width), dtype=torch.bool, device=masked.device)
            path_ref = _path_mask_from_traj(
                traj_raw[b],
                height,
                width,
                masked.device,
                width_ratio,
                trajectory_mode=trajectory_mode,
                projection_mode=projection_mode,
                forward_m=forward_m,
                lateral_m=lateral_m,
            )
            target_fraction = min(float(path_ref.float().mean().item()), float(sky_ratio))
            sky_h = max(1, min(height, int(round(height * target_fraction))))
            mask[:sky_h, :] = True
        else:
            raise ValueError(f"unknown path grounding mask mode: {mode}")
        masked[b, :, :, mask] = 0.0
    return masked


def _path_iou_from_traj(
    traj_a: torch.Tensor,
    traj_b: torch.Tensor,
    *,
    width_ratio: float,
    trajectory_mode: str,
    projection_mode: str,
    forward_m: float,
    lateral_m: float,
    mask_size: int = 64,
) -> float:
    device = traj_a.device
    mask_a = _path_mask_from_traj(
        traj_a,
        mask_size,
        mask_size,
        device,
        width_ratio,
        trajectory_mode=trajectory_mode,
        projection_mode=projection_mode,
        forward_m=forward_m,
        lateral_m=lateral_m,
    )
    mask_b = _path_mask_from_traj(
        traj_b,
        mask_size,
        mask_size,
        device,
        width_ratio,
        trajectory_mode=trajectory_mode,
        projection_mode=projection_mode,
        forward_m=forward_m,
        lateral_m=lateral_m,
    )
    union = float((mask_a | mask_b).float().sum().detach().item())
    if union <= 0.0:
        return 0.0
    inter = float((mask_a & mask_b).float().sum().detach().item())
    return inter / union


def _trajectory_distance(traj_a: torch.Tensor, traj_b: torch.Tensor) -> float:
    a = traj_a[..., :2].float()
    b = traj_b[..., :2].float()
    steps = min(a.size(0), b.size(0))
    if steps <= 0:
        return 0.0
    a = a[:steps]
    b = b[:steps]
    diff = a - b
    return float((torch.norm(diff, dim=-1).mean() + torch.norm(diff[-1])).detach().item())


def _continuous_consistency_soft_targets(
    c_labels: torch.Tensor,
    traj_raw: torch.Tensor,
    group_ids: List[str],
    source_types: List[str],
    *,
    mode: str,
    near_sources: set[str],
    hard_negative_sources: set[str],
    gamma: float,
    min_soft: float,
    distance_tau: float,
    path_width_ratio: float,
    trajectory_mode: str,
    projection_mode: str,
    forward_m: float,
    lateral_m: float,
) -> torch.Tensor:
    targets = c_labels.clone()
    if mode == "none":
        return targets
    grouped: Dict[str, List[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(idx)
    positive_by_group: Dict[str, int] = {}
    for group_id, indices in grouped.items():
        positives = [
            idx for idx in indices
            if float(c_labels[idx].detach().item()) > 0.5
        ]
        if positives:
            positive_by_group[group_id] = positives[0]

    min_soft = min(max(float(min_soft), 0.0), 1.0)
    for idx, source in enumerate(source_types):
        if float(c_labels[idx].detach().item()) > 0.5:
            continue
        if source in hard_negative_sources or source not in near_sources:
            continue
        pos_idx = positive_by_group.get(str(group_ids[idx]))
        if pos_idx is None:
            continue
        if mode == "path_iou":
            similarity = _path_iou_from_traj(
                traj_raw[idx],
                traj_raw[pos_idx],
                width_ratio=path_width_ratio,
                trajectory_mode=trajectory_mode,
                projection_mode=projection_mode,
                forward_m=forward_m,
                lateral_m=lateral_m,
            )
            target_value = max(min_soft, float(similarity) ** float(gamma))
        elif mode == "trajectory_distance":
            distance = _trajectory_distance(traj_raw[idx], traj_raw[pos_idx])
            target_value = math.exp(-distance / max(float(distance_tau), 1e-6))
            target_value = max(min_soft, min(1.0, target_value))
        else:
            raise ValueError(
                "unknown consistency_soft_target_mode: "
                f"{mode!r}; expected none/path_iou/trajectory_distance"
            )
        targets[idx] = float(target_value)
    return targets


def _supported_set_consistency_targets(
    c_labels: torch.Tensor,
    traj_raw: torch.Tensor,
    candidate_quality_scores: torch.Tensor | None,
    group_ids: List[str],
    source_types: List[str],
    base_supervision_mask: torch.Tensor,
    *,
    positive_sources: set[str],
    unknown_sources: set[str],
    hard_negative_sources: set[str],
    positive_quality_threshold: float,
    unknown_quality_threshold: float,
    positive_target: float,
    unknown_target: float,
    use_quality_as_target: bool,
    require_geometry_for_positive: bool,
    positive_path_iou_threshold: float,
    positive_distance_threshold: float,
    path_width_ratio: float,
    trajectory_mode: str,
    projection_mode: str,
    forward_m: float,
    lateral_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build supported-set labels for a multi-solution consistency task.

    GT remains positive, clear cross-scene/time/trajectory mismatches remain
    negative, and same-scene high-quality perturbations can become soft
    positives. Ambiguous same-scene candidates are removed from BCE instead of
    being treated as false negatives.
    """
    targets = c_labels.clone()
    supervision_mask = base_supervision_mask.clone()
    quality = (
        candidate_quality_scores
        if candidate_quality_scores is not None
        else torch.full_like(c_labels, float("nan"))
    )

    grouped: Dict[str, List[int]] = {}
    for idx, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(idx)
    gt_by_group: Dict[str, int] = {}
    for group_id, indices in grouped.items():
        positives = [
            idx
            for idx in indices
            if float(c_labels[idx].detach().item()) > 0.5
        ]
        if positives:
            gt_by_group[group_id] = positives[0]

    positive_target = min(max(float(positive_target), 0.0), 1.0)
    unknown_target = min(max(float(unknown_target), 0.0), 1.0)
    for idx, source in enumerate(source_types):
        if float(c_labels[idx].detach().item()) > 0.5:
            targets[idx] = 1.0
            continue
        source = str(source)
        if source in hard_negative_sources:
            targets[idx] = 0.0
            continue
        if source not in positive_sources and source not in unknown_sources:
            continue

        q = float(quality[idx].detach().item()) if torch.isfinite(quality[idx]) else math.nan
        gt_idx = gt_by_group.get(str(group_ids[idx]))
        geometry_supported = False
        if gt_idx is not None:
            path_iou = _path_iou_from_traj(
                traj_raw[idx],
                traj_raw[gt_idx],
                width_ratio=path_width_ratio,
                trajectory_mode=trajectory_mode,
                projection_mode=projection_mode,
                forward_m=forward_m,
                lateral_m=lateral_m,
            )
            distance = _trajectory_distance(traj_raw[idx], traj_raw[gt_idx])
            geometry_supported = (
                path_iou >= float(positive_path_iou_threshold)
                or distance <= float(positive_distance_threshold)
            )

        quality_supported = (
            math.isfinite(q)
            and q >= float(positive_quality_threshold)
        )
        if source in positive_sources and quality_supported and (
            geometry_supported or not require_geometry_for_positive
        ):
            target_value = positive_target
            if use_quality_as_target and math.isfinite(q):
                target_value = max(target_value, min(0.98, q))
            targets[idx] = target_value
            supervision_mask[idx] = True
        elif (
            source in unknown_sources
            and math.isfinite(q)
            and q >= float(unknown_quality_threshold)
        ):
            targets[idx] = unknown_target
            supervision_mask[idx] = False
        elif source in unknown_sources:
            supervision_mask[idx] = False

    return targets, supervision_mask


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
    out_cand.flatten()[cand_idx[:k]] = True
    out_wrong.flatten()[wrong_idx[:k]] = True
    return out_cand, out_wrong


def _mask_future_for_exclusive_path_grounding(
    future_images: torch.Tensor,
    traj_raw: torch.Tensor,
    wrong_traj_raw: torch.Tensor,
    width_ratio: float,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cand_masked = future_images.clone()
    wrong_masked = future_images.clone()
    bsz, _, _, height, width = future_images.shape
    fractions = torch.zeros((bsz,), dtype=torch.float32, device=future_images.device)
    for b in range(bsz):
        cand = _path_mask_from_traj(
            traj_raw[b],
            height,
            width,
            future_images.device,
            width_ratio,
            trajectory_mode=trajectory_mode,
            projection_mode=projection_mode,
            forward_m=forward_m,
            lateral_m=lateral_m,
        )
        wrong = _path_mask_from_traj(
            wrong_traj_raw[b],
            height,
            width,
            future_images.device,
            width_ratio,
            trajectory_mode=trajectory_mode,
            projection_mode=projection_mode,
            forward_m=forward_m,
            lateral_m=lateral_m,
        )
        cand_excl, wrong_excl = _equal_area_exclusive_masks(cand, wrong)
        cand_masked[b, :, :, cand_excl] = 0.0
        wrong_masked[b, :, :, wrong_excl] = 0.0
        fractions[b] = cand_excl.float().mean()
    return cand_masked, wrong_masked, fractions


def _wrong_traj_controls(
    traj_raw: torch.Tensor,
    labels: torch.Tensor,
    group_ids: List[str],
    *,
    selection: str = "trajectory_distance",
    width_ratio: float = 0.10,
    trajectory_mode: str = "cumulative",
    projection_mode: str = "relative",
    forward_m: float = 40.0,
    lateral_m: float = 10.0,
    mask_height: int = 224,
    mask_width: int = 224,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wrong = traj_raw.clone()
    mask = torch.zeros((traj_raw.size(0),), dtype=torch.bool, device=traj_raw.device)
    distance = torch.zeros((traj_raw.size(0),), dtype=torch.float32, device=traj_raw.device)
    grouped: Dict[str, List[int]] = {}
    for i, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(i)

    xy = traj_raw[..., :2].float()
    path_masks: Dict[int, torch.Tensor] = {}

    def traj_distance(i: int, j: int) -> torch.Tensor:
        diff = xy[i] - xy[j]
        mean_l2 = torch.norm(diff, dim=-1).mean()
        final_l2 = torch.norm(diff[-1])
        return mean_l2 + final_l2

    def path_mask(i: int) -> torch.Tensor:
        if i not in path_masks:
            path_masks[i] = _path_mask_from_traj(
                traj_raw[i],
                mask_height,
                mask_width,
                traj_raw.device,
                width_ratio,
                trajectory_mode=trajectory_mode,
                projection_mode=projection_mode,
                forward_m=forward_m,
                lateral_m=lateral_m,
            )
        return path_masks[i]

    def select_other(i: int, candidates: List[int]) -> tuple[int | None, torch.Tensor]:
        if not candidates:
            return None, distance[i]
        scores: List[tuple[float, float, float, int, torch.Tensor]] = []
        for j in candidates:
            dist = traj_distance(i, j)
            if selection == "mask_iou":
                cand_mask = path_mask(i)
                wrong_mask = path_mask(j)
                union = float((cand_mask | wrong_mask).float().sum().detach().item())
                inter = float((cand_mask & wrong_mask).float().sum().detach().item())
                iou = inter / max(union, 1.0)
                exclusive = float(
                    ((cand_mask & ~wrong_mask) | (wrong_mask & ~cand_mask))
                    .float()
                    .mean()
                    .detach()
                    .item()
                )
                scores.append((-iou, exclusive, float(dist.detach().item()), j, dist))
            else:
                scores.append((float(dist.detach().item()), 0.0, 0.0, j, dist))
        scores.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
        _, _, _, other, dist = scores[0]
        return other, dist

    for indices in grouped.values():
        if len(indices) < 2:
            continue
        pos = [i for i in indices if float(labels[i].detach().item()) > 0.5]
        neg = [i for i in indices if i not in pos]
        for i in indices:
            other: int | None = None
            if i in pos and neg:
                other, selected_distance = select_other(i, neg)
                distance[i] = selected_distance.detach()
            elif pos and i not in pos:
                other = pos[0]
                distance[i] = traj_distance(i, other).detach()
            else:
                candidates = [j for j in indices if j != i]
                other, selected_distance = select_other(i, candidates)
                distance[i] = selected_distance.detach()
            if other is not None:
                wrong[i] = traj_raw[other]
                mask[i] = True
    return wrong, mask, distance


def _group_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: List[str],
    source_types: List[str],
    source_weights: Dict[str, float],
    source_margins: Dict[str, float],
    margin: float,
) -> Tuple[torch.Tensor, int]:
    losses: List[torch.Tensor] = []
    grouped: Dict[str, List[int]] = {}
    for i, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(i)

    for indices in grouped.values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=logits.device)
        g_logits = logits.index_select(0, idx)
        g_labels = labels.index_select(0, idx)
        pos_mask = g_labels > 0.5
        neg_mask = ~pos_mask
        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            continue
        pos_scores = g_logits[pos_mask]
        neg_scores = g_logits[neg_mask]
        neg_flags = neg_mask.detach().cpu().tolist()
        neg_indices = [indices[j] for j, is_neg in enumerate(neg_flags) if is_neg]
        neg_sources = [source_types[j] for j in neg_indices]
        neg_margins = _weights_from_strings(
            neg_sources,
            source_margins,
            logits.device,
            default=float(margin),
        )
        pair_loss = F.relu(
            neg_margins[None, :] - (pos_scores[:, None] - neg_scores[None, :])
        )
        neg_weights = _weights_from_strings(
            neg_sources,
            source_weights,
            logits.device,
            default=1.0,
        )
        pair_loss = pair_loss * neg_weights[None, :]
        losses.append(pair_loss.mean())

    if not losses:
        return torch.tensor(0.0, device=logits.device), 0
    return torch.stack(losses).mean(), len(losses)


def _group_soft_target_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_ids: List[str],
    source_types: List[str],
    source_weights: Dict[str, float],
    *,
    margin: float,
    min_target_gap: float,
    eligible_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, int]:
    losses: List[torch.Tensor] = []
    pair_count = 0
    grouped: Dict[str, List[int]] = {}
    for i, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(i)

    for indices in grouped.values():
        if eligible_mask is not None:
            indices = [
                i for i in indices
                if bool(eligible_mask[i].detach().item())
            ]
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=logits.device)
        g_logits = logits.index_select(0, idx)
        g_targets = targets.index_select(0, idx)
        target_gap = g_targets[:, None] - g_targets[None, :]
        pair_mask = target_gap > float(min_target_gap)
        if not bool(pair_mask.any()):
            continue
        logit_gap = g_logits[:, None] - g_logits[None, :]
        pair_margin = float(margin) * target_gap.clamp_min(0.0)
        pair_loss = F.relu(pair_margin - logit_gap)
        weights = _weights_from_strings(
            [source_types[i] for i in indices],
            source_weights,
            logits.device,
            default=1.0,
        )
        pair_weight = weights[None, :].expand_as(pair_loss)
        losses.append((pair_loss[pair_mask] * pair_weight[pair_mask]).mean())
        pair_count += int(pair_mask.float().sum().detach().item())

    if not losses:
        return torch.tensor(0.0, device=logits.device), 0
    return torch.stack(losses).mean(), pair_count


def _group_hard_negative_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: List[str],
    source_types: List[str],
    source_weights: Dict[str, float],
    neg_margin: float,
    neg_target: float,
    targets: torch.Tensor | None = None,
    hard_sources: set[str] | None = None,
    max_target: float = 0.5,
) -> Tuple[torch.Tensor, int]:
    losses: List[torch.Tensor] = []
    grouped: Dict[str, List[int]] = {}
    for i, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(i)

    for indices in grouped.values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=logits.device)
        g_logits = logits.index_select(0, idx)
        if targets is None:
            g_labels = labels.index_select(0, idx)
            pos_mask = g_labels > 0.5
            neg_mask = ~pos_mask
            if not bool(pos_mask.any()) or not bool(neg_mask.any()):
                continue
        else:
            g_targets = targets.index_select(0, idx)
            neg_mask = g_targets <= float(max_target)
            if hard_sources:
                source_mask = torch.tensor(
                    [str(source_types[i]) in hard_sources for i in indices],
                    dtype=torch.bool,
                    device=logits.device,
                )
                neg_mask = neg_mask & source_mask
            if not bool(neg_mask.any()):
                continue

        neg_scores = g_logits[neg_mask]
        neg_flags = neg_mask.detach().cpu().tolist()
        neg_indices = [indices[j] for j, is_neg in enumerate(neg_flags) if is_neg]
        neg_sources = [source_types[j] for j in neg_indices]
        neg_weights = _weights_from_strings(
            neg_sources,
            source_weights,
            logits.device,
            default=1.0,
        )

        # Suppress the highest-scoring negative in each group so hard negatives
        # are not merely ranked below positives, but are actively pushed down.
        weighted_neg_scores = neg_scores + neg_weights.log()
        hard_idx = int(torch.argmax(weighted_neg_scores).item())
        hard_neg_score = neg_scores[hard_idx]
        hard_neg_weight = neg_weights[hard_idx]
        hard_neg_prob = torch.sigmoid(hard_neg_score)
        hard_neg_loss = F.relu(hard_neg_prob - neg_target + neg_margin)
        losses.append(hard_neg_loss * hard_neg_weight)

    if not losses:
        return torch.tensor(0.0, device=logits.device), 0
    return torch.stack(losses).mean(), len(losses)


def _trajectory_progress_value(
    traj_raw: torch.Tensor,
    *,
    mode: str,
    scale: float,
) -> torch.Tensor:
    xy = traj_raw[..., :2]
    if xy.ndim != 3 or xy.shape[1] == 0:
        return torch.zeros((traj_raw.shape[0],), device=traj_raw.device)
    if mode == "path_length":
        origin = torch.zeros_like(xy[:, :1, :])
        prev = torch.cat([origin, xy[:, :-1, :]], dim=1)
        value = torch.norm(xy - prev, p=2, dim=-1).sum(dim=1)
    elif mode == "forward":
        value = xy[:, -1, 0].clamp_min(0.0)
    elif mode == "final_displacement":
        value = torch.norm(xy[:, -1, :], p=2, dim=-1)
    else:
        raise ValueError(f"unknown progress_alignment_mode: {mode}")
    return value / max(float(scale), 1e-6)


def _progress_alignment_rank_loss(
    image_progress: torch.Tensor,
    traj_raw: torch.Tensor,
    labels: torch.Tensor,
    group_ids: List[str],
    source_types: List[str],
    *,
    mode: str,
    scale: float,
    hard_sources: set[str],
    near_sources: set[str],
    hard_margin: float,
    near_margin: float,
    near_weight: float,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    traj_progress = _trajectory_progress_value(
        traj_raw,
        mode=mode,
        scale=scale,
    ).to(dtype=image_progress.dtype)
    align_error = (image_progress - traj_progress).abs()
    grouped: Dict[str, List[int]] = {}
    for i, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(i)

    losses: List[torch.Tensor] = []
    pair_count = 0
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        pos_indices = [
            i for i in indices
            if float(labels[i].detach().item()) > 0.5
        ]
        if not pos_indices:
            continue
        pos_idx = pos_indices[0]
        pos_error = align_error[pos_idx]
        for idx in indices:
            if idx == pos_idx:
                continue
            source = str(source_types[idx])
            if source in hard_sources:
                margin = hard_margin
                weight = 1.0
            elif source in near_sources:
                if near_margin <= 0.0 or near_weight <= 0.0:
                    continue
                margin = near_margin
                weight = near_weight
            else:
                margin = hard_margin
                weight = 1.0
            losses.append(F.relu(float(margin) - (align_error[idx] - pos_error)) * weight)
            pair_count += 1
    if not losses:
        return image_progress.sum() * 0.0, 0, align_error.detach()
    return torch.stack(losses).mean(), pair_count, align_error.detach()


def _progress_alignment_soft_target_rank_loss(
    image_progress: torch.Tensor,
    traj_raw: torch.Tensor,
    targets: torch.Tensor,
    group_ids: List[str],
    source_types: List[str],
    source_weights: Dict[str, float],
    *,
    mode: str,
    scale: float,
    margin: float,
    min_target_gap: float,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    traj_progress = _trajectory_progress_value(
        traj_raw,
        mode=mode,
        scale=scale,
    ).to(dtype=image_progress.dtype)
    align_error = (image_progress - traj_progress).abs()
    grouped: Dict[str, List[int]] = {}
    for i, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(i)

    losses: List[torch.Tensor] = []
    pair_count = 0
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=image_progress.device)
        g_error = align_error.index_select(0, idx)
        g_targets = targets.index_select(0, idx)
        target_gap = g_targets[:, None] - g_targets[None, :]
        pair_mask = target_gap > float(min_target_gap)
        if not bool(pair_mask.any()):
            continue
        # If candidate i has a higher consistency target than j, its visual
        # progress error should be lower than j's by a margin scaled by target gap.
        error_gap = g_error[None, :] - g_error[:, None]
        pair_loss = F.relu(float(margin) * target_gap - error_gap)
        weights = _weights_from_strings(
            [source_types[i] for i in indices],
            source_weights,
            image_progress.device,
            default=1.0,
        )
        pair_weight = weights[None, :].expand_as(pair_loss)
        losses.append((pair_loss[pair_mask] * pair_weight[pair_mask]).mean())
        pair_count += int(pair_mask.float().sum().detach().item())

    if not losses:
        return image_progress.sum() * 0.0, 0, align_error.detach()
    return torch.stack(losses).mean(), pair_count, align_error.detach()


def run_consistency_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: Dict[str, Any],
    training: bool,
    max_steps: int = 0,
) -> Dict[str, float]:
    """Consistency Critic 的单 epoch 训练/验证 - 多维度评估"""
    model.train(training)

    # 多维度损失权重
    lambda_c = float(cfg.get("lambda_consistency", 1.0))
    lambda_v = float(cfg.get("lambda_validity", 0.5))
    lambda_speed = float(cfg.get("lambda_speed_consistency", 0.3))
    lambda_steering = float(cfg.get("lambda_steering_consistency", 0.3))
    lambda_progress = float(cfg.get("lambda_progress_consistency", 0.2))
    lambda_temporal = float(cfg.get("lambda_temporal_coherence", 0.2))
    lambda_group_rank = float(cfg.get("lambda_group_ranking", 0.0))
    lambda_group_hard_negative = float(
        cfg.get("lambda_group_hard_negative", 0.0)
    )
    lambda_progress_alignment = float(cfg.get("lambda_progress_alignment", 0.0))
    progress_alignment_target_mode = str(
        cfg.get("progress_alignment_target_mode", "hard")
    )
    progress_alignment_mode = str(
        cfg.get("progress_alignment_mode", "final_displacement")
    )
    progress_alignment_scale = float(cfg.get("progress_alignment_scale", 40.0))
    progress_alignment_min_target_gap = float(
        cfg.get("progress_alignment_min_target_gap", 0.08)
    )
    progress_alignment_hard_margin = float(
        cfg.get("progress_alignment_hard_margin", 0.05)
    )
    progress_alignment_near_margin = float(
        cfg.get("progress_alignment_near_margin", 0.01)
    )
    progress_alignment_near_weight = float(
        cfg.get("progress_alignment_near_weight", 0.15)
    )
    progress_alignment_hard_sources = {
        str(item)
        for item in cfg.get(
            "progress_alignment_hard_sources",
            ["image_swap", "time_shift_future", "traj_swap", "reverse_traj"],
        )
    }
    progress_alignment_near_sources = {
        str(item)
        for item in cfg.get(
            "progress_alignment_near_sources",
            ["perturb_speed", "perturb_lateral", "perturb_heading"],
        )
    }
    lambda_future_traj_geometry = float(
        cfg.get("lambda_future_traj_geometry", 0.0)
    )
    lambda_path_grounding = float(cfg.get("lambda_path_grounding", 0.0))
    path_grounding_margin = float(cfg.get("path_grounding_margin", 0.02))
    path_grounding_sky_weight = float(cfg.get("path_grounding_sky_weight", 1.0))
    lambda_path_sky_contrast = float(cfg.get("lambda_path_sky_contrast", 0.0))
    path_sky_contrast_margin = float(cfg.get("path_sky_contrast_margin", 0.02))
    path_grounding_path_width = float(cfg.get("path_grounding_path_width", 0.10))
    path_grounding_sky_ratio = float(cfg.get("path_grounding_sky_ratio", 0.25))
    path_grounding_positive_only = bool(cfg.get("path_grounding_positive_only", True))
    lambda_trajectory_specific_grounding = float(
        cfg.get("lambda_trajectory_specific_grounding", 0.0)
    )
    trajectory_specific_margin = float(
        cfg.get("trajectory_specific_grounding_margin", 0.01)
    )
    trajectory_specific_exclusive = bool(
        cfg.get("trajectory_specific_grounding_exclusive", False)
    )
    trajectory_specific_wrong_selection = str(
        cfg.get("trajectory_specific_wrong_selection", "trajectory_distance")
    )
    path_grounding_trajectory_mode = str(
        cfg.get("path_grounding_trajectory_mode", "cumulative")
    )
    path_grounding_projection_mode = str(
        cfg.get("path_grounding_projection_mode", "relative")
    )
    path_grounding_forward_m = float(cfg.get("path_grounding_forward_m", 40.0))
    path_grounding_lateral_m = float(cfg.get("path_grounding_lateral_m", 10.0))
    path_grounding_score_key = str(
        cfg.get("path_grounding_score_key", "consistency_logit")
    )
    trajectory_specific_score_key = str(
        cfg.get("trajectory_specific_grounding_score_key", path_grounding_score_key)
    )
    lambda_path_evidence_consistency = float(
        cfg.get("lambda_path_evidence_consistency", 0.0)
    )
    lambda_history_counterfactual = float(
        cfg.get("lambda_history_counterfactual", 0.0)
    )
    history_counterfactual_margin = float(
        cfg.get("history_counterfactual_margin", 0.05)
    )
    history_counterfactual_score_key = str(
        cfg.get("history_counterfactual_score_key", "consistency_logit")
    )
    history_counterfactual_positive_only = bool(
        cfg.get("history_counterfactual_positive_only", True)
    )
    history_counterfactual_swap_ego = bool(
        cfg.get("history_counterfactual_swap_ego", False)
    )
    lambda_motion_rule_attribute = float(
        cfg.get("lambda_motion_rule_attribute", 0.0)
    )
    lambda_motion_rule_match = float(cfg.get("lambda_motion_rule_match", 0.0))
    lambda_motion_rule_rank = float(cfg.get("lambda_motion_rule_rank", 0.0))
    motion_rule_attribute_min_target = float(
        cfg.get("motion_rule_attribute_min_target", 0.8)
    )
    motion_rule_match_use_soft_targets = bool(
        cfg.get("motion_rule_match_use_soft_targets", True)
    )
    motion_rule_attribute_weight_mode = str(
        cfg.get("motion_rule_attribute_weight_mode", "threshold")
    )
    motion_rule_rank_margin = float(cfg.get("motion_rule_rank_margin", 0.12))
    motion_rule_rank_min_target_gap = float(
        cfg.get("motion_rule_rank_min_target_gap", 0.10)
    )
    lambda_trajectory_reasonableness = float(
        cfg.get("lambda_trajectory_reasonableness", 0.0)
    )
    lambda_image_trajectory_consistency_head = float(
        cfg.get("lambda_image_trajectory_consistency_head", 0.0)
    )
    candidate_quality_score_weight = float(
        cfg.get("candidate_quality_score_weight", 0.0)
    )
    candidate_quality_target_mode = str(
        cfg.get("candidate_quality_target_mode", "blend")
    )
    candidate_quality_allowed_sources = {
        str(item)
        for item in cfg.get(
            "candidate_quality_allowed_sources",
            ["gt_pos", "perturb_speed", "perturb_lateral", "perturb_heading"],
        )
    }
    consistency_supervision_sources = {
        str(item)
        for item in cfg.get("consistency_supervision_sources", [])
    }
    consistency_ignored_sources = {
        str(item)
        for item in cfg.get("consistency_ignored_sources", [])
    }
    auxiliary_consistency_supervision_sources = {
        str(item)
        for item in cfg.get("auxiliary_consistency_supervision_sources", [])
    }
    auxiliary_consistency_ignored_sources = {
        str(item)
        for item in cfg.get("auxiliary_consistency_ignored_sources", [])
    }
    trajectory_reasonableness_allowed_sources = {
        str(item)
        for item in cfg.get(
            "trajectory_reasonableness_allowed_sources",
            candidate_quality_allowed_sources,
        )
    }
    trajectory_reasonableness_ignored_sources = {
        str(item)
        for item in cfg.get("trajectory_reasonableness_ignored_sources", [])
    }
    trajectory_reasonableness_source_weights = {
        str(k): float(v)
        for k, v in cfg.get(
            "trajectory_reasonableness_source_weights", {}
        ).items()
    }
    group_rank_margin = float(cfg.get("group_ranking_margin", 0.2))
    group_ranking_target_mode = str(
        cfg.get("group_ranking_target_mode", "hard")
    )
    group_ranking_min_target_gap = float(
        cfg.get("group_ranking_min_target_gap", 0.08)
    )
    group_hard_negative_margin = float(
        cfg.get("group_hard_negative_margin", 0.05)
    )
    group_hard_negative_target = float(
        cfg.get("group_hard_negative_target", 0.30)
    )
    group_hard_negative_target_mode = str(
        cfg.get("group_hard_negative_target_mode", "hard")
    )
    group_hard_negative_max_target = float(
        cfg.get("group_hard_negative_max_target", 0.30)
    )
    group_hard_negative_sources = {
        str(item)
        for item in cfg.get(
            "group_hard_negative_sources",
            ["image_swap", "time_shift_future", "traj_swap", "reverse_traj"],
        )
    }
    source_weight_cfg = {
        str(k): float(v)
        for k, v in cfg.get("consistency_source_weights", {}).items()
    }
    source_margin_cfg = {
        str(k): float(v)
        for k, v in cfg.get("consistency_source_margins", {}).items()
    }
    quality_weight_cfg = {
        str(k): float(v)
        for k, v in cfg.get("label_quality_weights", {}).items()
    }
    source_soft_target_cfg = {
        str(k): float(v)
        for k, v in cfg.get("consistency_source_soft_targets", {}).items()
    }
    consistency_soft_target_mode = str(
        cfg.get("consistency_soft_target_mode", "fixed_source")
    )
    auxiliary_consistency_target_mode = str(
        cfg.get("auxiliary_consistency_target_mode", "hard")
    )
    consistency_positive_mask_mode = str(
        cfg.get("consistency_positive_mask_mode", "hard")
    )
    soft_positive_target_threshold = float(
        cfg.get("soft_positive_target_threshold", 0.55)
    )
    soft_target_near_sources = {
        str(item)
        for item in cfg.get(
            "consistency_soft_target_near_sources",
            ["perturb_speed", "perturb_lateral", "perturb_heading"],
        )
    }
    soft_target_hard_negative_sources = {
        str(item)
        for item in cfg.get(
            "consistency_soft_target_hard_negative_sources",
            ["image_swap", "time_shift_future", "traj_swap", "reverse_traj"],
        )
    }
    soft_target_gamma = float(cfg.get("soft_target_gamma", 1.5))
    soft_target_min = float(cfg.get("soft_target_min", 0.05))
    soft_target_distance_tau = float(cfg.get("soft_target_distance_tau", 2.0))
    supported_set_target_mode = str(
        cfg.get("supported_set_target_mode", "disabled")
    )
    supported_set_enabled = supported_set_target_mode not in {
        "disabled",
        "off",
        "none",
    }
    supported_set_positive_sources = {
        str(item)
        for item in cfg.get(
            "supported_set_positive_sources",
            ["perturb_speed", "perturb_lateral", "perturb_heading"],
        )
    }
    supported_set_unknown_sources = {
        str(item)
        for item in cfg.get(
            "supported_set_unknown_sources",
            ["perturb_speed", "perturb_lateral", "perturb_heading"],
        )
    }
    supported_set_hard_negative_sources = {
        str(item)
        for item in cfg.get(
            "supported_set_hard_negative_sources",
            [
                "image_swap",
                "time_shift_future",
                "traj_swap",
                "reverse_traj",
                "high_pdm_image_mismatch",
            ],
        )
    }
    supported_set_positive_quality_threshold = float(
        cfg.get("supported_set_positive_quality_threshold", 0.78)
    )
    supported_set_unknown_quality_threshold = float(
        cfg.get("supported_set_unknown_quality_threshold", 0.45)
    )
    supported_set_positive_target = float(
        cfg.get("supported_set_positive_target", 0.86)
    )
    supported_set_unknown_target = float(
        cfg.get("supported_set_unknown_target", 0.50)
    )
    supported_set_use_quality_as_target = bool(
        cfg.get("supported_set_use_quality_as_target", True)
    )
    supported_set_require_geometry_for_positive = bool(
        cfg.get("supported_set_require_geometry_for_positive", False)
    )
    supported_set_positive_path_iou_threshold = float(
        cfg.get("supported_set_positive_path_iou_threshold", 0.35)
    )
    supported_set_positive_distance_threshold = float(
        cfg.get("supported_set_positive_distance_threshold", 2.5)
    )
    validity_negative_weight = float(cfg.get("validity_negative_weight", 1.0))
    consistency_class_balanced_loss = bool(
        cfg.get("consistency_class_balanced_loss", False)
    )
    consistency_negative_loss_weight = float(
        cfg.get("consistency_negative_loss_weight", 1.0)
    )
    
    # 正样本权重
    c_pw = torch.tensor(
        cfg.get("consistency_positive_weight", cfg["positive_weight"]),
        device=device,
    )
    v_pw = torch.tensor(
        cfg.get("validity_positive_weight", cfg["positive_weight"]),
        device=device,
    )
    
    # 多维度损失函数
    criterion_speed = nn.BCEWithLogitsLoss()
    criterion_steering = nn.BCEWithLogitsLoss()
    criterion_progress = nn.BCEWithLogitsLoss()
    criterion_temporal = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_c_loss = 0.0
    total_v_loss = 0.0
    total_speed_loss = 0.0
    total_steering_loss = 0.0
    total_progress_loss = 0.0
    total_temporal_loss = 0.0
    total_group_rank_loss = 0.0
    total_group_hard_negative_loss = 0.0
    total_progress_alignment_loss = 0.0
    total_progress_alignment_pairs = 0.0
    total_progress_alignment_error = 0.0
    total_future_traj_geometry_loss = 0.0
    total_path_grounding_loss = 0.0
    total_path_sky_contrast_loss = 0.0
    total_trajectory_specific_grounding_loss = 0.0
    total_trajectory_specific_positive_controls = 0.0
    total_trajectory_specific_wrong_distance = 0.0
    total_trajectory_specific_exclusive_fraction = 0.0
    total_history_counterfactual_loss = 0.0
    total_history_counterfactual_pairs = 0.0
    total_motion_rule_attribute_loss = 0.0
    total_motion_rule_match_loss = 0.0
    total_motion_rule_attribute_pairs = 0.0
    total_motion_rule_rank_loss = 0.0
    total_motion_rule_rank_pairs = 0.0
    total_trajectory_reasonableness_loss = 0.0
    total_trajectory_reasonableness_pairs = 0.0
    total_trajectory_reasonableness_abs_error = 0.0
    
    total_c_correct = 0.0
    total_v_correct = 0.0
    total_speed_correct = 0.0
    total_steering_correct = 0.0
    total_progress_correct = 0.0
    total_temporal_correct = 0.0
    total_c_tp = 0.0
    total_c_fp = 0.0
    total_c_fn = 0.0
    total_c_tn = 0.0
    total_c_pos_score = 0.0
    total_c_neg_score = 0.0
    total_c_pos_count = 0.0
    total_c_neg_count = 0.0
    
    total_samples = 0
    log_interval = int(cfg["log_interval"])
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.float16

    if training:
        for epoch_sampler in (
            getattr(loader, "sampler", None),
            getattr(loader, "batch_sampler", None),
        ):
            if hasattr(epoch_sampler, "set_epoch"):
                epoch_sampler.set_epoch(epoch)

    for step, batch in enumerate(loader, start=1):
        h_imgs = batch["history_images"].to(device, non_blocking=True)
        f_imgs = batch["future_images"].to(device, non_blocking=True)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["candidate_traj"].to(device, non_blocking=True)
        traj_raw = batch.get("candidate_traj_raw", batch["candidate_traj"]).to(
            device, non_blocking=True,
        )
        c_labels = batch["consistency_label"].to(device, non_blocking=True)
        v_labels = batch["validity_label"].to(device, non_blocking=True)
        candidate_quality_scores = batch.get("candidate_quality_score")
        if candidate_quality_scores is not None:
            candidate_quality_scores = candidate_quality_scores.to(
                device,
                non_blocking=True,
            )
        bs = c_labels.size(0)
        source_types = _metadata_list(batch, "source_type", "unknown", bs)
        group_ids = _metadata_list(batch, "group_id", "unknown", bs)
        label_qualities = _metadata_list(
            batch, "label_quality", "clean_negative", bs,
        )
        
        # 多维度标签（如果存在）
        speed_labels = batch.get("speed_consistency_label", c_labels).to(device, non_blocking=True)
        steering_labels = batch.get("steering_consistency_label", c_labels).to(device, non_blocking=True)
        progress_labels = batch.get("progress_consistency_label", c_labels).to(device, non_blocking=True)
        temporal_labels = batch.get("temporal_coherence_label", c_labels).to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out = model(h_imgs, f_imgs, ego, traj)
            
                # 多维度损失计算
                c_targets = c_labels.clone()
                if consistency_soft_target_mode in {"path_iou", "trajectory_distance"}:
                    c_targets = _continuous_consistency_soft_targets(
                        c_labels,
                        traj_raw,
                        group_ids,
                        source_types,
                        mode=consistency_soft_target_mode,
                        near_sources=soft_target_near_sources,
                        hard_negative_sources=soft_target_hard_negative_sources,
                        gamma=soft_target_gamma,
                        min_soft=soft_target_min,
                        distance_tau=soft_target_distance_tau,
                        path_width_ratio=path_grounding_path_width,
                        trajectory_mode=path_grounding_trajectory_mode,
                        projection_mode=path_grounding_projection_mode,
                        forward_m=path_grounding_forward_m,
                        lateral_m=path_grounding_lateral_m,
                    )
                elif consistency_soft_target_mode in {"fixed_source", "source", "none"} and source_soft_target_cfg:
                    for source_name, target_value in source_soft_target_cfg.items():
                        mask = torch.tensor(
                            [
                                float(src == source_name)
                                for src in source_types
                            ],
                            dtype=torch.bool,
                            device=device,
                        )
                        mask = mask & (c_labels <= 0.5)
                        if bool(mask.any()):
                            c_targets = torch.where(
                                mask,
                                torch.full_like(c_targets, target_value),
                                c_targets,
                            )
                elif consistency_soft_target_mode not in {"fixed_source", "source", "none"}:
                    raise ValueError(
                        "unknown consistency_soft_target_mode: "
                        f"{consistency_soft_target_mode!r}"
                    )
                if (
                    candidate_quality_score_weight > 0.0
                    and candidate_quality_scores is not None
                ):
                    quality_source_mask = torch.tensor(
                        [
                            src in candidate_quality_allowed_sources
                            for src in source_types
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                    quality_mask = (
                        torch.isfinite(candidate_quality_scores)
                        & quality_source_mask
                    )
                    if bool(quality_mask.any()):
                        quality_targets = candidate_quality_scores.clamp(0.0, 1.0)
                        blend = min(max(candidate_quality_score_weight, 0.0), 1.0)
                        if candidate_quality_target_mode == "blend":
                            blended_targets = (
                                (1.0 - blend) * c_targets
                                + blend * quality_targets
                            )
                        elif candidate_quality_target_mode == "override":
                            blended_targets = quality_targets
                        elif candidate_quality_target_mode == "max":
                            blended_targets = torch.maximum(
                                c_targets,
                                quality_targets,
                            )
                        elif candidate_quality_target_mode == "override_non_gt_preserve_gt":
                            blended_targets = torch.where(
                                c_labels > 0.5,
                                torch.maximum(c_targets, quality_targets),
                                quality_targets,
                            )
                        else:
                            raise ValueError(
                                "unknown candidate_quality_target_mode: "
                                f"{candidate_quality_target_mode!r}"
                            )
                        c_targets = torch.where(
                            quality_mask,
                            blended_targets,
                            c_targets,
                        )
                consistency_supervision_mask = _source_mask_from_strings(
                    source_types,
                    consistency_supervision_sources,
                    consistency_ignored_sources,
                    device,
                )
                auxiliary_supervision_mask = _source_mask_from_strings(
                    source_types,
                    auxiliary_consistency_supervision_sources,
                    auxiliary_consistency_ignored_sources,
                    device,
                )
                if supported_set_enabled:
                    (
                        c_targets,
                        supported_supervision_mask,
                    ) = _supported_set_consistency_targets(
                        c_targets,
                        traj_raw,
                        candidate_quality_scores,
                        group_ids,
                        source_types,
                        consistency_supervision_mask,
                        positive_sources=supported_set_positive_sources,
                        unknown_sources=supported_set_unknown_sources,
                        hard_negative_sources=supported_set_hard_negative_sources,
                        positive_quality_threshold=supported_set_positive_quality_threshold,
                        unknown_quality_threshold=supported_set_unknown_quality_threshold,
                        positive_target=supported_set_positive_target,
                        unknown_target=supported_set_unknown_target,
                        use_quality_as_target=supported_set_use_quality_as_target,
                        require_geometry_for_positive=(
                            supported_set_require_geometry_for_positive
                        ),
                        positive_path_iou_threshold=(
                            supported_set_positive_path_iou_threshold
                        ),
                        positive_distance_threshold=(
                            supported_set_positive_distance_threshold
                        ),
                        path_width_ratio=path_grounding_path_width,
                        trajectory_mode=path_grounding_trajectory_mode,
                        projection_mode=path_grounding_projection_mode,
                        forward_m=path_grounding_forward_m,
                        lateral_m=path_grounding_lateral_m,
                    )
                    consistency_supervision_mask = supported_supervision_mask
                if auxiliary_consistency_target_mode == "soft":
                    aux_consistency_targets = c_targets.detach()
                elif auxiliary_consistency_target_mode == "hard":
                    aux_consistency_targets = c_labels
                else:
                    raise ValueError(
                        "unknown auxiliary_consistency_target_mode: "
                        f"{auxiliary_consistency_target_mode!r}"
                    )
                if consistency_positive_mask_mode == "soft":
                    positive_mask_for_aux = (
                        c_targets.detach() >= soft_positive_target_threshold
                    )
                elif consistency_positive_mask_mode == "hard":
                    positive_mask_for_aux = c_labels > 0.5
                else:
                    raise ValueError(
                        "unknown consistency_positive_mask_mode: "
                        f"{consistency_positive_mask_mode!r}"
                    )
                positive_labels_for_controls = positive_mask_for_aux.float()

                c_loss_each = F.binary_cross_entropy_with_logits(
                    out["consistency_logit"],
                    c_targets,
                    pos_weight=c_pw,
                    reduction="none",
                )
                c_weights = _weights_from_strings(source_types, source_weight_cfg, device)
                quality_weights = _weights_from_strings(
                    label_qualities, quality_weight_cfg, device,
                )
                c_weights = (
                    c_weights
                    * quality_weights
                    * consistency_supervision_mask.float()
                )
                if consistency_class_balanced_loss:
                    pos_mask = positive_mask_for_aux
                    neg_mask = ~pos_mask
                    pos_loss = (c_loss_each * c_weights * pos_mask.float()).sum()
                    pos_loss = pos_loss / (
                        (c_weights * pos_mask.float()).sum().clamp_min(1.0)
                    )
                    neg_loss = (c_loss_each * c_weights * neg_mask.float()).sum()
                    neg_loss = neg_loss / (
                        (c_weights * neg_mask.float()).sum().clamp_min(1.0)
                    )
                    if bool(pos_mask.any()) and bool(neg_mask.any()):
                        loss_c = (
                            pos_loss
                            + consistency_negative_loss_weight * neg_loss
                        ) / (1.0 + consistency_negative_loss_weight)
                    elif bool(pos_mask.any()):
                        loss_c = pos_loss
                    else:
                        loss_c = neg_loss
                else:
                    loss_c = (
                        (c_loss_each * c_weights).sum()
                        / c_weights.sum().clamp_min(1.0)
                    )
                if (
                    lambda_image_trajectory_consistency_head > 0.0
                    and "image_trajectory_consistency_logit" in out
                ):
                    image_c_loss_each = F.binary_cross_entropy_with_logits(
                        out["image_trajectory_consistency_logit"],
                        c_targets.detach(),
                        reduction="none",
                    )
                    loss_image_trajectory_consistency_head = (
                        (image_c_loss_each * c_weights).sum()
                        / c_weights.sum().clamp_min(1.0)
                    )
                else:
                    loss_image_trajectory_consistency_head = (
                        out["consistency_logit"].sum() * 0.0
                    )
                v_loss_each = F.binary_cross_entropy_with_logits(
                    out["validity_logit"],
                    v_labels,
                    pos_weight=v_pw,
                    reduction="none",
                )
                v_weights = torch.where(
                    v_labels > 0.5,
                    torch.ones_like(v_labels),
                    torch.full_like(v_labels, validity_negative_weight),
                )
                loss_v = (
                    (v_loss_each * v_weights).sum()
                    / v_weights.sum().clamp_min(1.0)
                )
                if (
                    lambda_trajectory_reasonableness > 0.0
                    and "trajectory_reasonableness_logit" in out
                    and candidate_quality_scores is not None
                ):
                    reason_source_mask = _source_mask_from_strings(
                        source_types,
                        trajectory_reasonableness_allowed_sources,
                        trajectory_reasonableness_ignored_sources,
                        device,
                    )
                    reason_mask = (
                        torch.isfinite(candidate_quality_scores)
                        & reason_source_mask
                    )
                    if bool(reason_mask.any()):
                        reason_targets = torch.where(
                            torch.isfinite(candidate_quality_scores),
                            candidate_quality_scores.clamp(0.0, 1.0),
                            torch.zeros_like(candidate_quality_scores),
                        )
                        reason_loss_each = F.binary_cross_entropy_with_logits(
                            out["trajectory_reasonableness_logit"],
                            reason_targets,
                            reduction="none",
                        )
                        reason_weights = _weights_from_strings(
                            source_types,
                            trajectory_reasonableness_source_weights,
                            device,
                        )
                        reason_weights = reason_weights * reason_mask.float()
                        loss_trajectory_reasonableness = (
                            (reason_loss_each * reason_weights).sum()
                            / reason_weights.sum().clamp_min(1.0)
                        )
                        trajectory_reasonableness_pair_count = (
                            reason_mask.float().sum()
                        )
                        trajectory_reasonableness_abs_error = (
                            (
                                torch.sigmoid(
                                    out["trajectory_reasonableness_logit"]
                                )
                                - reason_targets
                            ).abs()
                            * reason_mask.float()
                        ).sum()
                    else:
                        loss_trajectory_reasonableness = (
                            out["consistency_logit"].sum() * 0.0
                        )
                        trajectory_reasonableness_pair_count = (
                            out["consistency_logit"].sum() * 0.0
                        )
                        trajectory_reasonableness_abs_error = (
                            out["consistency_logit"].sum() * 0.0
                        )
                else:
                    loss_trajectory_reasonableness = (
                        out["consistency_logit"].sum() * 0.0
                    )
                    trajectory_reasonableness_pair_count = (
                        out["consistency_logit"].sum() * 0.0
                    )
                    trajectory_reasonableness_abs_error = (
                        out["consistency_logit"].sum() * 0.0
                    )
                loss_speed = criterion_speed(out["speed_consistency_logit"], speed_labels)
                loss_steering = criterion_steering(out["steering_consistency_logit"], steering_labels)
                loss_progress = criterion_progress(out["progress_consistency_logit"], progress_labels)
                loss_temporal = criterion_temporal(out["temporal_coherence_logit"], temporal_labels)
                if group_ranking_target_mode == "soft":
                    loss_group_rank, _rank_groups = _group_soft_target_ranking_loss(
                        out["consistency_logit"],
                        c_targets.detach(),
                        group_ids,
                        source_types,
                        source_weight_cfg,
                        margin=group_rank_margin,
                        min_target_gap=group_ranking_min_target_gap,
                        eligible_mask=consistency_supervision_mask.detach(),
                    )
                elif group_ranking_target_mode == "hard":
                    loss_group_rank, _rank_groups = _group_ranking_loss(
                        out["consistency_logit"],
                        c_labels,
                        group_ids,
                        source_types,
                        source_weight_cfg,
                        source_margin_cfg,
                        group_rank_margin,
                    )
                else:
                    raise ValueError(
                        "unknown group_ranking_target_mode: "
                        f"{group_ranking_target_mode!r}"
                    )
                if group_hard_negative_target_mode == "soft":
                    (
                        loss_group_hard_negative,
                        _hard_neg_groups,
                    ) = _group_hard_negative_loss(
                        out["consistency_logit"],
                        c_labels,
                        group_ids,
                        source_types,
                        source_weight_cfg,
                        group_hard_negative_margin,
                        group_hard_negative_target,
                        targets=c_targets.detach(),
                        hard_sources=group_hard_negative_sources,
                        max_target=group_hard_negative_max_target,
                    )
                elif group_hard_negative_target_mode == "hard":
                    (
                        loss_group_hard_negative,
                        _hard_neg_groups,
                    ) = _group_hard_negative_loss(
                        out["consistency_logit"],
                        c_labels,
                        group_ids,
                        source_types,
                        source_weight_cfg,
                        group_hard_negative_margin,
                        group_hard_negative_target,
                    )
                else:
                    raise ValueError(
                        "unknown group_hard_negative_target_mode: "
                        f"{group_hard_negative_target_mode!r}"
                    )
                if (
                    lambda_progress_alignment > 0.0
                    and "progress_alignment_value" in out
                ):
                    if progress_alignment_target_mode == "soft":
                        (
                            loss_progress_alignment,
                            progress_alignment_pairs,
                            progress_alignment_error,
                        ) = _progress_alignment_soft_target_rank_loss(
                            out["progress_alignment_value"],
                            traj_raw,
                            c_targets.detach(),
                            group_ids,
                            source_types,
                            source_weight_cfg,
                            mode=progress_alignment_mode,
                            scale=progress_alignment_scale,
                            margin=progress_alignment_hard_margin,
                            min_target_gap=progress_alignment_min_target_gap,
                        )
                    elif progress_alignment_target_mode == "hard":
                        (
                            loss_progress_alignment,
                            progress_alignment_pairs,
                            progress_alignment_error,
                        ) = _progress_alignment_rank_loss(
                            out["progress_alignment_value"],
                            traj_raw,
                            c_labels,
                            group_ids,
                            source_types,
                            mode=progress_alignment_mode,
                            scale=progress_alignment_scale,
                            hard_sources=progress_alignment_hard_sources,
                            near_sources=progress_alignment_near_sources,
                            hard_margin=progress_alignment_hard_margin,
                            near_margin=progress_alignment_near_margin,
                            near_weight=progress_alignment_near_weight,
                        )
                    else:
                        raise ValueError(
                            "unknown progress_alignment_target_mode: "
                            f"{progress_alignment_target_mode!r}"
                        )
                    progress_alignment_pair_count = torch.tensor(
                        float(progress_alignment_pairs),
                        device=device,
                    )
                    progress_alignment_error_sum = progress_alignment_error.sum()
                else:
                    loss_progress_alignment = out["consistency_logit"].sum() * 0.0
                    progress_alignment_pair_count = out["consistency_logit"].sum() * 0.0
                    progress_alignment_error_sum = out["consistency_logit"].sum() * 0.0
                if (
                    lambda_future_traj_geometry > 0.0
                    and "future_traj_geometry_pred" in out
                    and "future_traj_geometry_target" in out
                ):
                    pos_mask = positive_mask_for_aux
                    if bool(pos_mask.any()):
                        loss_future_traj_geometry = F.smooth_l1_loss(
                            out["future_traj_geometry_pred"][pos_mask],
                            out["future_traj_geometry_target"][pos_mask],
                        )
                    else:
                        loss_future_traj_geometry = out[
                            "future_traj_geometry_pred"
                        ].sum() * 0.0
                else:
                    loss_future_traj_geometry = out[
                        "consistency_logit"
                    ].sum() * 0.0
                if "future_consistency_evidence_logit" in out:
                    loss_future_consistency_evidence = _masked_bce_with_logits(
                        out["future_consistency_evidence_logit"],
                        aux_consistency_targets,
                        auxiliary_supervision_mask,
                    )
                else:
                    loss_future_consistency_evidence = out["consistency_logit"].sum() * 0.0
                if (
                    lambda_path_evidence_consistency > 0.0
                    and "path_evidence_logit" in out
                ):
                    loss_path_evidence_consistency = _masked_bce_with_logits(
                        out["path_evidence_logit"],
                        aux_consistency_targets,
                        auxiliary_supervision_mask,
                    )
                else:
                    loss_path_evidence_consistency = out["consistency_logit"].sum() * 0.0
                if (
                    lambda_motion_rule_attribute > 0.0
                    and "visual_motion_rule_pred" in out
                    and "traj_motion_rule_target" in out
                ):
                    if motion_rule_attribute_weight_mode == "soft_target":
                        attr_weights = c_targets.detach().clamp(0.0, 1.0)
                        motion_rule_mask = attr_weights > 0.0
                    elif motion_rule_attribute_weight_mode == "threshold":
                        motion_rule_mask = c_targets >= motion_rule_attribute_min_target
                        attr_weights = motion_rule_mask.float()
                    else:
                        raise ValueError(
                            "unknown motion_rule_attribute_weight_mode: "
                            f"{motion_rule_attribute_weight_mode!r}"
                        )
                    if bool(motion_rule_mask.any()):
                        attr_loss_each = F.smooth_l1_loss(
                            out["visual_motion_rule_pred"],
                            out["traj_motion_rule_target"].detach(),
                            reduction="none",
                        ).mean(dim=-1)
                        loss_motion_rule_attribute = (
                            attr_loss_each[motion_rule_mask]
                            * attr_weights[motion_rule_mask]
                        ).sum() / attr_weights[motion_rule_mask].sum().clamp_min(1e-4)
                        motion_rule_attribute_pair_count = (
                            attr_weights[motion_rule_mask] > 0.0
                        ).float().sum()
                    else:
                        loss_motion_rule_attribute = out["consistency_logit"].sum() * 0.0
                        motion_rule_attribute_pair_count = (
                            out["consistency_logit"].sum() * 0.0
                        )
                else:
                    loss_motion_rule_attribute = out["consistency_logit"].sum() * 0.0
                    motion_rule_attribute_pair_count = (
                        out["consistency_logit"].sum() * 0.0
                    )
                if (
                    lambda_motion_rule_match > 0.0
                    and "motion_rule_match_logit" in out
                ):
                    motion_rule_match_target = (
                        c_targets if motion_rule_match_use_soft_targets else c_labels
                    )
                    loss_motion_rule_match = _masked_bce_with_logits(
                        out["motion_rule_match_logit"],
                        motion_rule_match_target.detach(),
                        auxiliary_supervision_mask,
                    )
                else:
                    loss_motion_rule_match = out["consistency_logit"].sum() * 0.0
                if (
                    lambda_motion_rule_rank > 0.0
                    and "motion_rule_match_logit" in out
                ):
                    (
                        loss_motion_rule_rank,
                        motion_rule_rank_pair_count_int,
                    ) = _group_soft_target_ranking_loss(
                        out["motion_rule_match_logit"],
                        c_targets.detach(),
                        group_ids,
                        source_types,
                        source_weight_cfg,
                        margin=motion_rule_rank_margin,
                        min_target_gap=motion_rule_rank_min_target_gap,
                        eligible_mask=auxiliary_supervision_mask.detach(),
                    )
                    motion_rule_rank_pair_count = torch.tensor(
                        float(motion_rule_rank_pair_count_int),
                        device=device,
                    )
                else:
                    loss_motion_rule_rank = out["consistency_logit"].sum() * 0.0
                    motion_rule_rank_pair_count = (
                        out["consistency_logit"].sum() * 0.0
                    )
                if (
                    training
                    and lambda_history_counterfactual > 0.0
                    and bs > 1
                ):
                    if history_counterfactual_score_key not in out:
                        raise KeyError(
                            "history counterfactual score key missing from "
                            f"model outputs: {history_counterfactual_score_key}"
                        )
                    perm = torch.roll(torch.arange(bs, device=device), shifts=1)
                    cf_h_imgs = h_imgs.index_select(0, perm)
                    cf_ego = (
                        ego.index_select(0, perm)
                        if history_counterfactual_swap_ego
                        else ego
                    )
                    out_hist_cf = model(cf_h_imgs, f_imgs, cf_ego, traj)
                    if history_counterfactual_score_key not in out_hist_cf:
                        raise KeyError(
                            "history counterfactual score key missing from "
                            "counterfactual outputs: "
                            f"{history_counterfactual_score_key}"
                        )
                    if history_counterfactual_positive_only:
                        history_cf_mask = positive_mask_for_aux
                    else:
                        history_cf_mask = torch.ones_like(c_labels, dtype=torch.bool)
                    if bool(history_cf_mask.any()):
                        orig_cf_score = torch.sigmoid(
                            out[history_counterfactual_score_key]
                        )
                        wrong_history_score = torch.sigmoid(
                            out_hist_cf[history_counterfactual_score_key]
                        )
                        loss_history_counterfactual = F.relu(
                            history_counterfactual_margin
                            - (
                                orig_cf_score[history_cf_mask]
                                - wrong_history_score[history_cf_mask]
                            )
                        ).mean()
                        history_counterfactual_pair_count = history_cf_mask.float().sum()
                    else:
                        loss_history_counterfactual = (
                            out["consistency_logit"].sum() * 0.0
                        )
                        history_counterfactual_pair_count = (
                            out["consistency_logit"].sum() * 0.0
                        )
                else:
                    loss_history_counterfactual = out["consistency_logit"].sum() * 0.0
                    history_counterfactual_pair_count = (
                        out["consistency_logit"].sum() * 0.0
                    )
                if "physics_support_logit" in out:
                    loss_physics_support = _masked_bce_with_logits(
                        out["physics_support_logit"],
                        aux_consistency_targets,
                        auxiliary_supervision_mask,
                    )
                    loss_action_support = _masked_bce_with_logits(
                        out["action_support_logit"],
                        aux_consistency_targets,
                        auxiliary_supervision_mask,
                    )
                    loss_future_support = _masked_bce_with_logits(
                        out["future_support_logit"],
                        aux_consistency_targets,
                        auxiliary_supervision_mask,
                    )
                    loss_consistency_fuse = _masked_bce_with_logits(
                        out["consistency_fuse_logit"],
                        aux_consistency_targets,
                        auxiliary_supervision_mask,
                    )
                else:
                    zero = out["consistency_logit"].sum() * 0.0
                    loss_physics_support = zero
                    loss_action_support = zero
                    loss_future_support = zero
                    loss_consistency_fuse = zero
                if training and (
                    lambda_path_grounding > 0.0
                    or lambda_path_sky_contrast > 0.0
                    or lambda_trajectory_specific_grounding > 0.0
                ):
                    path_f_imgs = _mask_future_for_path_grounding(
                        f_imgs,
                        traj_raw,
                        mode="path",
                        width_ratio=path_grounding_path_width,
                        sky_ratio=path_grounding_sky_ratio,
                        trajectory_mode=path_grounding_trajectory_mode,
                        projection_mode=path_grounding_projection_mode,
                        forward_m=path_grounding_forward_m,
                        lateral_m=path_grounding_lateral_m,
                    )
                    out_path = model(h_imgs, path_f_imgs, ego, traj)
                    if (
                        path_grounding_score_key not in out
                        or path_grounding_score_key not in out_path
                    ):
                        if path_grounding_score_key != "consistency_logit":
                            raise KeyError(
                                "path grounding score key missing from model outputs: "
                                f"{path_grounding_score_key}"
                            )
                        path_grounding_base_key = "consistency_logit"
                    else:
                        path_grounding_base_key = path_grounding_score_key
                    orig_score = torch.sigmoid(out[path_grounding_base_key])
                    path_score = torch.sigmoid(out_path[path_grounding_base_key])
                    if path_grounding_positive_only:
                        grounding_mask = positive_mask_for_aux
                    else:
                        grounding_mask = torch.ones_like(c_labels, dtype=torch.bool)
                    loss_path_sky_contrast = out["consistency_logit"].sum() * 0.0
                    if (
                        lambda_path_grounding > 0.0
                        or lambda_path_sky_contrast > 0.0
                    ) and bool(grounding_mask.any()):
                        sky_f_imgs = _mask_future_for_path_grounding(
                            f_imgs,
                            traj_raw,
                            mode="sky",
                            width_ratio=path_grounding_path_width,
                            sky_ratio=path_grounding_sky_ratio,
                            trajectory_mode=path_grounding_trajectory_mode,
                            projection_mode=path_grounding_projection_mode,
                            forward_m=path_grounding_forward_m,
                            lateral_m=path_grounding_lateral_m,
                        )
                        out_sky = model(h_imgs, sky_f_imgs, ego, traj)
                        if path_grounding_base_key not in out_sky:
                            if path_grounding_base_key != "consistency_logit":
                                raise KeyError(
                                    "path grounding score key missing from sky output: "
                                    f"{path_grounding_base_key}"
                                )
                            sky_score_key = "consistency_logit"
                        else:
                            sky_score_key = path_grounding_base_key
                        sky_score = torch.sigmoid(out_sky[sky_score_key])
                        orig_sel = orig_score[grounding_mask]
                        path_sel = path_score[grounding_mask]
                        sky_sel = sky_score[grounding_mask]
                        loss_path_sensitivity = F.relu(
                            path_grounding_margin - (orig_sel.detach() - path_sel)
                        ).mean()
                        loss_sky_invariance = F.smooth_l1_loss(
                            sky_sel,
                            orig_sel.detach(),
                        )
                        loss_path_grounding = (
                            loss_path_sensitivity
                            + path_grounding_sky_weight * loss_sky_invariance
                        )
                        loss_path_sky_contrast = F.relu(
                            path_sky_contrast_margin
                            - (
                                (orig_sel.detach() - path_sel)
                                - (orig_sel.detach() - sky_sel)
                            )
                        ).mean()
                    else:
                        loss_path_grounding = out["consistency_logit"].sum() * 0.0
                    if lambda_path_sky_contrast <= 0.0:
                        loss_path_sky_contrast = out["consistency_logit"].sum() * 0.0
                    if lambda_trajectory_specific_grounding > 0.0:
                        wrong_traj_raw, wrong_mask, wrong_distance = _wrong_traj_controls(
                            traj_raw,
                            positive_labels_for_controls,
                            group_ids,
                            selection=trajectory_specific_wrong_selection,
                            width_ratio=path_grounding_path_width,
                            trajectory_mode=path_grounding_trajectory_mode,
                            projection_mode=path_grounding_projection_mode,
                            forward_m=path_grounding_forward_m,
                            lateral_m=path_grounding_lateral_m,
                            mask_height=int(f_imgs.shape[-2]),
                            mask_width=int(f_imgs.shape[-1]),
                        )
                        traj_specific_mask = positive_mask_for_aux & wrong_mask
                        if bool(traj_specific_mask.any()):
                            if trajectory_specific_exclusive:
                                cand_spec_f_imgs, wrong_f_imgs, spec_fractions = (
                                    _mask_future_for_exclusive_path_grounding(
                                        f_imgs,
                                        traj_raw,
                                        wrong_traj_raw,
                                        width_ratio=path_grounding_path_width,
                                        trajectory_mode=path_grounding_trajectory_mode,
                                        projection_mode=path_grounding_projection_mode,
                                        forward_m=path_grounding_forward_m,
                                        lateral_m=path_grounding_lateral_m,
                                    )
                                )
                                out_cand_spec = model(h_imgs, cand_spec_f_imgs, ego, traj)
                                if (
                                    trajectory_specific_score_key not in out
                                    or trajectory_specific_score_key not in out_cand_spec
                                ):
                                    if trajectory_specific_score_key != "consistency_logit":
                                        raise KeyError(
                                            "trajectory-specific score key missing from "
                                            "candidate-exclusive output: "
                                            f"{trajectory_specific_score_key}"
                                        )
                                    traj_spec_score_key = "consistency_logit"
                                else:
                                    traj_spec_score_key = trajectory_specific_score_key
                                path_score_for_specific = torch.sigmoid(
                                    out_cand_spec[traj_spec_score_key]
                                )
                            else:
                                wrong_f_imgs = _mask_future_for_path_grounding(
                                    f_imgs,
                                    wrong_traj_raw,
                                    mode="path",
                                    width_ratio=path_grounding_path_width,
                                    sky_ratio=path_grounding_sky_ratio,
                                    trajectory_mode=path_grounding_trajectory_mode,
                                    projection_mode=path_grounding_projection_mode,
                                    forward_m=path_grounding_forward_m,
                                    lateral_m=path_grounding_lateral_m,
                                )
                                if (
                                    trajectory_specific_score_key not in out
                                    or trajectory_specific_score_key not in out_path
                                ):
                                    if trajectory_specific_score_key != "consistency_logit":
                                        raise KeyError(
                                            "trajectory-specific score key missing from "
                                            "path-masked output: "
                                            f"{trajectory_specific_score_key}"
                                        )
                                    traj_spec_score_key = "consistency_logit"
                                else:
                                    traj_spec_score_key = trajectory_specific_score_key
                                path_score_for_specific = torch.sigmoid(
                                    out_path[traj_spec_score_key]
                                )
                                spec_fractions = torch.zeros_like(wrong_distance)
                            out_wrong = model(h_imgs, wrong_f_imgs, ego, traj)
                            if traj_spec_score_key not in out_wrong:
                                if traj_spec_score_key != "consistency_logit":
                                    raise KeyError(
                                        "trajectory-specific score key missing from "
                                        f"wrong-path output: {traj_spec_score_key}"
                                    )
                                wrong_score_key = "consistency_logit"
                            else:
                                wrong_score_key = traj_spec_score_key
                            wrong_score = torch.sigmoid(out_wrong[wrong_score_key])
                            loss_trajectory_specific_grounding = F.relu(
                                trajectory_specific_margin
                                - (
                                    wrong_score[traj_specific_mask]
                                    - path_score_for_specific[traj_specific_mask]
                                )
                            ).mean()
                        else:
                            loss_trajectory_specific_grounding = (
                                out["consistency_logit"].sum() * 0.0
                            )
                            spec_fractions = torch.zeros_like(wrong_distance)
                        traj_specific_control_count = traj_specific_mask.float().sum()
                        traj_specific_wrong_distance = torch.where(
                            traj_specific_mask,
                            wrong_distance,
                            torch.zeros_like(wrong_distance),
                        ).sum()
                        traj_specific_exclusive_fraction = torch.where(
                            traj_specific_mask,
                            spec_fractions,
                            torch.zeros_like(spec_fractions),
                        ).sum()
                    else:
                        loss_trajectory_specific_grounding = (
                            out["consistency_logit"].sum() * 0.0
                        )
                        traj_specific_control_count = out["consistency_logit"].sum() * 0.0
                        traj_specific_wrong_distance = out["consistency_logit"].sum() * 0.0
                        traj_specific_exclusive_fraction = out["consistency_logit"].sum() * 0.0
                else:
                    loss_path_grounding = out["consistency_logit"].sum() * 0.0
                    loss_path_sky_contrast = out["consistency_logit"].sum() * 0.0
                    loss_trajectory_specific_grounding = (
                        out["consistency_logit"].sum() * 0.0
                    )
                    traj_specific_control_count = out["consistency_logit"].sum() * 0.0
                    traj_specific_wrong_distance = out["consistency_logit"].sum() * 0.0
                    traj_specific_exclusive_fraction = out["consistency_logit"].sum() * 0.0
             
                # 加权组合
                loss = (lambda_c * loss_c + 
                       lambda_v * loss_v + 
                       lambda_speed * loss_speed +
                       lambda_steering * loss_steering +
                       lambda_progress * loss_progress +
                       lambda_temporal * loss_temporal +
                       lambda_group_rank * loss_group_rank +
                       lambda_group_hard_negative * loss_group_hard_negative +
                       lambda_progress_alignment * loss_progress_alignment +
                       lambda_future_traj_geometry * loss_future_traj_geometry +
                       lambda_path_grounding * loss_path_grounding +
                       lambda_path_sky_contrast * loss_path_sky_contrast +
                       lambda_trajectory_specific_grounding * loss_trajectory_specific_grounding +
                       float(cfg.get("lambda_future_consistency_evidence", 0.0)) * loss_future_consistency_evidence +
                       lambda_path_evidence_consistency * loss_path_evidence_consistency +
                       lambda_history_counterfactual * loss_history_counterfactual +
                       lambda_motion_rule_attribute * loss_motion_rule_attribute +
                       lambda_motion_rule_match * loss_motion_rule_match +
                       lambda_motion_rule_rank * loss_motion_rule_rank +
                       lambda_trajectory_reasonableness * loss_trajectory_reasonableness +
                       lambda_image_trajectory_consistency_head * loss_image_trajectory_consistency_head +
                       float(cfg.get("lambda_hierarchical_consistency", 0.0)) * (
                           loss_physics_support +
                           loss_action_support +
                           loss_future_support +
                           loss_consistency_fuse
                       ))
            
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = c_labels.size(0)
        
        # 多维度准确率计算
        c_scores = torch.sigmoid(out["consistency_logit"])
        c_preds = (c_scores >= 0.5).float()
        v_preds = (torch.sigmoid(out["validity_logit"]) >= 0.5).float()
        speed_preds = (torch.sigmoid(out["speed_consistency_logit"]) >= 0.5).float()
        steering_preds = (torch.sigmoid(out["steering_consistency_logit"]) >= 0.5).float()
        progress_preds = (torch.sigmoid(out["progress_consistency_logit"]) >= 0.5).float()
        temporal_preds = (torch.sigmoid(out["temporal_coherence_logit"]) >= 0.5).float()

        total_loss += loss.detach().item() * bs
        total_c_loss += loss_c.detach().item() * bs
        total_v_loss += loss_v.detach().item() * bs
        total_speed_loss += loss_speed.detach().item() * bs
        total_steering_loss += loss_steering.detach().item() * bs
        total_progress_loss += loss_progress.detach().item() * bs
        total_temporal_loss += loss_temporal.detach().item() * bs
        total_group_rank_loss += loss_group_rank.detach().item() * bs
        total_group_hard_negative_loss += (
            loss_group_hard_negative.detach().item() * bs
        )
        total_progress_alignment_loss += loss_progress_alignment.detach().item() * bs
        total_progress_alignment_pairs += float(
            progress_alignment_pair_count.detach().item()
        )
        total_progress_alignment_error += float(
            progress_alignment_error_sum.detach().item()
        )
        total_future_traj_geometry_loss += (
            loss_future_traj_geometry.detach().item() * bs
        )
        total_path_grounding_loss += loss_path_grounding.detach().item() * bs
        total_path_sky_contrast_loss += loss_path_sky_contrast.detach().item() * bs
        total_trajectory_specific_grounding_loss += (
            loss_trajectory_specific_grounding.detach().item() * bs
        )
        total_trajectory_specific_positive_controls += float(
            traj_specific_control_count.detach().item()
        )
        total_trajectory_specific_wrong_distance += float(
            traj_specific_wrong_distance.detach().item()
        )
        total_trajectory_specific_exclusive_fraction += float(
            traj_specific_exclusive_fraction.detach().item()
        )
        total_history_counterfactual_loss += (
            loss_history_counterfactual.detach().item() * bs
        )
        total_history_counterfactual_pairs += float(
            history_counterfactual_pair_count.detach().item()
        )
        total_motion_rule_attribute_loss += (
            loss_motion_rule_attribute.detach().item() * bs
        )
        total_motion_rule_match_loss += loss_motion_rule_match.detach().item() * bs
        total_motion_rule_attribute_pairs += float(
            motion_rule_attribute_pair_count.detach().item()
        )
        total_motion_rule_rank_loss += loss_motion_rule_rank.detach().item() * bs
        total_motion_rule_rank_pairs += float(
            motion_rule_rank_pair_count.detach().item()
        )
        total_trajectory_reasonableness_loss += (
            loss_trajectory_reasonableness.detach().item() * bs
        )
        total_trajectory_reasonableness_pairs += float(
            trajectory_reasonableness_pair_count.detach().item()
        )
        total_trajectory_reasonableness_abs_error += float(
            trajectory_reasonableness_abs_error.detach().item()
        )
        
        total_c_correct += (c_preds == c_labels).float().sum().item()
        total_v_correct += (v_preds == v_labels).float().sum().item()
        total_speed_correct += (speed_preds == speed_labels).float().sum().item()
        total_steering_correct += (steering_preds == steering_labels).float().sum().item()
        total_progress_correct += (progress_preds == progress_labels).float().sum().item()
        total_temporal_correct += (temporal_preds == temporal_labels).float().sum().item()
        c_pos_mask = c_labels > 0.5
        c_neg_mask = ~c_pos_mask
        total_c_tp += ((c_preds > 0.5) & c_pos_mask).float().sum().item()
        total_c_fp += ((c_preds > 0.5) & c_neg_mask).float().sum().item()
        total_c_fn += ((c_preds <= 0.5) & c_pos_mask).float().sum().item()
        total_c_tn += ((c_preds <= 0.5) & c_neg_mask).float().sum().item()
        total_c_pos_score += c_scores[c_pos_mask].sum().item()
        total_c_neg_score += c_scores[c_neg_mask].sum().item()
        total_c_pos_count += c_pos_mask.float().sum().item()
        total_c_neg_count += c_neg_mask.float().sum().item()
        
        total_samples += bs

        if is_main_process() and step % log_interval == 0:
            phase = "Train" if training else "Val"
            print(
                f"[{phase}] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.detach().item():.4f} "
                f"c_loss={loss_c.detach().item():.4f} "
                f"v_loss={loss_v.detach().item():.4f} "
                f"rank_loss={loss_group_rank.detach().item():.4f} "
                f"hard_neg_loss={loss_group_hard_negative.detach().item():.4f} "
                f"prog_align={loss_progress_alignment.detach().item():.4f} "
                f"path_ground_loss={loss_path_grounding.detach().item():.4f} "
                f"path_sky_contrast={loss_path_sky_contrast.detach().item():.4f} "
                f"traj_spec_loss={loss_trajectory_specific_grounding.detach().item():.4f} "
                f"traj_spec_pos={traj_specific_control_count.detach().item():.0f} "
                f"traj_spec_excl={traj_specific_exclusive_fraction.detach().item():.4f} "
                f"hist_cf={loss_history_counterfactual.detach().item():.4f} "
                f"motion_attr={loss_motion_rule_attribute.detach().item():.4f} "
                f"motion_match={loss_motion_rule_match.detach().item():.4f} "
                f"motion_rank={loss_motion_rule_rank.detach().item():.4f} "
                f"reason={loss_trajectory_reasonableness.detach().item():.4f} "
                f"img_c={loss_image_trajectory_consistency_head.detach().item():.4f}",
                flush=True,
            )
        if max_steps and step >= max_steps:
            break
        if sigterm_received():
            if is_main_process():
                phase = "训练" if training else "验证"
                print(f"[WARNING] SIGTERM 中断{phase}，已完成 step={step}/{len(loader)}")
            break

    metrics = torch.tensor(
        [
            total_loss, total_c_loss, total_v_loss,
            total_speed_loss, total_steering_loss, total_progress_loss, total_temporal_loss,
            total_group_rank_loss, total_group_hard_negative_loss,
            total_progress_alignment_loss,
            total_future_traj_geometry_loss,
            total_path_grounding_loss, total_trajectory_specific_grounding_loss,
            total_path_sky_contrast_loss,
            total_c_correct, total_v_correct,
            total_speed_correct, total_steering_correct, total_progress_correct, total_temporal_correct,
            total_c_tp, total_c_fp, total_c_fn, total_c_tn,
            total_c_pos_score, total_c_neg_score,
            total_c_pos_count, total_c_neg_count,
            float(total_samples),
            total_progress_alignment_pairs,
            total_progress_alignment_error,
            total_trajectory_specific_positive_controls,
            total_trajectory_specific_wrong_distance,
            total_trajectory_specific_exclusive_fraction,
            total_history_counterfactual_loss,
            total_history_counterfactual_pairs,
            total_motion_rule_attribute_loss,
            total_motion_rule_match_loss,
            total_motion_rule_attribute_pairs,
            total_motion_rule_rank_loss,
            total_motion_rule_rank_pairs,
            total_trajectory_reasonableness_loss,
            total_trajectory_reasonableness_pairs,
            total_trajectory_reasonableness_abs_error,
        ],
        dtype=torch.float64,
        device=device,
    )
    metrics = reduce_mean(metrics)
    n = max(float(metrics[28].item()), 1.0)
    progress_alignment_pair_count = max(float(metrics[29].item()), 0.0)
    progress_alignment_error_sum = float(metrics[30].item())
    traj_spec_control_count = max(float(metrics[31].item()), 0.0)
    traj_spec_wrong_distance = float(metrics[32].item())
    traj_spec_exclusive_fraction = float(metrics[33].item())
    history_cf_pair_count = max(float(metrics[35].item()), 0.0)
    motion_rule_attribute_pair_count = max(float(metrics[38].item()), 0.0)
    motion_rule_rank_pair_count = max(float(metrics[40].item()), 0.0)
    trajectory_reasonableness_pair_count = max(float(metrics[42].item()), 0.0)
    trajectory_reasonableness_abs_error = float(metrics[43].item())
    c_tp = float(metrics[20].item())
    c_fp = float(metrics[21].item())
    c_fn = float(metrics[22].item())
    c_tn = float(metrics[23].item())
    c_pos_score_sum = float(metrics[24].item())
    c_neg_score_sum = float(metrics[25].item())
    c_pos_count = max(float(metrics[26].item()), 0.0)
    c_neg_count = max(float(metrics[27].item()), 0.0)
    c_precision = c_tp / max(c_tp + c_fp, 1.0)
    c_recall = c_tp / max(c_tp + c_fn, 1.0)
    c_tnr = c_tn / max(c_tn + c_fp, 1.0)
    c_balanced_acc = 0.5 * (c_recall + c_tnr)
    c_pos_score_mean = c_pos_score_sum / max(c_pos_count, 1.0)
    c_neg_score_mean = c_neg_score_sum / max(c_neg_count, 1.0)
    c_score_gap = c_pos_score_mean - c_neg_score_mean
    return {
        "loss": float(metrics[0].item() / n),
        "c_loss": float(metrics[1].item() / n),
        "v_loss": float(metrics[2].item() / n),
        "speed_loss": float(metrics[3].item() / n),
        "steering_loss": float(metrics[4].item() / n),
        "progress_loss": float(metrics[5].item() / n),
        "temporal_loss": float(metrics[6].item() / n),
        "group_rank_loss": float(metrics[7].item() / n),
        "group_hard_negative_loss": float(metrics[8].item() / n),
        "progress_alignment_loss": float(metrics[9].item() / n),
        "progress_alignment_pairs": progress_alignment_pair_count,
        "progress_alignment_error_mean": (
            progress_alignment_error_sum / max(n, 1.0)
        ),
        "future_traj_geometry_loss": float(metrics[10].item() / n),
        "path_grounding_loss": float(metrics[11].item() / n),
        "path_sky_contrast_loss": float(metrics[12].item() / n),
        "trajectory_specific_grounding_loss": float(metrics[13].item() / n),
        "trajectory_specific_positive_controls": traj_spec_control_count,
        "trajectory_specific_wrong_distance_mean": (
            traj_spec_wrong_distance / max(traj_spec_control_count, 1.0)
        ),
        "trajectory_specific_exclusive_fraction_mean": (
            traj_spec_exclusive_fraction / max(traj_spec_control_count, 1.0)
        ),
        "history_counterfactual_loss": float(metrics[34].item() / n),
        "history_counterfactual_pairs": history_cf_pair_count,
        "motion_rule_attribute_loss": float(metrics[36].item() / n),
        "motion_rule_match_loss": float(metrics[37].item() / n),
        "motion_rule_attribute_pairs": motion_rule_attribute_pair_count,
        "motion_rule_rank_loss": float(metrics[39].item() / n),
        "motion_rule_rank_pairs": motion_rule_rank_pair_count,
        "trajectory_reasonableness_loss": float(metrics[41].item() / n),
        "trajectory_reasonableness_pairs": trajectory_reasonableness_pair_count,
        "trajectory_reasonableness_mae": (
            trajectory_reasonableness_abs_error
            / max(trajectory_reasonableness_pair_count, 1.0)
        ),
        "c_acc": float(metrics[14].item() / n),
        "v_acc": float(metrics[15].item() / n),
        "speed_acc": float(metrics[16].item() / n),
        "steering_acc": float(metrics[17].item() / n),
        "progress_acc": float(metrics[18].item() / n),
        "temporal_acc": float(metrics[19].item() / n),
        "c_precision": c_precision,
        "c_recall": c_recall,
        "c_tnr": c_tnr,
        "c_balanced_acc": c_balanced_acc,
        "c_pos_score_mean": c_pos_score_mean,
        "c_neg_score_mean": c_neg_score_mean,
        "c_score_gap": c_score_gap,
    }


def build_dataloader(
    cfg: Dict[str, Any],
    index_path: str,
    training: bool,
    epoch: int = 0,
) -> DataLoader:
    dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=training)
    num_workers = int(cfg["num_workers"])
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg.get("persistent_workers", True),
        )
        loader_kwargs["prefetch_factor"] = int(
            cfg.get("prefetch_factor", 4),
        )
    sampler = None
    ranking_cfg = cfg.get("ranking", {})
    lambda_group_rank = float(
        cfg.get("lambda_group_ranking", ranking_cfg.get("loss_weight", 0.0)),
    )
    use_group_batches = bool(
        ranking_cfg.get("group_batches", lambda_group_rank > 0.0),
    )
    if use_group_batches:
        difficulty_cfg = cfg.get("difficulty_sampling", {})
        n_per_epoch = int(difficulty_cfg.get("num_samples_per_epoch", 0))
        if n_per_epoch <= 0:
            n_per_epoch = len(dataset.samples)
        sample_difficulties = None
        if bool(difficulty_cfg.get("enabled", False)) and training:
            from iac_difficulty_sampler import assign_difficulty

            sample_difficulties = [
                assign_difficulty(sample) for sample in dataset.samples
            ]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        batch_sampler = GroupRankingBatchSampler(
            samples=dataset.samples,
            batch_size=int(cfg["batch_size"]),
            num_samples_per_epoch=n_per_epoch,
            seed=int(cfg.get("seed", 42)),
            rank=rank,
            world_size=world_size,
            source_weights=cfg.get("consistency_source_weights", {}),
            hard_negative_sources=ranking_cfg.get(
                "hard_negative_sources",
                cfg.get("hard_negative_sources", ()),
            ),
            max_negatives_per_group=int(
                ranking_cfg.get("max_negatives_per_group", 0)
                if training
                else 0
            ),
            sample_difficulties=sample_difficulties,
            difficulty_mix=difficulty_cfg.get("mix", ()),
        )
        batch_sampler.set_epoch(epoch)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )

    # iWorld-Bench style difficulty-stratified sampling (single-GPU
    # path). When DDP is active we wrap the stratified sampler with
    # DistributedSampler-equivalent shuffling (different shuffle per
    # rank) so each rank sees a different reordering of the same
    # stratified index stream.
    difficulty_cfg = cfg.get("difficulty_sampling", {})
    use_difficulty = bool(difficulty_cfg.get("enabled", False)) and training
    if use_difficulty:
        from iac_difficulty_sampler import (
            DifficultyStratifiedSampler,
            assign_difficulty,
        )
        # Map raw index source_type → difficulty bucket
        buckets = {1: [], 2: [], 3: [], 4: []}
        for i, s in enumerate(dataset.samples):
            if s.get("consistency_label", 1) == 1:
                continue  # positives handled inside the stratified sampler
            buckets[max(1, assign_difficulty(s))].append(i)

        # num_samples_per_epoch == 0 means "auto" — fall back to
        # batch_size × 100 (≈ one epoch over 35k anchors per rank),
        # bounded by the actual dataset size. Without this fallback
        # a config like the shipped one (num_samples_per_epoch=0)
        # would silently produce a zero-length sampler and the
        # DataLoader would emit zero batches.
        cfg_n = int(difficulty_cfg.get("num_samples_per_epoch", 0))
        if cfg_n <= 0:
            cfg_n = max(len(dataset.samples), int(cfg["batch_size"]) * 100)
        n_per_epoch = cfg_n
        mix = tuple(difficulty_cfg.get("mix", (0.30, 0.30, 0.25, 0.15)))
        pos_ratio = float(difficulty_cfg.get("positive_ratio", 0.25))
        seed = int(cfg.get("seed", 42))

        base_sampler = DifficultyStratifiedSampler(
            samples=dataset.samples,
            num_samples_per_epoch=n_per_epoch,
            mix=mix,
            positive_ratio=pos_ratio,
            seed=seed,
        )
        # initialise epoch counter so external set_epoch(epoch) propagates
        base_sampler.set_epoch(epoch)

        if dist.is_available() and dist.is_initialized():
            # Custom wrapper: keep base_sampler's stratified draws but
            # apply rank-aware shuffling so each rank sees different
            # ordering while preserving the difficulty mix.
            class _RankedStratifiedSampler:
                def __init__(self, base, world_size: int, rank: int) -> None:
                    self.base = base
                    self.world_size = world_size
                    self.rank = rank

                def set_epoch(self, epoch: int) -> None:
                    self.base.set_epoch(epoch * max(self.world_size, 1) + self.rank)

                def __iter__(self):
                    import random as _r
                    indices = list(self.base)
                    rng = _r.Random(self.base.seed + self.base.epoch + self.rank)
                    rng.shuffle(indices)
                    return iter(indices)

                def __len__(self) -> int:
                    return len(self.base)

            sampler = _RankedStratifiedSampler(
                base_sampler,
                world_size=dist.get_world_size(),
                rank=dist.get_rank(),
            )
        else:
            sampler = base_sampler
    elif dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=training, drop_last=training)

    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=sampler is None and training,
        sampler=sampler,
        drop_last=training,
        **loader_kwargs,
    )


def _preflight_indices(num_items: int, max_samples: int, seed: int) -> List[int]:
    if max_samples <= 0 or num_items <= max_samples:
        return list(range(num_items))
    rng = random.Random(seed)
    picked = {0, num_items - 1}
    picked.update(rng.sample(range(num_items), max_samples - len(picked)))
    return sorted(picked)


def validate_index_image_paths(
    cfg: Dict[str, Any],
    index_paths: Sequence[str],
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return

    for index_path in index_paths:
        dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=False)
        indices = _preflight_indices(len(dataset), max_samples, int(cfg.get("seed", 42)))
        missing: List[str] = []
        checked = 0
        for idx in indices:
            sample = dataset.samples[idx]
            image_paths = (
                dataset.selected_image_paths(
                    sample, "history_images", dataset.history_num_frames,
                )
                + dataset.selected_image_paths(
                    sample, "future_images", dataset.future_num_frames,
                )
            )
            for path in image_paths:
                checked += 1
                if not path.exists():
                    missing.append(str(path))
                    if len(missing) >= 10:
                        break
            if len(missing) >= 10:
                break
        if missing:
            preview = "\n  ".join(missing)
            raise FileNotFoundError(
                f"索引图片预检失败: {index_path}\n"
                f"image_root={cfg['image_root']}\n"
                f"检查样本数={len(indices)}, 图片数={checked}\n"
                f"缺失示例:\n  {preview}"
            )
        print(
            f"[Preflight] {index_path}: "
            f"checked_samples={len(indices)} checked_images={checked}",
            flush=True,
        )


def save_checkpoint(
    work_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    best_val_loss: float,
    is_best: bool,
    tag: str = "latest",
    interrupted: bool = False,
    best_metric_name: str = "val_loss",
    best_metric_value: float | None = None,
) -> None:
    if best_metric_value is None:
        best_metric_value = best_val_loss
    state = {
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "best_val_loss": best_val_loss,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "interrupted": interrupted,
        "checkpoint_tag": tag,
    }
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / f"{tag}.pth")
    if tag == "latest" and not interrupted:
        torch.save(state, checkpoint_dir / f"epoch_{epoch}.pth")
    if tag == "latest" and is_best and not interrupted:
        torch.save(state, checkpoint_dir / "best.pth")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.work_dir is not None:
        cfg["work_dir"] = args.work_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.baseline_mode is not None:
        cfg["baseline_mode"] = args.baseline_mode
    if cfg.get("model_type") != "consistency":
        raise ValueError("新版训练入口只支持 model_type='consistency' 的 IAC 配置。")

    # 注册 SIGTERM 信号处理器，收到终止信号时优雅退出
    signal.signal(signal.SIGTERM, _sigterm_handler)

    dist_info = setup_distributed()
    set_seed(int(cfg["seed"]) + dist_info["rank"])

    device = torch.device(
        f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu"
    )
    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(work_dir / "config_snapshot.json")
    if is_main_process():
        with (work_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    if is_main_process() and int(args.preflight_samples) > 0:
        validate_index_image_paths(
            cfg,
            [cfg["train_index"], cfg["val_index"]],
            int(args.preflight_samples),
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    train_loader = build_dataloader(cfg, cfg["train_index"], training=True, epoch=0)
    val_loader = build_dataloader(cfg, cfg["val_index"], training=False)

    model = ConsistencyCriticModel(cfg).to(device)
    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]] if torch.cuda.is_available() else None,
            output_device=dist_info["local_rank"] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    optimizer_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )

    def _checkpoint_score(metrics: Dict[str, float]) -> float:
        mode = str(cfg.get("checkpoint_metric", "val_loss"))
        if mode == "val_loss":
            return -float(metrics["loss"])
        if mode == "val_c_score_gap":
            return float(metrics.get("c_score_gap", 0.0))
        if mode == "val_c_balanced_acc":
            return float(metrics.get("c_balanced_acc", 0.0))
        if mode == "val_iac_consistency":
            return (
                float(metrics.get("c_score_gap", 0.0))
                + 0.25 * float(metrics.get("c_balanced_acc", 0.0))
                - 0.05 * float(metrics.get("group_rank_loss", 0.0))
            )
        if mode == "val_iac_precision":
            return (
                0.55 * float(metrics.get("c_precision", 0.0))
                + 0.35 * float(metrics.get("c_tnr", 0.0))
                + 0.20 * float(metrics.get("c_balanced_acc", 0.0))
                + 0.10 * float(metrics.get("c_score_gap", 0.0))
                - 0.03 * float(metrics.get("group_rank_loss", 0.0))
            )
        raise ValueError(f"unknown checkpoint_metric: {mode}")

    best_val_loss = math.inf
    best_metric_name = str(cfg.get("checkpoint_metric", "val_loss"))
    best_metric_value = -math.inf
    total_epochs = int(cfg["epochs"])
    start_time = time.time()

    if is_main_process():
        print("=" * 60)
        print("NuPlan IAC Consistency Critic Training")
        print(f"Config: {cfg['_config_path']}")
        print(f"Work dir: {work_dir}")
        print(f"Device: {device}")
        print(f"World size: {dist_info['world_size']}")
        if torch.cuda.is_available():
            mem_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            print(f"GPU memory: {mem_total:.1f} GB")
        print("=" * 60)

    try:
        for epoch in range(1, total_epochs + 1):
            train_metrics = run_consistency_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                cfg=cfg,
                training=True,
                max_steps=args.max_train_steps or 0,
            )
            val_metrics = run_consistency_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                cfg=cfg,
                training=False,
                max_steps=args.max_val_steps or 0,
            )

            current_metric_value = _checkpoint_score(val_metrics)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
            is_best = current_metric_value > best_metric_value
            if is_best:
                best_metric_value = current_metric_value

            if is_main_process():
                print(
                    f"[Epoch {epoch}/{total_epochs}] "
                    f"loss={train_metrics['loss']:.4f} "
                    f"c_acc={train_metrics['c_acc']:.4f} "
                    f"c_gap={train_metrics.get('c_score_gap', 0.0):.4f} "
                    f"v_acc={train_metrics['v_acc']:.4f} "
                    f"speed_acc={train_metrics['speed_acc']:.4f} "
                    f"steering_acc={train_metrics['steering_acc']:.4f} "
                    f"progress_acc={train_metrics['progress_acc']:.4f} "
                    f"temporal_acc={train_metrics['temporal_acc']:.4f} "
                    f"rank_loss={train_metrics.get('group_rank_loss', 0.0):.4f} "
                    f"traj_spec_loss={train_metrics.get('trajectory_specific_grounding_loss', 0.0):.4f} "
                    f"traj_spec_pos={train_metrics.get('trajectory_specific_positive_controls', 0.0):.0f} "
                    f"traj_spec_dist={train_metrics.get('trajectory_specific_wrong_distance_mean', 0.0):.2f} "
                    f"traj_spec_excl={train_metrics.get('trajectory_specific_exclusive_fraction_mean', 0.0):.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_c_acc={val_metrics['c_acc']:.4f} "
                    f"val_c_bal={val_metrics.get('c_balanced_acc', 0.0):.4f} "
                    f"val_c_recall={val_metrics.get('c_recall', 0.0):.4f} "
                    f"val_c_gap={val_metrics.get('c_score_gap', 0.0):.4f} "
                    f"val_path_sky={val_metrics.get('path_sky_contrast_loss', 0.0):.4f} "
                    f"val_v_acc={val_metrics['v_acc']:.4f} "
                    f"val_rank_loss={val_metrics.get('group_rank_loss', 0.0):.4f} "
                    f"ckpt_metric={best_metric_name}:{current_metric_value:.4f}"
                )
                if epoch % int(cfg["save_interval"]) == 0:
                    save_checkpoint(
                        work_dir=work_dir,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        best_val_loss=best_val_loss,
                        is_best=is_best,
                        best_metric_name=best_metric_name,
                        best_metric_value=best_metric_value,
                    )

            # 收到 SIGTERM 时保存当前进度并退出
            if sigterm_received():
                if is_main_process():
                    print(f"[WARNING] 收到终止信号，保存 epoch={epoch} 的 interrupted checkpoint...")
                    save_checkpoint(
                        work_dir=work_dir,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        best_val_loss=best_val_loss,
                        is_best=False,
                        best_metric_name=best_metric_name,
                        best_metric_value=best_metric_value,
                        tag=f"interrupted_epoch_{epoch}",
                        interrupted=True,
                    )
                    print("[WARNING] interrupted checkpoint 已保存，训练提前退出")
                break
    except Exception as e:
        # 打印详细错误信息，包含 GPU 显存状态，便于定位 OOM 等问题
        rank = dist_info["rank"]
        print(f"\n[ERROR][rank={rank}] 训练异常: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            print(
                f"[ERROR][rank={rank}] GPU 显存: "
                f"allocated={mem_alloc:.2f}GB, reserved={mem_reserved:.2f}GB",
                flush=True,
            )
        # 异常退出前只保存带 error 标记的 checkpoint，避免误用为正常结果。
        if is_main_process():
            try:
                print("[ERROR] 尝试保存 error checkpoint...", flush=True)
                save_checkpoint(
                    work_dir=work_dir,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    best_val_loss=best_val_loss,
                    is_best=False,
                    tag=f"error_epoch_{epoch}",
                    interrupted=True,
                )
                print(f"[ERROR] error checkpoint 已保存至 {work_dir}/checkpoints/", flush=True)
            except Exception:
                print("[ERROR] error checkpoint 保存失败", flush=True)
        cleanup_distributed()
        sys.exit(1)

    if is_main_process():
        elapsed = time.time() - start_time
        print("=" * 60)
        print("Training finished")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Best metric: {best_metric_name}={best_metric_value:.4f}")
        print(f"Elapsed seconds: {elapsed:.1f}")
        print("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
