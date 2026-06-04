#!/usr/bin/env python3
"""DINOv2-backboned Consistency Critic (minimal variant).

This is the *minimal* DINOv2 integration for IAC, deliberately scoped:

  ✔ 1️⃣ DINOv2-vits14 backbone replaces the 4-layer CNN
  ✔ 3️⃣ Explicit `diff / l2_norm / cos_sim` features concatenated into
     the fusion head (zero-cost, high-yield shortcut signal)
  ✘ 2️⃣ Multi-layer fusion (start with single layer [11] — multi-layer
     adds 6×proj params and overfits on the 357k anchor set)
  ✘ 4️⃣ AvgPool(k=2) (no ablation evidence, default off)
  ✘ 5️⃣ Ridge pretrain of layer weights (nuPlan data is too small to
     justify; PDF-cited "nuScenes SROCC 0.9275" has no source)
  ✘ 6️⃣ Geometric margin-ranking reg (likely hurts the consistency
     critic; one DINOv2 forward is wasted per training step)

Inherits everything else from train.py:
  - Same ConsistencyDataset
  - Same DDP / checkpoint / SIGTERM handling
  - Same eval flow (ConsistencyCriticModel alias)
  - Same difficulty-stratified sampler (D1..D4) when config enables it

Usage::

  # Smoke
  python train_dinov2_v5_minimal.py \
    --config configs/train_dinov2_v5_minimal.py \
    --work-dir work_dirs/iac_dinov2_v5_smoke \
    --epochs 1 --batch-size 32 --max-train-steps 20

  # Full
  python train_dinov2_v5_minimal.py \
    --config configs/train_dinov2_v5_minimal.py \
    --work-dir work_dirs/iac_dinov2_v5 \
    --epochs 5 --batch-size 32
"""

from __future__ import annotations

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

# ─────────────────────────── re-use train.py primitives ───────────────────────────
# We import after defining parse_args so that "python train_dinov2_v5_minimal.py
# --help" still works without the heavy torch.hub load.

_DINOV2_MODEL_NAMES = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14")
_DINOV2_SPECS: Dict[str, Dict[str, int]] = {
    "dinov2_vits14": {"feat_dim": 384, "n_blocks": 12},
    "dinov2_vitb14": {"feat_dim": 768, "n_blocks": 12},
    "dinov2_vitl14": {"feat_dim": 1024, "n_blocks": 24},
}


def _import_train():
    """Late import so that --help works before DINOv2 is loaded."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train  # type: ignore
    return train


# ─────────────────────────── CLI ───────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINOv2 Consistency Critic (minimal v5 variant)"
    )
    p.add_argument("--config", required=True, help="Python config path")
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--baseline-mode",
        choices=["full", "no_image", "ego_only", "no_traj", "traj_only"],
        default=None,
    )
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--preflight-samples", type=int, default=128)
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume model/optimizer state from a checkpoint.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable CUDA autocast mixed precision.",
    )
    p.add_argument(
        "--dinov2-model", type=str, default=None,
        choices=list(_DINOV2_MODEL_NAMES),
    )
    p.add_argument(
        "--dinov2-freeze",
        dest="dinov2_freeze",
        action="store_true",
        default=None,
        help="Freeze DINOv2 backbone (default: use config value).",
    )
    p.add_argument(
        "--dinov2-trainable",
        dest="dinov2_trainable",
        action="store_true",
        default=False,
        help="Unfreeze DINOv2 backbone for fine-tuning.",
    )
    p.add_argument(
        "--no-dinov2",
        dest="no_dinov2",
        action="store_true",
        default=False,
        help="Disable DINOv2 backbone and use the 4-layer CNN from train.py instead. "
        "Lets you A/B the D1-D4 sampling effect with the original backbone.",
    )
    return p.parse_args()


# ─────────────────────────── DINOv2 encoder (single-layer, no AvgPool) ───────────────────────────


class DINOv2Encoder(nn.Module):
    """DINOv2 single-layer encoder. Returns a (B, out_dim) vector per image batch.

    Differences from the PDF-supplied v3 script:
      * Single layer only (default [11]).
      * No AvgPool — DINOv2 patch tokens are used as-is.
      * No Ridge pretrain — layer weights are uniform 1.0.
      * mean() pool over patch tokens (excluding the CLS token) for a
        dense single-vector representation; cheaper than concatenating
        6 layers and avoids overfitting on 357k samples.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        layer_index: int = 11,
        out_dim: int = 256,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in _DINOV2_SPECS:
            raise ValueError(
                f"Unsupported DINOv2 model '{model_name}'. "
                f"Choose from {list(_DINOV2_MODEL_NAMES)}."
            )
        spec = _DINOV2_SPECS[model_name]
        n_blocks = spec["n_blocks"]
        if layer_index < 0 or layer_index >= n_blocks:
            raise ValueError(
                f"layer_index {layer_index} out of range [0, {n_blocks - 1}]"
            )
        self.model_name = model_name
        self.layer_index = layer_index
        self.feat_dim = spec["feat_dim"]
        self.freeze = freeze

        # Prefer the already-cached torch hub checkout on servers. This avoids
        # occasional torch.hub network/trust-list stalls during evaluation.
        hub_dir = os.environ.get("DINOV2_HUB_DIR")
        if not hub_dir:
            torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
            cached_hub = torch_home / "hub" / "facebookresearch_dinov2_main"
            if cached_hub.exists():
                hub_dir = str(cached_hub)
        if hub_dir and Path(hub_dir).exists():
            self.model = torch.hub.load(hub_dir, model_name, source="local")
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # Keep DINOv2 in eval mode (BN stats frozen) even if the
            # surrounding module is in train mode.
            self.model.eval()
        return self

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 expects 14*N input; 224 is fine.
        if x.shape[-2] != 224 or x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B*T, 3, H, W) → (B*T, out_dim)."""
        x = self._resize(images)
        ctx = torch.no_grad if self.freeze else torch.enable_grad
        with ctx():
            outputs = self.model.get_intermediate_layers(
                x, n=self.layer_index + 1, return_class_token=True,
            )
        # outputs: tuple of (patch_tokens, cls_token) per layer
        # patch_tokens: (B*T, n_patches, feat_dim); cls_token: (B*T, feat_dim)
        patch_tokens, cls_token = outputs[self.layer_index]
        # mean-pool patch tokens (robust to image resizing)
        feat = patch_tokens.mean(dim=1)
        return self.proj(feat)


# ─────────────────────────── Critic model ───────────────────────────


class DINOv2ConsistencyCritic(nn.Module):
    """Minimal DINOv2 critic: single-layer backbone + explicit distance fusion.

    When ``cfg['dinov2']['enabled']`` is False the model falls back to
    the original 4-layer CNN backbone from train.py, with the same
    fusion head and explicit-distance option. This makes the trainer
    a strict superset of train.py so that running it with dinov2
    disabled is a clean A/B against train.py itself.

    Shape contract identical to train.ConsistencyCriticModel so that
    eval_dinov2_critic.py and benchmark_wam.py work without modification.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        mcfg = cfg["model"]
        dcfg = cfg.get("dinov2", {})

        img_dim = int(mcfg["image_feature_dim"])
        act_dim = int(mcfg["action_feature_dim"])
        hidden = int(mcfg["hidden_dim"])
        fusion_dim = int(mcfg.get("fusion_dim", 256))
        dropout = float(mcfg.get("dropout", 0.0))
        ego_dim = int(cfg["ego_state_dim"])
        traj_steps = int(cfg["candidate_traj_steps"])
        consistency_traj_steps = int(
            cfg.get(
                "consistency_traj_steps",
                min(int(cfg.get("future_num_frames", traj_steps)), traj_steps),
            )
        )
        traj_d = int(cfg["traj_dim"])
        self.baseline_mode = str(cfg.get("baseline_mode", "full"))
        self.consistency_traj_steps = consistency_traj_steps
        self.use_dinov2 = bool(dcfg.get("enabled", True))
        self.use_explicit_distance = bool(dcfg.get("use_explicit_distance", True))

        if self.use_dinov2:
            model_name = str(dcfg.get("model_name", "dinov2_vits14"))
            layer_index = int(dcfg.get("layer_index", 11))
            freeze = bool(dcfg.get("freeze", True))
            self.image_encoder = DINOv2Encoder(
                model_name=model_name,
                layer_index=layer_index,
                out_dim=img_dim,
                freeze=freeze,
            )
            self.history_proj = self.image_encoder.proj
            self.future_proj = self.image_encoder.proj
        else:
            # Fall back to the original 4-layer CNN backbone. We
            # import lazily so that a pure DINOv2 run never has to
            # load train.py's ConsistencyCriticModel class.
            from train import ConsistencyCriticModel as _CNNCritic  # type: ignore
            cnn = _CNNCritic(cfg)
            self.image_encoder = cnn  # for state-dict symmetry
            self.history_proj = cnn.history_proj
            self.future_proj = cnn.future_proj
            self._cnn_shared_backbone = cnn.shared_backbone

        self.consistency_traj_encoder = nn.Sequential(
            nn.Linear(consistency_traj_steps * traj_d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, act_dim),
            nn.ReLU(inplace=True),
        )
        self.validity_traj_encoder = nn.Sequential(
            nn.Linear(traj_steps * traj_d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, act_dim),
            nn.ReLU(inplace=True),
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, act_dim),
            nn.ReLU(inplace=True),
        )

        # consistency_dim = hist + fut (+ diff + l2 + cos if explicit) + traj + ego
        consistency_dim = img_dim * 2 + act_dim * 2
        if self.use_explicit_distance:
            consistency_dim += img_dim + 1 + 1

        self.shared_fusion = nn.Sequential(
            nn.Linear(consistency_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.validity_fusion = nn.Sequential(
            nn.Linear(act_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Same 6 heads as train.py.
        self.consistency_head = nn.Linear(fusion_dim, 1)
        self.speed_consistency_head = nn.Linear(fusion_dim, 1)
        self.steering_consistency_head = nn.Linear(fusion_dim, 1)
        self.progress_consistency_head = nn.Linear(fusion_dim, 1)
        self.temporal_coherence_head = nn.Linear(fusion_dim, 1)
        self.validity_head = nn.Linear(fusion_dim, 1)

    def _encode_images(
        self, images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (B, T, 3, H, W) → (B, out_dim)."""
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        if self.use_dinov2:
            feat = self.image_encoder(flat)
        else:
            feat = self._cnn_shared_backbone(flat).flatten(1)
            feat = self.history_proj(feat)
        return feat.reshape(b, t, -1).mean(dim=1)

    def forward(
        self,
        history_images: torch.Tensor,
        future_images: torch.Tensor,
        ego_state: torch.Tensor,
        candidate_traj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_hist = self._encode_images(history_images)
        z_fut = self._encode_images(future_images)
        consistency_traj = candidate_traj[:, : self.consistency_traj_steps, :]
        z_traj_cons = self.consistency_traj_encoder(consistency_traj.flatten(1))
        z_traj_val = self.validity_traj_encoder(candidate_traj.flatten(1))
        z_ego = self.ego_encoder(ego_state)

        mode = self.baseline_mode
        if mode in {"no_image", "ego_only", "traj_only"}:
            z_hist = torch.zeros_like(z_hist)
            z_fut = torch.zeros_like(z_fut)
        if mode in {"no_traj", "ego_only"}:
            z_traj_cons = torch.zeros_like(z_traj_cons)
            z_traj_val = torch.zeros_like(z_traj_val)
        if mode == "traj_only":
            z_ego = torch.zeros_like(z_ego)

        parts: List[torch.Tensor] = [z_hist, z_fut]
        if self.use_explicit_distance:
            diff = z_hist - z_fut
            l2_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
            cos_sim = F.cosine_similarity(z_hist, z_fut, dim=-1).unsqueeze(-1)
            parts.extend([diff, l2_norm, cos_sim])
        parts.extend([z_traj_cons, z_ego])
        z_all = torch.cat(parts, dim=-1)
        z_shared = self.shared_fusion(z_all)
        z_validity = self.validity_fusion(
            torch.cat([z_traj_val, z_ego], dim=-1)
        )

        return {
            "consistency_logit": self.consistency_head(z_shared).squeeze(-1),
            "speed_consistency_logit": self.speed_consistency_head(z_shared).squeeze(-1),
            "steering_consistency_logit": self.steering_consistency_head(z_shared).squeeze(-1),
            "progress_consistency_logit": self.progress_consistency_head(z_shared).squeeze(-1),
            "temporal_coherence_logit": self.temporal_coherence_head(z_shared).squeeze(-1),
            "validity_logit": self.validity_head(z_validity).squeeze(-1),
        }


# Public alias so eval_critic.py / benchmark_wam.py can use either trainer
# interchangeably. train.py's eval_critic does `from train import
# ConsistencyCriticModel`, so we set the same attribute on this module too —
# but only after train.py is imported.
ConsistencyCriticModel = DINOv2ConsistencyCritic  # late rebound below


# ─────────────────────────── main ───────────────────────────


def main() -> None:
    args = parse_args()
    train = _import_train()

    # Late-bound alias: some downstream scripts do
    # `from train import ConsistencyCriticModel` and won't see DINOv2.
    # We keep the local name consistent but do NOT mutate train.
    globals()["ConsistencyDataset"] = train.ConsistencyDataset
    globals()["load_config"] = train.load_config
    globals()["run_consistency_epoch"] = train.run_consistency_epoch
    globals()["build_dataloader"] = train.build_dataloader
    globals()["validate_index_image_paths"] = train.validate_index_image_paths
    globals()["save_checkpoint"] = train.save_checkpoint
    globals()["setup_distributed"] = train.setup_distributed
    globals()["cleanup_distributed"] = train.cleanup_distributed
    globals()["set_seed"] = train.set_seed
    globals()["is_main_process"] = train.is_main_process
    globals()["sigterm_received"] = train.sigterm_received
    globals()["_sigterm_handler"] = train._sigterm_handler

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
        raise ValueError("model_type must be 'consistency'.")
    # Apply CLI overrides for DINOv2 settings
    dcfg = cfg.setdefault("dinov2", {})
    if args.dinov2_model is not None:
        dcfg["model_name"] = args.dinov2_model
    if args.dinov2_freeze:
        dcfg["freeze"] = True
    if args.dinov2_trainable:
        dcfg["freeze"] = False
    if args.no_dinov2:
        dcfg["enabled"] = False
    if args.amp:
        cfg["amp"] = True

    signal.signal(signal.SIGTERM, _sigterm_handler)
    dist_info = setup_distributed()
    set_seed(int(cfg["seed"]) + dist_info["rank"])

    device = torch.device(
        f"cuda:{dist_info['local_rank']}" if torch.cuda.is_available() else "cpu"
    )
    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    from train import ensure_parent  # type: ignore
    ensure_parent(work_dir / "config_snapshot.json")
    if is_main_process():
        with (work_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    if is_main_process() and int(args.preflight_samples) > 0:
        validate_index_image_paths(
            cfg, [cfg["train_index"], cfg["val_index"]], int(args.preflight_samples),
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    train_loader = build_dataloader(cfg, cfg["train_index"], training=True, epoch=0)
    val_loader = build_dataloader(cfg, cfg["val_index"], training=False)

    model = DINOv2ConsistencyCritic(cfg).to(device)
    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]] if torch.cuda.is_available() else None,
            output_device=dist_info["local_rank"] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )

    best_val_loss = math.inf
    total_epochs = int(cfg["epochs"])
    start_epoch = 1
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        target_model = model.module if isinstance(model, DDP) else model
        missing, unexpected = target_model.load_state_dict(
            checkpoint["model"],
            strict=False,
        )
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        interrupted = bool(checkpoint.get("interrupted", False))
        if interrupted:
            # The interrupted checkpoint may have been written after a
            # one-step validation pass. Reset best tracking for the resumed
            # run so best.pth reflects a full validation epoch in this workdir.
            best_val_loss = math.inf
        ckpt_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = ckpt_epoch if interrupted else ckpt_epoch + 1
        if is_main_process():
            print(
                f"[Resume] loaded {resume_path} "
                f"epoch={ckpt_epoch} start_epoch={start_epoch} "
                f"best_val_loss={best_val_loss}"
            )
            if missing:
                print(f"[Resume][WARNING] missing keys: {missing[:8]}")
            if unexpected:
                print(f"[Resume][WARNING] unexpected keys: {unexpected[:8]}")
    start_time = time.time()

    if is_main_process():
        print("=" * 60)
        print("DINOv2 Consistency Critic v5 (minimal, ablation-aware)")
        print(f"  Backbone        : {'DINOv2 ' + dcfg.get('model_name','dinov2_vits14') + ' layer[' + str(dcfg.get('layer_index',11)) + ']' if dcfg.get('enabled', True) else '4-layer CNN (from train.py)'}")
        print(f"  DINOv2 freeze   : {dcfg.get('freeze', True) if dcfg.get('enabled', True) else 'N/A'}")
        print(f"  Explicit dist   : {dcfg.get('use_explicit_distance', True)}")
        group_batches = bool(
            cfg.get("ranking", {}).get(
                "group_batches",
                float(cfg.get("lambda_group_ranking", 0.0)) > 0.0,
            )
        )
        print(f"  Group batches   : {group_batches}")
        print(f"  D1-D4 sampling  : {cfg.get('difficulty_sampling', {}).get('enabled', False)}")
        print(f"  Work dir        : {work_dir}")
        print(f"  World size      : {dist_info['world_size']}")
        if torch.cuda.is_available():
            mem_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            print(f"  GPU memory      : {mem_total:.1f} GB")
        print("=" * 60)

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            train_metrics = run_consistency_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                device=device, epoch=epoch, cfg=cfg, training=True,
                max_steps=args.max_train_steps or 0,
            )
            val_metrics = run_consistency_epoch(
                model=model, loader=val_loader, optimizer=optimizer,
                device=device, epoch=epoch, cfg=cfg, training=False,
                max_steps=args.max_val_steps or 0,
            )
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
            if is_main_process():
                print(
                    f"[Epoch {epoch}/{total_epochs}] "
                    f"loss={train_metrics['loss']:.4f} "
                    f"c_acc={train_metrics['c_acc']:.4f} "
                    f"v_acc={train_metrics['v_acc']:.4f} "
                    f"rank_loss={train_metrics.get('group_rank_loss', 0.0):.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_c_acc={val_metrics['c_acc']:.4f} "
                    f"val_v_acc={val_metrics['v_acc']:.4f} "
                    f"val_rank_loss={val_metrics.get('group_rank_loss', 0.0):.4f}"
                )
                if epoch % int(cfg["save_interval"]) == 0:
                    save_checkpoint(
                        work_dir=work_dir, epoch=epoch, model=model,
                        optimizer=optimizer, cfg=cfg,
                        best_val_loss=best_val_loss, is_best=is_best,
                    )
            if sigterm_received():
                if is_main_process():
                    print(f"[WARNING] SIGTERM at epoch={epoch}, saving interrupted ckpt...")
                    save_checkpoint(
                        work_dir=work_dir, epoch=epoch, model=model,
                        optimizer=optimizer, cfg=cfg,
                        best_val_loss=best_val_loss, is_best=False,
                        tag=f"interrupted_epoch_{epoch}", interrupted=True,
                    )
                break
    except Exception as e:
        rank = dist_info["rank"]
        print(f"\n[ERROR][rank={rank}] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        if torch.cuda.is_available():
            ma = torch.cuda.memory_allocated(device) / (1024 ** 3)
            mr = torch.cuda.memory_reserved(device) / (1024 ** 3)
            print(f"[ERROR] GPU mem allocated={ma:.2f}GB reserved={mr:.2f}GB")
        cleanup_distributed()
        sys.exit(1)

    if is_main_process():
        elapsed = time.time() - start_time
        print("=" * 60)
        print("Training finished")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Elapsed:      {elapsed:.1f}s")
        print("=" * 60)
    cleanup_distributed()


if __name__ == "__main__":
    main()
