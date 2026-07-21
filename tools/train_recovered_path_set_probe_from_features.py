"""Train a multi-modal recovered-path set probe from cached visual features.

Instead of predicting one average future path, this probe predicts K candidate
paths plus mixture logits. It is trained with a winner-take-minADE objective,
mirroring trajectory forecasting minADE@K. This is designed for ambiguous
future images where multiple speed/lateral/heading variants are visually
plausible.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_recovered_path_probe import _ade, _fde, _trajectory_features


class RecoveredPathSetProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        steps: int,
        traj_dim: int,
        num_modes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.steps = int(steps)
        self.traj_dim = int(traj_dim)
        self.num_modes = int(num_modes)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.path_head = nn.Linear(hidden_dim, self.num_modes * self.steps * self.traj_dim)
        self.logit_head = nn.Linear(hidden_dim, self.num_modes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        paths = self.path_head(h).reshape(x.shape[0], self.num_modes, self.steps, self.traj_dim)
        logits = self.logit_head(h)
        return paths, logits


def _load_cache(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _split_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> List[tuple[torch.Tensor, torch.Tensor]]:
    idx = torch.randperm(x.shape[0]) if shuffle else torch.arange(x.shape[0])
    return [(x[t], y[t]) for t in (idx[i : i + batch_size] for i in range(0, x.shape[0], batch_size))]


def _mode_ade(paths: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target[:, None, :, :]
    return torch.norm(paths[..., :2] - target[..., :2], p=2, dim=-1).mean(dim=-1)


def _mode_fde(paths: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target[:, None, :, :]
    return torch.norm(paths[:, :, -1, :2] - target[:, :, -1, :2], p=2, dim=-1)


def _diversity_loss(paths: torch.Tensor, min_separation: float) -> torch.Tensor:
    if paths.shape[1] < 2:
        return paths.sum() * 0.0
    xy = paths[..., :2].flatten(2)
    dist = torch.cdist(xy, xy, p=2)
    eye = torch.eye(paths.shape[1], device=paths.device, dtype=torch.bool)[None]
    pair = dist.masked_select(~eye)
    return F.relu(float(min_separation) - pair).mean()


@torch.no_grad()
def _metrics(
    model: RecoveredPathSetProbe,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    min_ades: List[float] = []
    top_ades: List[float] = []
    min_fdes: List[float] = []
    for xb, yb in _split_batches(x, y, batch_size, False):
        paths, logits = model(xb.to(device))
        paths = paths.cpu()
        logits = logits.cpu()
        ades = _mode_ade(paths, yb)
        fdes = _mode_fde(paths, yb)
        best = torch.argmin(ades, dim=1)
        top = torch.argmax(logits, dim=1)
        min_ades.extend(float(v) for v in ades.gather(1, best[:, None]).squeeze(1).tolist())
        top_ades.extend(float(v) for v in ades.gather(1, top[:, None]).squeeze(1).tolist())
        min_fdes.extend(float(v) for v in fdes.gather(1, best[:, None]).squeeze(1).tolist())
    xs = sorted(min_ades)
    if not xs:
        return {}
    return {
        "num_samples": len(xs),
        "minade_mean": sum(min_ades) / len(min_ades),
        "topmode_ade_mean": sum(top_ades) / len(top_ades),
        "minfde_mean": sum(min_fdes) / len(min_fdes),
        "minade_p50": xs[int(0.50 * (len(xs) - 1))],
        "minade_p80": xs[int(0.80 * (len(xs) - 1))],
        "minade_p90": xs[int(0.90 * (len(xs) - 1))],
        "minade_p95": xs[int(0.95 * (len(xs) - 1))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shape-loss-weight", type=float, default=0.25)
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.05)
    parser.add_argument("--cls-loss-weight", type=float, default=0.20)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--diversity-min-separation", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train = _load_cache(args.train_cache)
    val = _load_cache(args.val_cache)
    x_train = train["x"].float()
    y_train = train["y"].float()
    x_val = val["x"].float()
    y_val = val["y"].float()
    steps = int(train["metadata"]["candidate_traj_steps"])
    traj_dim = int(train["metadata"]["traj_dim"])
    y_train = y_train[:, :steps, :traj_dim]
    y_val = y_val[:, :steps, :traj_dim]
    model = RecoveredPathSetProbe(
        int(x_train.shape[-1]),
        steps,
        traj_dim,
        int(args.num_modes),
        int(args.hidden_dim),
        float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = math.inf
    best_state = None
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for xb, yb in _split_batches(x_train, y_train, args.batch_size, True):
            xb = xb.to(device)
            yb = yb.to(device)
            paths, logits = model(xb)
            ades = _mode_ade(paths, yb)
            best_mode = torch.argmin(ades.detach(), dim=1)
            best_paths = paths[torch.arange(paths.shape[0], device=device), best_mode]
            loss = F.smooth_l1_loss(best_paths, yb)
            if args.shape_loss_weight:
                loss = loss + float(args.shape_loss_weight) * F.smooth_l1_loss(
                    _trajectory_features(best_paths), _trajectory_features(yb)
                )
            if args.smoothness_loss_weight and paths.shape[2] > 2:
                pred_vel = best_paths[:, 1:, :2] - best_paths[:, :-1, :2]
                tgt_vel = yb[:, 1:, :2] - yb[:, :-1, :2]
                loss = loss + float(args.smoothness_loss_weight) * F.smooth_l1_loss(pred_vel, tgt_vel)
            if args.cls_loss_weight:
                loss = loss + float(args.cls_loss_weight) * F.cross_entropy(logits, best_mode)
            if args.diversity_weight:
                loss = loss + float(args.diversity_weight) * _diversity_loss(
                    paths,
                    float(args.diversity_min_separation),
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = _metrics(model, x_val, y_val, args.batch_size, device)
        record = {"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), **val_metrics}
        history.append(record)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(args.epochs):
            print(json.dumps(record, ensure_ascii=False))
        val_minade = float(val_metrics.get("minade_mean", math.inf))
        if val_minade < best:
            best = val_minade
            best_state = {
                "probe": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert best_state is not None
    metadata = {
        "kind": "recovered_path_set_probe_from_features",
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "input_dim": int(x_train.shape[-1]),
        "steps": steps,
        "traj_dim": traj_dim,
        "num_modes": int(args.num_modes),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "best_epoch": int(best_state["epoch"]),
        "best_val_metrics": best_state["val_metrics"],
        "train_metadata": train["metadata"],
        "val_metadata": val["metadata"],
        "history": history,
    }
    torch.save({**best_state, "metadata": metadata}, out / "recovered_path_set_probe.pt")
    (out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out / "recovered_path_set_probe.pt")


if __name__ == "__main__":
    main()
