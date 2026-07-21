"""Train recovered-path probes from cached frozen visual features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_recovered_path_probe import RecoveredPathProbe, _ade, _fde, _trajectory_features


def _load_cache(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _split_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> List[tuple[torch.Tensor, torch.Tensor]]:
    idx = torch.randperm(x.shape[0]) if shuffle else torch.arange(x.shape[0])
    batches: List[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, x.shape[0], batch_size):
        take = idx[start : start + batch_size]
        batches.append((x[take], y[take]))
    return batches


@torch.no_grad()
def _metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> Dict[str, Any]:
    model.eval()
    ades: List[float] = []
    fdes: List[float] = []
    for xb, yb in _split_batches(x, y, batch_size, False):
        pred = model(xb.to(device)).cpu()
        ades.extend(float(v) for v in _ade(pred, yb).tolist())
        fdes.extend(float(v) for v in _fde(pred, yb).tolist())
    xs = sorted(ades)
    if not xs:
        return {}
    return {
        "num_samples": len(xs),
        "ade_mean": sum(ades) / len(ades),
        "fde_mean": sum(fdes) / len(fdes),
        "ade_p50": xs[int(0.50 * (len(xs) - 1))],
        "ade_p80": xs[int(0.80 * (len(xs) - 1))],
        "ade_p90": xs[int(0.90 * (len(xs) - 1))],
        "ade_p95": xs[int(0.95 * (len(xs) - 1))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shape-loss-weight", type=float, default=0.25)
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.05)
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
    model = RecoveredPathProbe(
        int(x_train.shape[-1]),
        steps,
        traj_dim,
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
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            if args.shape_loss_weight:
                loss = loss + float(args.shape_loss_weight) * F.smooth_l1_loss(
                    _trajectory_features(pred), _trajectory_features(yb)
                )
            if args.smoothness_loss_weight and pred.shape[1] > 2:
                pred_vel = pred[:, 1:, :2] - pred[:, :-1, :2]
                tgt_vel = yb[:, 1:, :2] - yb[:, :-1, :2]
                loss = loss + float(args.smoothness_loss_weight) * F.smooth_l1_loss(pred_vel, tgt_vel)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = _metrics(model, x_val, y_val, args.batch_size, device)
        record = {"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), **val_metrics}
        history.append(record)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(args.epochs):
            print(json.dumps(record, ensure_ascii=False))
        val_ade = float(val_metrics.get("ade_mean", math.inf))
        if val_ade < best:
            best = val_ade
            best_state = {
                "probe": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert best_state is not None
    metadata = {
        "kind": "recovered_path_probe_from_features",
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "input_dim": int(x_train.shape[-1]),
        "steps": steps,
        "traj_dim": traj_dim,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "best_epoch": int(best_state["epoch"]),
        "best_val_metrics": best_state["val_metrics"],
        "train_metadata": train["metadata"],
        "val_metadata": val["metadata"],
        "history": history,
    }
    torch.save({**best_state, "metadata": metadata}, out / "recovered_path_probe.pt")
    (out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out / "recovered_path_probe.pt")


if __name__ == "__main__":
    main()
