"""Train a recovered-path probe from future visual evidence.

The probe is deliberately not given the candidate trajectory as an input. It
uses frozen IAC visual features (history/future image encodings) and predicts
the GT future trajectory for positive rows. This gives IAC-PathBench a
recover-then-compare diagnostic: first infer the path supported by the future
image, then compare candidates against that recovered path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_positive(row: Dict[str, Any]) -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return str(row.get("source_type", "")) == "gt_pos"


def _positive_indices(dataset: Dataset[Any]) -> List[int]:
    rows = getattr(dataset, "samples", None)
    if rows is None:
        rows = getattr(dataset, "rows", None)
    if rows is None:
        return list(range(len(dataset)))
    return [idx for idx, row in enumerate(rows) if _is_positive(row)]


def _trajectory_features(traj: torch.Tensor) -> torch.Tensor:
    xy = traj[..., :2]
    origin = torch.zeros_like(xy[:, :1, :])
    prev = torch.cat([origin, xy[:, :-1, :]], dim=1)
    step = xy - prev
    step_dist = torch.norm(step, p=2, dim=-1)
    return torch.stack(
        [
            step_dist.mean(dim=1),
            step_dist.max(dim=1).values,
            xy[:, -1, 0],
            xy[:, -1, 1],
        ],
        dim=-1,
    )


def _ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred[..., :2] - target[..., :2], p=2, dim=-1).mean(dim=-1)


def _fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred[:, -1, :2] - target[:, -1, :2], p=2, dim=-1)


class RecoveredPathProbe(nn.Module):
    def __init__(self, input_dim: int, steps: int, traj_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.steps = int(steps)
        self.traj_dim = int(traj_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.steps * self.traj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(x.shape[0], self.steps, self.traj_dim)


def _probe_input(feats: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    z_hist = feats["z_hist"]
    z_fut = feats["z_fut"]
    hist_last = feats.get("hist_seq_last", z_hist)
    fut_first = feats.get("fut_seq", None)
    if fut_first is not None:
        fut_first = fut_first[:, 0, :]
    else:
        fut_first = z_fut
    fut_last = feats.get("fut_seq_last", z_fut)
    if mode == "future_only":
        return torch.cat([z_fut, fut_last - fut_first], dim=-1)
    if mode == "history_future":
        return torch.cat([z_hist, z_fut, z_fut - z_hist], dim=-1)
    if mode == "motion_rich":
        return torch.cat(
            [
                z_hist,
                z_fut,
                z_fut - z_hist,
                hist_last,
                fut_first,
                fut_last,
                fut_last - hist_last,
                fut_last - fut_first,
            ],
            dim=-1,
        )
    raise ValueError(f"unknown probe input mode: {mode}")


def _load_model_and_dataset(
    *,
    config_path: str,
    checkpoint_path: str,
    index_path: str,
    device: torch.device,
    model_kind: str,
):
    sys.path.insert(0, str(_repo_root()))
    from train import ConsistencyDataset, load_config  # type: ignore
    from benchmark_wam import _load_model  # type: ignore

    cfg = load_config(config_path)
    cfg["train_index"] = index_path
    cfg["val_index"] = index_path
    dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=False)
    model, info = _load_model(Path(checkpoint_path), cfg, device, model_kind)
    return cfg, model, dataset, info


def _batch_target(batch: Dict[str, Any], steps: int, traj_dim: int, device: torch.device) -> torch.Tensor:
    target = batch.get("candidate_traj_raw")
    if target is None:
        target = batch["candidate_traj"]
    target = target.to(device=device, dtype=torch.float32)
    if target.shape[-1] < traj_dim:
        target = F.pad(target, (0, traj_dim - target.shape[-1]))
    target = target[:, :steps, :traj_dim]
    if target.shape[1] < steps:
        target = F.pad(target, (0, 0, 0, steps - target.shape[1]))
    return target


@torch.no_grad()
def _evaluate(
    *,
    model: nn.Module,
    probe: RecoveredPathProbe,
    loader: DataLoader,
    device: torch.device,
    input_mode: str,
    steps: int,
    traj_dim: int,
) -> Dict[str, Any]:
    model.eval()
    probe.eval()
    ades: List[float] = []
    fdes: List[float] = []
    for batch in loader:
        hist = batch["history_images"].to(device, non_blocking=True)
        fut = batch["future_images"].to(device, non_blocking=True)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["candidate_traj"].to(device, non_blocking=True)
        target = _batch_target(batch, steps, traj_dim, device)
        feats = model.extract_probe_features(hist, fut, ego, traj)
        pred = probe(_probe_input(feats, input_mode))
        ades.extend(float(v) for v in _ade(pred, target).detach().cpu().tolist())
        fdes.extend(float(v) for v in _fde(pred, target).detach().cpu().tolist())
    if not ades:
        return {}
    sorted_ade = sorted(ades)
    return {
        "num_samples": len(ades),
        "ade_mean": sum(ades) / len(ades),
        "fde_mean": sum(fdes) / len(fdes),
        "ade_p50": sorted_ade[int(0.50 * (len(sorted_ade) - 1))],
        "ade_p80": sorted_ade[int(0.80 * (len(sorted_ade) - 1))],
        "ade_p90": sorted_ade[int(0.90 * (len(sorted_ade) - 1))],
        "ade_p95": sorted_ade[int(0.95 * (len(sorted_ade) - 1))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--val-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-kind", default="auto", choices=["auto", "cnn", "dinov2"])
    parser.add_argument("--input-mode", default="motion_rich", choices=["future_only", "history_future", "motion_rich"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--traj-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg, model, train_dataset, model_info = _load_model_and_dataset(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        index_path=args.train_index,
        device=device,
        model_kind=args.model_kind,
    )
    _, _, val_dataset, _ = _load_model_and_dataset(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        index_path=args.val_index,
        device=device,
        model_kind=args.model_kind,
    )

    train_indices = _positive_indices(train_dataset)
    val_indices = _positive_indices(val_dataset)
    if args.max_train_samples > 0:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_indices = val_indices[: args.max_val_samples]
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    steps = int(args.steps or cfg["candidate_traj_steps"])
    traj_dim = int(args.traj_dim or cfg["traj_dim"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    first = next(iter(train_loader))
    with torch.no_grad():
        feats = model.extract_probe_features(
            first["history_images"].to(device),
            first["future_images"].to(device),
            first["ego_state"].to(device),
            first["candidate_traj"].to(device),
        )
        input_dim = int(_probe_input(feats, args.input_mode).shape[-1])
    probe = RecoveredPathProbe(input_dim, steps, traj_dim, args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    best_state: Dict[str, Any] | None = None
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        probe.train()
        losses: List[float] = []
        for batch in train_loader:
            hist = batch["history_images"].to(device, non_blocking=True)
            fut = batch["future_images"].to(device, non_blocking=True)
            ego = batch["ego_state"].to(device, non_blocking=True)
            traj = batch["candidate_traj"].to(device, non_blocking=True)
            target = _batch_target(batch, steps, traj_dim, device)
            with torch.no_grad():
                feats = model.extract_probe_features(hist, fut, ego, traj)
                x = _probe_input(feats, args.input_mode)
            pred = probe(x)
            loss_path = F.smooth_l1_loss(pred, target)
            loss_shape = F.smooth_l1_loss(_trajectory_features(pred), _trajectory_features(target))
            loss = loss_path + 0.25 * loss_shape
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = _evaluate(
            model=model,
            probe=probe,
            loader=val_loader,
            device=device,
            input_mode=args.input_mode,
            steps=steps,
            traj_dim=traj_dim,
        )
        train_loss = sum(losses) / max(len(losses), 1)
        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        val_ade = float(val_metrics.get("ade_mean", math.inf))
        if val_ade < best_val:
            best_val = val_ade
            best_state = {
                "probe": probe.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert best_state is not None
    metadata = {
        "kind": "recovered_path_probe",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "model_info": model_info,
        "train_index": args.train_index,
        "val_index": args.val_index,
        "input_mode": args.input_mode,
        "input_dim": input_dim,
        "steps": steps,
        "traj_dim": traj_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "best_epoch": best_state["epoch"],
        "best_val_metrics": best_state["val_metrics"],
        "num_train_positive": len(train_indices),
        "num_val_positive": len(val_indices),
        "history": history,
    }
    torch.save({**best_state, "metadata": metadata}, out_dir / "recovered_path_probe.pt")
    (out_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("saved", out_dir / "recovered_path_probe.pt")


if __name__ == "__main__":
    main()
