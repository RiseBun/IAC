"""Extract frozen visual features for recovered-path probe experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from train_recovered_path_probe import _is_positive, _probe_input


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _positive_indices(dataset: Dataset[Any]) -> List[int]:
    rows = getattr(dataset, "samples", None)
    if rows is None:
        rows = getattr(dataset, "rows", None)
    if rows is None:
        return list(range(len(dataset)))
    return [idx for idx, row in enumerate(rows) if _is_positive(row)]


def _load_model_and_dataset(
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
    dataset = ConsistencyDataset(index_path=index_path, cfg=cfg, training=False)
    model, info = _load_model(Path(checkpoint_path), cfg, device, model_kind)
    return cfg, model, dataset, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-kind", default="auto", choices=["auto", "cnn", "dinov2"])
    parser.add_argument("--input-mode", default="motion_rich", choices=["future_only", "history_future", "motion_rich"])
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg, model, dataset, info = _load_model_and_dataset(
        args.config,
        args.checkpoint,
        args.index,
        device,
        args.model_kind,
    )
    indices = _positive_indices(dataset) if args.positive_only else list(range(len(dataset)))
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    sample_ids: List[str] = []
    group_ids: List[str] = []
    source_types: List[str] = []
    with torch.no_grad():
        for batch in loader:
            hist = batch["history_images"].to(device, non_blocking=True)
            fut = batch["future_images"].to(device, non_blocking=True)
            ego = batch["ego_state"].to(device, non_blocking=True)
            traj = batch["candidate_traj"].to(device, non_blocking=True)
            feats = model.extract_probe_features(hist, fut, ego, traj)
            xs.append(_probe_input(feats, args.input_mode).detach().cpu())
            ys.append(batch.get("candidate_traj_raw", batch["candidate_traj"]).detach().cpu().float())
            sample_ids.extend([str(x) for x in batch.get("sample_id", [])])
            group_ids.extend([str(x) for x in batch.get("group_id", [])])
            source_types.extend([str(x) for x in batch.get("source_type", [])])
    out = {
        "x": torch.cat(xs, dim=0),
        "y": torch.cat(ys, dim=0),
        "sample_id": sample_ids,
        "group_id": group_ids,
        "source_type": source_types,
        "metadata": {
            "kind": "recovered_path_feature_cache",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "index": args.index,
            "model_info": info,
            "input_mode": args.input_mode,
            "positive_only": bool(args.positive_only),
            "num_samples": len(indices),
            "candidate_traj_steps": int(cfg["candidate_traj_steps"]),
            "traj_dim": int(cfg["traj_dim"]),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(json.dumps(out["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
