"""Extract frozen visual features for recovered-path probe experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from train_recovered_path_probe import _is_positive, _probe_input


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source(row: Dict[str, Any]) -> str:
    for key in ("source_type", "action_type", "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _quality(row: Dict[str, Any], fields: Sequence[str]) -> float:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return score
    return math.nan


def _is_supported_member(
    row: Dict[str, Any],
    *,
    supported_sources: set[str],
    min_quality: float,
    quality_fields: Sequence[str],
) -> bool:
    if _is_positive(row):
        return True
    if _source(row) not in supported_sources:
        return False
    quality = _quality(row, quality_fields)
    return math.isfinite(quality) and quality >= float(min_quality)


def _positive_indices(
    dataset: Dataset[Any],
    *,
    supported_sources: set[str],
    min_quality: float,
    quality_fields: Sequence[str],
) -> List[int]:
    rows = getattr(dataset, "samples", None)
    if rows is None:
        rows = getattr(dataset, "rows", None)
    if rows is None:
        return list(range(len(dataset)))
    return [
        idx
        for idx, row in enumerate(rows)
        if _is_supported_member(
            row,
            supported_sources=supported_sources,
            min_quality=min_quality,
            quality_fields=quality_fields,
        )
    ]


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
    parser.add_argument(
        "--supported-sources",
        default="",
        help="Comma-separated non-GT sources to include as supported members when quality is high.",
    )
    parser.add_argument("--min-quality", type=float, default=0.76)
    parser.add_argument(
        "--quality-fields",
        default="candidate_quality_score,official_epdms_score,epdms_score,official_pdm_score,pdms_score,planning_score",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print extraction progress every N batches; set 0 to disable.",
    )
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
    supported_sources = {
        item.strip() for item in args.supported_sources.split(",") if item.strip()
    }
    quality_fields = [
        item.strip() for item in args.quality_fields.split(",") if item.strip()
    ]
    indices = (
        _positive_indices(
            dataset,
            supported_sources=supported_sources,
            min_quality=args.min_quality,
            quality_fields=quality_fields,
        )
        if args.positive_only
        else list(range(len(dataset)))
    )
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(indices)
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
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
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
            if args.log_every > 0 and (
                batch_idx == 1 or batch_idx % int(args.log_every) == 0 or batch_idx == len(loader)
            ):
                done = min(batch_idx * int(args.batch_size), len(indices))
                elapsed = max(time.time() - start_time, 1e-6)
                print(
                    json.dumps(
                        {
                            "kind": "recovered_path_feature_progress",
                            "output": args.output,
                            "batch": batch_idx,
                            "num_batches": len(loader),
                            "samples": done,
                            "num_samples": len(indices),
                            "samples_per_sec": done / elapsed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
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
            "supported_sources": sorted(supported_sources),
            "min_quality": float(args.min_quality),
            "quality_fields": quality_fields,
            "num_samples": len(indices),
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
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
