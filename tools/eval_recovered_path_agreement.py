"""Evaluate candidate ranking against a recovered path probe.

This tool implements the recover-then-compare diagnostic:

1. recover an ego path from history/future images without reading a candidate;
2. compare every candidate trajectory in the group to that recovered path;
3. report whether GT, current winner, and near-neighbor misses are supported by
   the recovered path confidence set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_recovered_path_probe import RecoveredPathProbe, _ade, _fde, _probe_input


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                row = json.loads(line)
                row["_row_index"] = idx
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source(row: Dict[str, Any], wam_key: str) -> str:
    for key in ("source_type", "action_type", wam_key, "wam_name", "sample_type", "wam"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _is_positive(row: Dict[str, Any], wam_key: str) -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return _source(row, wam_key) == "gt_pos"


def _group_id(row: Dict[str, Any], group_key: str) -> str | None:
    value = row.get(group_key) or row.get("anchor_id") or row.get("group_id")
    if value is not None:
        return str(value)
    sample_id = row.get("sample_id")
    if sample_id is None:
        return None
    sample_id = str(sample_id)
    if "__" in sample_id:
        return sample_id.rsplit("__", 1)[0]
    return sample_id


def _groups(rows: Iterable[Dict[str, Any]], group_key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gid = _group_id(row, group_key)
        if gid is not None:
            out[gid].append(row)
    return out


def _traj_tensor(row: Dict[str, Any], steps: int, traj_dim: int, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(row.get("candidate_traj", []), dtype=torch.float32, device=device)
    if tensor.ndim != 2:
        tensor = torch.zeros((steps, traj_dim), dtype=torch.float32, device=device)
    if tensor.shape[-1] < traj_dim:
        tensor = torch.nn.functional.pad(tensor, (0, traj_dim - tensor.shape[-1]))
    tensor = tensor[:steps, :traj_dim]
    if tensor.shape[0] < steps:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, steps - tensor.shape[0]))
    return tensor


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


class ManifestWithIndex(Dataset[Any]):
    def __init__(self, base_dataset: Dataset[Any]) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.base[idx])
        item["row_index"] = int(idx)
        return item


@torch.no_grad()
def _predict_recovered_paths(
    *,
    rows: Sequence[Dict[str, Any]],
    config_path: str,
    checkpoint_path: str,
    probe_path: str,
    image_root: str | None,
    device: torch.device,
    model_kind: str,
    batch_size: int,
    num_workers: int,
) -> tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import WAMManifestDataset, _load_model  # type: ignore
    from train import load_config  # type: ignore

    cfg = load_config(config_path)
    dataset = ManifestWithIndex(WAMManifestDataset(list(rows), cfg, image_root))
    model, model_info = _load_model(Path(checkpoint_path), cfg, device, model_kind)
    bundle = torch.load(probe_path, map_location="cpu", weights_only=False)
    meta = bundle["metadata"]
    probe = RecoveredPathProbe(
        int(meta["input_dim"]),
        int(meta["steps"]),
        int(meta["traj_dim"]),
        int(meta["hidden_dim"]),
        float(meta.get("dropout", 0.0)),
    ).to(device)
    probe.load_state_dict(bundle["probe"], strict=True)
    input_mode = str(
        meta.get(
            "input_mode",
            meta.get("train_metadata", {}).get("input_mode", "motion_rich"),
        )
    )
    model.eval()
    probe.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    recovered: Dict[int, torch.Tensor] = {}
    for batch in loader:
        row_indices = batch["row_index"].detach().cpu().tolist()
        hist = batch["history_images"].to(device, non_blocking=True)
        fut = batch["future_images"].to(device, non_blocking=True)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["candidate_traj"].to(device, non_blocking=True)
        feats = model.extract_probe_features(hist, fut, ego, traj)
        pred = probe(_probe_input(feats, input_mode)).detach().cpu()
        for idx, item in zip(row_indices, pred):
            recovered[int(idx)] = item
    return recovered, {"model_info": model_info, "probe_metadata": meta}


def _recovered_summary(
    rows: Sequence[Dict[str, Any]],
    recovered: Dict[int, torch.Tensor],
    *,
    group_key: str,
    wam_key: str,
    score_key: str,
    radius: float | None,
    conformal_quantile: float,
    recover_mode: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    near_sources = {"perturb_speed", "perturb_lateral", "perturb_heading"}
    grouped = _groups(rows, group_key)
    per_group: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []
    gt_ades: List[float] = []
    winner_ades: List[float] = []
    recovered_top1_hits: List[float] = []
    current_top1_hits: List[float] = []
    current_winner_supported: List[float] = []
    gt_supported: List[float] = []
    ambiguity_set_sizes: List[float] = []
    miss_sources: Counter[str] = Counter()
    support_categories: Counter[str] = Counter()

    steps = next(iter(recovered.values())).shape[0] if recovered else 0
    traj_dim = next(iter(recovered.values())).shape[1] if recovered else 0
    device = torch.device("cpu")

    group_items: List[Dict[str, Any]] = []
    for gid, group in grouped.items():
        if len(group) < 2:
            continue
        positives = [row for row in group if _is_positive(row, wam_key)]
        if not positives:
            continue
        positive = positives[0]
        gt_row_index = int(positive["_row_index"])
        group_rec = recovered.get(gt_row_index)
        if group_rec is None:
            continue
        group_rec = group_rec.to(device)
        candidate_metrics: List[Dict[str, Any]] = []
        for row in group:
            idx = int(row["_row_index"])
            rec = group_rec if recover_mode == "group_gt_future" else recovered.get(idx)
            if rec is None:
                continue
            rec = rec.to(device)
            traj = _traj_tensor(row, steps, traj_dim, device).unsqueeze(0)
            pred = rec.unsqueeze(0)
            ade = float(_ade(pred, traj).item())
            fde = float(_fde(pred, traj).item())
            out = dict(row)
            out["recovered_path_ade"] = ade
            out["recovered_path_fde"] = fde
            out["recovered_path_agreement"] = -ade
            scored_rows.append(out)
            candidate_metrics.append(
                {
                    "row": row,
                    "row_index": idx,
                    "source": _source(row, wam_key),
                    "score": float(row.get(score_key, row.get("iac_consistency", 0.0))),
                    "ade": ade,
                    "fde": fde,
                    "is_positive": row is positive,
                }
            )
        if len(candidate_metrics) < 2:
            continue
        current_ranked = sorted(candidate_metrics, key=lambda item: item["score"], reverse=True)
        recovered_ranked = sorted(candidate_metrics, key=lambda item: item["ade"])
        gt_item = next(item for item in candidate_metrics if item["is_positive"])
        current_winner = current_ranked[0]
        recovered_winner = recovered_ranked[0]
        group_items.append(
            {
                "gid": gid,
                "positive": positive,
                "candidate_metrics": candidate_metrics,
                "gt_item": gt_item,
                "current_winner": current_winner,
                "recovered_winner": recovered_winner,
            }
        )
        gt_ades.append(float(gt_item["ade"]))

    radius_value = (
        float(radius)
        if radius is not None
        else float(_percentile(gt_ades, conformal_quantile * 100.0) or 0.0)
    )

    for item_group in group_items:
        gid = item_group["gid"]
        positive = item_group["positive"]
        candidate_metrics = item_group["candidate_metrics"]
        gt_item = item_group["gt_item"]
        current_winner = item_group["current_winner"]
        recovered_winner = item_group["recovered_winner"]
        current_hit = current_winner["is_positive"]
        recovered_hit = recovered_winner["is_positive"]
        winner_ades.append(float(current_winner["ade"]))
        current_top1_hits.append(float(current_hit))
        recovered_top1_hits.append(float(recovered_hit))
        if not current_hit:
            miss_sources[str(current_winner["source"])] += 1

        effective_radius = radius_value
        supported = [item for item in candidate_metrics if float(item["ade"]) <= effective_radius]
        ambiguity_set_sizes.append(float(len(supported)))
        gt_is_supported = float(gt_item["ade"] <= effective_radius)
        winner_is_supported = float(current_winner["ade"] <= effective_radius)
        gt_supported.append(gt_is_supported)
        current_winner_supported.append(winner_is_supported)
        if current_hit:
            category = "hit"
        elif winner_is_supported and current_winner["source"] in near_sources:
            category = "recovered_ambiguous_near_miss"
        elif gt_item["ade"] < current_winner["ade"]:
            category = "recovered_prefers_gt"
        else:
            category = "recovered_prefers_winner_or_error"
        support_categories[category] += 1
        per_group.append(
            {
                "group_id": gid,
                "category": category,
                "current_top1_hit": bool(current_hit),
                "recovered_top1_hit": bool(recovered_hit),
                "current_winner_source": current_winner["source"],
                "recovered_winner_source": recovered_winner["source"],
                "gt_recovered_ade": gt_item["ade"],
                "current_winner_recovered_ade": current_winner["ade"],
                "recovered_winner_ade": recovered_winner["ade"],
                "gt_minus_current_winner_ade": gt_item["ade"] - current_winner["ade"],
                "ambiguity_radius": effective_radius,
                "ambiguity_set_size": len(supported),
                "current_winner_supported": bool(winner_is_supported),
                "gt_supported": bool(gt_is_supported),
                "current_winner_sample_id": current_winner["row"].get("sample_id"),
                "gt_sample_id": positive.get("sample_id"),
            }
        )

    summary = {
        "num_groups": len(per_group),
        "score_key": score_key,
        "recover_mode": recover_mode,
        "ambiguity_radius": radius_value,
        "conformal_quantile": conformal_quantile,
        "current_hard_top1": _mean(current_top1_hits),
        "recovered_path_top1": _mean(recovered_top1_hits),
        "mean_gt_recovered_ade": _mean(gt_ades),
        "mean_current_winner_recovered_ade": _mean(winner_ades),
        "gt_ade_lt_current_winner_frac": _mean(
            float(item["gt_recovered_ade"] < item["current_winner_recovered_ade"])
            for item in per_group
        ),
        "current_winner_supported_frac": _mean(current_winner_supported),
        "gt_supported_frac": _mean(gt_supported),
        "mean_ambiguity_set_size": _mean(ambiguity_set_sizes),
        "support_categories": dict(support_categories),
        "miss_source_distribution": dict(miss_sources),
    }
    return summary, per_group, scored_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--model-kind", default="auto", choices=["auto", "cnn", "dinov2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-key", default="iac_consistency")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--ambiguity-radius", type=float)
    parser.add_argument("--conformal-quantile", type=float, default=0.90)
    parser.add_argument(
        "--recover-mode",
        default="row_future",
        choices=["row_future", "group_gt_future"],
        help=(
            "row_future recovers a path from each row's own future image; "
            "group_gt_future uses the GT row future for all candidates and is "
            "only a near-trajectory ambiguity diagnostic."
        ),
    )
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-per-group")
    parser.add_argument("--output-scored-rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = _load_jsonl(Path(args.scores))
    recovered, info = _predict_recovered_paths(
        rows=rows,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        probe_path=args.probe,
        image_root=args.image_root,
        device=device,
        model_kind=args.model_kind,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary, per_group, scored_rows = _recovered_summary(
        rows,
        recovered,
        group_key=args.group_key,
        wam_key=args.wam_key,
        score_key=args.score_key,
        radius=args.ambiguity_radius,
        conformal_quantile=args.conformal_quantile,
        recover_mode=args.recover_mode,
    )
    summary["probe_info"] = info
    out_path = Path(args.output_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_per_group:
        _write_jsonl(Path(args.output_per_group), per_group)
    if args.output_scored_rows:
        _write_jsonl(Path(args.output_scored_rows), scored_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
