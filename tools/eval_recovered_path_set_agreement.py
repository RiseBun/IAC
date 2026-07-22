"""Evaluate candidates against a multi-modal recovered-path set."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from train_recovered_path_probe import _probe_input
from train_recovered_path_set_probe_from_features import RecoveredPathSetProbe


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _groups(rows: Iterable[Dict[str, Any]], group_key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gid = _group_id(row, group_key)
        if gid is not None:
            out[gid].append(row)
    return out


def _traj_tensor(row: Dict[str, Any], steps: int, traj_dim: int) -> torch.Tensor:
    tensor = torch.tensor(row.get("candidate_traj", []), dtype=torch.float32)
    if tensor.ndim != 2:
        tensor = torch.zeros((steps, traj_dim), dtype=torch.float32)
    if tensor.shape[-1] < traj_dim:
        tensor = torch.nn.functional.pad(tensor, (0, traj_dim - tensor.shape[-1]))
    tensor = tensor[:steps, :traj_dim]
    if tensor.shape[0] < steps:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, steps - tensor.shape[0]))
    return tensor


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _angle_wrap(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _path_lengths(paths: torch.Tensor) -> torch.Tensor:
    if paths.shape[-2] < 2:
        return torch.zeros(paths.shape[:-2], dtype=paths.dtype, device=paths.device)
    deltas = paths[..., 1:, :2] - paths[..., :-1, :2]
    return torch.norm(deltas, p=2, dim=-1).sum(dim=-1)


def _path_headings(paths: torch.Tensor) -> torch.Tensor:
    if paths.shape[-2] < 2:
        return torch.zeros(paths.shape[:-2], dtype=paths.dtype, device=paths.device)
    delta = paths[..., -1, :2] - paths[..., 0, :2]
    return torch.atan2(delta[..., 1], delta[..., 0])


def _mode_path_iou(paths: torch.Tensor, traj: torch.Tensor, radius: float) -> torch.Tensor:
    if paths.numel() == 0:
        return torch.zeros((0,), dtype=traj.dtype)
    dist = torch.cdist(paths[..., :2], traj[None, :, :2], p=2)
    pred_hits = (dist.min(dim=2).values <= float(radius)).float().sum(dim=1)
    traj_hits = (dist.min(dim=1).values <= float(radius)).float().sum(dim=1)
    union = float(paths.shape[1] + traj.shape[0]) - 0.5 * (pred_hits + traj_hits)
    overlap = 0.5 * (pred_hits + traj_hits)
    return overlap / torch.clamp(union, min=1.0)


def _agreement_metrics(
    paths: torch.Tensor,
    logits: torch.Tensor,
    traj: torch.Tensor,
    *,
    ade_scale: float,
    fde_scale: float,
    heading_scale: float,
    progress_scale: float,
    path_iou_radius: float,
    ade_weight: float,
    fde_weight: float,
    heading_weight: float,
    progress_weight: float,
    path_iou_weight: float,
    logit_bias: float,
) -> Dict[str, float]:
    diff = paths[..., :2] - traj[None, :, :2]
    mode_ade = torch.norm(diff, p=2, dim=-1).mean(dim=-1)
    mode_fde = torch.norm(paths[:, -1, :2] - traj[None, -1, :2], p=2, dim=-1)
    heading_err = _angle_wrap(_path_headings(paths) - _path_headings(traj[None])).abs()
    progress_err = (_path_lengths(paths) - _path_lengths(traj[None])).abs()
    path_iou = _mode_path_iou(paths, traj, path_iou_radius)
    agreement_logit = (
        float(logit_bias)
        - float(ade_weight) * mode_ade / max(float(ade_scale), 1e-6)
        - float(fde_weight) * mode_fde / max(float(fde_scale), 1e-6)
        - float(heading_weight) * heading_err / max(float(heading_scale), 1e-6)
        - float(progress_weight) * progress_err / max(float(progress_scale), 1e-6)
        + float(path_iou_weight) * path_iou
    )
    best = int(torch.argmax(agreement_logit).item())
    top = int(torch.argmax(logits).item())
    return {
        "minade": float(torch.min(mode_ade).item()),
        "topmode_ade": float(mode_ade[top].item()),
        "best_mode": float(best),
        "best_mode_ade": float(mode_ade[best].item()),
        "best_mode_fde": float(mode_fde[best].item()),
        "best_mode_heading_error": float(heading_err[best].item()),
        "best_mode_progress_error": float(progress_err[best].item()),
        "best_mode_path_iou": float(path_iou[best].item()),
        "agreement_logit": float(agreement_logit[best].item()),
        "agreement_prob": _sigmoid(float(agreement_logit[best].item())),
    }


def _minade(paths: torch.Tensor, traj: torch.Tensor) -> float:
    diff = paths[..., :2] - traj[None, :, :2]
    ade = torch.norm(diff, p=2, dim=-1).mean(dim=-1)
    return float(torch.min(ade).item())


def _topmode_ade(paths: torch.Tensor, logits: torch.Tensor, traj: torch.Tensor) -> float:
    idx = int(torch.argmax(logits).item())
    diff = paths[idx, :, :2] - traj[:, :2]
    return float(torch.norm(diff, p=2, dim=-1).mean().item())


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
def _predict_path_sets(
    rows: Sequence[Dict[str, Any]],
    *,
    config_path: str,
    checkpoint_path: str,
    probe_path: str,
    image_root: str | None,
    device: torch.device,
    model_kind: str,
    batch_size: int,
    num_workers: int,
) -> tuple[Dict[int, tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import WAMManifestDataset, _load_model  # type: ignore
    from train import load_config  # type: ignore

    cfg = load_config(config_path)
    dataset = ManifestWithIndex(WAMManifestDataset(list(rows), cfg, image_root))
    model, info = _load_model(Path(checkpoint_path), cfg, device, model_kind)
    bundle = torch.load(probe_path, map_location="cpu", weights_only=False)
    meta = bundle["metadata"]
    probe = RecoveredPathSetProbe(
        int(meta["input_dim"]),
        int(meta["steps"]),
        int(meta["traj_dim"]),
        int(meta["num_modes"]),
        int(meta["hidden_dim"]),
        float(meta.get("dropout", 0.0)),
    ).to(device)
    probe.load_state_dict(bundle["probe"], strict=True)
    input_mode = str(meta.get("train_metadata", {}).get("input_mode", "motion_rich"))
    model.eval()
    probe.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    out: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for batch in loader:
        idxs = batch["row_index"].detach().cpu().tolist()
        hist = batch["history_images"].to(device, non_blocking=True)
        fut = batch["future_images"].to(device, non_blocking=True)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["candidate_traj"].to(device, non_blocking=True)
        feats = model.extract_probe_features(hist, fut, ego, traj)
        paths, logits = probe(_probe_input(feats, input_mode))
        paths = paths.detach().cpu()
        logits = logits.detach().cpu()
        for idx, p, l in zip(idxs, paths, logits):
            out[int(idx)] = (p, l)
    return out, {"model_info": info, "probe_metadata": meta}


def _summarize(
    rows: Sequence[Dict[str, Any]],
    path_sets: Dict[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    group_key: str,
    wam_key: str,
    score_key: str,
    conformal_quantile: float,
    recover_mode: str,
    agreement_args: Dict[str, float],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    near_sources = {"perturb_speed", "perturb_lateral", "perturb_heading"}
    grouped = _groups(rows, group_key)
    scored_rows: List[Dict[str, Any]] = []
    group_items = []
    gt_minades: List[float] = []
    steps = next(iter(path_sets.values()))[0].shape[1] if path_sets else 0
    traj_dim = next(iter(path_sets.values()))[0].shape[2] if path_sets else 0
    for gid, group in grouped.items():
        positives = [row for row in group if _is_positive(row, wam_key)]
        if not positives or len(group) < 2:
            continue
        positive = positives[0]
        positive_path_set = path_sets.get(int(positive["_row_index"]))
        if recover_mode == "group_gt_future" and positive_path_set is None:
            continue
        candidates = []
        for row in group:
            idx = int(row["_row_index"])
            if recover_mode == "group_gt_future":
                path_set = positive_path_set
            else:
                path_set = path_sets.get(idx)
            if path_set is None:
                continue
            paths, logits = path_set
            traj = _traj_tensor(row, steps, traj_dim)
            metrics = _agreement_metrics(paths, logits, traj, **agreement_args)
            minade = metrics["minade"]
            topade = metrics["topmode_ade"]
            out = dict(row)
            out["base_iac_consistency"] = row.get("iac_consistency")
            out["recovered_set_minade"] = minade
            out["recovered_set_topmode_ade"] = topade
            out["recovered_set_best_mode"] = int(metrics["best_mode"])
            out["recovered_set_best_mode_ade"] = metrics["best_mode_ade"]
            out["recovered_set_best_mode_fde"] = metrics["best_mode_fde"]
            out["recovered_set_heading_error"] = metrics["best_mode_heading_error"]
            out["recovered_set_progress_error"] = metrics["best_mode_progress_error"]
            out["recovered_set_path_iou"] = metrics["best_mode_path_iou"]
            out["recovered_set_agreement_logit"] = metrics["agreement_logit"]
            out["recovered_set_agreement"] = metrics["agreement_prob"]
            scored_rows.append(out)
            candidates.append(
                {
                    "row": row,
                    "source": _source(row, wam_key),
                    "score": float(row.get(score_key, row.get("iac_consistency", 0.0))),
                    "minade": minade,
                    "topade": topade,
                    "agreement": metrics["agreement_prob"],
                    "agreement_logit": metrics["agreement_logit"],
                    "is_positive": row is positive,
                }
            )
        if len(candidates) < 2:
            continue
        gt = next(item for item in candidates if item["is_positive"])
        gt_minades.append(float(gt["minade"]))
        group_items.append((gid, positive, candidates, gt))
    radius = float(_percentile(gt_minades, conformal_quantile * 100.0) or 0.0)
    for row in scored_rows:
        row["recovered_set_conformal_radius"] = radius
        row["recovered_set_supported"] = float(
            float(row.get("recovered_set_minade", 1e9)) <= radius
        )

    per_group: List[Dict[str, Any]] = []
    current_hits: List[float] = []
    recovered_hits: List[float] = []
    gt_better: List[float] = []
    winner_supported: List[float] = []
    gt_supported: List[float] = []
    set_sizes: List[float] = []
    miss_sources: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    high_pdm_above_gt: List[float] = []
    for gid, positive, candidates, gt in group_items:
        current = sorted(candidates, key=lambda item: item["score"], reverse=True)[0]
        recovered = sorted(candidates, key=lambda item: item["agreement"], reverse=True)[0]
        current_hit = current["is_positive"]
        recovered_hit = recovered["is_positive"]
        supported = [item for item in candidates if item["minade"] <= radius]
        current_supported = current["minade"] <= radius
        gt_is_supported = gt["minade"] <= radius
        high_pdm_rows = [
            item
            for item in candidates
            if item["source"] == "high_pdm_image_mismatch"
        ]
        if high_pdm_rows:
            high_pdm_above_gt.append(
                float(max(item["agreement"] for item in high_pdm_rows) > gt["agreement"])
            )
        current_hits.append(float(current_hit))
        recovered_hits.append(float(recovered_hit))
        gt_better.append(float(gt["minade"] < current["minade"]))
        winner_supported.append(float(current_supported))
        gt_supported.append(float(gt_is_supported))
        set_sizes.append(float(len(supported)))
        if not current_hit:
            miss_sources[current["source"]] += 1
        if current_hit:
            category = "hit"
        elif current_supported and current["source"] in near_sources:
            category = "set_ambiguous_near_miss"
        elif gt["minade"] < current["minade"]:
            category = "set_prefers_gt"
        else:
            category = "set_prefers_winner_or_error"
        categories[category] += 1
        per_group.append(
            {
                "group_id": gid,
                "category": category,
                "current_top1_hit": bool(current_hit),
                "recovered_set_top1_hit": bool(recovered_hit),
                "current_winner_source": current["source"],
                "recovered_winner_source": recovered["source"],
                "gt_minade": gt["minade"],
                "current_winner_minade": current["minade"],
                "recovered_winner_minade": recovered["minade"],
                "gt_recovered_set_agreement": gt["agreement"],
                "current_winner_recovered_set_agreement": current["agreement"],
                "recovered_winner_agreement": recovered["agreement"],
                "gt_better_than_current_winner": bool(gt["minade"] < current["minade"]),
                "ambiguity_radius": radius,
                "ambiguity_set_size": len(supported),
                "current_winner_supported": bool(current_supported),
                "gt_supported": bool(gt_is_supported),
                "gt_sample_id": positive.get("sample_id"),
                "current_winner_sample_id": current["row"].get("sample_id"),
            }
        )
    summary = {
        "num_groups": len(per_group),
        "score_key": score_key,
        "conformal_quantile": conformal_quantile,
        "ambiguity_radius": radius,
        "current_hard_top1": _mean(current_hits),
        "recovered_set_top1": _mean(recovered_hits),
        "mean_gt_minade": _mean(gt_minades),
        "gt_minade_lt_current_winner_frac": _mean(gt_better),
        "current_winner_supported_frac": _mean(winner_supported),
        "gt_supported_frac": _mean(gt_supported),
        "high_pdm_mismatch_above_gt_rate": _mean(high_pdm_above_gt),
        "mean_ambiguity_set_size": _mean(set_sizes),
        "support_categories": dict(categories),
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
    parser.add_argument("--conformal-quantile", type=float, default=0.90)
    parser.add_argument(
        "--recover-mode",
        default="group_gt_future",
        choices=["row_future", "group_gt_future"],
        help=(
            "group_gt_future recovers one K-set from the GT/future row and "
            "compares every candidate in the group against it. This is the "
            "pure recover-then-compare setting for image-to-supported-set."
        ),
    )
    parser.add_argument("--agreement-ade-scale", type=float, default=2.0)
    parser.add_argument("--agreement-fde-scale", type=float, default=4.0)
    parser.add_argument("--agreement-heading-scale", type=float, default=0.75)
    parser.add_argument("--agreement-progress-scale", type=float, default=4.0)
    parser.add_argument("--agreement-path-iou-radius", type=float, default=1.5)
    parser.add_argument("--agreement-ade-weight", type=float, default=1.0)
    parser.add_argument("--agreement-fde-weight", type=float, default=0.35)
    parser.add_argument("--agreement-heading-weight", type=float, default=0.25)
    parser.add_argument("--agreement-progress-weight", type=float, default=0.20)
    parser.add_argument("--agreement-path-iou-weight", type=float, default=1.0)
    parser.add_argument("--agreement-logit-bias", type=float, default=2.0)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-per-group")
    parser.add_argument("--output-scored-rows")
    parser.add_argument(
        "--output-agreement-scores",
        help="Optional WAM score JSONL whose iac_consistency is recovered-set agreement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = _load_jsonl(Path(args.scores))
    path_sets, info = _predict_path_sets(
        rows,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        probe_path=args.probe,
        image_root=args.image_root,
        device=device,
        model_kind=args.model_kind,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary, per_group, scored_rows = _summarize(
        rows,
        path_sets,
        group_key=args.group_key,
        wam_key=args.wam_key,
        score_key=args.score_key,
        conformal_quantile=args.conformal_quantile,
        recover_mode=args.recover_mode,
        agreement_args={
            "ade_scale": args.agreement_ade_scale,
            "fde_scale": args.agreement_fde_scale,
            "heading_scale": args.agreement_heading_scale,
            "progress_scale": args.agreement_progress_scale,
            "path_iou_radius": args.agreement_path_iou_radius,
            "ade_weight": args.agreement_ade_weight,
            "fde_weight": args.agreement_fde_weight,
            "heading_weight": args.agreement_heading_weight,
            "progress_weight": args.agreement_progress_weight,
            "path_iou_weight": args.agreement_path_iou_weight,
            "logit_bias": args.agreement_logit_bias,
        },
    )
    summary["probe_info"] = info
    summary["recover_mode"] = args.recover_mode
    summary["agreement_config"] = {
        "ade_scale": args.agreement_ade_scale,
        "fde_scale": args.agreement_fde_scale,
        "heading_scale": args.agreement_heading_scale,
        "progress_scale": args.agreement_progress_scale,
        "path_iou_radius": args.agreement_path_iou_radius,
        "ade_weight": args.agreement_ade_weight,
        "fde_weight": args.agreement_fde_weight,
        "heading_weight": args.agreement_heading_weight,
        "progress_weight": args.agreement_progress_weight,
        "path_iou_weight": args.agreement_path_iou_weight,
        "logit_bias": args.agreement_logit_bias,
    }
    out = Path(args.output_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_per_group:
        _write_jsonl(Path(args.output_per_group), per_group)
    if args.output_scored_rows:
        _write_jsonl(
            Path(args.output_scored_rows),
            sorted(scored_rows, key=lambda row: int(row.get("_row_index", 0))),
        )
    if args.output_agreement_scores:
        agreement_rows = []
        for row in sorted(scored_rows, key=lambda row: int(row.get("_row_index", 0))):
            out_row = dict(row)
            out_row["iac_consistency"] = float(out_row["recovered_set_agreement"])
            out_row["score_fusion_label"] = "recovered_set_agreement"
            agreement_rows.append(out_row)
        _write_jsonl(Path(args.output_agreement_scores), agreement_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
