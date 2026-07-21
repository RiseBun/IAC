"""Audit built-in future-geometry recovery in the current IAC model.

Some DINOv2 IAC configs already include ``future_traj_geometry_pred``: an
image-only geometry prediction head trained from future visual features. This
tool evaluates whether that head explains ranking failures without training a
new probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader


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
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _groups(rows: Iterable[Dict[str, Any]], group_key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gid = _group_id(row, group_key)
        if gid is not None:
            out[gid].append(row)
    return out


def _traj_geometry(row: Dict[str, Any]) -> List[float]:
    traj = row.get("candidate_traj") or []
    xy: List[tuple[float, float]] = []
    yaw: List[float] = []
    for pt in traj:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            xy.append((float(pt[0]), float(pt[1])))
            yaw.append(float(pt[2]) if len(pt) > 2 else 0.0)
    if not xy:
        return [0.0] * 8
    prev = [(0.0, 0.0)] + xy[:-1]
    step = [((x - px) ** 2 + (y - py) ** 2) ** 0.5 for (x, y), (px, py) in zip(xy, prev)]
    final_x, final_y = xy[-1]
    yaw_delta = yaw[-1] - yaw[0] if yaw else 0.0
    return [
        sum(step),
        (final_x**2 + final_y**2) ** 0.5,
        final_x,
        abs(final_y),
        sum(step) / len(step),
        max(step),
        yaw_delta,
        abs(yaw_delta),
    ]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


@torch.no_grad()
def _predict_geometry(
    *,
    rows: Sequence[Dict[str, Any]],
    config_path: str,
    checkpoint_path: str,
    image_root: str | None,
    device: torch.device,
    model_kind: str,
    batch_size: int,
    num_workers: int,
) -> tuple[Dict[int, List[float]], Dict[str, Any]]:
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import WAMManifestDataset, _load_model  # type: ignore
    from train import load_config  # type: ignore

    cfg = load_config(config_path)
    dataset = WAMManifestDataset(list(rows), cfg, image_root)
    model, info = _load_model(Path(checkpoint_path), cfg, device, model_kind)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out: Dict[int, List[float]] = {}
    offset = 0
    for batch in loader:
        hist = batch["history_images"].to(device, non_blocking=True)
        fut = batch["future_images"].to(device, non_blocking=True)
        ego = batch["ego_state"].to(device, non_blocking=True)
        traj = batch["candidate_traj"].to(device, non_blocking=True)
        model_out = model(hist, fut, ego, traj)
        pred = model_out.get("future_traj_geometry_pred")
        if pred is None:
            raise RuntimeError("model output does not contain future_traj_geometry_pred")
        pred_cpu = pred.detach().cpu().tolist()
        for item in pred_cpu:
            out[offset] = [float(v) for v in item]
            offset += 1
    return out, info


def _l1(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(abs(float(a[i]) - float(b[i])) for i in range(n)) / n


def _summarize(
    rows: Sequence[Dict[str, Any]],
    pred_geom: Dict[int, List[float]],
    *,
    group_key: str,
    wam_key: str,
    score_key: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    grouped = _groups(rows, group_key)
    per_group: List[Dict[str, Any]] = []
    current_hits: List[float] = []
    geom_hits: List[float] = []
    gt_better_than_winner: List[float] = []
    miss_sources: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    near_sources = {"perturb_speed", "perturb_lateral", "perturb_heading"}
    for gid, group in grouped.items():
        positives = [row for row in group if _is_positive(row, wam_key)]
        if not positives or len(group) < 2:
            continue
        positive = positives[0]
        pred = pred_geom.get(int(positive["_row_index"]))
        if pred is None:
            continue
        items = []
        for row in group:
            err = _l1(pred, _traj_geometry(row))
            items.append(
                {
                    "row": row,
                    "source": _source(row, wam_key),
                    "score": float(row.get(score_key, row.get("iac_consistency", 0.0))),
                    "geom_error": err,
                    "is_positive": row is positive,
                }
            )
        current = sorted(items, key=lambda x: x["score"], reverse=True)[0]
        geom = sorted(items, key=lambda x: x["geom_error"])[0]
        gt = next(item for item in items if item["is_positive"])
        current_hit = current["is_positive"]
        geom_hit = geom["is_positive"]
        current_hits.append(float(current_hit))
        geom_hits.append(float(geom_hit))
        gt_better = float(gt["geom_error"] < current["geom_error"])
        gt_better_than_winner.append(gt_better)
        if current_hit:
            category = "hit"
        elif current["source"] in near_sources and abs(gt["geom_error"] - current["geom_error"]) <= 0.05:
            category = "geometry_ambiguous_near_miss"
        elif gt_better:
            category = "geometry_prefers_gt"
        else:
            category = "geometry_prefers_winner_or_error"
        categories[category] += 1
        if not current_hit:
            miss_sources[current["source"]] += 1
        per_group.append(
            {
                "group_id": gid,
                "category": category,
                "current_top1_hit": bool(current_hit),
                "geometry_top1_hit": bool(geom_hit),
                "current_winner_source": current["source"],
                "geometry_winner_source": geom["source"],
                "gt_geometry_error": gt["geom_error"],
                "current_winner_geometry_error": current["geom_error"],
                "geometry_winner_error": geom["geom_error"],
                "gt_better_than_current_winner": bool(gt_better),
                "gt_sample_id": positive.get("sample_id"),
                "current_winner_sample_id": current["row"].get("sample_id"),
            }
        )
    return (
        {
            "num_groups": len(per_group),
            "score_key": score_key,
            "current_hard_top1": _mean(current_hits),
            "future_geometry_top1": _mean(geom_hits),
            "gt_geometry_error_lt_current_winner_frac": _mean(gt_better_than_winner),
            "categories": dict(categories),
            "miss_source_distribution": dict(miss_sources),
        },
        per_group,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--model-kind", default="auto", choices=["auto", "cnn", "dinov2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-key", default="iac_consistency")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-per-group")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = _load_jsonl(Path(args.scores))
    pred, info = _predict_geometry(
        rows=rows,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        image_root=args.image_root,
        device=device,
        model_kind=args.model_kind,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary, per_group = _summarize(
        rows,
        pred,
        group_key=args.group_key,
        wam_key=args.wam_key,
        score_key=args.score_key,
    )
    summary["model_info"] = info
    out = Path(args.output_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_per_group:
        _write_jsonl(Path(args.output_per_group), per_group)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
