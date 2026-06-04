#!/usr/bin/env python3
"""Export failure cases for the DINOv2 IAC critic.

This script intentionally keeps outputs analysis-friendly:
- per-sample consistency FP/FN rows
- per-sample validity FP/FN rows
- ranking top-1 failure groups
- aggregate summary and threshold sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import ConsistencyDataset, load_config  # noqa: E402
from train_dinov2_v5_minimal import DINOv2ConsistencyCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export IAC failure cases")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=4096)
    p.add_argument("--max-ranking-groups", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="failure_cases")
    return p.parse_args()


def sigmoid_np(x: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(x).detach().cpu().numpy().astype(float)


def safe_get(sample: Dict[str, Any], key: str, default: Any = "") -> Any:
    value = sample.get(key, default)
    if value is None:
        return default
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def selected_paths(sample: Dict[str, Any], key: str, n: int = 2) -> str:
    paths = sample.get(key, [])
    if not isinstance(paths, list):
        return ""
    return "|".join(str(p) for p in paths[-n:])


def traj_summary(sample: Dict[str, Any]) -> Dict[str, Any]:
    traj = sample.get("candidate_traj", [])
    if not isinstance(traj, list) or not traj:
        return {"traj_len": 0}
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {"traj_len": len(traj)}
    start = arr[0, :2].tolist() if arr.shape[1] >= 2 else []
    end = arr[-1, :2].tolist() if arr.shape[1] >= 2 else []
    displacement = float(np.linalg.norm(arr[-1, :2] - arr[0, :2])) if arr.shape[1] >= 2 else None
    return {
        "traj_len": int(arr.shape[0]),
        "traj_start_xy": start,
        "traj_end_xy": end,
        "traj_displacement": displacement,
    }


def base_row(
    index: int,
    sample: Dict[str, Any],
    c_prob: float,
    v_prob: float,
    c_pred: int,
    v_pred: int,
) -> Dict[str, Any]:
    row = {
        "index": index,
        "sample_id": safe_get(sample, "sample_id", index),
        "group_id": safe_get(sample, "group_id", safe_get(sample, "anchor_id", "")),
        "anchor_id": safe_get(sample, "anchor_id", ""),
        "scene_name": safe_get(sample, "scene_name", ""),
        "timestamp_us": safe_get(sample, "timestamp_us", ""),
        "source_type": safe_get(sample, "source_type", "unknown"),
        "label_quality": safe_get(sample, "label_quality", ""),
        "perturb_type": safe_get(sample, "perturb_type", ""),
        "perturb_level": safe_get(sample, "perturb_level", ""),
        "perturb_magnitude": safe_get(sample, "perturb_magnitude", ""),
        "consistency_label": int(float(sample.get("consistency_label", 0)) > 0.5),
        "validity_label": int(float(sample.get("validity_label", 0)) > 0.5),
        "consistency_prob": round(float(c_prob), 8),
        "validity_prob": round(float(v_prob), 8),
        "consistency_pred": int(c_pred),
        "validity_pred": int(v_pred),
        "consistency_margin_abs": round(abs(float(c_prob) - 0.5), 8),
        "history_images_tail": selected_paths(sample, "history_images"),
        "future_images_tail": selected_paths(sample, "future_images"),
    }
    row.update(traj_summary(sample))
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def eval_samples(
    model: torch.nn.Module,
    dataset: ConsistencyDataset,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> Dict[str, Any]:
    from torch.utils.data import DataLoader, Subset

    limit = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    subset = Subset(dataset, list(range(limit)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    c_logits: List[torch.Tensor] = []
    v_logits: List[torch.Tensor] = []
    c_labels: List[torch.Tensor] = []
    v_labels: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            out = model(
                batch["history_images"].to(device, non_blocking=True),
                batch["future_images"].to(device, non_blocking=True),
                batch["ego_state"].to(device, non_blocking=True),
                batch["candidate_traj"].to(device, non_blocking=True),
            )
            c_logits.append(out["consistency_logit"].cpu())
            v_logits.append(out["validity_logit"].cpu())
            c_labels.append(batch["consistency_label"].cpu())
            v_labels.append(batch["validity_label"].cpu())
            if step % 20 == 0:
                print(f"[Samples] step={step}/{len(loader)} rows={min(step * batch_size, limit)}", flush=True)

    return {
        "c_probs": sigmoid_np(torch.cat(c_logits)),
        "v_probs": sigmoid_np(torch.cat(v_logits)),
        "c_labels": torch.cat(c_labels).numpy().astype(int),
        "v_labels": torch.cat(v_labels).numpy().astype(int),
        "indices": list(range(limit)),
    }


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tnr": float(tnr),
        "fpr": float(1.0 - tnr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_sweep(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = [metrics_at_threshold(probs, labels, float(t)) for t in thresholds]
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_balanced = max(rows, key=lambda r: (r["recall"] + r["tnr"]) / 2.0)
    operating = {}
    for target in (0.5, 0.7, 0.8, 0.9):
        candidates = [r for r in rows if r["recall"] >= target]
        operating[f"recall>={target:.1f}"] = max(candidates, key=lambda r: r["precision"]) if candidates else None
    return {
        "default_0.5": metrics_at_threshold(probs, labels, 0.5),
        "best_f1": best_f1,
        "best_balanced_accuracy": best_balanced,
        "operating_points": operating,
    }


def source_counter(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("source_type", "unknown")) for row in rows).most_common())


def export_classification_failures(
    dataset: ConsistencyDataset,
    eval_data: Dict[str, Any],
    threshold: float,
    out_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    c_probs = eval_data["c_probs"]
    v_probs = eval_data["v_probs"]
    c_labels = eval_data["c_labels"]
    v_labels = eval_data["v_labels"]
    rows: List[Dict[str, Any]] = []
    validity_rows: List[Dict[str, Any]] = []

    for index, c_prob, v_prob, c_label, v_label in zip(
        eval_data["indices"], c_probs, v_probs, c_labels, v_labels
    ):
        c_pred = int(c_prob >= threshold)
        v_pred = int(v_prob >= threshold)
        sample = dataset.samples[index]
        row = base_row(index, sample, float(c_prob), float(v_prob), c_pred, v_pred)
        if c_pred != int(c_label):
            row["error_type"] = "FN" if int(c_label) == 1 else "FP"
            rows.append(row)
        if v_pred != int(v_label):
            vrow = dict(row)
            vrow["error_type"] = "validity_FN" if int(v_label) == 1 else "validity_FP"
            validity_rows.append(vrow)

    rows.sort(key=lambda r: (r["error_type"], -abs(float(r["consistency_prob"]) - threshold)))
    validity_rows.sort(key=lambda r: (r["error_type"], -abs(float(r["validity_prob"]) - threshold)))

    write_csv(out_dir / f"{prefix}_classification_failures.csv", rows)
    write_json(out_dir / f"{prefix}_classification_failures.json", rows)
    write_csv(out_dir / f"{prefix}_validity_failures.csv", validity_rows)
    write_json(out_dir / f"{prefix}_validity_failures.json", validity_rows)

    fn_rows = [r for r in rows if r["error_type"] == "FN"]
    fp_rows = [r for r in rows if r["error_type"] == "FP"]
    return {
        "classification_failures": len(rows),
        "consistency_fn": len(fn_rows),
        "consistency_fp": len(fp_rows),
        "validity_failures": len(validity_rows),
        "fn_by_source_type": source_counter(fn_rows),
        "fp_by_source_type": source_counter(fp_rows),
        "validity_by_source_type": source_counter(validity_rows),
    }


def ndcg(scores: List[float], labels: List[int], k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = np.asarray(labels, dtype=np.float64)[order]
    discounts = 1.0 / np.log2(np.arange(len(gains)) + 2)
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(np.asarray(labels, dtype=np.float64))[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def export_ranking_failures(
    model: torch.nn.Module,
    dataset: ConsistencyDataset,
    device: torch.device,
    batch_size: int,
    max_groups: int,
    out_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        gid = sample.get("group_id") or sample.get("anchor_id") or f"{sample.get('scene_name', 'unknown')}::{sample.get('timestamp_us', idx)}"
        grouped[str(gid)].append({"index": idx, "label": int(float(sample.get("consistency_label", 0)) > 0.5)})

    groups = [
        (gid, items)
        for gid, items in grouped.items()
        if len(items) >= 2 and any(item["label"] == 1 for item in items)
    ]
    if max_groups > 0:
        groups = groups[:max_groups]

    failures: List[Dict[str, Any]] = []
    all_metrics = []
    model.eval()
    with torch.no_grad():
        for group_i, (gid, items) in enumerate(groups, start=1):
            scores: List[float] = []
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                samples = [dataset[item["index"]] for item in chunk]
                out = model(
                    torch.stack([s["history_images"] for s in samples]).to(device),
                    torch.stack([s["future_images"] for s in samples]).to(device),
                    torch.stack([s["ego_state"] for s in samples]).to(device),
                    torch.stack([s["candidate_traj"] for s in samples]).to(device),
                )
                scores.extend(torch.sigmoid(out["consistency_logit"]).detach().cpu().numpy().astype(float).tolist())

            labels = [int(item["label"]) for item in items]
            order = np.argsort(scores)[::-1].tolist()
            sorted_labels = [labels[i] for i in order]
            top1_hit = int(sorted_labels[0] == 1)
            pos_ranks = [rank + 1 for rank, i in enumerate(order) if labels[i] == 1]
            mrr = 1.0 / pos_ranks[0] if pos_ranks else 0.0
            group_metrics = {
                "top1_hit": top1_hit,
                "mrr": mrr,
                "ndcg@3": ndcg(scores, labels, 3),
                "ndcg@5": ndcg(scores, labels, 5),
            }
            all_metrics.append(group_metrics)

            if not top1_hit:
                top_item = items[order[0]]
                best_pos_order_idx = next(i for i in order if labels[i] == 1)
                top_sample = dataset.samples[top_item["index"]]
                pos_sample = dataset.samples[best_pos_order_idx]
                candidate_rows = []
                for rank, item_idx in enumerate(order, start=1):
                    item = items[item_idx]
                    sample = dataset.samples[item["index"]]
                    candidate_rows.append(
                        {
                            "rank": rank,
                            "index": item["index"],
                            "sample_id": safe_get(sample, "sample_id", item["index"]),
                            "source_type": safe_get(sample, "source_type", "unknown"),
                            "label": labels[item_idx],
                            "score": round(float(scores[item_idx]), 8),
                            "perturb_type": safe_get(sample, "perturb_type", ""),
                            "perturb_level": safe_get(sample, "perturb_level", ""),
                            "perturb_magnitude": safe_get(sample, "perturb_magnitude", ""),
                            "history_images_tail": selected_paths(sample, "history_images"),
                            "future_images_tail": selected_paths(sample, "future_images"),
                        }
                    )
                failures.append(
                    {
                        "group_id": gid,
                        "scene_name": safe_get(top_sample, "scene_name", safe_get(pos_sample, "scene_name", "")),
                        "timestamp_us": safe_get(top_sample, "timestamp_us", safe_get(pos_sample, "timestamp_us", "")),
                        "num_candidates": len(items),
                        "positive_rank": pos_ranks[0] if pos_ranks else None,
                        "top_score": round(float(scores[order[0]]), 8),
                        "best_positive_score": round(float(scores[best_pos_order_idx]), 8),
                        "score_gap_top_minus_positive": round(float(scores[order[0]] - scores[best_pos_order_idx]), 8),
                        "top_index": top_item["index"],
                        "top_sample_id": safe_get(top_sample, "sample_id", top_item["index"]),
                        "top_source_type": safe_get(top_sample, "source_type", "unknown"),
                        "positive_index": best_pos_order_idx,
                        "positive_sample_id": safe_get(pos_sample, "sample_id", best_pos_order_idx),
                        "candidates": candidate_rows,
                    }
                )

            if group_i % 100 == 0 or group_i == len(groups):
                print(f"[Ranking export] group={group_i}/{len(groups)} failures={len(failures)}", flush=True)

    failures.sort(key=lambda r: (-float(r["score_gap_top_minus_positive"]), int(r["positive_rank"] or 999)))
    write_json(out_dir / f"{prefix}_ranking_top1_failures.json", failures)
    flat_rows = []
    for failure in failures:
        base = {k: v for k, v in failure.items() if k != "candidates"}
        for candidate in failure["candidates"]:
            row = dict(base)
            row.update({f"candidate_{k}": v for k, v in candidate.items()})
            flat_rows.append(row)
    write_csv(out_dir / f"{prefix}_ranking_top1_failures.csv", flat_rows)

    top_types = Counter(str(f["top_source_type"]) for f in failures)
    return {
        "ranking_groups": len(groups),
        "ranking_top1_failures": len(failures),
        "ranking_top1_hit_rate": 1.0 - len(failures) / max(len(groups), 1),
        "ranking_failure_top_source_type": dict(top_types.most_common()),
        "ranking_mean_ndcg@3": float(np.mean([m["ndcg@3"] for m in all_metrics])) if all_metrics else 0.0,
        "ranking_mean_ndcg@5": float(np.mean([m["ndcg@5"] for m in all_metrics])) if all_metrics else 0.0,
        "ranking_mean_mrr": float(np.mean([m["mrr"] for m in all_metrics])) if all_metrics else 0.0,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv2ConsistencyCritic(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    index_key = "val_index" if args.split == "val" else "train_index"
    dataset = ConsistencyDataset(index_path=cfg[index_key], cfg=cfg, training=False)

    print(f"Loaded checkpoint epoch={checkpoint.get('epoch')} best_val_loss={checkpoint.get('best_val_loss')}")
    print(f"Dataset {args.split}: {len(dataset)} rows")

    eval_data = eval_samples(model, dataset, device, args.batch_size, args.max_samples)
    class_summary = export_classification_failures(dataset, eval_data, args.threshold, out_dir, args.prefix)
    rank_summary = export_ranking_failures(
        model,
        dataset,
        device,
        args.batch_size,
        args.max_ranking_groups,
        out_dir,
        args.prefix,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "split": args.split,
        "max_samples": min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset),
        "threshold": args.threshold,
        "classification": class_summary,
        "threshold_sweep_consistency": threshold_sweep(eval_data["c_probs"], eval_data["c_labels"]),
        "threshold_sweep_validity": threshold_sweep(eval_data["v_probs"], eval_data["v_labels"]),
        "ranking": rank_summary,
        "outputs": {
            "classification_failures_csv": str(out_dir / f"{args.prefix}_classification_failures.csv"),
            "classification_failures_json": str(out_dir / f"{args.prefix}_classification_failures.json"),
            "validity_failures_csv": str(out_dir / f"{args.prefix}_validity_failures.csv"),
            "validity_failures_json": str(out_dir / f"{args.prefix}_validity_failures.json"),
            "ranking_top1_failures_csv": str(out_dir / f"{args.prefix}_ranking_top1_failures.csv"),
            "ranking_top1_failures_json": str(out_dir / f"{args.prefix}_ranking_top1_failures.json"),
        },
    }
    write_json(out_dir / f"{args.prefix}_failure_summary.json", summary)

    lines = [
        "# IAC Failure Case Summary",
        "",
        f"- checkpoint epoch: {summary['checkpoint_epoch']}",
        f"- split/max_samples: {args.split}/{summary['max_samples']}",
        f"- threshold: {args.threshold}",
        "",
        "## Classification",
        f"- total failures: {class_summary['classification_failures']}",
        f"- FN: {class_summary['consistency_fn']}",
        f"- FP: {class_summary['consistency_fp']}",
        f"- FN by source_type: {class_summary['fn_by_source_type']}",
        f"- FP by source_type: {class_summary['fp_by_source_type']}",
        "",
        "## Threshold Sweep",
        f"- default 0.5: {summary['threshold_sweep_consistency']['default_0.5']}",
        f"- best F1: {summary['threshold_sweep_consistency']['best_f1']}",
        f"- best balanced accuracy: {summary['threshold_sweep_consistency']['best_balanced_accuracy']}",
        "",
        "## Ranking",
        f"- groups: {rank_summary['ranking_groups']}",
        f"- top1 failures: {rank_summary['ranking_top1_failures']}",
        f"- top1 hit rate: {rank_summary['ranking_top1_hit_rate']:.6f}",
        f"- top wrong source_type: {rank_summary['ranking_failure_top_source_type']}",
    ]
    (out_dir / f"{args.prefix}_failure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
