#!/usr/bin/env python3
"""Analyze per-sample IAC benchmark scores.

`benchmark_wam.py` produces `wam_iac_scores.jsonl`. This script turns that
row-level output into a compact failure report: confusion metrics, negative
recall by source type, perturbation curves, and the highest-confidence errors.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


META_KEYS = (
    "sample_id",
    "anchor_id",
    "group_id",
    "scene_name",
    "token",
    "timestamp_us",
    "source_type",
    "sample_type",
    "action_type",
    "perturb_type",
    "perturb_level",
    "perturb_magnitude",
    "wam_name",
    "model_name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze wam_iac_scores.jsonl")
    parser.add_argument("--scores", required=True, help="Path to wam_iac_scores.jsonl")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--csv-errors",
        default=None,
        help="Optional CSV path for the highest-confidence errors.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _label(row: Dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        value = row.get("label") if key == "consistency_label" else None
    if value is None:
        return None
    return int(float(value) >= 0.5)


def _score(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise KeyError(f"Missing score field: {key}")
    return float(value)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _head_metrics(rows: Sequence[Dict[str, Any]], score_key: str, label_key: str, threshold: float) -> Dict[str, Any]:
    labeled = [row for row in rows if _label(row, label_key) is not None]
    if not labeled:
        return {"num_labeled": 0}

    tp = fp = fn = tn = 0
    pos_scores: List[float] = []
    neg_scores: List[float] = []
    for row in labeled:
        label = _label(row, label_key)
        assert label is not None
        pred = int(_score(row, score_key) >= threshold)
        if label == 1:
            pos_scores.append(_score(row, score_key))
        else:
            neg_scores.append(_score(row, score_key))
        tp += int(pred == 1 and label == 1)
        fp += int(pred == 1 and label == 0)
        fn += int(pred == 0 and label == 1)
        tn += int(pred == 0 and label == 0)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    fpr = fp / (fp + tn) if fp + tn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "num_labeled": len(labeled),
        "num_positive": len(pos_scores),
        "num_negative": len(neg_scores),
        "accuracy": (tp + tn) / len(labeled),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tnr": tnr,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_score_mean": _mean(pos_scores),
        "negative_score_mean": _mean(neg_scores),
    }


def _threshold_sweep(rows: Sequence[Dict[str, Any]], score_key: str, label_key: str) -> Dict[str, Any]:
    labeled = [
        row for row in rows
        if _label(row, label_key) is not None
    ]
    if not labeled:
        return {"num_labeled": 0}
    scores = [_score(row, score_key) for row in labeled]
    eps = 1e-6
    thresholds = sorted({min(scores) - eps, max(scores) + eps, *scores})
    if not thresholds:
        return {"num_labeled": len(labeled)}

    best_balanced: Dict[str, Any] | None = None
    best_f1: Dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = _head_metrics(labeled, score_key, label_key, threshold)
        recall = metrics.get("recall")
        tnr = metrics.get("tnr")
        balanced = (
            (float(recall) + float(tnr)) / 2.0
            if recall is not None and tnr is not None
            else None
        )
        candidate = dict(metrics)
        candidate["threshold"] = threshold
        candidate["balanced_accuracy"] = balanced
        if (
            balanced is not None
            and (
                best_balanced is None
                or balanced > float(best_balanced["balanced_accuracy"])
                or (
                    balanced == float(best_balanced["balanced_accuracy"])
                    and (candidate.get("f1") or 0.0) > (best_balanced.get("f1") or 0.0)
                )
            )
        ):
            best_balanced = candidate
        if (
            candidate.get("f1") is not None
            and (
                best_f1 is None
                or float(candidate["f1"]) > float(best_f1["f1"])
                or (
                    candidate["f1"] == best_f1["f1"]
                    and (balanced or 0.0) > (best_f1.get("balanced_accuracy") or 0.0)
                )
            )
        ):
            best_f1 = candidate

    return {
        "num_labeled": len(labeled),
        "best_balanced_accuracy": best_balanced,
        "best_f1": best_f1,
    }


def _group_summary(rows: Sequence[Dict[str, Any]], key: str, threshold: float) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or row.get("sample_type") or "unknown")].append(row)

    out: Dict[str, Any] = {}
    for value, items in sorted(groups.items()):
        out[value] = {
            "count": len(items),
            "mean_consistency": _mean(_score(row, "iac_consistency") for row in items),
            "mean_validity": _mean(_score(row, "iac_validity") for row in items if "iac_validity" in row),
            "consistency": _head_metrics(items, "iac_consistency", "consistency_label", threshold),
        }
    return out


def _graded_curve(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("perturb_magnitude") is None:
            continue
        ptype = row.get("perturb_type") or row.get("action_type") or row.get("source_type") or "perturb"
        level = row.get("perturb_level", "unknown")
        groups[f"{ptype}:{level}"].append(row)
    return {
        key: {
            "count": len(items),
            "mean_consistency": _mean(_score(row, "iac_consistency") for row in items),
            "mean_perturb_magnitude": _mean(float(row["perturb_magnitude"]) for row in items),
        }
        for key, items in sorted(groups.items())
    }


def _calibration(rows: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    bins: List[Dict[str, Any]] = []
    labeled = [row for row in rows if _label(row, "consistency_label") is not None]
    if not labeled:
        return {"bins": bins}
    for i in range(10):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        def in_score_bin(row: Dict[str, Any]) -> bool:
            score = _score(row, "iac_consistency")
            if i == 9:
                return lo <= score <= hi
            return lo <= score < hi

        in_bin = [
            row for row in labeled
            if in_score_bin(row)
        ]
        if not in_bin:
            bins.append({"lo": lo, "hi": hi, "count": 0})
            continue
        labels = [_label(row, "consistency_label") for row in in_bin]
        preds = [int(_score(row, "iac_consistency") >= threshold) for row in in_bin]
        bins.append({
            "lo": lo,
            "hi": hi,
            "count": len(in_bin),
            "mean_score": _mean(_score(row, "iac_consistency") for row in in_bin),
            "positive_rate": _mean(float(label) for label in labels if label is not None),
            "accuracy_at_threshold": _mean(float(pred == label) for pred, label in zip(preds, labels) if label is not None),
        })
    return {"bins": bins}


def _compact_row(row: Dict[str, Any], row_index: int) -> Dict[str, Any]:
    compact = {"row_index": row_index}
    for key in META_KEYS:
        if key in row:
            compact[key] = row[key]
    compact["consistency_label"] = _label(row, "consistency_label")
    compact["validity_label"] = _label(row, "validity_label")
    compact["iac_consistency"] = _score(row, "iac_consistency")
    if "iac_validity" in row:
        compact["iac_validity"] = _score(row, "iac_validity")
    for key in ("history_images", "future_images"):
        value = row.get(key)
        if isinstance(value, list):
            compact[key] = value
    return compact


def _top_errors(rows: Sequence[Dict[str, Any]], threshold: float, top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    false_pos: List[tuple[float, int, Dict[str, Any]]] = []
    false_neg: List[tuple[float, int, Dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        label = _label(row, "consistency_label")
        if label is None:
            continue
        score = _score(row, "iac_consistency")
        pred = int(score >= threshold)
        if pred == 1 and label == 0:
            false_pos.append((score, idx, row))
        elif pred == 0 and label == 1:
            false_neg.append((1.0 - score, idx, row))

    false_pos.sort(key=lambda item: item[0], reverse=True)
    false_neg.sort(key=lambda item: item[0], reverse=True)
    return {
        "false_positives": [_compact_row(row, idx) for _, idx, row in false_pos[:top_k]],
        "false_negatives": [_compact_row(row, idx) for _, idx, row in false_neg[:top_k]],
    }


def _write_error_csv(path: Path, errors: Dict[str, List[Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "error_type",
        "row_index",
        "sample_id",
        "anchor_id",
        "group_id",
        "source_type",
        "action_type",
        "perturb_type",
        "perturb_level",
        "perturb_magnitude",
        "consistency_label",
        "iac_consistency",
        "validity_label",
        "iac_validity",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for error_type, rows in errors.items():
            for row in rows:
                flat = {key: row.get(key) for key in fieldnames}
                flat["error_type"] = error_type
                writer.writerow(flat)


def main() -> None:
    args = parse_args()
    scores_path = Path(args.scores)
    rows = _read_jsonl(scores_path)
    if not rows:
        raise ValueError(f"No rows found in {scores_path}")

    errors = _top_errors(rows, args.threshold, args.top_k)
    report = {
        "scores": str(scores_path),
        "threshold": args.threshold,
        "num_samples": len(rows),
        "source_counts": dict(Counter(str(row.get("source_type", "unknown")) for row in rows)),
        "consistency": _head_metrics(rows, "iac_consistency", "consistency_label", args.threshold),
        "validity": _head_metrics(rows, "iac_validity", "validity_label", args.threshold),
        "threshold_sweep": {
            "consistency": _threshold_sweep(rows, "iac_consistency", "consistency_label"),
            "validity": _threshold_sweep(rows, "iac_validity", "validity_label"),
        },
        "by_source_type": _group_summary(rows, "source_type", args.threshold),
        "by_action_type": _group_summary(rows, "action_type", args.threshold),
        "graded_perturbation_curve": _graded_curve(rows),
        "calibration": _calibration(rows, args.threshold),
        "top_errors": errors,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv_errors:
        _write_error_csv(Path(args.csv_errors), errors)

    cm = report["consistency"]
    print(f"rows={len(rows)}")
    if cm.get("num_labeled", 0):
        print(
            "consistency "
            f"acc={cm['accuracy']:.4f} "
            f"tnr={cm['tnr']:.4f} "
            f"recall={cm['recall']:.4f} "
            f"fp={cm['fp']} fn={cm['fn']}"
        )
        sweep = report["threshold_sweep"]["consistency"]
        best = sweep.get("best_balanced_accuracy") or {}
        if best:
            print(
                "consistency best_balanced "
                f"thr={best['threshold']:.4f} "
                f"bal_acc={best['balanced_accuracy']:.4f} "
                f"acc={best['accuracy']:.4f} "
                f"tnr={best['tnr']:.4f} "
                f"recall={best['recall']:.4f}"
            )
    print(f"report={output}")
    if args.csv_errors:
        print(f"errors_csv={args.csv_errors}")


if __name__ == "__main__":
    main()
