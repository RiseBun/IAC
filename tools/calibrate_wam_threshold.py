#!/usr/bin/env python3
"""Group-disjoint threshold calibration for WAM/IAC scores.

This tool separates score calibration from causal diagnostics. It chooses a
threshold on calibration groups, then reports metrics on held-out groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="wam_iac_scores.jsonl")
    parser.add_argument("--output", required=True, help="Output JSON report")
    parser.add_argument("--score-key", default="iac_consistency")
    parser.add_argument("--label-key", default="consistency_label")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--calib-fraction", type=float, default=0.5)
    parser.add_argument(
        "--metric",
        choices=("balanced_accuracy", "f1", "precision_at_recall"),
        default="balanced_accuracy",
    )
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label(row: Dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None and key == "consistency_label":
        value = row.get("label")
    if value is None:
        return None
    return int(float(value) >= 0.5)


def source(row: Dict[str, Any]) -> str:
    return str(row.get("source_type") or row.get("sample_type") or "unknown")


def group_id(row: Dict[str, Any], key: str) -> str:
    value = row.get(key) or row.get("anchor_id") or row.get("token")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", "unknown"))
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def split_rows(
    rows: Sequence[Dict[str, Any]],
    group_key: str,
    calib_fraction: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    calib: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    frac = max(0.05, min(float(calib_fraction), 0.95))
    for row in rows:
        gid = group_id(row, group_key)
        (calib if stable_unit_interval(gid) < frac else eval_rows).append(row)
    return calib, eval_rows


def safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def metrics_at(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Dict[str, Any]:
    preds = [int(score >= threshold) for score in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "threshold": float(threshold),
        "num_samples": len(labels),
        "num_positive": int(sum(labels)),
        "num_negative": int(len(labels) - sum(labels)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "tnr": float(tnr),
        "fpr": float(1.0 - tnr),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + tnr)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "positive_score_mean": safe_mean(s for s, y in zip(scores, labels) if y == 1),
        "negative_score_mean": safe_mean(s for s, y in zip(scores, labels) if y == 0),
    }


def valid_scores(
    rows: Sequence[Dict[str, Any]],
    score_key: str,
    label_key: str,
) -> tuple[List[float], List[int], List[Dict[str, Any]]]:
    scores: List[float] = []
    labels: List[int] = []
    kept: List[Dict[str, Any]] = []
    for row in rows:
        if score_key not in row:
            continue
        y = label(row, label_key)
        if y is None:
            continue
        scores.append(float(row[score_key]))
        labels.append(y)
        kept.append(row)
    return scores, labels, kept


def choose_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    metric: str,
    min_recall: float,
) -> Dict[str, Any]:
    if not scores:
        raise ValueError("no calibration scores")
    candidates = sorted(set(float(s) for s in scores))
    candidates = [min(candidates) - 1e-6] + candidates + [max(candidates) + 1e-6]
    best: Dict[str, Any] | None = None
    for threshold in candidates:
        m = metrics_at(scores, labels, threshold)
        if metric == "balanced_accuracy":
            key = (m["balanced_accuracy"], m["f1"], m["tnr"])
        elif metric == "f1":
            key = (m["f1"], m["balanced_accuracy"], m["tnr"])
        else:
            if m["recall"] < min_recall:
                continue
            key = (m["precision"], m["balanced_accuracy"], m["tnr"])
        m["_selection_key"] = key
        if best is None or key > best["_selection_key"]:
            best = m
    if best is None:
        best = max(
            (metrics_at(scores, labels, t) for t in candidates),
            key=lambda m: (m["recall"], m["balanced_accuracy"]),
        )
    best.pop("_selection_key", None)
    return best


def by_source_metrics(
    rows: Sequence[Dict[str, Any]],
    score_key: str,
    label_key: str,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[source(row)].append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for name, items in sorted(grouped.items()):
        scores, labels, _ = valid_scores(items, score_key, label_key)
        if labels:
            out[name] = metrics_at(scores, labels, threshold)
    return out


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.scores))
    calib_rows, eval_rows = split_rows(rows, args.group_key, args.calib_fraction)
    calib_scores, calib_labels, calib_kept = valid_scores(
        calib_rows, args.score_key, args.label_key
    )
    eval_scores, eval_labels, eval_kept = valid_scores(
        eval_rows, args.score_key, args.label_key
    )
    selected = choose_threshold(
        calib_scores,
        calib_labels,
        args.metric,
        args.min_recall,
    )
    threshold = float(selected["threshold"])
    report = {
        "scores": str(args.scores),
        "score_key": args.score_key,
        "label_key": args.label_key,
        "group_key": args.group_key,
        "split": {
            "calib_fraction": args.calib_fraction,
            "calib_rows": len(calib_kept),
            "eval_rows": len(eval_kept),
            "calib_groups": len({group_id(row, args.group_key) for row in calib_kept}),
            "eval_groups": len({group_id(row, args.group_key) for row in eval_kept}),
        },
        "selection_metric": args.metric,
        "selected_threshold": threshold,
        "calibration_selected": selected,
        "calibration_default": metrics_at(
            calib_scores, calib_labels, args.default_threshold
        ),
        "eval_selected": metrics_at(eval_scores, eval_labels, threshold),
        "eval_default": metrics_at(eval_scores, eval_labels, args.default_threshold),
        "eval_by_source_selected": by_source_metrics(
            eval_kept, args.score_key, args.label_key, threshold
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["eval_selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
