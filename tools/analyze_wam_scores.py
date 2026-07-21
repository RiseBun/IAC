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
import math
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

IMAGE_COUNTERFACTUAL_SOURCES = {
    "image_swap",
    "time_shift_future",
}
TRAJECTORY_COUNTERFACTUAL_SOURCES = {
    "traj_swap",
    "perturb_lateral",
    "perturb_heading",
    "perturb_speed",
    "reverse_traj",
}
GEOMETRY_PERTURB_SOURCES = {
    "perturb_lateral",
    "perturb_heading",
    "perturb_speed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze wam_iac_scores.jsonl")
    parser.add_argument("--scores", required=True, help="Path to wam_iac_scores.jsonl")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=None)
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


def _source(row: Dict[str, Any]) -> str:
    return str(row.get("source_type") or row.get("sample_type") or "unknown")


def _anchor_key(row: Dict[str, Any]) -> str | None:
    for key in ("anchor_id", "group_id", "token"):
        if row.get(key) is not None:
            return str(row[key])
    sample_id = row.get("sample_id")
    if sample_id is None:
        return None
    sample_id = str(sample_id)
    source = row.get("source_type") or row.get("sample_type")
    if source is not None:
        suffix = f"__{source}"
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    if "__" in sample_id:
        return sample_id.rsplit("__", 1)[0]
    return sample_id


def _family_rows(rows: Sequence[Dict[str, Any]], sources: set[str]) -> List[Dict[str, Any]]:
    return [row for row in rows if _source(row) in sources]


def _source_means(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_source(row)].append(row)
    return {
        source: {
            "count": len(items),
            "mean_consistency": _mean(_score(row, "iac_consistency") for row in items),
            "mean_validity": _mean(_score(row, "iac_validity") for row in items if "iac_validity" in row),
        }
        for source, items in sorted(groups.items())
    }


def _pairwise_delta_by_source(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compare each counterfactual candidate against the GT row in its group."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _anchor_key(row)
        if key is not None:
            groups[key].append(row)

    deltas: Dict[str, List[float]] = defaultdict(list)
    gt_missing = 0
    for items in groups.values():
        positives = [
            row for row in items
            if _source(row) == "gt_pos" or _label(row, "consistency_label") == 1
        ]
        if not positives:
            gt_missing += 1
            continue
        gt_score = max(_score(row, "iac_consistency") for row in positives)
        for row in items:
            source = _source(row)
            if source == "gt_pos":
                continue
            deltas[source].append(gt_score - _score(row, "iac_consistency"))

    out = {
        source: {
            "count": len(values),
            "mean_delta_from_gt": _mean(values),
            "median_delta_from_gt": float(np.median(values)) if values else None,
            "fraction_gt_higher": _mean(float(value > 0.0) for value in values),
        }
        for source, values in sorted(deltas.items())
    }
    out["_meta"] = {
        "num_groups": len(groups),
        "groups_without_gt": gt_missing,
    }
    return out


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Average-rank implementation to avoid scipy dependency."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    if float(rx.std()) <= 1e-12 or float(ry.std()) <= 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _trajectory_tolerance_curve(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """iWorld-style graded perturbation check.

    A useful consistency score should usually fall as trajectory perturbation
    magnitude increases. This is not a learned metric; it is a monotonicity
    diagnostic over the benchmark's own perturbation metadata.
    """
    out: Dict[str, Any] = {}
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_items: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("perturb_magnitude") is None:
            continue
        source = _source(row)
        if source not in TRAJECTORY_COUNTERFACTUAL_SOURCES:
            continue
        ptype = str(row.get("perturb_type") or source)
        by_type[ptype].append(row)
        all_items.append(row)

    for ptype, items in sorted(by_type.items()):
        magnitudes = [float(row["perturb_magnitude"]) for row in items]
        scores = [_score(row, "iac_consistency") for row in items]
        out[ptype] = {
            "count": len(items),
            "mean_perturb_magnitude": _mean(magnitudes),
            "mean_consistency": _mean(scores),
            "score_vs_magnitude_spearman": _spearman(magnitudes, scores),
            "desired_spearman_sign": "negative",
        }

    if all_items:
        magnitudes = [float(row["perturb_magnitude"]) for row in all_items]
        scores = [_score(row, "iac_consistency") for row in all_items]
        out["_overall"] = {
            "count": len(all_items),
            "mean_perturb_magnitude": _mean(magnitudes),
            "mean_consistency": _mean(scores),
            "score_vs_magnitude_spearman": _spearman(magnitudes, scores),
            "desired_spearman_sign": "negative",
        }
    return out


def _family_metric(
    rows: Sequence[Dict[str, Any]],
    sources: set[str],
    threshold: float,
) -> Dict[str, Any]:
    items = _family_rows(rows, sources)
    return {
        "sources": sorted(sources),
        "count": len(items),
        "mean_consistency": _mean(_score(row, "iac_consistency") for row in items),
        "mean_validity": _mean(_score(row, "iac_validity") for row in items if "iac_validity" in row),
        "confusion_at_threshold": _head_metrics(items, "iac_consistency", "consistency_label", threshold),
    }


def _safe_mean_delta(pairwise: Dict[str, Dict[str, Any]], sources: set[str]) -> float | None:
    values = [
        float(item["mean_delta_from_gt"])
        for source, item in pairwise.items()
        if source in sources and item.get("mean_delta_from_gt") is not None
    ]
    return _mean(values)


def _iworld_style_diagnostics(rows: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Translate iWorld-Bench checks into IAC-native diagnostics.

    iWorld-Bench asks whether generated videos obey camera controls, tolerate
    graded trajectory checks, and close memory loops. For IAC, we ask the same
    first-principles questions using available row metadata and critic scores:

    - action control: do matched GT futures score above image/time counterfactuals?
    - trajectory tolerance: do larger trajectory perturbations lower scores?
    - memory ability: are reverse-trajectory counterfactuals suppressed?
    - shortcut risk: is score movement dominated by trajectory priors while
      subtle future-image counterfactuals remain weak?
    """
    gt_rows = [
        row for row in rows
        if _source(row) == "gt_pos" or _label(row, "consistency_label") == 1
    ]
    gt_mean = _mean(_score(row, "iac_consistency") for row in gt_rows)
    source_means = _source_means(rows)
    pairwise = _pairwise_delta_by_source(rows)

    image_family = _family_metric(rows, IMAGE_COUNTERFACTUAL_SOURCES, threshold)
    traj_family = _family_metric(rows, TRAJECTORY_COUNTERFACTUAL_SOURCES, threshold)
    geom_family = _family_metric(rows, GEOMETRY_PERTURB_SOURCES, threshold)
    reverse_family = _family_metric(rows, {"reverse_traj"}, threshold)

    for source, item in source_means.items():
        if source == "gt_pos" or gt_mean is None or item.get("mean_consistency") is None:
            item["delta_from_gt_mean"] = None
        else:
            item["delta_from_gt_mean"] = float(gt_mean - item["mean_consistency"])

    image_delta = _safe_mean_delta(pairwise, IMAGE_COUNTERFACTUAL_SOURCES)
    traj_delta = _safe_mean_delta(pairwise, TRAJECTORY_COUNTERFACTUAL_SOURCES)
    time_shift_delta = None
    if pairwise.get("time_shift_future", {}).get("mean_delta_from_gt") is not None:
        time_shift_delta = float(pairwise["time_shift_future"]["mean_delta_from_gt"])
    reverse_delta = None
    if pairwise.get("reverse_traj", {}).get("mean_delta_from_gt") is not None:
        reverse_delta = float(pairwise["reverse_traj"]["mean_delta_from_gt"])

    image_tnr = image_family["confusion_at_threshold"].get("tnr")
    traj_tnr = traj_family["confusion_at_threshold"].get("tnr")
    geom_tnr = geom_family["confusion_at_threshold"].get("tnr")

    warnings: List[str] = []
    if image_delta is None or image_delta < 0.03:
        warnings.append("future_counterfactual_delta_weak")
    if time_shift_delta is None or time_shift_delta < 0.03:
        warnings.append("time_shift_future_delta_weak")
    if traj_delta is not None and image_delta is not None and traj_delta > 1.5 * max(image_delta, 1e-6):
        warnings.append("trajectory_delta_dominates_future_delta")
    if traj_tnr is not None and traj_tnr < 0.5:
        warnings.append("trajectory_counterfactual_false_positives_high")
    if geom_tnr is not None and geom_tnr < 0.5:
        warnings.append("geometry_perturb_false_positives_high")
    if image_tnr is not None and traj_tnr is not None and image_tnr - traj_tnr > 0.35:
        warnings.append("image_swaps_solved_but_trajectory_counterfactuals_confused")

    tolerance = _trajectory_tolerance_curve(rows)
    rho = tolerance.get("_overall", {}).get("score_vs_magnitude_spearman")
    if rho is not None and rho > -0.1:
        warnings.append("trajectory_tolerance_not_monotonic")

    diagnostics = {
        "inspiration": "iWorld-Bench action-control / trajectory-tolerance / memory checks translated to IAC row-level diagnostics.",
        "gt_pos_mean_consistency": gt_mean,
        "source_means": source_means,
        "pairwise_delta_from_gt_by_source": pairwise,
        "action_control_proxy": {
            "definition": "GT consistency score minus image/time counterfactual score within the same group.",
            "image_counterfactual_sources": sorted(IMAGE_COUNTERFACTUAL_SOURCES),
            "mean_pairwise_delta": image_delta,
            "time_shift_future_delta": time_shift_delta,
            "family": image_family,
        },
        "trajectory_following_proxy": {
            "definition": "GT consistency score minus trajectory counterfactual score within the same group.",
            "trajectory_counterfactual_sources": sorted(TRAJECTORY_COUNTERFACTUAL_SOURCES),
            "mean_pairwise_delta": traj_delta,
            "geometry_perturb_family": geom_family,
            "family": traj_family,
        },
        "trajectory_tolerance_curve": tolerance,
        "memory_ability_proxy": {
            "definition": "Reverse-trajectory counterfactual suppression, when reverse_traj rows exist.",
            "reverse_traj_delta": reverse_delta,
            "family": reverse_family,
        },
        "shortcut_assessment": {
            "future_delta": image_delta,
            "time_shift_future_delta": time_shift_delta,
            "trajectory_delta": traj_delta,
            "image_family_tnr": image_tnr,
            "trajectory_family_tnr": traj_tnr,
            "geometry_family_tnr": geom_tnr,
            "warnings": warnings,
            "is_shortcut_prone": bool(warnings),
        },
    }
    path_causal = _path_causal_summary(rows)
    if path_causal.get("count", 0):
        diagnostics["path_grounding_causal_test"] = path_causal
    traj_specific = _trajectory_specific_causal_summary(rows)
    if traj_specific.get("count", 0):
        diagnostics["trajectory_specific_causal_test"] = traj_specific
    return diagnostics


def _path_causal_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("iac_consistency_path_masked") is not None
        and row.get("iac_consistency_sky_masked") is not None
    ]
    if not valid:
        return {"count": 0}

    def delta(row: Dict[str, Any], masked_key: str) -> float:
        return _score(row, "iac_consistency") - float(row[masked_key])

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[_source(row)].append(row)

    by_source: Dict[str, Dict[str, Any]] = {}
    for source, items in sorted(grouped.items()):
        path_d = [delta(row, "iac_consistency_path_masked") for row in items]
        sky_d = [delta(row, "iac_consistency_sky_masked") for row in items]
        by_source[source] = {
            "count": len(items),
            "mean_path_delta": _mean(path_d),
            "mean_sky_delta": _mean(sky_d),
            "mean_path_minus_sky_delta": _mean(p - s for p, s in zip(path_d, sky_d)),
            "path_delta_gt_sky_fraction": _mean(float(p > s) for p, s in zip(path_d, sky_d)),
        }

    path_d = [delta(row, "iac_consistency_path_masked") for row in valid]
    sky_d = [delta(row, "iac_consistency_sky_masked") for row in valid]
    diff = [p - s for p, s in zip(path_d, sky_d)]
    return {
        "count": len(valid),
        "definition": "Equal-area causal mask: trajectory-projected path ROI versus top-image sky/background control ROI.",
        "mean_path_delta": _mean(path_d),
        "mean_sky_delta": _mean(sky_d),
        "mean_path_minus_sky_delta": _mean(diff),
        "path_delta_gt_sky_fraction": _mean(float(p > s) for p, s in zip(path_d, sky_d)),
        "mean_path_mask_fraction": _mean(float(row.get("path_mask_fraction", 0.0)) for row in valid),
        "mean_sky_mask_fraction": _mean(float(row.get("sky_mask_fraction", 0.0)) for row in valid),
        "is_path_grounded": bool((_mean(diff) or 0.0) > 0.01),
        "by_source_type": by_source,
    }


def _trajectory_specific_causal_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("iac_consistency_path_masked") is not None
        and row.get("iac_consistency_wrong_path_masked") is not None
        and row.get("wrong_path_source_type") is not None
    ]
    if not valid:
        return {"count": 0}

    def delta(row: Dict[str, Any], masked_key: str) -> float:
        return _score(row, "iac_consistency") - float(row[masked_key])

    def summarize(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        cand_d = [delta(row, "iac_consistency_path_masked") for row in items]
        wrong_d = [delta(row, "iac_consistency_wrong_path_masked") for row in items]
        diff = [c - w for c, w in zip(cand_d, wrong_d)]
        return {
            "count": len(items),
            "mean_candidate_path_delta": _mean(cand_d),
            "mean_wrong_path_delta": _mean(wrong_d),
            "mean_candidate_minus_wrong_delta": _mean(diff),
            "candidate_delta_gt_wrong_fraction": _mean(
                float(c > w) for c, w in zip(cand_d, wrong_d)
            ),
        }

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[_source(row)].append(row)
    positives = [
        row for row in valid
        if _label(row, "consistency_label") == 1
    ]
    overall = summarize(valid)
    return {
        **overall,
        "definition": (
            "Same-group causal mask: current candidate path ROI versus wrong "
            "candidate path ROI. This tests trajectory-specific path grounding."
        ),
        "positive_rows": summarize(positives) if positives else {"count": 0},
        "mean_wrong_path_mask_fraction": _mean(
            float(row.get("wrong_path_mask_fraction", 0.0)) for row in valid
        ),
        "is_trajectory_specific_path_grounded": bool(
            (overall.get("mean_candidate_minus_wrong_delta") or 0.0) > 0.01
        ),
        "by_source_type": {
            source: summarize(items)
            for source, items in sorted(grouped.items())
        },
    }


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

    threshold = args.threshold
    if threshold is None:
        threshold = float(_threshold_sweep(rows, "iac_consistency", "consistency_label")["best_balanced_accuracy"]["threshold"])
    errors = _top_errors(rows, threshold, args.top_k)
    report = {
        "scores": str(scores_path),
        "threshold": threshold,
        "num_samples": len(rows),
        "source_counts": dict(Counter(str(row.get("source_type", "unknown")) for row in rows)),
        "consistency": _head_metrics(rows, "iac_consistency", "consistency_label", threshold),
        "validity": _head_metrics(rows, "iac_validity", "validity_label", threshold),
        "threshold_sweep": {
            "consistency": _threshold_sweep(rows, "iac_consistency", "consistency_label"),
            "validity": _threshold_sweep(rows, "iac_validity", "validity_label"),
        },
        "by_source_type": _group_summary(rows, "source_type", threshold),
        "by_action_type": _group_summary(rows, "action_type", threshold),
        "graded_perturbation_curve": _graded_curve(rows),
        "calibration": _calibration(rows, threshold),
        "iworld_style_diagnostics": _iworld_style_diagnostics(rows, threshold),
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
        diag = report["iworld_style_diagnostics"]["shortcut_assessment"]
        print(
            "iworld_style "
            f"future_delta={diag.get('future_delta')} "
            f"traj_delta={diag.get('trajectory_delta')} "
            f"geom_tnr={diag.get('geometry_family_tnr')} "
            f"shortcut_prone={diag.get('is_shortcut_prone')}"
        )
    print(f"report={output}")
    if args.csv_errors:
        print(f"errors_csv={args.csv_errors}")


if __name__ == "__main__":
    main()
