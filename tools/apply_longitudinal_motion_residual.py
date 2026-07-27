#!/usr/bin/env python3
"""Apply a source-blind, geometry-gated longitudinal motion residual.

This is the portable form of the independent SCOPE/TIRF experiment:

* the current IAC winner is the within-group reference;
* candidate geometry decides whether a row is primarily longitudinal;
* a lower-is-better motion evidence energy is compared with the reference;
* the signed contribution is written explicitly in logit space;
* source labels are used only after scoring for diagnostic metrics.

Unlike a learned fusion MLP, every score change can be traced to trajectory
geometry, relative evidence, confidence and one non-negative scalar weight.
No parameter sweep is implemented here: tune on a separate calibration split,
then pass one frozen configuration to each held-out split.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source(row: Mapping[str, Any]) -> str:
    for key in (
        "source_type",
        "action_type",
        "wam_name",
        "sample_type",
        "source",
        "wam",
    ):
        if row.get(key) is not None:
            return str(row[key])
    return "unknown"


def _group_id(row: Mapping[str, Any]) -> str:
    value = row.get("group_id") or row.get("anchor_id")
    if value is not None:
        return str(value)
    sample_id = str(row.get("sample_id", ""))
    return sample_id.rsplit("__", 1)[0] if "__" in sample_id else sample_id


def _sample_id(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id", ""))


def _is_positive(row: Mapping[str, Any]) -> bool:
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    if row.get("is_positive") is not None:
        return bool(row["is_positive"])
    return _source(row) == "gt_pos"


def _logit(probability: float) -> float:
    value = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _trajectory(
    values: Any,
    *,
    mode: str,
    reference_steps: int,
) -> np.ndarray:
    trajectory = np.asarray(values, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < reference_steps:
        raise ValueError(
            "candidate_traj must be a two-dimensional array with at least "
            f"{reference_steps} steps; received {trajectory.shape}"
        )
    if trajectory.shape[1] < 2:
        raise ValueError("candidate_traj needs forward and lateral coordinates")
    xy = trajectory[:, :2].copy()
    if mode == "deltas":
        xy = np.cumsum(xy, axis=0)
    elif mode != "positions":
        raise ValueError(f"unknown trajectory mode: {mode}")

    origin = np.zeros((1, 2), dtype=np.float64)
    steps = np.diff(np.vstack([origin, xy]), axis=0)
    derived_heading = np.arctan2(steps[:, 1], steps[:, 0])
    if trajectory.shape[1] >= 3:
        heading = trajectory[:, 2].copy()
        if mode == "deltas":
            heading = np.cumsum(heading)
    else:
        heading = derived_heading
    return np.column_stack([xy, heading])[:reference_steps]


def trajectory_geometry(
    reference_values: Any,
    candidate_values: Any,
    *,
    trajectory_mode: str = "positions",
    reference_steps: int = 4,
) -> Dict[str, float]:
    """Decompose candidate-reference deviation in the reference tangent frame."""

    reference = _trajectory(
        reference_values,
        mode=trajectory_mode,
        reference_steps=reference_steps,
    )
    candidate = _trajectory(
        candidate_values,
        mode=trajectory_mode,
        reference_steps=reference_steps,
    )
    tangent = np.stack(
        [np.cos(reference[:, 2]), np.sin(reference[:, 2])],
        axis=1,
    )
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    displacement = candidate[:, :2] - reference[:, :2]
    along = np.sum(displacement * tangent, axis=1)
    lateral = np.sum(displacement * normal, axis=1)

    origin = np.zeros((1, 2), dtype=np.float64)
    reference_step = np.diff(np.vstack([origin, reference[:, :2]]), axis=0)
    candidate_step = np.diff(np.vstack([origin, candidate[:, :2]]), axis=0)
    reference_step_length = np.linalg.norm(reference_step, axis=1)
    candidate_step_length = np.linalg.norm(candidate_step, axis=1)
    speed_profile_difference = candidate_step_length - reference_step_length

    longitudinal = float(
        np.sqrt(np.mean(np.square(along)))
        + 0.5 * np.sqrt(np.mean(np.square(speed_profile_difference)))
    )
    lateral_value = float(np.sqrt(np.mean(np.square(lateral))))
    path_scale = max(float(reference_step_length.sum()), 5.0)
    heading_value = float(
        path_scale
        * np.sqrt(
            np.mean(
                np.square(
                    _wrap_angle(candidate[:, 2] - reference[:, 2])
                )
            )
        )
    )
    total = longitudinal + lateral_value + heading_value
    return {
        "longitudinal_m": longitudinal,
        "lateral_m": lateral_value,
        "heading_equivalent_m": heading_value,
        "longitudinal_share": longitudinal / max(total, 1e-8),
    }


def _align_evidence(
    primary_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    evidence_by_key = {
        (_group_id(row), _sample_id(row)): row for row in evidence_rows
    }
    aligned: List[Mapping[str, Any]] = []
    missing: List[Tuple[str, str]] = []
    for row in primary_rows:
        key = (_group_id(row), _sample_id(row))
        evidence = evidence_by_key.get(key)
        if evidence is None:
            missing.append(key)
        else:
            aligned.append(evidence)
    if missing:
        examples = ", ".join(f"{group}/{sample}" for group, sample in missing[:3])
        raise ValueError(
            f"missing {len(missing)} aligned evidence rows; examples: {examples}"
        )
    return aligned


def _group_indices(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_group_id(row)].append(index)
    return groups


def apply_longitudinal_residual(
    primary_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    primary_score_key: str = "iac_consistency",
    output_score_key: str = "iac_consistency_interpretable",
    evidence_key: str = "flow_speed_energy",
    weight: float,
    share_threshold: float = 0.5,
    minimum_longitudinal_m: float = 1.0,
    minimum_evidence_spread: float = 0.0,
    evidence_margin: float = 0.0,
    evidence_clip: float = 3.0,
    trajectory_mode: str = "positions",
    reference_steps: int = 4,
) -> List[Dict[str, Any]]:
    """Return scored rows without consulting labels or source types."""

    if weight < 0.0:
        raise ValueError("weight must be non-negative")
    if not 0.0 <= share_threshold < 1.0:
        raise ValueError("share_threshold must be in [0, 1)")
    aligned = _align_evidence(primary_rows, evidence_rows)
    output = [dict(row) for row in primary_rows]
    groups = _group_indices(primary_rows)
    for group, indices in groups.items():
        reference_index = max(
            indices,
            key=lambda index: (
                float(primary_rows[index][primary_score_key]),
                _sample_id(primary_rows[index]),
            ),
        )
        evidence_values = np.asarray(
            [float(aligned[index][evidence_key]) for index in indices],
            dtype=np.float64,
        )
        spread = float(evidence_values.max() - evidence_values.min())
        reference_energy = float(aligned[reference_index][evidence_key])
        reference_row = primary_rows[reference_index]

        for index in indices:
            row = output[index]
            base_score = float(primary_rows[index][primary_score_key])
            geometry = {
                "longitudinal_m": 0.0,
                "lateral_m": 0.0,
                "heading_equivalent_m": 0.0,
                "longitudinal_share": 0.0,
            }
            confidence = 0.0
            relative_energy = float(aligned[index][evidence_key]) - reference_energy
            clipped_relative = 0.0
            active = False
            if index != reference_index and spread >= minimum_evidence_spread:
                geometry = trajectory_geometry(
                    reference_row["candidate_traj"],
                    primary_rows[index]["candidate_traj"],
                    trajectory_mode=trajectory_mode,
                    reference_steps=reference_steps,
                )
                if (
                    geometry["longitudinal_m"] >= minimum_longitudinal_m
                    and geometry["longitudinal_share"] >= share_threshold
                ):
                    magnitude = min(
                        max(abs(relative_energy) - evidence_margin, 0.0),
                        evidence_clip,
                    )
                    dominance = (
                        geometry["longitudinal_share"] - share_threshold
                    ) / max(1.0 - share_threshold, 1e-6)
                    strength = min(
                        geometry["longitudinal_m"]
                        / max(minimum_longitudinal_m, 0.25),
                        2.0,
                    ) / 2.0
                    confidence = float(
                        np.clip(dominance, 0.0, 1.0) * strength
                    )
                    clipped_relative = math.copysign(magnitude, relative_energy)
                    active = magnitude > 0.0 and confidence > 0.0

            logit_contribution = (
                -float(weight) * clipped_relative * confidence if active else 0.0
            )
            row[output_score_key] = _sigmoid(_logit(base_score) + logit_contribution)
            row["interpretable_longitudinal_residual"] = {
                "reference_group": group,
                "reference_sample_id": _sample_id(reference_row),
                "reference_primary_score": float(
                    reference_row[primary_score_key]
                ),
                "evidence_key": evidence_key,
                "candidate_evidence": float(aligned[index][evidence_key]),
                "reference_evidence": reference_energy,
                "relative_evidence": relative_energy,
                "clipped_relative_evidence": clipped_relative,
                "evidence_spread": spread,
                **geometry,
                "confidence": confidence,
                "weight": float(weight),
                "logit_contribution": logit_contribution,
                "active": bool(active),
            }
    return output


def _ranking_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
) -> Dict[str, Any]:
    groups = _group_indices(rows)
    top1: List[float] = []
    reciprocal_ranks: List[float] = []
    pairwise: Dict[str, List[float]] = defaultdict(list)
    for indices in groups.values():
        positives = [index for index in indices if _is_positive(rows[index])]
        if not positives:
            continue
        ranked = sorted(
            indices,
            key=lambda index: float(rows[index][score_key]),
            reverse=True,
        )
        positive_set = set(positives)
        top1.append(float(ranked[0] in positive_set))
        positive_rank = min(
            rank
            for rank, index in enumerate(ranked, start=1)
            if index in positive_set
        )
        reciprocal_ranks.append(1.0 / positive_rank)
        positive = max(positives, key=lambda index: float(rows[index][score_key]))
        positive_score = float(rows[positive][score_key])
        for index in indices:
            if index in positive_set:
                continue
            pairwise[_source(rows[index])].append(
                float(positive_score > float(rows[index][score_key]))
            )
    return {
        "groups": len(top1),
        "top1": float(np.mean(top1)) if top1 else None,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None,
        "pairwise_by_source_report_only": {
            source: {
                "accuracy": float(np.mean(values)),
                "count": len(values),
            }
            for source, values in sorted(pairwise.items())
        },
    }


def _audit(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    primary_score_key: str,
    output_score_key: str,
) -> Dict[str, Any]:
    groups_before = _group_indices(before)
    groups_after = _group_indices(after)
    transitions: Counter[str] = Counter()
    active_by_source: Dict[str, List[float]] = defaultdict(list)
    active_groups = 0
    for group in sorted(set(groups_before) & set(groups_after)):
        before_indices = groups_before[group]
        after_indices = groups_after[group]
        before_winner = max(
            before_indices,
            key=lambda index: float(before[index][primary_score_key]),
        )
        after_winner = max(
            after_indices,
            key=lambda index: float(after[index][output_score_key]),
        )
        transitions[
            f"{_source(before[before_winner])}->{_source(after[after_winner])}"
        ] += 1
        group_active = False
        for index in after_indices:
            active = bool(
                after[index]["interpretable_longitudinal_residual"]["active"]
            )
            group_active = group_active or active
            active_by_source[_source(after[index])].append(float(active))
        active_groups += int(group_active)
    total_groups = len(groups_after)
    return {
        "transformation_uses_source_labels": False,
        "source_labels_are_report_only": True,
        "base": _ranking_metrics(before, score_key=primary_score_key),
        "final": _ranking_metrics(after, score_key=output_score_key),
        "active_group_rate": active_groups / total_groups if total_groups else None,
        "active_row_rate_by_source_report_only": {
            source: float(np.mean(values))
            for source, values in sorted(active_by_source.items())
        },
        "winner_transitions_report_only": transitions.most_common(30),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", required=True)
    parser.add_argument("--evidence-rows", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--primary-score-key", default="iac_consistency")
    parser.add_argument(
        "--output-score-key",
        default="iac_consistency_interpretable",
    )
    parser.add_argument("--evidence-key", default="flow_speed_energy")
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--share-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-longitudinal-m", type=float, default=1.0)
    parser.add_argument("--minimum-evidence-spread", type=float, default=0.0)
    parser.add_argument("--evidence-margin", type=float, default=0.0)
    parser.add_argument("--evidence-clip", type=float, default=3.0)
    parser.add_argument(
        "--trajectory-mode",
        choices=["positions", "deltas"],
        default="positions",
    )
    parser.add_argument("--reference-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = _read_jsonl(Path(args.primary_scores))
    evidence = _read_jsonl(Path(args.evidence_rows))
    scored = apply_longitudinal_residual(
        primary,
        evidence,
        primary_score_key=args.primary_score_key,
        output_score_key=args.output_score_key,
        evidence_key=args.evidence_key,
        weight=args.weight,
        share_threshold=args.share_threshold,
        minimum_longitudinal_m=args.minimum_longitudinal_m,
        minimum_evidence_spread=args.minimum_evidence_spread,
        evidence_margin=args.evidence_margin,
        evidence_clip=args.evidence_clip,
        trajectory_mode=args.trajectory_mode,
        reference_steps=args.reference_steps,
    )
    _write_jsonl(Path(args.output_scores), scored)
    report = _audit(
        primary,
        scored,
        primary_score_key=args.primary_score_key,
        output_score_key=args.output_score_key,
    )
    report["config"] = {
        "primary_scores": args.primary_scores,
        "evidence_rows": args.evidence_rows,
        "primary_score_key": args.primary_score_key,
        "output_score_key": args.output_score_key,
        "evidence_key": args.evidence_key,
        "weight": args.weight,
        "share_threshold": args.share_threshold,
        "minimum_longitudinal_m": args.minimum_longitudinal_m,
        "minimum_evidence_spread": args.minimum_evidence_spread,
        "evidence_margin": args.evidence_margin,
        "evidence_clip": args.evidence_clip,
        "trajectory_mode": args.trajectory_mode,
        "reference_steps": args.reference_steps,
    }
    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
