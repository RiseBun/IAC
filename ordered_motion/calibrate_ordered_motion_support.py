#!/usr/bin/env python3
"""Freeze three-state ordered-motion thresholds on a validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import _pathfix  # noqa: F401

from ordered_motion_support import (  # noqa: E402
    SupportDecisionConfig,
    aggregate_segment_evidence,
    calibrate_energy_thresholds,
)


DEFAULT_ACCEPTABLE = "gt_pos,perturb_speed,perturb_lateral,perturb_heading"
DEFAULT_HARD = "image_swap,time_shift_future,traj_swap,reverse_traj,high_pdm_image_mismatch"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    acceptable = _csv(args.acceptable_sources)
    hard = _csv(args.hard_sources)
    if acceptable & hard:
        raise ValueError("acceptable and hard source sets must be disjoint")
    input_path = Path(args.scores)
    records: list[tuple[float, bool]] = []
    excluded = {"unknown_source": 0, "low_coverage": 0, "uncertainty": 0}
    for row in _read_jsonl(input_path):
        source = str(row.get("source_type", ""))
        if source not in acceptable and source not in hard:
            excluded["unknown_source"] += 1
            continue
        evidence = aggregate_segment_evidence(row)
        if float(evidence["evidence_coverage"]) < float(
            args.min_evidence_coverage
        ):
            excluded["low_coverage"] += 1
            continue
        uncertainty = evidence["mean_normalized_uncertainty"]
        if args.max_mean_normalized_uncertainty is not None:
            if uncertainty is None and args.require_uncertainty:
                excluded["uncertainty"] += 1
                continue
            if (
                uncertainty is not None
                and float(uncertainty) > args.max_mean_normalized_uncertainty
            ):
                excluded["uncertainty"] += 1
                continue
        energy = evidence["visibility_aware_energy"]
        if energy is not None:
            records.append((float(energy), source in acceptable))

    selection = calibrate_energy_thresholds(
        records,
        min_supported_precision=float(args.min_supported_precision),
        min_unsupported_precision=float(args.min_unsupported_precision),
        min_supported_precision_lower_bound=args.min_supported_precision_lower_bound,
        min_unsupported_precision_lower_bound=args.min_unsupported_precision_lower_bound,
        confidence_z=float(args.confidence_z),
    )
    config = SupportDecisionConfig(
        support_energy_max=selection["support_energy_max"],
        unsupported_energy_min=selection["unsupported_energy_min"],
        min_evidence_coverage=float(args.min_evidence_coverage),
        max_mean_normalized_uncertainty=args.max_mean_normalized_uncertainty,
        require_uncertainty=bool(args.require_uncertainty),
    )
    output = {
        "kind": "ordered_motion_support_calibration_v1",
        "calibration_only_uses_source_labels": True,
        "inference_uses_source_labels": False,
        "validation_scores": str(input_path),
        "validation_scores_sha256": _sha256(input_path),
        "acceptable_sources": sorted(acceptable),
        "hard_sources": sorted(hard),
        "excluded_rows": excluded,
        "selection": selection,
        "support_decision_config": config.to_dict(),
    }
    _write_json(Path(args.output_config), output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--acceptable-sources", default=DEFAULT_ACCEPTABLE)
    parser.add_argument("--hard-sources", default=DEFAULT_HARD)
    parser.add_argument("--min-supported-precision", type=float, default=0.95)
    parser.add_argument("--min-unsupported-precision", type=float, default=0.95)
    parser.add_argument("--min-supported-precision-lower-bound", type=float)
    parser.add_argument("--min-unsupported-precision-lower-bound", type=float)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--min-evidence-coverage", type=float, default=0.6)
    parser.add_argument("--max-mean-normalized-uncertainty", type=float)
    parser.add_argument("--require-uncertainty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    calibrate(parse_args())
