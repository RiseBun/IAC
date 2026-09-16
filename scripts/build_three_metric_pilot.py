#!/usr/bin/env python3
"""Build an AS/RCS/GS diagnostic bundle from precomputed legacy reports.

The builder deliberately leaves Grounding Score unavailable unless an external
logged/simulated reference report is supplied.  Legacy CFAC/CCFC/FAU names are
retained only as compatibility metadata. The alignment input is legacy
flow-structure component evidence, not the complete yaw-plus-progress AS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_diagnostic(path: Path) -> dict[str, Any]:
    report = _load(path)
    normal = report.get("summary", {}).get("controls", {}).get("normal", {})
    count = int(normal.get("count", 0) or 0)
    return {
        "metric_id": "AS",
        "metric_name": "Alignment Score",
        "legacy_alias": "CFAC/MAS",
        "status": "diagnostic" if count else "unavailable",
        "score_definition": "legacy_flow_structure_component",
        "claim_boundary": "not a complete AS result; coarse ordinal progress is absent",
        "coverage": normal.get("interval_coverage"),
        "reliable_interval_fraction": normal.get("reliable_interval_fraction"),
        "median_residual_px": normal.get("median_residual_px"),
        "median_direction_cosine": normal.get("median_direction_cosine"),
        "controls": report.get("summary", {}).get("controls"),
        "source_report": str(path),
    }


def _rcs(path: Path) -> dict[str, Any]:
    report = _load(path)
    summary = report.get("summary", {})
    normal = summary.get("controls", {}).get("normal", {})
    return {
        "metric_id": "RCS",
        "metric_name": "Response Consistency Score",
        "legacy_alias": "CCFC",
        "status": "pilot" if normal.get("interval_coverage") is not None else "unavailable",
        "coverage": normal.get("interval_coverage"),
        "median_direction_cosine": normal.get("median_direction_cosine"),
        "response_gain_median": summary.get("normal_response_gain_median"),
        "temporal_persistence": normal.get("temporal_persistence"),
        "controls": summary.get("controls"),
        "source_report": str(path),
    }


def _gs(path: Path, model: str) -> dict[str, Any]:
    """Load an explicitly supplied structural GS candidate report.

    The report is never inferred from AS/RCS inputs. A missing model entry
    remains unavailable and retains the protocol's no-zero-fill contract.
    """
    report = _load(path)
    model_table = report.get("models", {})
    source = model_table.get(model)
    if source is None and isinstance(model_table, dict):
        source = next(
            (value for key, value in model_table.items() if str(key).lower() == model.lower()),
            None,
        )
    if source is None:
        source = report
    if not isinstance(source, dict) or source.get("score_median") is None:
        return {
            "metric_id": "GS",
            "metric_name": "Grounding Score",
            "legacy_alias": "FAU",
            "status": "unavailable",
            "reason": "grounding_candidate_report_missing_model_or_score",
            "source_report": str(path),
        }
    return {
        "metric_id": "GS",
        "metric_name": "Grounding Score",
        "legacy_alias": "FAU",
        "status": "candidate",
        "score": source.get("score_median"),
        "coverage": source.get("coverage"),
        "identity_shuffle_mean_score": source.get("identity_shuffle_mean_score"),
        "identity_shuffle_q95_mean_score": source.get("identity_shuffle_q95_mean_score"),
        "source_report": str(path),
        "claim_boundary": "external reality grounding; not future-to-action causality",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--as", "--mas", dest="alignment", action="append", metavar="MODEL=PATH",
        default=[], help="legacy alignment component report (--mas is deprecated)",
    )
    parser.add_argument("--rcs", action="append", metavar="MODEL=PATH", default=[])
    parser.add_argument("--gs", action="append", metavar="MODEL=PATH", default=[])
    args = parser.parse_args()
    models: dict[str, dict[str, Any]] = {}
    for item in args.alignment:
        model, path = item.split("=", 1)
        models.setdefault(model, {})["alignment_score"] = _as_diagnostic(Path(path))
    for item in args.rcs:
        model, path = item.split("=", 1)
        models.setdefault(model, {})["response_consistency_score"] = _rcs(Path(path))
    for item in args.gs:
        model, path = item.split("=", 1)
        models.setdefault(model, {})["grounding_score"] = _gs(Path(path), model)
    for model, values in models.items():
        values.setdefault("grounding_score", {
            "metric_id": "GS",
            "metric_name": "Grounding Score",
            "legacy_alias": "FAU",
            "status": "unavailable",
            "reason": "external_logged_or_simulated_future_reference_not_joined_in_this_structural_pilot",
            "claim_boundary": "must not be inferred from flow smoothness or affine explainability",
        })
    result = {
        "protocol": "iac-wam-three-metric-v1",
        "status": "pilot_grounding_pending",
        "canonical_metrics": [
            "Alignment Score",
            "Response Consistency Score",
            "Grounding Score",
        ],
        "legacy_aliases": {"CFAC/MAS": "Alignment Score", "CCFC": "Response Consistency Score", "FAU": "Grounding Score components"},
        "models": models,
        "missing_value_policy": "unavailable_or_abstain_never_zero_fill",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
