#!/usr/bin/env python3
"""Convert ordered-motion segment ledgers into three-state support decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import _pathfix  # noqa: F401

from ordered_motion_support import (  # noqa: E402
    SupportDecisionConfig,
    score_row,
    summarize_scored_rows,
)


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config) if args.config else None
    if config_path is not None:
        config = SupportDecisionConfig.from_mapping(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
    else:
        if args.support_energy_max is None or args.unsupported_energy_min is None:
            raise ValueError(
                "provide --config or both energy thresholds; validation-frozen "
                "--config is required for formal evaluation"
            )
        config = SupportDecisionConfig(
            support_energy_max=float(args.support_energy_max),
            unsupported_energy_min=float(args.unsupported_energy_min),
            min_evidence_coverage=float(args.min_evidence_coverage),
            max_mean_normalized_uncertainty=(
                float(args.max_mean_normalized_uncertainty)
                if args.max_mean_normalized_uncertainty is not None
                else None
            ),
            require_uncertainty=bool(args.require_uncertainty),
        )
    config.validate()
    input_path = Path(args.scores)
    scored = [score_row(row, config) for row in _read_jsonl(input_path)]
    output_path = Path(args.output_scores)
    _write_jsonl(output_path, scored)
    summary = {
        "kind": "ordered_motion_three_state_support",
        "inference_uses_source_labels": False,
        "scores": str(input_path),
        "scores_sha256": _sha256(input_path),
        "output_scores": str(output_path),
        "output_scores_sha256": _sha256(output_path),
        "config": config.to_dict(),
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": _sha256(config_path) if config_path is not None else None,
        "metrics": summarize_scored_rows(scored),
    }
    if args.output_summary:
        _write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--config")
    parser.add_argument("--support-energy-max", type=float)
    parser.add_argument("--unsupported-energy-min", type=float)
    parser.add_argument("--min-evidence-coverage", type=float, default=0.6)
    parser.add_argument("--max-mean-normalized-uncertainty", type=float)
    parser.add_argument("--require-uncertainty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
