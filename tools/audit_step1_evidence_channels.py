#!/usr/bin/env python3
"""Aggregate the shared Step1 evidence channels at source/twin level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CHANNELS = (
    "yaw_direction",
    "lateral_direction",
    "longitudinal_order",
    "expansion_response",
    "rotation_response",
)


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(draws)]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def audit(path: Path, *, min_intervals: int = 2, bootstrap_draws: int = 5000, seed: int = 20260912) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    by_source: dict[str, dict[str, Any]] = {}
    for row in report.get("rows", []):
        source = str(row.get("source_key") or "")
        if source and source not in by_source:
            by_source[source] = row
    pairs = [row.get("counterfactual_pair_evidence", {}) for row in by_source.values()]
    channels: dict[str, Any] = {}
    for channel_index, channel in enumerate(CHANNELS):
        values = []
        eligible = []
        for pair in pairs:
            item = pair.get(channel, {}) if isinstance(pair, dict) else {}
            intervals = int(item.get("usable_intervals", 0) or 0)
            accuracy = item.get("direction_accuracy")
            if intervals >= min_intervals and accuracy is not None and np.isfinite(float(accuracy)):
                eligible.append(item)
                values.append(float(accuracy) > 0.5)
        hits = np.asarray(values, dtype=float)
        ci = _bootstrap(hits, bootstrap_draws, seed + channel_index)
        channels[channel] = {
            "pairs_total": len(pairs),
            "pairs_evaluable": len(eligible),
            "coverage": len(eligible) / max(len(pairs), 1),
            "direction_accuracy": float(np.mean(hits)) if len(hits) else None,
            "direction_ci95": ci,
            "median_interval_direction_accuracy": float(np.median([float(item["direction_accuracy"]) for item in eligible])) if eligible else None,
            "status_counts": {
                status: sum(str(item.get("status")) == status for item in (pair.get(channel, {}) for pair in pairs if isinstance(pair, dict)))
                for status in ("scored", "weak", "inverted", "unavailable")
            },
            "formal_gate_candidate": bool(
                len(eligible) / max(len(pairs), 1) >= 0.90
                and ci is not None
                and ci[0] >= 0.75
            ),
        }
    return {
        "protocol": "iac-step1-evidence-channel-audit-v1",
        "source_report": str(path),
        "pair_count": len(pairs),
        "min_intervals": min_intervals,
        "channels": channels,
        "claim_boundary": "Source/twin aggregation of the shared evidence layer. It is not a cross-architecture promotion result and does not establish future-to-action causality.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=2, metavar=("MODEL", "REPORT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {model: audit(Path(path)) for model, path in args.model}
    result = {"protocol": "iac-step1-evidence-channel-audit-v1", "models": models, "claim_boundary": "Evidence-channel pilot only; no universal MAS/RCS promotion."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({model: report["channels"] for model, report in models.items()}, indent=2))


if __name__ == "__main__":
    main()
