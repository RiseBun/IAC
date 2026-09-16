#!/usr/bin/env python3
"""Compose the frozen RCS v2 macro score from formal channel reports."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaw", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs" / "response_consistency_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    yaw = json.loads(args.yaw.read_text(encoding="utf-8"))
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    yaw_score = float(yaw["RCS_yaw"])
    progress_score = float(progress["endpoint_accuracy"])
    weights = config["composite"]["weights"]
    composite = float(weights["yaw"] * yaw_score + weights["relative_progress"] * progress_score)
    yaw_total = int(yaw["contract_valid_pairs"])
    yaw_hits = int(round(yaw_score * yaw_total))
    progress_total = int(progress["scored_groups"])
    progress_hits = int(progress["hits"])
    yaw_values = [1.0] * yaw_hits + [0.0] * (yaw_total - yaw_hits)
    progress_values = [1.0] * progress_hits + [0.0] * (progress_total - progress_hits)
    stats = config["statistics"]
    rng = random.Random(int(stats["bootstrap_seed"]))
    draws = []
    for _ in range(int(stats["bootstrap_draws"])):
        yaw_draw = sum(rng.choice(yaw_values) for _ in yaw_values) / len(yaw_values)
        progress_draw = sum(rng.choice(progress_values) for _ in progress_values) / len(progress_values)
        draws.append(float(weights["yaw"] * yaw_draw + weights["relative_progress"] * progress_draw))
    draws.sort()
    ci = [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
    progress_promoted = bool(progress.get("promotion_pass"))
    report = {
        "protocol": config["protocol"],
        "status": "formal" if progress_promoted else "not_promoted",
        "definition": config["composite"]["official_score"],
        "weights": weights,
        "channels": {
            "yaw": {"score": yaw_score, "count": yaw_total},
            "relative_progress": {"score": progress_score, "count": progress_total},
        },
        "relative_progress_promotion_pass": progress_promoted,
        "RCS": composite,
        "RCS_100": 100.0 * composite,
        "RCS_bootstrap_95ci": ci,
        "claim_boundary": config["claim_boundary"],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
