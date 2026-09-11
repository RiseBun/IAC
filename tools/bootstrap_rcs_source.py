#!/usr/bin/env python3
"""Bootstrap RCS direction and persistence at the source/twin level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ci(values: np.ndarray, *, draws: int, seed: int) -> list[float] | None:
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        sample = values[rng.integers(0, values.size, size=values.size)]
        stats[i] = float(np.mean(sample))
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def summarize(path: Path, *, draws: int, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs") or []
    comparable = [
        row for row in pairs
        if row.get("status") == "scored"
        and row.get("action_direction_match") is not None
    ]
    matches = np.asarray(
        [1.0 if row["action_direction_match"] else 0.0 for row in comparable],
        dtype=np.float64,
    )
    persistence = np.asarray(
        [float(row["temporal_persistence"]) for row in pairs
         if row.get("status") == "scored" and row.get("temporal_persistence") is not None],
        dtype=np.float64,
    )
    return {
        "source_unit": "source_key_or_twin",
        "input": str(path),
        "pair_count": len(pairs),
        "direction_pairs": int(matches.size),
        "direction_accuracy": float(matches.mean()) if matches.size else None,
        "direction_bootstrap_ci95": _ci(matches, draws=draws, seed=seed),
        "persistence_count": int(persistence.size),
        "temporal_persistence_mean": float(persistence.mean()) if persistence.size else None,
        "temporal_persistence_bootstrap_ci95": _ci(persistence, draws=draws, seed=seed + 1),
        "claim_boundary": "Source-level uncertainty for structural RCS; not metric distance or causality.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    report = {
        "protocol": "iac-rcs-source-bootstrap-v1",
        "draws": int(args.draws),
        "seed": int(args.seed),
        "models": [summarize(path, draws=args.draws, seed=args.seed + i * 17)
                   for i, path in enumerate(args.input)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
