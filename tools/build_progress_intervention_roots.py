#!/usr/bin/env python3
"""Build validation-only fast/slow action roots from matched action manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _trajectory(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).squeeze()
    if result.shape == (3, 8):
        result = result.T
    if result.shape != (8, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"expected a finite [8,3] trajectory, got {result.shape}")
    return result


def _mean_action(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = 0.5 * (left + right)
    result[:, 2] = np.arctan2(
        np.sin(left[:, 2]) + np.sin(right[:, 2]),
        np.cos(left[:, 2]) + np.cos(right[:, 2]),
    )
    return result


def _read_pairs(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for manifest in sorted(root.glob("shard_*/manifest.json")):
        for row in json.loads(manifest.read_text(encoding="utf-8")):
            role = str(row["branch_mode"])
            if role not in {"left", "right"}:
                continue
            source = str(row["source_key"])
            if role in pairs[source]:
                raise ValueError(f"duplicate {source}/{role}")
            pairs[source][role] = row
    complete = {source: pair for source, pair in pairs.items() if set(pair) == {"left", "right"}}
    if not complete:
        raise ValueError(f"no complete matched pairs found under {root}")
    return complete


def _stratum(row: dict[str, Any]) -> str:
    with Path(row["source_sample"]).open("rb") as handle:
        sample = pickle.load(handle)
    return str((sample.get("metadata") or {}).get("stratum", "unknown"))


def _select(
    pairs: dict[str, dict[str, dict[str, Any]]],
    quotas: dict[str, int],
    seed: str,
) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for source, pair in pairs.items():
        grouped[_stratum(pair["left"])].append(source)
    selected: list[str] = []
    for stratum, quota in quotas.items():
        ranked = sorted(
            grouped.get(stratum, []),
            key=lambda source: hashlib.sha256(f"{seed}:{source}".encode()).hexdigest(),
        )
        if len(ranked) < quota:
            raise ValueError(f"{stratum}: requested {quota}, found {len(ranked)}")
        selected.extend(ranked[:quota])
    return sorted(selected)


def build_roots(
    *,
    input_root: Path,
    output_root: Path,
    fast_scale: float,
    slow_scale: float,
    quotas: dict[str, int],
    selection_seed: str,
) -> dict[str, Any]:
    if not 0.0 < slow_scale < fast_scale:
        raise ValueError("scales must satisfy 0 < slow < fast")
    pairs = _read_pairs(input_root)
    selected = _select(pairs, quotas, selection_seed)
    rows = {"fast": [], "slow": []}
    audit_rows: list[dict[str, Any]] = []
    for source in selected:
        pair = pairs[source]
        base = _mean_action(
            _trajectory(pair["left"]["action_trajectory"]),
            _trajectory(pair["right"]["action_trajectory"]),
        )
        stratum = _stratum(pair["left"])
        for role, scale in (("fast", fast_scale), ("slow", slow_scale)):
            action = base.copy()
            action[:, :2] *= scale
            rows[role].append(
                {
                    "source_key": source,
                    "source_sample": pair["left"]["source_sample"],
                    "predicted_action_trajectory": action.tolist(),
                    "branch_mode": role,
                    "action_trajectory_source": "validation_only_scaled_matched_action",
                }
            )
        audit_rows.append(
            {
                "source_key": source,
                "stratum": stratum,
                "fast_scale": fast_scale,
                "slow_scale": slow_scale,
                "max_yaw_difference_rad": 0.0,
            }
        )
    for role in ("fast", "slow"):
        directory = output_root / role / "shard_0"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps(rows[role], indent=2) + "\n", encoding="utf-8"
        )
    report = {
        "protocol": "validation-only-orthogonal-progress-intervention-v1",
        "source_count": len(selected),
        "quotas": quotas,
        "selection_seed": selection_seed,
        "fast_scale": fast_scale,
        "slow_scale": slow_scale,
        "yaw_identical_by_construction": True,
        "sources": audit_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fast-scale", type=float, default=1.25)
    parser.add_argument("--slow-scale", type=float, default=0.75)
    parser.add_argument("--quota", action="append", required=True, help="STRATUM=COUNT")
    parser.add_argument("--selection-seed", default="progress-pilot-v1")
    args = parser.parse_args()
    quotas = {name: int(count) for name, count in (item.split("=", 1) for item in args.quota)}
    report = build_roots(
        input_root=args.input_root,
        output_root=args.output_root,
        fast_scale=args.fast_scale,
        slow_scale=args.slow_scale,
        quotas=quotas,
        selection_seed=args.selection_seed,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
