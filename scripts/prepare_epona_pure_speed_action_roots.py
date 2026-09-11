#!/usr/bin/env python3
"""Adapt pure-speed twin manifests to the Epona matched-action runner.

The Epona probe predates the pure-speed preparation format and expects one
``shard_*/manifest.json`` per action branch.  This adapter only rewrites
metadata; it does not generate images or alter the trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(
    input_manifest: Path,
    output_root: Path,
    *,
    shard_size: int = 64,
    speed_role: str | None = None,
) -> dict[str, Any]:
    rows = _rows(input_manifest)
    if speed_role is not None:
        rows = [row for row in rows if str(row.get("speed_role")) == speed_role]
        if not rows:
            raise ValueError(f"no rows with speed_role={speed_role}: {input_manifest}")
    if not rows:
        raise ValueError(f"empty pure-speed manifest: {input_manifest}")
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = str(row.get("source_key") or "")
        if not source:
            raise ValueError("every pure-speed row requires source_key")
        if source in by_source:
            raise ValueError(f"duplicate source_key in branch manifest: {source}")
        source_sample = row.get("source_sample")
        if not source_sample:
            raise ValueError(f"{source}: missing source_sample; regenerate pure-speed roots")
        sample_path = Path(str(source_sample))
        if not sample_path.is_file():
            raise ValueError(f"{source}: source_sample does not exist: {sample_path}")
        trajectory = row.get("action_trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != 8:
            raise ValueError(f"{source}: expected eight-point action_trajectory")
        by_source[source] = {
            "source_key": source,
            "source_sample": str(sample_path),
            "predicted_action_trajectory": trajectory,
            "intervention_type": row.get("intervention_type"),
            "speed_role": row.get("speed_role"),
            "twin_id": row.get("twin_id"),
            "scene_group": row.get("scene_group"),
            "stratum": row.get("stratum"),
            "action_trajectory_source": row.get("action_trajectory_source"),
        }

    ordered = [by_source[key] for key in sorted(by_source)]
    output_root.mkdir(parents=True, exist_ok=True)
    manifests: list[str] = []
    for shard_index in range((len(ordered) + shard_size - 1) // shard_size):
        shard = ordered[shard_index * shard_size : (shard_index + 1) * shard_size]
        shard_dir = output_root / f"shard_{shard_index}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        path = shard_dir / "manifest.json"
        path.write_text(json.dumps(shard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifests.append(str(path))
    report = {
        "protocol": "iac-pure-speed-epona-action-adapter-v1",
        "input_manifest": str(input_manifest),
        "source_count": len(ordered),
        "branch": speed_role or ordered[0]["speed_role"],
        "intervention_type": "pure_speed",
        "shard_size": shard_size,
        "manifest_paths": manifests,
        "images_pending": True,
    }
    (output_root / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--speed-role", choices=("fast", "slow"), required=True)
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("shard-size must be positive")
    print(json.dumps(prepare(args.input_manifest, args.output_root, shard_size=args.shard_size, speed_role=args.speed_role), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
