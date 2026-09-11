#!/usr/bin/env python3
"""Create native DriveWAM intervention samples from pure-speed action rows."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any


def prepare(input_manifest: Path, output_root: Path, *, speed_role: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in input_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if str(row.get("speed_role")) == speed_role]
    if not rows:
        raise ValueError(f"no {speed_role} rows in {input_manifest}")
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(sorted(rows, key=lambda item: str(item["source_key"]))):
        source = str(row["source_key"])
        if source in seen:
            raise ValueError(f"duplicate source_key: {source}")
        seen.add(source)
        source_sample = Path(str(row["source_sample"]))
        with source_sample.open("rb") as handle:
            sample = copy.deepcopy(pickle.load(handle))
        trajectory = row.get("action_trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != 8:
            raise ValueError(f"{source}: expected eight action poses")
        future = list(sample.get("future_trajectory") or [])
        if len(future) < 8:
            raise ValueError(f"{source}: source sample has fewer than eight future poses")
        for step, pose in enumerate(trajectory):
            future[step]["pose"] = [float(value) for value in pose[:3]]
        sample["future_trajectory"] = future
        metadata = sample.setdefault("metadata", {})
        metadata.update({
            "source_key": source,
            "action_trajectory": trajectory,
            "action_trajectory_source": "validation_only_scaled_action",
            "future_images_source": "drivewam_generated_pure_speed_action",
            "intervention_type": "pure_speed",
            "speed_role": speed_role,
            "twin_id": row.get("twin_id"),
            "scene_group": row.get("scene_group"),
            "stratum": row.get("stratum"),
        })
        path = output_root / f"sample_{index:06d}.pkl"
        with path.open("wb") as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        written.append({"sample": str(path), "source_key": source, "speed_role": speed_role})
    (output_root / "manifest.json").write_text(json.dumps(written, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"protocol": "iac-pure-speed-drivewam-sample-adapter-v1", "speed_role": speed_role, "source_count": len(written), "output": str(output_root)}
    (output_root / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--speed-role", choices=("fast", "slow"), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_manifest, args.output_root, speed_role=args.speed_role), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
