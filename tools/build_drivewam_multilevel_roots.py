#!/usr/bin/env python3
"""Create native DriveWAM four-level pure-speed action roots from NavSim PKLs."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
from pathlib import Path
from typing import Any


ROLES = {"stop": 0.0, "slow": 0.5, "normal": 1.0, "fast": 1.5}


def build(input_root: Path, output_root: Path, seed_base: int = 9_103_000) -> dict[str, Any]:
    source_paths = sorted(input_root.glob("shard_*/*.pkl"))
    if not source_paths:
        raise ValueError(f"no source PKLs under {input_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    rows: list[dict[str, Any]] = []
    for index, source_path in enumerate(source_paths):
        sample = pickle.loads(source_path.read_bytes())
        metadata = sample.get("metadata") or {}
        source_key = str(metadata.get("source_key") or metadata.get("sample_id") or source_path.stem)
        base = [list(map(float, item.get("pose", item)[:3])) for item in sample.get("future_trajectory") or []]
        if len(base) != 8:
            raise ValueError(f"{source_path}: expected eight future poses")
        seed = seed_base + index
        x0 = base[0][0]
        for role, scale in ROLES.items():
            branch = copy.deepcopy(sample)
            action = [[x0 + scale * (point[0] - x0), point[1], point[2]] for point in base]
            for future, pose in zip(branch["future_trajectory"], action):
                future["pose"] = pose
            branch["metadata"].update({
                "source_key": source_key,
                "counterfactual_group_id": source_key,
                "history_fingerprint": source_key,
                "nuisance_seed": seed,
                "protocol": "iac-pure-speed-multilevel-v1",
                "intervention_type": "pure_speed",
                "speed_role": role,
                "longitudinal_scale": scale,
                "action_trajectory": action,
                "action_trajectory_source": "validation_only_scaled_longitudinal",
                "future_images_source": "drivewam_generated_pure_speed_multilevel_pending",
                "wam_model_id": "drivewam_navsim",
                "source_sample_original": str(source_path),
            })
            shard_dir = output_root / role / f"shard_{index // 64}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            branch_path = shard_dir / f"sample_{index:06d}.pkl"
            branch_path.write_bytes(pickle.dumps(branch, protocol=pickle.HIGHEST_PROTOCOL))
            rows.append({
                "sample_id": f"{source_key}::{role}",
                "source_key": source_key,
                "counterfactual_group_id": source_key,
                "history_fingerprint": source_key,
                "nuisance_seed": seed,
                "speed_role": role,
                "intervention_type": "pure_speed",
                "longitudinal_scale": scale,
                "source_sample": str(branch_path),
                "source_sample_original": str(source_path),
                "action_trajectory": action,
                "action_trajectory_source": "validation_only_scaled_longitudinal",
                "future_images_source": "drivewam_generated_pure_speed_multilevel_pending",
                "wam_model_id": "drivewam_navsim",
                "images_pending": True,
            })
    for role in ROLES:
        for shard_dir in sorted((output_root / role).glob("shard_*")):
            subset = [row for row in rows if row["speed_role"] == role and Path(row["source_sample"]).parent == shard_dir]
            (shard_dir / "manifest.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"protocol": "iac-pure-speed-multilevel-v1", "status": "images_pending", "source_count": len(source_paths), "branch_count": len(rows), "roles": ROLES, "output": str(output_root), "same_history_seed_contract": True}
    (output_root / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=9_103_000)
    args = parser.parse_args()
    print(json.dumps(build(args.input_root, args.output_root, args.seed_base), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
