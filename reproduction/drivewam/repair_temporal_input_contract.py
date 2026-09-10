#!/usr/bin/env python3
"""Repair legacy DriveWAM pickles without overwriting their source archive."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .build_inputs import _model_image_times
except ImportError:  # Direct CLI execution sets the script directory on sys.path.
    from build_inputs import _model_image_times


def repair_sample(sample: dict[str, Any], *, source: str) -> dict[str, Any]:
    images = np.asarray(sample["images"])
    history = np.asarray(sample["history_poses"])
    future = list(sample["future_trajectory"])
    history_count = len(history)
    future_count = len(future)
    if history_count != 4 or future_count != 8:
        raise ValueError(f"{source}: expected 4 history poses and 8 future poses")
    if images.ndim != 4 or len(images) != history_count + future_count:
        raise ValueError(f"{source}: legacy images must be 4 history + 8 future")
    metadata = dict(sample.get("metadata") or {})
    image_paths = list(metadata.get("image_paths") or [])
    if image_paths and len(image_paths) != len(images):
        raise ValueError(f"{source}: metadata image_paths do not match images")

    repaired = dict(sample)
    repaired["images"] = np.ascontiguousarray(images[history_count - 1 :])
    metadata["image_paths"] = image_paths[history_count - 1 :] if image_paths else []
    future_times = list(metadata.get("future_times_s") or [])
    try:
        input_image_times = _model_image_times(future_times)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    metadata.update({
        "input_image_contract": "drivewam_current_plus_8_future_v1",
        "input_image_times_s": input_image_times,
        "temporal_contract_repair": {
            "source": source,
            "legacy_contract": "4_history_plus_8_future",
            "selected_indices": list(range(history_count - 1, history_count + future_count)),
        },
    })
    repaired["metadata"] = metadata
    if len(repaired["images"]) != 1 + future_count:
        raise AssertionError("repaired DriveWAM input must contain current + future frames")
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-per-shard", type=int, default=0)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root:
        raise ValueError("output root must differ from input root")

    written = []
    for shard in sorted(path for path in input_root.glob("shards/shard_*") if path.is_dir()):
        source_paths = sorted(shard.glob("sample_*.pkl"))
        if args.limit_per_shard > 0:
            source_paths = source_paths[: args.limit_per_shard]
        target_shard = output_root / "shards" / shard.name
        target_shard.mkdir(parents=True, exist_ok=True)
        manifest = []
        for source_path in source_paths:
            with source_path.open("rb") as handle:
                sample = pickle.load(handle)
            repaired = repair_sample(sample, source=str(source_path))
            target = target_shard / source_path.name
            with target.open("wb") as handle:
                pickle.dump(repaired, handle, protocol=pickle.HIGHEST_PROTOCOL)
            manifest.append({
                "sample": str(target),
                "source_sample": str(source_path),
                "source_key": str((repaired.get("metadata") or {}).get("source_key") or ""),
                "input_image_contract": "drivewam_current_plus_8_future_v1",
            })
            written.append(target)
        (target_shard / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "repair_summary.json").write_text(
        json.dumps({
            "input_root": str(input_root),
            "output_root": str(output_root),
            "samples": len(written),
            "source_archive_preserved": True,
            "input_image_contract": "drivewam_current_plus_8_future_v1",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_root), "samples": len(written)}, indent=2))


if __name__ == "__main__":
    main()
