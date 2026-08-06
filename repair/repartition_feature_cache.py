#!/usr/bin/env python3
"""Reorder existing feature caches to match a repaired JSONL split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def _read_sample_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [str(json.loads(line)["sample_id"]) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repartition_cache(
    caches: Sequence[Mapping[str, Any]], desired_sample_ids: Sequence[str]
) -> dict[str, Any]:
    if not caches:
        raise ValueError("at least one input cache is required")
    keys = set(caches[0])
    if any(set(cache) != keys for cache in caches[1:]):
        raise ValueError("input caches must contain identical keys")
    locations: dict[str, tuple[int, int]] = {}
    row_counts: list[int] = []
    for cache_index, cache in enumerate(caches):
        ids = [str(value) for value in cache["sample_id"]]
        row_counts.append(len(ids))
        for row_index, sample_id in enumerate(ids):
            if sample_id in locations:
                raise ValueError(f"duplicate sample_id across caches: {sample_id}")
            locations[sample_id] = (cache_index, row_index)
    missing = [sample_id for sample_id in desired_sample_ids if sample_id not in locations]
    if missing:
        raise ValueError(f"desired rows are absent from input caches: {missing[:10]}")
    if len(set(desired_sample_ids)) != len(desired_sample_ids):
        raise ValueError("desired sample_id list contains duplicates")

    result: dict[str, Any] = {}
    for key in sorted(keys):
        values = [cache[key] for cache in caches]
        if all(
            isinstance(value, torch.Tensor) and value.shape[0] == count
            for value, count in zip(values, row_counts)
        ):
            result[key] = torch.stack(
                [values[cache_index][row_index] for cache_index, row_index in map(locations.__getitem__, desired_sample_ids)]
            )
        elif all(
            isinstance(value, (list, tuple)) and len(value) == count
            for value, count in zip(values, row_counts)
        ):
            result[key] = [
                values[cache_index][row_index]
                for cache_index, row_index in map(locations.__getitem__, desired_sample_ids)
            ]
        elif key == "metadata":
            result[key] = dict(values[0])
        elif any(value != values[0] for value in values[1:]):
            raise ValueError(f"non-row cache value differs for key {key!r}")
        else:
            result[key] = values[0]
    metadata = dict(result.get("metadata", {}))
    metadata.update(
        {
            "rows": len(desired_sample_ids),
            "repartitioned_without_feature_reextraction": True,
            "repartition_source_cache_count": len(caches),
        }
    )
    result["metadata"] = metadata
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", action="append", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output-cache", required=True)
    args = parser.parse_args()
    input_paths = [Path(value) for value in args.input_cache]
    row_path = Path(args.rows)
    caches = [torch.load(path, map_location="cpu", weights_only=False) for path in input_paths]
    result = repartition_cache(caches, _read_sample_ids(row_path))
    output = Path(args.output_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(result, temporary)
    temporary.replace(output)
    summary = {
        "kind": "iac_feature_cache_repartition_v1",
        "input_caches": [str(path) for path in input_paths],
        "input_cache_sha256": [_sha256(path) for path in input_paths],
        "rows": str(row_path),
        "rows_sha256": _sha256(row_path),
        "output_cache": str(output),
        "output_cache_sha256": _sha256(output),
        "output_rows": len(result["sample_id"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
