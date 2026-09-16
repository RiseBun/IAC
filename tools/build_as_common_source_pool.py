#!/usr/bin/env python3
"""Select a model-output-blind, stratified common source pool for AS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank(source_key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{source_key}".encode()).hexdigest()


def select_sources(
    benchmark_rows: list[dict[str, Any]],
    available_rows: list[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    benchmark = {str(row["source_key"]): row for row in benchmark_rows}
    available: dict[str, dict[str, Any]] = {}
    for row in available_rows:
        source_key = str(row.get("source_key") or "")
        source_sample = (row.get("lineage") or {}).get("source_sample")
        if source_key in benchmark and source_sample and source_key not in available:
            available[source_key] = row

    selected: list[dict[str, Any]] = []
    for stratum, count in quotas.items():
        candidates = [
            key for key, row in benchmark.items()
            if row.get("stratum") == stratum and key in available
        ]
        candidates.sort(key=lambda key: _rank(key, seed))
        if len(candidates) < count:
            raise ValueError(f"stratum {stratum!r} has {len(candidates)} available sources, needs {count}")
        for key in candidates[:count]:
            source = available[key]
            selected.append({
                "pool_index": len(selected),
                "source_key": key,
                "benchmark_id": benchmark[key].get("benchmark_id"),
                "scene_group": benchmark[key].get("scene_group"),
                "stratum": stratum,
                "source_sample": source["lineage"]["source_sample"],
                "selection_hash": _rank(key, seed),
                "selection_uses_model_output": False,
            })
    return selected


def _materialize_symlinks(rows: list[dict[str, Any]], root: Path) -> None:
    target_dir = root / "drivewam_samples_logged"
    target_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = Path(row["source_sample"]).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        link = target_dir / f"sample_{int(row['pool_index']):06d}.pkl"
        if link.exists() or link.is_symlink():
            if link.resolve() != source:
                raise FileExistsError(f"refusing to replace {link}")
            continue
        os.symlink(source, link)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--available-manifest", type=Path, required=True)
    parser.add_argument("--quotas", help='JSON object, e.g. {"lateral_turn":4}')
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        help="Repeatable STRATUM=COUNT form; avoids shell-specific JSON quoting.",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path)
    args = parser.parse_args()

    quotas = {str(key): int(value) for key, value in json.loads(args.quotas).items()} if args.quotas else {}
    for item in args.quota:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"invalid --quota {item!r}; expected STRATUM=COUNT")
        quotas[key] = int(value)
    if not quotas or any(value < 1 for value in quotas.values()):
        raise ValueError("all stratum quotas must be positive")
    rows = select_sources(
        _read_jsonl(args.benchmark),
        _read_jsonl(args.available_manifest),
        quotas,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if args.sample_root is not None:
        _materialize_symlinks(rows, args.sample_root)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "source_count": len(rows),
        "quotas": quotas,
        "seed": args.seed,
        "selection_uses_model_output": False,
        "sample_root": str(args.sample_root.resolve()) if args.sample_root else None,
    }, indent=2))


if __name__ == "__main__":
    main()
