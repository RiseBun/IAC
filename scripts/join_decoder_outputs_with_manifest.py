#!/usr/bin/env python3
"""Strictly join decoder outputs to their candidate-blind manifest metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def join_rows(
    manifests: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in manifests:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in by_id:
            raise ValueError(f"manifest sample_id is empty or duplicated: {sample_id!r}")
        by_id[sample_id] = row
    seen: set[str] = set()
    joined: list[dict[str, Any]] = []
    for row in outputs:
        sample_id = str(row.get("sample_id") or "")
        if sample_id in seen:
            raise ValueError(f"decoder sample_id is duplicated: {sample_id!r}")
        if sample_id not in by_id:
            raise ValueError(f"decoder sample_id is absent from manifest: {sample_id!r}")
        seen.add(sample_id)
        joined.append({**by_id[sample_id], **row})
    missing = sorted(set(by_id) - seen)
    if missing:
        raise ValueError(
            f"decoder outputs are missing {len(missing)} manifest rows; first={missing[0]!r}"
        )
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--decoder-output", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined = join_rows(_read(args.manifest), _read(args.decoder_output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            for row in joined
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(joined), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
