#!/usr/bin/env python3
"""Compare source-key strata across two benchmark manifests."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row: dict[str, Any]) -> str:
    return str(row.get("source_key") or (row.get("lineage") or {}).get("source_key") or row.get("sample_id"))


def stratum(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") or {}
    return row.get("stratum") or metadata.get("stratum") or (row.get("controls") or {}).get("stratum")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    left_rows = read_jsonl(args.left)
    right_rows = read_jsonl(args.right)
    left: dict[str, set[str | None]] = defaultdict(set)
    right: dict[str, set[str | None]] = defaultdict(set)
    for row in left_rows:
        left[key(row)].add(stratum(row))
    for row in right_rows:
        right[key(row)].add(stratum(row))
    overlap = sorted(set(left) & set(right))
    cross = Counter()
    mismatches = []
    for source_key in overlap:
        l = sorted(left[source_key], key=lambda value: str(value))
        r = sorted(right[source_key], key=lambda value: str(value))
        cross[(l[0] if len(l) == 1 else "MULTI", r[0] if len(r) == 1 else "MULTI")] += 1
        if set(l) != set(r):
            mismatches.append({"source_key": source_key, "left": l, "right": r})
    output = {
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "left_unique_source_keys": len(left),
        "right_unique_source_keys": len(right),
        "overlap_unique_source_keys": len(overlap),
        "left_strata_on_overlap": dict(Counter(next(iter(left[k])) if len(left[k]) == 1 else "MULTI" for k in overlap)),
        "right_strata_on_overlap": dict(Counter(next(iter(right[k])) if len(right[k]) == 1 else "MULTI" for k in overlap)),
        "cross_tab": {f"{a}->{b}": n for (a, b), n in sorted(cross.items())},
        "stratum_mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
