#!/usr/bin/env python3
"""Filter an AS manifest and visual probe to a declared source pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-probe", type=Path, required=True)
    args = parser.parse_args()

    pool_rows = _jsonl(args.pool)
    pool = {str(row["source_key"]) for row in pool_rows}
    manifest = [row for row in _jsonl(args.manifest) if str(row.get("source_key")) in pool]
    found = {str(row.get("source_key")) for row in manifest}
    missing = sorted(pool - found)
    if missing:
        raise ValueError(f"manifest lacks {len(missing)} declared pool sources")

    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    manifest_ids = {str(row["sample_id"]) for row in manifest}
    probe_rows = [row for row in probe.get("rows", []) if str(row.get("sample_id")) in manifest_ids]
    probe_ids = {str(row.get("sample_id")) for row in probe_rows}
    missing_probe = sorted(manifest_ids - probe_ids)
    if missing_probe:
        raise ValueError(f"probe lacks {len(missing_probe)} selected manifest rows")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_probe.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    output_probe = {key: value for key, value in probe.items() if key != "rows"}
    output_probe.update({
        "source_probe": str(args.probe),
        "source_pool": str(args.pool),
        "rows": probe_rows,
    })
    args.output_probe.write_text(json.dumps(output_probe, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_count": len(found),
        "manifest_rows": len(manifest),
        "probe_rows": len(probe_rows),
        "selection_uses_scores": False,
    }, indent=2))


if __name__ == "__main__":
    main()
