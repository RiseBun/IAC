#!/usr/bin/env python3
"""Join counterfactual manifests with candidate-blind Level-1 outputs."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = _read(args.manifest)
    scores = {str(row["sample_id"]): row for row in _read(args.scores)}
    manifest_ids = {str(row["sample_id"]) for row in manifests}
    missing = sorted(manifest_ids - set(scores))
    extra = sorted(set(scores) - manifest_ids)
    if missing or extra:
        raise ValueError(
            f"manifest/score sample mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    rows = []
    for manifest in manifests:
        sample_id = str(manifest["sample_id"])
        score = scores[sample_id]
        if score.get("candidate_bank_used_by_decoder") is not False:
            raise ValueError(f"{sample_id}: Level-1 output is not candidate-blind")
        rows.append({**manifest, **score})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
