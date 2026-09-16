#!/usr/bin/env python3
"""Score RCS from a paired benchmark manifest and frozen AS visual results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.response_consistency import score_response_consistency


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--as-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "response_consistency_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    as_payload = json.loads(args.as_result.read_text(encoding="utf-8"))
    # AS lineage is retained as a diagnostic.  RCS has a stricter, paired
    # same-history contract of its own; inheriting AS's single-clip boundary
    # gate would reject valid common-history counterfactual comparisons.
    lineage = as_payload.get("aggregate", {}).get("input_lineage_audit", {})
    result = score_response_consistency(_jsonl(args.manifest), as_payload.get("rows", []), config)
    result.update(
        {
            "manifest": str(args.manifest),
            "as_result": str(args.as_result),
            "config": str(args.config),
            "as_input_lineage_audit": lineage,
            "as_lineage_used_as_formal_gate": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
