#!/usr/bin/env python3
"""Report label metrics for already-frozen ordered-motion support decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_ACCEPTABLE = "gt_pos,perturb_speed,perturb_lateral,perturb_heading"
DEFAULT_HARD = "image_swap,time_shift_future,traj_swap,reverse_traj,high_pdm_image_mismatch"
STATES = ("supported", "unsupported", "insufficient_evidence")


def _csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    acceptable_sources: set[str],
    hard_sources: set[str],
) -> dict[str, Any]:
    if acceptable_sources & hard_sources:
        raise ValueError("acceptable and hard source sets must be disjoint")
    by_state: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_state = 0
    unknown_source = 0
    for row in rows:
        state = str(row.get("ordered_motion_support_state", ""))
        source = str(row.get("source_type", ""))
        if state not in STATES:
            unknown_state += 1
            continue
        if source not in acceptable_sources and source not in hard_sources:
            unknown_source += 1
        by_state[state][source] += 1
        by_source[source][state] += 1

    supported_known = sum(by_state["supported"][source] for source in acceptable_sources | hard_sources)
    unsupported_known = sum(by_state["unsupported"][source] for source in acceptable_sources | hard_sources)
    supported_correct = sum(by_state["supported"][source] for source in acceptable_sources)
    unsupported_correct = sum(by_state["unsupported"][source] for source in hard_sources)
    per_source = {}
    for source in sorted(by_source):
        counts = by_source[source]
        total = sum(counts.values())
        per_source[source] = {
            "rows": total,
            "state_counts": {state: counts[state] for state in STATES},
            "state_fractions": {
                state: _ratio(counts[state], total) for state in STATES
            },
        }
    return {
        "rows": len(rows),
        "unknown_state_rows": unknown_state,
        "unknown_source_rows": unknown_source,
        "supported_precision": _ratio(supported_correct, supported_known),
        "supported_decisions": supported_known,
        "unsupported_precision": _ratio(unsupported_correct, unsupported_known),
        "unsupported_decisions": unsupported_known,
        "state_source_counts": {
            state: dict(sorted(by_state[state].items())) for state in STATES
        },
        "per_source": per_source,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--acceptable-sources", default=DEFAULT_ACCEPTABLE)
    parser.add_argument("--hard-sources", default=DEFAULT_HARD)
    args = parser.parse_args()
    score_path = Path(args.scores)
    with score_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = {
        "kind": "ordered_motion_support_label_audit_v1",
        "source_labels_used_for_report_only": True,
        "thresholds_or_decisions_modified": False,
        "scores": str(score_path),
        "scores_sha256": _sha256(score_path),
        "acceptable_sources": sorted(_csv(args.acceptable_sources)),
        "hard_sources": sorted(_csv(args.hard_sources)),
        "metrics": audit_rows(
            rows,
            acceptable_sources=_csv(args.acceptable_sources),
            hard_sources=_csv(args.hard_sources),
        ),
    }
    output = Path(args.output_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
