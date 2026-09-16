#!/usr/bin/env python3
"""Audit whether execution rollouts use the same action as visual evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _source(value: Any) -> str:
    return str(value or "").split("::", 1)[0]


def audit(manifest: Path, rollout: Path, *, exact_tolerance: float = 1e-6, approximate_tolerance: float = 1e-2) -> dict[str, Any]:
    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rollout_rows = [json.loads(line) for line in rollout.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        by_source.setdefault(_source(row.get("source_key")), []).append(row)
    distances: list[float] = []
    exact = approximate = 0
    matched = 0
    source_overlap = 0
    branch_counts: Counter[str] = Counter()
    details = []
    for row in rollout_rows:
        source = _source(row.get("source_key"))
        options = by_source.get(source, [])
        if not options:
            continue
        source_overlap += 1
        f = np.asarray(row.get("action_trajectory"), dtype=float)
        times = np.asarray(row.get("future_times_s"), dtype=float)
        if f.ndim != 2 or f.shape[1] != 3 or len(times) != len(f):
            continue
        indices = [int(np.argmin(np.abs(times - t))) for t in (1.0, 2.0, 3.0, 4.0)]
        target = f[indices]
        candidates = []
        for option in options:
            action = np.asarray(option.get("action_trajectory"), dtype=float)
            if action.shape != target.shape:
                continue
            candidates.append((float(np.max(np.abs(action - target))), str(option.get("branch_role") or "unknown")))
        if not candidates:
            continue
        distance, branch = min(candidates)
        matched += 1
        distances.append(distance)
        exact += distance <= exact_tolerance
        approximate += distance <= approximate_tolerance
        branch_counts[branch] += 1
        if len(details) < 20:
            details.append({"source_key": source, "rollout_branch": row.get("branch_id"), "nearest_visual_branch": branch, "max_abs_coordinate_error": distance})
    return {
        "protocol": "iac-action-alignment-audit-v1",
        "status": "passed" if matched and exact == matched else "failed_action_mismatch",
        "visual_manifest_rows": len(manifest_rows),
        "rollout_rows": len(rollout_rows),
        "source_overlap_rows": source_overlap,
        "matched_rows": matched,
        "exact_matches": exact,
        "approximate_matches": approximate,
        "exact_tolerance": exact_tolerance,
        "approximate_tolerance": approximate_tolerance,
        "nearest_visual_branch_counts": dict(branch_counts),
        "nearest_max_abs_error_median": float(np.median(distances)) if distances else None,
        "nearest_max_abs_error_p95": float(np.quantile(distances, 0.95)) if distances else None,
        "examples": details,
        "decision": "do_not_use_joint_correlation_as_same_action_validation" if exact != matched else "same_action_alignment_passed",
        "claim_boundary": "Source overlap is not action alignment; model, seed, branch and trajectory identity must be verified before joint validity claims.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.rollout)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "source_overlap_rows", "matched_rows", "exact_matches", "approximate_matches", "nearest_max_abs_error_median", "nearest_max_abs_error_p95", "decision")}, indent=2))


if __name__ == "__main__":
    main()
