#!/usr/bin/env python3
"""Score ordinal pure-speed response from a manifest and visual probe JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.progress_response import score_progress_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-action-delta", type=float, default=0.5)
    parser.add_argument("--min-inlier-fraction", type=float, default=0.20)
    parser.add_argument("--min-intervals", type=int, default=2)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    visual_payload = json.loads(args.visual.read_text(encoding="utf-8"))
    report = score_progress_response(
        manifest,
        list(visual_payload.get("rows") or []),
        min_action_delta=args.min_action_delta,
        min_inlier_fraction=args.min_inlier_fraction,
        min_intervals=args.min_intervals,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: report[key] for key in ("protocol", "declared_groups", "eligible_pairs", "scored_pairs", "coverage", "hits", "ordering_accuracy")}
    summary["four_role"] = {key: report["four_role"][key] for key in ("declared_groups", "scored_groups", "coverage", "adjacent_accuracy", "monotonic_accuracy")}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
