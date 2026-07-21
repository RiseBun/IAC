"""Audit visual indistinguishability for IAC-PathBench v3.2.

This tool classifies hard top-1 misses using recovered-path conformal support.
A near speed/lateral/heading winner is accepted as visually indistinguishable
only when both GT and winner are inside the recovered-path support set and the
score gap is small.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-group-output")
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--wam-key", default="wam_name")
    parser.add_argument("--score-key", default="iac_consistency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(_repo_root()))
    from benchmark_wam import _summary  # type: ignore

    rows = _load_jsonl(Path(args.scores))
    summary = _summary(rows, args.wam_key, args.group_key, args.score_key)
    recovered = summary.get("recovered_set_metrics", {})
    visual = (
        summary.get("iac_pathbench_v3", {})
        .get("diagnostic_metrics", {})
        .get("visual_indistinguishability", {})
    )
    report = {
        "scores": args.scores,
        "score_key": args.score_key,
        "num_groups": recovered.get("num_groups"),
        "hard_top1": summary.get("iac_pathbench_v3", {})
        .get("secondary_ranking_metrics", {})
        .get("hard_top1"),
        "visual_indistinguishability": visual,
        "support_categories": recovered.get("support_categories"),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.per_group_output:
        records = recovered.get("per_group")
        if records is None:
            records = []
            # _summary removes per_group before attaching metrics, so recompute it by
            # calling the lower-level helper when detailed records are requested.
            from benchmark_wam import _recovered_set_metrics  # type: ignore

            recovered_with_groups = _recovered_set_metrics(
                rows,
                args.group_key,
                args.wam_key,
            )
            records = recovered_with_groups.get("per_group", [])
        sample_path = Path(args.per_group_output)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
