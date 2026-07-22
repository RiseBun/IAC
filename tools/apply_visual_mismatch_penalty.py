"""Apply a one-sided visual mismatch penalty to IAC score rows.

The gate score is a non-mismatch probability. Unlike logit fusion, this tool
does not reward high non-mismatch scores; it only subtracts a penalty when the
gate falls below a threshold. This preserves ambiguous near-trajectory ranking
while reducing hard visual-time mismatches.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _group_id(row: Dict[str, Any]) -> Any:
    return row.get("group_id") or row.get("anchor_id") or row.get("sample_id")


def _score_fields(rows: Sequence[Dict[str, Any]]) -> set[str]:
    fields = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("iac_consistency") and isinstance(value, (int, float)):
                fields.add(key)
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", required=True)
    parser.add_argument("--gate-scores", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label", default="visual_mismatch_penalty")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_rows = _load_rows(Path(args.primary_scores))
    gate_rows = _load_rows(Path(args.gate_scores))
    if len(primary_rows) != len(gate_rows):
        raise ValueError(f"row mismatch: primary={len(primary_rows)} gate={len(gate_rows)}")
    fields = sorted(_score_fields(primary_rows))
    gate_threshold_logit = _logit(float(args.threshold))
    out: List[Dict[str, Any]] = []
    for idx, (primary, gate) in enumerate(zip(primary_rows, gate_rows)):
        if _group_id(primary) != _group_id(gate):
            raise ValueError(f"group mismatch at row {idx}")
        gate_logit = float(gate.get("visual_non_mismatch_logit", _logit(gate["iac_consistency"])))
        penalty = max(0.0, gate_threshold_logit - gate_logit)
        row = dict(primary)
        for field in fields:
            row[field] = _sigmoid(_logit(float(primary[field])) - float(args.weight) * penalty)
        row["visual_mismatch_penalty"] = float(penalty)
        row["visual_non_mismatch_logit"] = gate_logit
        row["visual_non_mismatch"] = float(gate.get("visual_non_mismatch", gate.get("iac_consistency")))
        row["score_fusion_label"] = args.label
        row["score_fusion_aux"] = {
            "gate_scores": args.gate_scores,
            "weight": float(args.weight),
            "threshold": float(args.threshold),
        }
        out.append(row)
    _write_jsonl(Path(args.output_scores), out)
    print(f"wrote {args.output_scores}")


if __name__ == "__main__":
    main()
