"""Fuse multiple IAC score JSONL files in logit space."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _score_fields(rows: Iterable[Dict[str, Any]]) -> set[str]:
    fields: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith("iac_consistency") and isinstance(value, (int, float)):
                fields.add(key)
    return fields


def _group_id(row: Dict[str, Any]) -> Any:
    return row.get("group_id") or row.get("anchor_id") or row.get("sample_id")


def _recompute_delta_fields(row: Dict[str, Any]) -> None:
    if "iac_consistency" not in row:
        return
    score = float(row["iac_consistency"])
    if "iac_consistency_path_masked" in row:
        row["path_mask_delta"] = score - float(row["iac_consistency_path_masked"])
    if "iac_consistency_sky_masked" in row:
        row["sky_mask_delta"] = score - float(row["iac_consistency_sky_masked"])
    if "path_mask_delta" in row and "sky_mask_delta" in row:
        row["path_minus_sky_delta"] = (
            float(row["path_mask_delta"]) - float(row["sky_mask_delta"])
        )
    if "iac_consistency_wrong_path_masked" in row:
        row["wrong_path_delta"] = score - float(
            row["iac_consistency_wrong_path_masked"]
        )
    if "path_mask_delta" in row and "wrong_path_delta" in row:
        row["candidate_minus_wrong_path_delta"] = (
            float(row["path_mask_delta"]) - float(row["wrong_path_delta"])
        )
    if "iac_consistency_candidate_exclusive_path_masked" in row:
        row["candidate_exclusive_path_delta"] = score - float(
            row["iac_consistency_candidate_exclusive_path_masked"]
        )
    if "iac_consistency_wrong_exclusive_path_masked" in row:
        row["wrong_exclusive_path_delta"] = score - float(
            row["iac_consistency_wrong_exclusive_path_masked"]
        )
    if "candidate_exclusive_path_delta" in row and "wrong_exclusive_path_delta" in row:
        row["candidate_minus_wrong_exclusive_path_delta"] = (
            float(row["candidate_exclusive_path_delta"])
            - float(row["wrong_exclusive_path_delta"])
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", required=True)
    parser.add_argument(
        "--aux",
        action="append",
        default=[],
        metavar="PATH:WEIGHT",
        help="Auxiliary score JSONL plus logit-space weight. May repeat.",
    )
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--label", default="fused")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_rows = _load_rows(Path(args.primary_scores))
    aux_specs: List[tuple[Path, float]] = []
    for raw in args.aux:
        path_raw, sep, weight_raw = raw.rpartition(":")
        if not sep:
            raise SystemExit(f"--aux must be PATH:WEIGHT, got {raw!r}")
        aux_specs.append((Path(path_raw), float(weight_raw)))
    aux_rows = [(path, weight, _load_rows(path)) for path, weight in aux_specs]

    fields = _score_fields(primary_rows)
    for _, _, rows in aux_rows:
        fields &= _score_fields(rows)
    fields = set(sorted(fields))
    if not fields:
        raise SystemExit("no shared iac_consistency* fields to fuse")

    out_path = Path(args.output_scores)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, primary in enumerate(primary_rows):
            row = dict(primary)
            group = _group_id(primary)
            for path, _, rows in aux_rows:
                if idx >= len(rows):
                    raise ValueError(f"{path} has fewer rows than primary")
                if _group_id(rows[idx]) != group:
                    raise ValueError(
                        f"row {idx} group mismatch: {group!r} vs {_group_id(rows[idx])!r}"
                    )
            for field in fields:
                fused_logit = _logit(primary[field])
                for _, weight, rows in aux_rows:
                    fused_logit += weight * _logit(rows[idx][field])
                row[field] = _sigmoid(fused_logit)
            row["score_fusion_label"] = args.label
            row["score_fusion_aux"] = [
                {"path": str(path), "weight": weight}
                for path, weight, _ in aux_rows
            ]
            _recompute_delta_fields(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
