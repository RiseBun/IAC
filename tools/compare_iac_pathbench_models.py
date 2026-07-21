#!/usr/bin/env python3
"""Compare frozen IAC-PathBench results across model families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Label=path/to/wam_iac_summary.json. Repeat for each model family.",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    return parser.parse_args()


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_item(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"expected label=path, got {raw!r}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"missing label in {raw!r}")
    return label, Path(path.strip())


def _extract(summary: Dict[str, Any]) -> Dict[str, Any]:
    v2 = summary.get("iac_pathbench_v2", {})
    primary = v2.get("primary_scientific_metrics", {})
    secondary = v2.get("secondary_ranking_metrics", {})
    diagnostic = v2.get("diagnostic_metrics", {})
    protocol = v2.get("protocol", {})
    return {
        "protocol_name": protocol.get("name"),
        "hard_top1_is_secondary": protocol.get("hard_top1_is_secondary"),
        "hard_top1": secondary.get("hard_top1"),
        "mrr": secondary.get("mrr"),
        "exact_path_delta": primary.get("exact_path_delta"),
        "path_minus_sky_delta": primary.get("path_minus_sky_delta"),
        "ambiguity_adjusted_top1": primary.get("ambiguity_adjusted_top1"),
        "likely_model_error_fraction": diagnostic.get("likely_model_error_fraction"),
        "raw_miss_fraction": diagnostic.get("raw_miss_fraction"),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "model",
        "protocol",
        "hard top1",
        "MRR",
        "exact-path delta",
        "path-minus-sky",
        "ambiguity-adjusted top1",
        "likely model error",
        "raw miss",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    _fmt(row["protocol_name"]),
                    _fmt(row["hard_top1"]),
                    _fmt(row["mrr"]),
                    _fmt(row["exact_path_delta"], 4),
                    _fmt(row["path_minus_sky_delta"], 4),
                    _fmt(row["ambiguity_adjusted_top1"]),
                    _fmt(row["likely_model_error_fraction"]),
                    _fmt(row["raw_miss_fraction"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for item in args.summary:
        label, path = _parse_item(item)
        summary = _load(path)
        extracted = _extract(summary)
        extracted["label"] = label
        extracted["path"] = str(path)
        rows.append(extracted)

    rows.sort(key=lambda row: row["label"])
    payload = {"rows": rows}
    table = _markdown_table(rows)
    print(table)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_out:
        out_path = Path(args.markdown_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
