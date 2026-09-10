#!/usr/bin/env python3
"""Apply a frozen real-domain RAFT refinement-uncertainty interval gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    threshold = float(calibration["threshold"])
    output = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows = record["flow_structure"]["rows"]
        for row in rows:
            uncertainty = (row.get("refinement_uncertainty") or {}).get("median")
            passed = bool(
                row.get("input_available")
                and uncertainty is not None
                and float(uncertainty) <= threshold
            )
            row["input_available"] = passed
            row["uncertainty_gate"] = {
                "passed": passed,
                "threshold": threshold,
                "calibration_domain": calibration["reference_domain"],
                "calibration_protocol": calibration["protocol"],
            }
        count = sum(bool(row["input_available"]) for row in rows)
        record["flow_structure"]["available_interval_fraction"] = count / max(len(rows), 1)
        record["flow_structure"]["measurement_available"] = bool(rows) and count == len(rows)
        record["flow_structure"]["status"] = (
            "usable" if rows and count == len(rows) else "partial" if count else "abstain"
        )
        output.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output))
    print(json.dumps({
        "records": len(output),
        "threshold": threshold,
        "available_intervals": sum(
            bool(row["input_available"])
            for record in output for row in record["flow_structure"]["rows"]
        ),
        "total_intervals": sum(len(record["flow_structure"]["rows"]) for record in output),
    }, indent=2))


if __name__ == "__main__":
    main()
