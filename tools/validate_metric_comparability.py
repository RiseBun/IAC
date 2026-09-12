#!/usr/bin/env python3
"""Validate frozen MAS/RCS directional-yaw comparability contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.metric_contract import validate_directional_yaw_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        result = validate_directional_yaw_config(config)
        result["config"] = str(path)
        reports.append(result)
    print(json.dumps({"status": "valid", "configs": reports}, indent=2))


if __name__ == "__main__":
    main()
