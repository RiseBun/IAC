#!/usr/bin/env python3
"""Export a privacy-preserving descriptor-only GS reference release."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

DESCRIPTORS = (
    "median_flow_magnitude_px",
    "horizontal_flow_center",
    "vertical_flow_center",
    "divergence",
    "curl",
)


def _pseudonym(source_key: str, salt: str) -> str:
    if not salt:
        raise ValueError("hmac salt is required; raw source keys must not be released")
    digest = hmac.new(salt.encode("utf-8"), source_key.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"gs:{digest[:32]}"


def export(rows: list[dict[str, Any]], *, salt: str, release_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not release_id.strip():
        raise ValueError("release_id is required")
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        source = str(row.get("source_key") or row.get("sample_id") or "")
        if not source:
            raise ValueError("every row requires source_key")
        try:
            interval = int(row.get("interval_index"))
        except (TypeError, ValueError) as error:
            raise ValueError("every row requires integer interval_index") from error
        key = (_pseudonym(source, salt), interval)
        if key in seen:
            raise ValueError(f"duplicate source/interval: {key}")
        seen.add(key)
        clean: dict[str, Any] = {
            "source_key": key[0],
            "interval_index": interval,
            "input_available": bool(row.get("input_available", True)),
        }
        if row.get("stratum") is not None:
            clean["stratum"] = str(row["stratum"])
        for descriptor in DESCRIPTORS:
            value = row.get(descriptor)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"non-numeric descriptor: {descriptor}") from error
            if not (value == value and abs(value) != float("inf")):
                raise ValueError(f"non-finite descriptor: {descriptor}")
            clean[descriptor] = value
        output.append(clean)
    manifest = {
        "protocol": "iac-gs-reference-descriptor-release-v1",
        "release_id": release_id,
        "source_count": len({row["source_key"] for row in output}),
        "interval_count": len(output),
        "source_key_scheme": "hmac_sha256_truncated_128bit",
        "contains": ["source_key_pseudonym", "interval_index", "stratum", *DESCRIPTORS],
        "omits": ["images", "camera_calibration", "ego_state", "trajectory", "raw_source_key", "private_paths"],
        "claim_boundary": "Descriptor-only release enables GS recomputation; it is not a release of logged future images or ground-truth trajectories.",
        "missing_value_policy": "unavailable_never_zero_fill",
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="private flattened GS reference JSONL")
    parser.add_argument("--output", type=Path, required=True, help="descriptor-only public JSONL")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--salt", required=True, help="operator-held HMAC salt; never commit it")
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    output, manifest = export(rows, salt=args.salt, release_id=args.release_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in output), encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
