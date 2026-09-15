"""Join a candidate-blind visual probe with benchmark-v3 action states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iac_new.history_conditioned_as import aggregate, score_row


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("real_logged", "real_counterpart", "wam"), required=True)
    parser.add_argument(
        "--as-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "as_history_conditioned_v1.json",
    )
    parser.add_argument(
        "--visual-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "visual_output_layer_v1.json",
    )
    args = parser.parse_args()
    as_config = json.loads(args.as_config.read_text(encoding="utf-8"))
    visual_protocol = json.loads(args.visual_config.read_text(encoding="utf-8"))
    visual_config = visual_protocol["outputs"]
    manifest_rows = list(map(json.loads, args.manifest.open(encoding="utf-8")))
    manifest = {row["sample_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("manifest contains duplicate sample_id values")
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    probe_rows = probe.get("rows", [])
    probe_by_id = {row["sample_id"]: row for row in probe_rows}
    if len(probe_by_id) != len(probe_rows):
        raise ValueError("visual probe contains duplicate sample_id values")
    unknown_probe_ids = sorted(set(probe_by_id) - set(manifest))
    if unknown_probe_ids:
        raise ValueError(f"visual probe contains {len(unknown_probe_ids)} rows absent from manifest")
    rows = []
    for record in manifest_rows:
        row = probe_by_id.get(record["sample_id"], {"intervals": []})
        rows.append(score_row(row, record, config=visual_config))
    output = {
        "protocol": as_config["protocol"],
        "visual_protocol": visual_protocol["protocol"],
        "as_config": str(args.as_config),
        "visual_config": str(args.visual_config),
        "mode": args.mode,
        "manifest": str(args.manifest),
        "probe": str(args.probe),
        "aggregate": aggregate(rows, mode=args.mode, config=as_config),
        "rows": rows,
        "claim_boundary": "AS compares independently extracted coarse visual yaw/progress with the supplied trajectory. It is not exact SE(3), physical realism, or proof that the trajectory causally controls image generation. WAM action is not ground truth; real_logged is the calibration/upper-bound condition.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
