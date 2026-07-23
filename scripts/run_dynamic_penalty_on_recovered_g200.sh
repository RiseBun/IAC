#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${IAC_ROOT:-}" ]]; then
  ROOT="$IAC_ROOT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniforge3/envs/drivingworld/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${FALLBACK_PYTHON_BIN:-python3}"
fi
export PYTHONUNBUFFERED=1

RECOVERED_DIR="${RECOVERED_DIR:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_recovered_set_vnext/g200_recovered_set_k8}"
DYNAMIC_DIR="${DYNAMIC_DIR:-work_dirs/iac_dynamic_evidence_recovered_g200}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_dynamic_evidence_recovered_g200/regular_calibrated_penalty}"
RECOVERED_ALPHA="${RECOVERED_ALPHA:-0.4}"
CALIB_SPLIT="${CALIB_SPLIT:-regular}"
THRESHOLD_GRID="${THRESHOLD_GRID:--0.5,-0.25,0,0.25,0.5,0.75,1.0,1.25}"
BETA_GRID="${BETA_GRID:-0.02,0.05,0.1,0.2,0.35,0.5,0.75,1.0,1.5,2.0}"
export WORK_DIR

"$PYTHON_BIN" -m py_compile tools/apply_dynamic_evidence_penalty.py
mkdir -p "$WORK_DIR"

CALIB_PRIMARY="$RECOVERED_DIR/${CALIB_SPLIT}_fused_consistency_path_recovered_a${RECOVERED_ALPHA}_scores.jsonl"
CALIB_DYNAMIC="$DYNAMIC_DIR/${CALIB_SPLIT}/fused_dynamic_rows.jsonl"
CALIB_SUMMARY="$WORK_DIR/${CALIB_SPLIT}_calibration_summary.json"
export CALIB_SUMMARY

"$PYTHON_BIN" tools/apply_dynamic_evidence_penalty.py \
  --primary-scores "$CALIB_PRIMARY" \
  --dynamic-rows "$CALIB_DYNAMIC" \
  --output-scores "$WORK_DIR/${CALIB_SPLIT}_calibrated_best_scores.jsonl" \
  --output-summary "$CALIB_SUMMARY" \
  --threshold-grid="$THRESHOLD_GRID" \
  --beta-grid="$BETA_GRID" \
  --label "dynamic_penalty_regular_calibrated_a${RECOVERED_ALPHA}" \
  --sweep

read -r THRESHOLD BETA < <("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["CALIB_SUMMARY"]).read_text(encoding="utf-8"))
final = summary["final"]
print(final["threshold"], final["beta"])
PY
)

for split in regular low_iou holdout; do
  "$PYTHON_BIN" tools/apply_dynamic_evidence_penalty.py \
    --primary-scores "$RECOVERED_DIR/${split}_fused_consistency_path_recovered_a${RECOVERED_ALPHA}_scores.jsonl" \
    --dynamic-rows "$DYNAMIC_DIR/${split}/fused_dynamic_rows.jsonl" \
    --output-scores "$WORK_DIR/${split}_fixed_scores.jsonl" \
    --output-summary "$WORK_DIR/${split}_fixed_summary.json" \
    --threshold "$THRESHOLD" \
    --beta "$BETA" \
    --label "dynamic_penalty_regular_fixed_a${RECOVERED_ALPHA}"
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

work_dir = Path(os.environ.get(
    "WORK_DIR",
    "work_dirs/iac_dynamic_evidence_recovered_g200/regular_calibrated_penalty",
))
rows = []
for split in ("regular", "low_iou", "holdout"):
    data = json.loads((work_dir / f"{split}_fixed_summary.json").read_text(encoding="utf-8"))
    rows.append({
        "split": split,
        "original_hard_top1": data["original"]["hard_top1"],
        "original_ambiguity_adjusted_top1": data["original"]["ambiguity_adjusted_top1"],
        "original_hard_above": data["original"]["hard_mismatch_above_gt_by_score"],
        "final_hard_top1": data["final"]["hard_top1"],
        "final_ambiguity_adjusted_top1": data["final"]["ambiguity_adjusted_top1"],
        "final_hard_above": data["final"]["hard_mismatch_above_gt_by_score"],
        "threshold": data["final"]["threshold"],
        "beta": data["final"]["beta"],
    })
out = {"work_dir": str(work_dir), "rows": rows}
(work_dir / "summary_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
for row in rows:
    print(row, flush=True)
PY
