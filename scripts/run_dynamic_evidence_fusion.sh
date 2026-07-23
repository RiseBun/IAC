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

RGBDIFF_EVAL_DIR="${RGBDIFF_EVAL_DIR:-work_dirs/iac_navsim_future_rgbdiff_motion_temporal_contrast_probe_g1000/scope_motion_evidence_eval}"
FLOW_EVAL_DIR="${FLOW_EVAL_DIR:-work_dirs/iac_flow_speed_g1000/evidence_eval}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_dynamic_evidence_fusion_g1000}"
WEIGHT_GRID="${WEIGHT_GRID:-1:0,1:0.025,1:0.05,1:0.075,1:0.1,1:0.125,1:0.15,1:0.175,1:0.2,1:0.25,1:0.3,1:0.35,1:0.4,1:0.5}"

"$PYTHON_BIN" -m py_compile tools/eval_dynamic_evidence_fusion.py
mkdir -p "$WORK_DIR"

"$PYTHON_BIN" tools/eval_dynamic_evidence_fusion.py \
  --scope-rows "$RGBDIFF_EVAL_DIR/normal_rows.jsonl" \
  --flow-rows "$FLOW_EVAL_DIR/normal_rows.jsonl" \
  --weight-grid "$WEIGHT_GRID" \
  --output-summary "$WORK_DIR/fine_summary.json" \
  --output-rows "$WORK_DIR/fine_best_rows.jsonl"

for control in reverse_future roll_future shuffle_future; do
  "$PYTHON_BIN" tools/eval_dynamic_evidence_fusion.py \
    --scope-rows "$RGBDIFF_EVAL_DIR/${control}_rows.jsonl" \
    --flow-rows "$FLOW_EVAL_DIR/${control}_rows.jsonl" \
    --weight-grid "1:0.125" \
    --output-summary "$WORK_DIR/${control}_summary.json" \
    --output-rows "$WORK_DIR/${control}_rows.jsonl"
done
