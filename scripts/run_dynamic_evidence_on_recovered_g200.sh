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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG="${CONFIG:-configs/train_navsim_future_rgbdiff_motion_temporal_contrast_probe.py}"
CKPT="${CKPT:-work_dirs/iac_navsim_future_rgbdiff_motion_temporal_contrast_probe_g1000/checkpoints/best.pth}"
FLOW_MODEL="${FLOW_MODEL:-work_dirs/iac_flow_speed_g1000/flow_speed_head.npz}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs}"
RECOVERED_DIR="${RECOVERED_DIR:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_recovered_set_vnext/g200_recovered_set_k8}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_dynamic_evidence_recovered_g200}"
FLOW_WEIGHT="${FLOW_WEIGHT:-0.125}"
EVAL_MAX_GROUPS="${EVAL_MAX_GROUPS:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
FLOW_WORKERS="${FLOW_WORKERS:-8}"

"$PYTHON_BIN" -m py_compile \
  tools/eval_scope_motion_evidence.py \
  tools/eval_flow_speed_evidence.py \
  tools/eval_dynamic_evidence_fusion.py

mkdir -p "$WORK_DIR"

for split in regular low_iou holdout; do
  INDEX="$RECOVERED_DIR/${split}_recovered_set_rows.jsonl"
  OUT_DIR="$WORK_DIR/$split"
  mkdir -p "$OUT_DIR"

  "$PYTHON_BIN" tools/eval_scope_motion_evidence.py \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --index "$INDEX" \
    --split val \
    --control normal \
    --max-groups "$EVAL_MAX_GROUPS" \
    --batch-size 8 \
    --num-workers "$NUM_WORKERS" \
    --output-summary "$OUT_DIR/rgbdiff_summary.json" \
    --output-rows "$OUT_DIR/rgbdiff_rows.jsonl"

  "$PYTHON_BIN" tools/eval_flow_speed_evidence.py \
    --index "$INDEX" \
    --image-root "$IMAGE_ROOT" \
    --model "$FLOW_MODEL" \
    --control normal \
    --workers "$FLOW_WORKERS" \
    --cache-dir "$WORK_DIR/flow_cache" \
    --max-groups "$EVAL_MAX_GROUPS" \
    --output-summary "$OUT_DIR/flow_summary.json" \
    --output-rows "$OUT_DIR/flow_rows.jsonl"

  "$PYTHON_BIN" tools/eval_dynamic_evidence_fusion.py \
    --scope-rows "$OUT_DIR/rgbdiff_rows.jsonl" \
    --flow-rows "$OUT_DIR/flow_rows.jsonl" \
    --weight-grid "1:${FLOW_WEIGHT}" \
    --output-summary "$OUT_DIR/fused_dynamic_summary.json" \
    --output-rows "$OUT_DIR/fused_dynamic_rows.jsonl"
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

work_dir = Path(os.environ.get("WORK_DIR", "work_dirs/iac_dynamic_evidence_recovered_g200"))
rows = []
for split in ("regular", "low_iou", "holdout"):
    fused_path = work_dir / split / "fused_dynamic_summary.json"
    rgb_path = work_dir / split / "rgbdiff_summary.json"
    flow_path = work_dir / split / "flow_summary.json"
    if not fused_path.exists():
        continue
    fused = json.loads(fused_path.read_text(encoding="utf-8"))["best"]
    rgb = json.loads(rgb_path.read_text(encoding="utf-8"))
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    rows.append({
        "split": split,
        "rgb_auc": rgb.get("positive_vs_hard_logit_auc"),
        "rgb_hard_above": rgb.get("hard_mismatch_energy_above_gt_group_rate"),
        "flow_auc": flow.get("positive_vs_hard_energy_auc"),
        "flow_hard_above": flow.get("hard_mismatch_energy_above_gt_group_rate"),
        "fused_auc": fused.get("auc"),
        "fused_hard_above": fused.get("hard_above"),
        "fused_image_swap_pair": fused.get("image_swap_pair"),
        "fused_time_shift_pair": fused.get("time_shift_pair"),
        "fused_traj_swap_pair": fused.get("traj_swap_pair"),
    })
out = {"work_dir": str(work_dir), "rows": rows}
(work_dir / "summary_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
for row in rows:
    print(row, flush=True)
PY
