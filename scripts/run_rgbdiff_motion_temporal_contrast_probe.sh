#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${IAC_ROOT:-}" ]]; then
  ROOT="$IAC_ROOT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniforge3/envs/drivingworld/bin/python}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]]; then
  export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-${TORCH_HOME}/hub/facebookresearch_dinov2_main}"
fi

CONFIG="${CONFIG:-configs/train_navsim_future_rgbdiff_motion_temporal_contrast_probe.py}"
BASE_CKPT="${BASE_CKPT:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_vnext/checkpoints/latest.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_rgbdiff_motion_temporal_contrast_probe}"
EPOCHS="${EPOCHS:-7}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-40}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-1}"
EVAL_MAX_GROUPS="${EVAL_MAX_GROUPS:-200}"

"$PYTHON_BIN" -m py_compile \
  iac_extensions/rgb_motion_head.py \
  train_scope_motion_head.py \
  tools/eval_scope_motion_evidence.py \
  "$CONFIG"

"$PYTHON_BIN" train_scope_motion_head.py \
  --config "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --resume-from "$BASE_CKPT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --max-val-steps "$MAX_VAL_STEPS" \
  --dinov2-freeze \
  --amp

CKPT="$WORK_DIR/checkpoints/best.pth"
EVAL_DIR="$WORK_DIR/scope_motion_evidence_eval"
mkdir -p "$EVAL_DIR"

for control in normal reverse_future roll_future shuffle_future zero_future; do
  "$PYTHON_BIN" tools/eval_scope_motion_evidence.py \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --split val \
    --control "$control" \
    --max-groups "$EVAL_MAX_GROUPS" \
    --batch-size 8 \
    --num-workers "$NUM_WORKERS" \
    --output-summary "$EVAL_DIR/${control}_summary.json" \
    --output-rows "$EVAL_DIR/${control}_rows.jsonl"
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get(
    "WORK_DIR",
    "work_dirs/iac_navsim_future_rgbdiff_motion_temporal_contrast_probe",
))
eval_dir = root / "scope_motion_evidence_eval"
rows = []
for path in sorted(eval_dir.glob("*_summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("source_summary", {})
    rows.append({
        "control": data["config"]["control"],
        "positive_vs_hard_logit_auc": data.get("positive_vs_hard_logit_auc"),
        "hard_mismatch_energy_above_gt_group_rate": data.get(
            "hard_mismatch_energy_above_gt_group_rate"
        ),
        "near_perturb_energy_above_gt_group_rate": data.get(
            "near_perturb_energy_above_gt_group_rate"
        ),
        "gt_energy": source.get("gt_pos", {}).get("energy_mean"),
        "time_shift_energy": source.get("time_shift_future", {}).get("energy_mean"),
        "image_swap_energy": source.get("image_swap", {}).get("energy_mean"),
        "traj_swap_energy": source.get("traj_swap", {}).get("energy_mean"),
    })
out = {"eval_dir": str(eval_dir), "rows": rows}
(eval_dir / "summary_table.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for row in rows:
    print(row, flush=True)
PY
