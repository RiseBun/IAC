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

FLOW_METHOD="${FLOW_METHOD:-raft_small}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_flow_speed_${FLOW_METHOD}_g1000}"
TRAIN_INDEX="${TRAIN_INDEX:-indices_navsim_future/consistency_train_official_pdms_highpdm_mismatch.jsonl}"
VAL_INDEX="${VAL_INDEX:-indices_navsim_future/consistency_val_official_pdms_highpdm_mismatch.jsonl}"
EVAL_INDEX="${EVAL_INDEX:-indices_navsim_future/consistency_val_official_pdms_highpdm_mismatch.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-144}"
MAX_TRAIN_POSITIVES="${MAX_TRAIN_POSITIVES:-1000}"
MAX_VAL_POSITIVES="${MAX_VAL_POSITIVES:-300}"
EVAL_MAX_GROUPS="${EVAL_MAX_GROUPS:-200}"
SEED="${SEED:-20260723}"
export WORK_DIR

"$PYTHON_BIN" -m py_compile \
  iac_extensions/flow_evidence.py \
  tools/flow_speed_head.py \
  tools/eval_flow_speed_evidence.py

mkdir -p "$WORK_DIR/evidence_eval"

"$PYTHON_BIN" tools/flow_speed_head.py fit \
  --train-index "$TRAIN_INDEX" \
  --val-index "$VAL_INDEX" \
  --image-root "$IMAGE_ROOT" \
  --output "$WORK_DIR/flow_speed_head.npz" \
  --method "$FLOW_METHOD" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --workers 1 \
  --cache-dir "$WORK_DIR/feature_cache" \
  --max-train-positives "$MAX_TRAIN_POSITIVES" \
  --max-val-positives "$MAX_VAL_POSITIVES" \
  --seed "$SEED"

for control in normal reverse_future roll_future shuffle_future; do
  "$PYTHON_BIN" tools/eval_flow_speed_evidence.py \
    --index "$EVAL_INDEX" \
    --image-root "$IMAGE_ROOT" \
    --model "$WORK_DIR/flow_speed_head.npz" \
    --control "$control" \
    --workers 1 \
    --cache-dir "$WORK_DIR/feature_cache" \
    --max-groups "$EVAL_MAX_GROUPS" \
    --seed "$SEED" \
    --output-summary "$WORK_DIR/evidence_eval/${control}_summary.json" \
    --output-rows "$WORK_DIR/evidence_eval/${control}_rows.jsonl"
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

work_dir = Path(os.environ.get("WORK_DIR", "work_dirs/iac_flow_speed_raft_small_g1000"))
rows = []
for control in ("normal", "reverse_future", "roll_future", "shuffle_future"):
    path = work_dir / "evidence_eval" / f"{control}_summary.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    pairwise = data.get("pairwise_accuracy", {})
    rows.append({
        "control": control,
        "auc": data.get("positive_vs_hard_energy_auc"),
        "hard_above": data.get("hard_mismatch_energy_above_gt_group_rate"),
        "near_above": data.get("near_perturb_energy_above_gt_group_rate"),
        "image_swap_pair": pairwise.get("gt_better_energy_vs_image_swap"),
        "time_shift_pair": pairwise.get("gt_better_energy_vs_time_shift_future"),
        "traj_swap_pair": pairwise.get("gt_better_energy_vs_traj_swap"),
    })
summary = {"work_dir": str(work_dir), "rows": rows}
(work_dir / "evidence_eval" / "summary_table.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for row in rows:
    print(row, flush=True)
PY
