#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$HOME/miniforge3/envs/drivingworld/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniforge3/envs/drivingworld/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
if [[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]]; then
  export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-${TORCH_HOME}/hub/facebookresearch_dinov2_main}"
fi

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_progress_alignment.py}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_progress_alignment}"
RESUME_FROM="${RESUME_FROM:-work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/checkpoints/latest.pth}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-100}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-128}"

REGULAR_INDEX="${REGULAR_INDEX:-indices_navsim_future/consistency_val.jsonl}"
LOW_IOU_INDEX="${LOW_IOU_INDEX:-indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl}"
LOW_IOU_REPORT="${LOW_IOU_REPORT:-indices_navsim_future/diagnostics/consistency_val_low_iou_g200_report.json}"
HOLDOUT_LOW_IOU_INDEX="${HOLDOUT_LOW_IOU_INDEX:-indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl}"
HOLDOUT_LOW_IOU_REPORT="${HOLDOUT_LOW_IOU_REPORT:-indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399_report.json}"
BENCH_MAX_GROUPS="${BENCH_MAX_GROUPS:-200}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_CAUSAL="${RUN_CAUSAL:-1}"
PROGRESS_FUSION_BETA="${PROGRESS_FUSION_BETA:-0.0}"
PROGRESS_FUSION_MODE="${PROGRESS_FUSION_MODE:-final_displacement}"
PROGRESS_FUSION_SCALE="${PROGRESS_FUSION_SCALE:-40.0}"
CAUSAL_ARGS=()
if [[ "$RUN_CAUSAL" != "0" ]]; then
  CAUSAL_ARGS=(
    --path-causal-metrics
    --trajectory-specific-causal-metrics
    --wrong-path-selection mask_iou
  )
fi

mkdir -p "$WORK_DIR"
mkdir -p "$(dirname "$LOW_IOU_INDEX")"

if [[ ! -s "$LOW_IOU_INDEX" ]]; then
  "$PYTHON_BIN" tools/build_low_iou_subset.py \
    --input "$REGULAR_INDEX" \
    --output "$LOW_IOU_INDEX" \
    --report "$LOW_IOU_REPORT" \
    --max-groups "$BENCH_MAX_GROUPS"
fi
if [[ ! -s "$HOLDOUT_LOW_IOU_INDEX" ]]; then
  "$PYTHON_BIN" tools/build_low_iou_subset.py \
    --input "$REGULAR_INDEX" \
    --output "$HOLDOUT_LOW_IOU_INDEX" \
    --report "$HOLDOUT_LOW_IOU_REPORT" \
    --max-groups "$BENCH_MAX_GROUPS" \
    --start-rank "$BENCH_MAX_GROUPS"
fi

echo "[IAC] GPU scope: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[IAC] Config:    ${CONFIG}"
echo "[IAC] Work dir:  ${WORK_DIR}"
echo "[IAC] Resume:    ${RESUME_FROM}"

"$PYTHON_BIN" -m py_compile \
  train.py \
  train_dinov2_v5_minimal.py \
  "$CONFIG"

if [[ "$RUN_TRAIN" != "0" ]]; then
  "$PYTHON_BIN" train_dinov2_v5_minimal.py \
    --config "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-train-steps "$MAX_TRAIN_STEPS" \
    --max-val-steps "$MAX_VAL_STEPS" \
    --preflight-samples "$PREFLIGHT_SAMPLES" \
    --resume-from "$RESUME_FROM" \
    --amp \
    2>&1 | tee "$WORK_DIR/train_progress_alignment.log"
else
  echo "[IAC] RUN_TRAIN=0, skipping training."
fi

if [[ -z "${CKPT:-}" ]]; then
  if [[ -f "$WORK_DIR/checkpoints/best.pth" ]]; then
    CKPT="$WORK_DIR/checkpoints/best.pth"
  elif [[ -f "$WORK_DIR/checkpoints/latest.pth" ]]; then
    CKPT="$WORK_DIR/checkpoints/latest.pth"
  else
    CKPT="$RESUME_FROM"
  fi
fi

run_eval() {
  local name="$1"
  local index_path="$2"
  local out_dir="$WORK_DIR/${name}"
  if [[ ! -f "$index_path" ]]; then
    echo "[IAC][SKIP] Missing index for ${name}: ${index_path}"
    return 0
  fi
  "$PYTHON_BIN" benchmark_wam.py \
    --input "$index_path" \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --model-kind dinov2 \
    --output-dir "$out_dir" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-groups "$BENCH_MAX_GROUPS" \
    --progress-fusion-beta "$PROGRESS_FUSION_BETA" \
    --progress-fusion-mode "$PROGRESS_FUSION_MODE" \
    --progress-fusion-scale "$PROGRESS_FUSION_SCALE" \
    "${CAUSAL_ARGS[@]}" \
    2>&1 | tee "$out_dir.log"

  "$PYTHON_BIN" tools/audit_iac_visual_indistinguishability.py \
    --scores "$out_dir/wam_iac_scores.jsonl" \
    --output "$out_dir/visual_indistinguishability_audit.json" \
    --per-group-output "$out_dir/visual_indistinguishability_groups.jsonl" \
    || true
}

run_eval "regular_g200" "$REGULAR_INDEX"
run_eval "low_iou_g200" "$LOW_IOU_INDEX"
run_eval "holdout_low_iou_g200" "$HOLDOUT_LOW_IOU_INDEX"

echo "[IAC] Progress alignment run complete: ${WORK_DIR}"
