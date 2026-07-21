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

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_path_evidence_segmented_vnext.py}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_path_evidence_segmented_vnext}"
RESUME_FROM="${RESUME_FROM:-work_dirs/iac_navsim_future_dinov2_path_evidence_vnext/checkpoints/latest.pth}"

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-100}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-128}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
BENCH_MAX_GROUPS="${BENCH_MAX_GROUPS:-200}"

mkdir -p "$WORK_DIR"

echo "[IAC] Config:   ${CONFIG}"
echo "[IAC] Work dir: ${WORK_DIR}"
echo "[IAC] Resume:   ${RESUME_FROM}"
echo "[IAC] GPU:      ${CUDA_VISIBLE_DEVICES}"

"$PYTHON_BIN" -m py_compile \
  train.py \
  train_dinov2_v5_minimal.py \
  benchmark_wam.py \
  tools/audit_iac_ambiguity.py \
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
    2>&1 | tee "$WORK_DIR/train.log"
else
  echo "[IAC] RUN_TRAIN=0, skipping training."
fi

if [[ "$RUN_EVAL" != "0" ]]; then
  CKPT="${CKPT:-$WORK_DIR/checkpoints/best.pth}"
  if [[ ! -f "$CKPT" ]]; then
    CKPT="$WORK_DIR/checkpoints/latest.pth"
  fi

  for split in regular low_iou holdout; do
    case "$split" in
      regular)
        INPUT="indices_navsim_future/consistency_val.jsonl"
        OUT_DIR="$WORK_DIR/regular_g200"
        ;;
      low_iou)
        INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl"
        OUT_DIR="$WORK_DIR/low_iou_g200"
        ;;
      holdout)
        INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl"
        OUT_DIR="$WORK_DIR/holdout_low_iou_g200"
        ;;
    esac

    mkdir -p "$OUT_DIR"
    "$PYTHON_BIN" benchmark_wam.py \
      --input "$INPUT" \
      --checkpoint "$CKPT" \
      --config "$CONFIG" \
      --model-kind dinov2 \
      --output-dir "$OUT_DIR" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-groups "$BENCH_MAX_GROUPS" \
      --consistency-score-key consistency_logit \
      --path-causal-metrics \
      --trajectory-specific-causal-metrics \
      --wrong-path-selection mask_iou \
      2>&1 | tee "$OUT_DIR.log"

    "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
      --scores "$OUT_DIR/wam_iac_scores.jsonl" \
      --output "$OUT_DIR/ambiguity_audit.json" \
      --per-sample-output "$OUT_DIR/ambiguity_groups.jsonl" \
      || true
  done
fi

echo "[IAC] segmented vNext run complete: ${WORK_DIR}"
