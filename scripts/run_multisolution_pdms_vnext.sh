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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]]; then
  export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-${TORCH_HOME}/hub/facebookresearch_dinov2_main}"
fi

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_multisolution_pdms_vnext.py}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_multisolution_pdms_vnext}"
RESUME_FROM="${RESUME_FROM:-work_dirs/iac_navsim_future_dinov2_structured_rules_vnext/checkpoints/latest.pth}"

EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1500}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-200}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-128}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

mkdir -p "$WORK_DIR"

echo "[IAC] Config:   ${CONFIG}"
echo "[IAC] Work dir: ${WORK_DIR}"
echo "[IAC] Resume:   ${RESUME_FROM}"
echo "[IAC] GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "[IAC] NPROC:    ${NPROC_PER_NODE}"
echo "[IAC] Steps:    train=${MAX_TRAIN_STEPS} val=${MAX_VAL_STEPS} epochs=${EPOCHS}"

"$PYTHON_BIN" -m py_compile \
  train.py \
  train_dinov2_v5_minimal.py \
  "$CONFIG"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  train_dinov2_v5_minimal.py \
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
