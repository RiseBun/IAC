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
CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_evidence.py}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_evidence_quick}"
TRAIN_INDEX="${TRAIN_INDEX:-indices_navsim_future/consistency_train.jsonl}"
VAL_INDEX="${VAL_INDEX:-indices_navsim_future/consistency_val.jsonl}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-500}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-4096}"
MAX_RANKING_GROUPS="${MAX_RANKING_GROUPS:-512}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-256}"

export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
if [[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]]; then
  export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-${TORCH_HOME}/hub/facebookresearch_dinov2_main}"
fi

mkdir -p "$WORK_DIR"

"$PYTHON_BIN" tools/audit_consistency_index.py \
  "$TRAIN_INDEX" \
  "$VAL_INDEX" \
  --fail-positive-exact-overlap 0.01 \
  --fail-positive-any-overlap 0.05 \
  --json-out "$WORK_DIR/index_audit.json"

"$PYTHON_BIN" train_dinov2_v5_minimal.py \
  --config "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --max-val-steps "$MAX_VAL_STEPS" \
  --preflight-samples "$PREFLIGHT_SAMPLES" \
  --amp \
  2>&1 | tee "$WORK_DIR/train.log"

"$PYTHON_BIN" benchmark_wam.py \
  --input "$VAL_INDEX" \
  --checkpoint "$WORK_DIR/checkpoints/best.pth" \
  --config "$CONFIG" \
  --model-kind dinov2 \
  --output-dir "$WORK_DIR/benchmark_val_${MAX_EVAL_SAMPLES}" \
  --max-samples "$MAX_EVAL_SAMPLES" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  2>&1 | tee "$WORK_DIR/benchmark.log"

"$PYTHON_BIN" tools/analyze_wam_scores.py \
  --scores "$WORK_DIR/benchmark_val_${MAX_EVAL_SAMPLES}/wam_iac_scores.jsonl" \
  --output "$WORK_DIR/benchmark_val_${MAX_EVAL_SAMPLES}/score_analysis.json" \
  --csv-errors "$WORK_DIR/benchmark_val_${MAX_EVAL_SAMPLES}/top_errors.csv" \
  --top-k 50
