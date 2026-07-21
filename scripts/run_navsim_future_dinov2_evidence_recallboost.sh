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

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_evidence_recallboost.py}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_evidence_recallboost}"
RESUME_FROM="${RESUME_FROM:-work_dirs/iac_navsim_future_dinov2_evidence_ddp4_resume/checkpoints/best.pth}"

EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1800}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-500}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-64}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
if [[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]]; then
  export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-${TORCH_HOME}/hub/facebookresearch_dinov2_main}"
fi

mkdir -p "$WORK_DIR"

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

BENCHMARK_CHECKPOINT="$WORK_DIR/checkpoints/best.pth"
if [[ ! -f "$BENCHMARK_CHECKPOINT" ]]; then
  BENCHMARK_CHECKPOINT="$WORK_DIR/checkpoints/latest.pth"
fi
if [[ ! -f "$BENCHMARK_CHECKPOINT" ]]; then
  BENCHMARK_CHECKPOINT="$RESUME_FROM"
fi

"$PYTHON_BIN" benchmark_wam.py \
  --input indices_navsim_future/consistency_val.jsonl \
  --checkpoint "$BENCHMARK_CHECKPOINT" \
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
