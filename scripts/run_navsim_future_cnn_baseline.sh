#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_cnn_5k}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-2000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-20000}"
MAX_RANKING_GROUPS="${MAX_RANKING_GROUPS:-2000}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-512}"
CONFIG="${CONFIG:-configs/train_navsim_future_stable.py}"
TRAIN_INDEX="${TRAIN_INDEX:-indices_navsim_future/consistency_train.jsonl}"
VAL_INDEX="${VAL_INDEX:-indices_navsim_future/consistency_val.jsonl}"

mkdir -p "$WORK_DIR"

"$PYTHON_BIN" tools/audit_consistency_index.py \
  "$TRAIN_INDEX" \
  "$VAL_INDEX" \
  --fail-positive-exact-overlap 0.01 \
  --fail-positive-any-overlap 0.05 \
  --json-out "$WORK_DIR/index_audit.json"

"$PYTHON_BIN" train.py \
  --config "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --max-val-steps "$MAX_VAL_STEPS" \
  --preflight-samples "$PREFLIGHT_SAMPLES" \
  2>&1 | tee "$WORK_DIR/train.log"

"$PYTHON_BIN" eval_critic.py \
  --checkpoint "$WORK_DIR/checkpoints/best.pth" \
  --split val \
  --batch-size 32 \
  --max-samples "$MAX_EVAL_SAMPLES" \
  --eval-ranking \
  --max-ranking-groups "$MAX_RANKING_GROUPS" \
  2>&1 | tee "$WORK_DIR/eval.log"

"$PYTHON_BIN" benchmark_wam.py \
  --input "$VAL_INDEX" \
  --checkpoint "$WORK_DIR/checkpoints/best.pth" \
  --output-dir "$WORK_DIR/benchmark_val_${MAX_EVAL_SAMPLES}" \
  --max-samples "$MAX_EVAL_SAMPLES" \
  --batch-size 32 \
  --num-workers "$NUM_WORKERS" \
  2>&1 | tee "$WORK_DIR/benchmark.log"

"$PYTHON_BIN" tools/analyze_wam_scores.py \
  --scores "$WORK_DIR/benchmark_val_${MAX_EVAL_SAMPLES}/wam_iac_scores.jsonl" \
  --output "$WORK_DIR/benchmark_val_${MAX_EVAL_SAMPLES}/score_analysis.json" \
  --csv-errors "$WORK_DIR/benchmark_val_${MAX_EVAL_SAMPLES}/top_errors.csv" \
  --top-k 50
