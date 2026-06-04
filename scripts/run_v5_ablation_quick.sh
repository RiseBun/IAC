#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_HOME="${TORCH_HOME:-/root/.cache/torch}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/root/autodl-tmp}"
export NUPLAN_DB_ROOT="${NUPLAN_DB_ROOT:-/root/autodl-tmp/data/cache/mini}"
export NUPLAN_INDEX_ROOT="${NUPLAN_INDEX_ROOT:-/root/autodl-tmp/nuplan/indices_v4}"
export PYTHONUNBUFFERED=1

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-200}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-4096}"
MAX_RANKING_GROUPS="${MAX_RANKING_GROUPS:-512}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-64}"
RUN_RANKING="${RUN_RANKING:-1}"

run_variant() {
  local name="$1"
  local config="$2"

  if [[ -n "${ONLY:-}" && "${ONLY}" != "${name}" ]]; then
    return 0
  fi

  local work_dir="work_dirs/${name}_quick_s${MAX_TRAIN_STEPS}"
  local log_dir="${work_dir}/logs"
  mkdir -p "${log_dir}"

  echo "============================================================"
  echo "[train] ${name}"
  echo "  config=${config}"
  echo "  work_dir=${work_dir}"
  echo "  steps=${MAX_TRAIN_STEPS} val_steps=${MAX_VAL_STEPS} batch=${BATCH_SIZE}"
  echo "============================================================"
  "${PYTHON_BIN}" train_dinov2_v5_minimal.py \
    --config "${config}" \
    --work-dir "${work_dir}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-train-steps "${MAX_TRAIN_STEPS}" \
    --max-val-steps "${MAX_VAL_STEPS}" \
    --preflight-samples "${PREFLIGHT_SAMPLES}" \
    2>&1 | tee "${log_dir}/train.log"

  local checkpoint="${work_dir}/checkpoints/best.pth"
  local ranking_args=()
  if [[ "${RUN_RANKING}" != "0" ]]; then
    ranking_args=(--eval-ranking --max-ranking-groups "${MAX_RANKING_GROUPS}")
  fi

  echo "============================================================"
  echo "[eval] ${name}"
  echo "  checkpoint=${checkpoint}"
  echo "  max_samples=${EVAL_MAX_SAMPLES} ranking_groups=${MAX_RANKING_GROUPS}"
  echo "============================================================"
  "${PYTHON_BIN}" eval_dinov2_critic.py \
    --checkpoint "${checkpoint}" \
    --split val \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --max-samples "${EVAL_MAX_SAMPLES}" \
    "${ranking_args[@]}" \
    --output-prefix "${name}_val_${EVAL_MAX_SAMPLES}_rank${MAX_RANKING_GROUPS}" \
    2>&1 | tee "${log_dir}/eval.log"
}

run_variant "v5_base_cnn" "configs/train_dinov2_v5_base_cnn.py"
run_variant "v5_d1d4" "configs/train_dinov2_v5_d1d4.py"
run_variant "v5_dist" "configs/train_dinov2_v5_dist.py"
run_variant "v5_minimal" "configs/train_dinov2_v5_minimal.py"
