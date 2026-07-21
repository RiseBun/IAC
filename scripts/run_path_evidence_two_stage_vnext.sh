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

STAGE1_CONFIG="${STAGE1_CONFIG:-configs/train_navsim_future_dinov2_path_evidence_stage1_strong_vnext.py}"
STAGE1_WORK_DIR="${STAGE1_WORK_DIR:-work_dirs/iac_navsim_future_dinov2_path_evidence_stage1_strong_vnext}"
STAGE1_RESUME_FROM="${STAGE1_RESUME_FROM:-work_dirs/iac_navsim_future_dinov2_path_evidence_vnext/checkpoints/latest.pth}"

STAGE2_CONFIG="${STAGE2_CONFIG:-configs/train_navsim_future_dinov2_path_evidence_fused_vnext.py}"
STAGE2_WORK_DIR="${STAGE2_WORK_DIR:-work_dirs/iac_navsim_future_dinov2_path_evidence_fused_vnext}"

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-13}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-14}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STAGE1_MAX_TRAIN_STEPS="${STAGE1_MAX_TRAIN_STEPS:-400}"
STAGE1_MAX_VAL_STEPS="${STAGE1_MAX_VAL_STEPS:-100}"
STAGE2_MAX_TRAIN_STEPS="${STAGE2_MAX_TRAIN_STEPS:-250}"
STAGE2_MAX_VAL_STEPS="${STAGE2_MAX_VAL_STEPS:-100}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
BENCH_MAX_GROUPS="${BENCH_MAX_GROUPS:-200}"

mkdir -p "$STAGE1_WORK_DIR" "$STAGE2_WORK_DIR"

"$PYTHON_BIN" -m py_compile \
  train.py \
  train_dinov2_v5_minimal.py \
  benchmark_wam.py \
  tools/audit_iac_ambiguity.py \
  tools/sweep_iac_fused_scores.py \
  "$STAGE1_CONFIG" \
  "$STAGE2_CONFIG"

if [[ "$RUN_STAGE1" != "0" ]]; then
  echo "[IAC] Stage 1 evidence strengthening"
  "$PYTHON_BIN" train_dinov2_v5_minimal.py \
    --config "$STAGE1_CONFIG" \
    --work-dir "$STAGE1_WORK_DIR" \
    --epochs "$STAGE1_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-train-steps "$STAGE1_MAX_TRAIN_STEPS" \
    --max-val-steps "$STAGE1_MAX_VAL_STEPS" \
    --preflight-samples "$PREFLIGHT_SAMPLES" \
    --resume-from "$STAGE1_RESUME_FROM" \
    --amp \
    2>&1 | tee "$STAGE1_WORK_DIR/train.log"
else
  echo "[IAC] RUN_STAGE1=0, skipping stage 1."
fi

STAGE2_RESUME_FROM="${STAGE2_RESUME_FROM:-$STAGE1_WORK_DIR/checkpoints/latest.pth}"
if [[ ! -f "$STAGE2_RESUME_FROM" ]]; then
  STAGE2_RESUME_FROM="$STAGE1_WORK_DIR/checkpoints/best.pth"
fi

if [[ "$RUN_STAGE2" != "0" ]]; then
  echo "[IAC] Stage 2 fused judge training from ${STAGE2_RESUME_FROM}"
  "$PYTHON_BIN" train_dinov2_v5_minimal.py \
    --config "$STAGE2_CONFIG" \
    --work-dir "$STAGE2_WORK_DIR" \
    --epochs "$STAGE2_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-train-steps "$STAGE2_MAX_TRAIN_STEPS" \
    --max-val-steps "$STAGE2_MAX_VAL_STEPS" \
    --preflight-samples "$PREFLIGHT_SAMPLES" \
    --resume-from "$STAGE2_RESUME_FROM" \
    --amp \
    2>&1 | tee "$STAGE2_WORK_DIR/train.log"
else
  echo "[IAC] RUN_STAGE2=0, skipping stage 2."
fi

if [[ "$RUN_EVAL" != "0" ]]; then
  CKPT="${CKPT:-$STAGE2_WORK_DIR/checkpoints/best.pth}"
  if [[ ! -f "$CKPT" ]]; then
    CKPT="$STAGE2_WORK_DIR/checkpoints/latest.pth"
  fi

  for score_key in consistency_logit path_evidence_logit; do
    for split in regular low_iou holdout; do
      case "$split" in
        regular)
          INPUT="indices_navsim_future/consistency_val.jsonl"
          OUT_DIR="$STAGE2_WORK_DIR/${score_key}/regular_g200"
          ;;
        low_iou)
          INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl"
          OUT_DIR="$STAGE2_WORK_DIR/${score_key}/low_iou_g200"
          ;;
        holdout)
          INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl"
          OUT_DIR="$STAGE2_WORK_DIR/${score_key}/holdout_low_iou_g200"
          ;;
      esac

      mkdir -p "$OUT_DIR"
      "$PYTHON_BIN" benchmark_wam.py \
        --input "$INPUT" \
        --checkpoint "$CKPT" \
        --config "$STAGE2_CONFIG" \
        --model-kind dinov2 \
        --output-dir "$OUT_DIR" \
        --batch-size "$EVAL_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --max-groups "$BENCH_MAX_GROUPS" \
        --consistency-score-key "$score_key" \
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
  done

  for split in regular low_iou holdout; do
    case "$split" in
      regular) split_dir="regular_g200" ;;
      low_iou) split_dir="low_iou_g200" ;;
      holdout) split_dir="holdout_low_iou_g200" ;;
    esac
    primary="$STAGE2_WORK_DIR/consistency_logit/${split_dir}/wam_iac_scores.jsonl"
    aux="$STAGE2_WORK_DIR/path_evidence_logit/${split_dir}/wam_iac_scores.jsonl"
    if [[ -f "$primary" && -f "$aux" ]]; then
      "$PYTHON_BIN" tools/sweep_iac_fused_scores.py \
        --primary-scores "$primary" \
        --aux-scores "$aux" \
        --alphas 0,0.1,0.2,0.3,0.5,1.0 \
        --output "$STAGE2_WORK_DIR/${split}_fused_sweep.json"
    fi
  done
else
  echo "[IAC] RUN_EVAL=0, skipping evaluation."
fi

echo "[IAC] two-stage vNext run complete."
