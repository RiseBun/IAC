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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"

OUT_ROOT="${OUT_ROOT:-work_dirs/iac_pathbench_g1000_2026_07_18}"
BENCH_MAX_GROUPS="${BENCH_MAX_GROUPS:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_CNN="${RUN_CNN:-1}"
RUN_DINOV2="${RUN_DINOV2:-1}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"

CNN_CKPT="${CNN_CKPT:-work_dirs/iac_navsim_future_cnn_3k/checkpoints/best.pth}"
DINO_CONFIG="${DINO_CONFIG:-configs/train_navsim_future_dinov2_path_evidence_vnext.py}"
DINO_CKPT="${DINO_CKPT:-work_dirs/iac_navsim_future_dinov2_path_evidence_vnext/checkpoints/best.pth}"
if [[ ! -f "$DINO_CKPT" ]]; then
  DINO_CKPT="work_dirs/iac_navsim_future_dinov2_path_evidence_vnext/checkpoints/latest.pth"
fi

LOW_IOU_INPUT="${LOW_IOU_INPUT:-indices_navsim_future/diagnostics/consistency_val_low_iou_g1000.jsonl}"
HOLDOUT_INPUT="${HOLDOUT_INPUT:-indices_navsim_future/diagnostics/consistency_val_low_iou_g1000_holdout_rank1000_1999.jsonl}"

mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" -m py_compile \
  benchmark_wam.py \
  tools/bootstrap_iac_pathbench_v2.py \
  tools/validate_iac_pathbench_protocol.py \
  tools/compare_iac_pathbench_models.py

run_one() {
  local label="$1"
  local model_kind="$2"
  local config="$3"
  local ckpt="$4"

  local config_args=()
  if [[ -n "$config" ]]; then
    config_args=(--config "$config")
  fi

  for score_key in consistency_logit path_evidence_logit; do
    for split in regular low_iou holdout; do
      local input
      local out_dir
      case "$split" in
      regular)
        input="indices_navsim_future/consistency_val.jsonl"
        out_dir="$OUT_ROOT/${label}/${score_key}/regular_g${BENCH_MAX_GROUPS}"
        ;;
        low_iou)
          input="$LOW_IOU_INPUT"
          out_dir="$OUT_ROOT/${label}/${score_key}/low_iou_g1000"
          ;;
        holdout)
          input="$HOLDOUT_INPUT"
          out_dir="$OUT_ROOT/${label}/${score_key}/holdout_low_iou_g1000"
          ;;
        *)
          echo "unknown split: $split" >&2
          exit 2
          ;;
      esac

      mkdir -p "$out_dir"
      echo "[IAC] ${label} ${split} ${score_key}: ${ckpt}"

      "$PYTHON_BIN" benchmark_wam.py \
        --input "$input" \
        --checkpoint "$ckpt" \
        "${config_args[@]}" \
        --model-kind "$model_kind" \
        --output-dir "$out_dir" \
        --batch-size "$EVAL_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --max-groups "$BENCH_MAX_GROUPS" \
        --consistency-score-key "$score_key" \
        --path-causal-metrics \
        --trajectory-specific-causal-metrics \
        --wrong-path-selection mask_iou \
        2>&1 | tee "$out_dir.log"

      "$PYTHON_BIN" tools/validate_iac_pathbench_protocol.py \
        "$out_dir/wam_iac_summary.json" \
        --json-out "$out_dir/protocol_validation.json"

      if [[ "$RUN_BOOTSTRAP" != "0" ]]; then
        "$PYTHON_BIN" tools/bootstrap_iac_pathbench_v2.py \
          --scores "$out_dir/wam_iac_scores.jsonl" \
          --output "$out_dir/bootstrap_ci95.json" \
          --num-bootstrap "$BOOTSTRAP_SAMPLES" \
          --seed 897
      fi
    done
  done

  local primary="$OUT_ROOT/${label}/consistency_logit/holdout_low_iou_g1000/wam_iac_scores.jsonl"
  local aux="$OUT_ROOT/${label}/path_evidence_logit/holdout_low_iou_g1000/wam_iac_scores.jsonl"
  if [[ -f "$primary" && -f "$aux" ]]; then
    "$PYTHON_BIN" tools/sweep_iac_fused_scores.py \
      --primary-scores "$primary" \
      --aux-scores "$aux" \
      --alphas 0,0.1,0.2,0.3,0.5,1.0 \
      --output "$OUT_ROOT/${label}/holdout_fused_sweep.json"
  fi
}

if [[ "$RUN_CNN" != "0" ]]; then
  run_one "cnn_3k" "cnn" "" "$CNN_CKPT"
fi

if [[ "$RUN_DINOV2" != "0" ]]; then
  run_one "dinov2_vnext" "dinov2" "$DINO_CONFIG" "$DINO_CKPT"
fi

echo "[IAC] g1000 run complete: ${OUT_ROOT}"
