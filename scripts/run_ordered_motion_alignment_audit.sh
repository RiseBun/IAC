#!/usr/bin/env bash
set -Eeuo pipefail

# Required local rows. No dataset, backbone or feature cache is uploaded.
TRAIN_ROWS=${TRAIN_ROWS:?set TRAIN_ROWS}
VAL_ROWS=${VAL_ROWS:?set VAL_ROWS}
EVAL_ROWS=${EVAL_ROWS:?set EVAL_ROWS}

PYTHON=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-work/ordered_motion_alignment}
TRAIN_CACHE=${TRAIN_CACHE:-$OUT_DIR/train_vjepa.pt}
VAL_CACHE=${VAL_CACHE:-$OUT_DIR/val_vjepa.pt}
EVAL_CACHE=${EVAL_CACHE:-$OUT_DIR/eval_vjepa.pt}
FEATURE_KEY=${FEATURE_KEY:-x_tokens}
VJEPA_MODEL=${VJEPA_MODEL:-facebook/vjepa2-vitl-fpc64-256}
VJEPA_BATCH_SIZE=${VJEPA_BATCH_SIZE:-1}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-20260728}
BATCH_SIZE=${BATCH_SIZE:-128}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-40}
PATIENCE=${PATIENCE:-6}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-0}
MAX_VAL_ROWS=${MAX_VAL_ROWS:-0}
RUN_RAW_FRAME_CONTROLS=${RUN_RAW_FRAME_CONTROLS:-0}

MODEL="$OUT_DIR/ordered_motion_alignment.pt"
STATUS="$OUT_DIR/run_status.json"
RESULT_ARCHIVE="${OUT_DIR%/}_results.tar.gz"

mkdir -p "$OUT_DIR"
start_epoch=$(date +%s)
run_state=failed

finish() {
  exit_code=$?
  end_epoch=$(date +%s)
  printf '{"status":"%s","exit_code":%d,"elapsed_seconds":%d}\n' \
    "$run_state" "$exit_code" "$((end_epoch-start_epoch))" > "$STATUS"
  if [[ -d "$OUT_DIR" ]]; then
    tar -czf "$RESULT_ARCHIVE" \
      --exclude='*_vjepa.pt' \
      -C "$OUT_DIR" .
  fi
}
trap finish EXIT

"$PYTHON" - <<'PY'
import torch
print({
    "python_ok": True,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
})
PY

ensure_cache() {
  rows=$1
  cache=$2
  if [[ -f "$cache" ]]; then
    return
  fi
  IMAGE_ROOT=${IMAGE_ROOT:?set IMAGE_ROOT when a V-JEPA cache is missing}
  mkdir -p "$(dirname "$cache")"
  "$PYTHON" tools/extract_vjepa_video_features.py \
    --index "$rows" \
    --image-root "$IMAGE_ROOT" \
    --model-name "$VJEPA_MODEL" \
    --output "$cache" \
    --token-summary-size 16 \
    --batch-size "$VJEPA_BATCH_SIZE" \
    --device "$DEVICE"
}

ensure_cache "$TRAIN_ROWS" "$TRAIN_CACHE"
ensure_cache "$VAL_ROWS" "$VAL_CACHE"
ensure_cache "$EVAL_ROWS" "$EVAL_CACHE"

"$PYTHON" tools/train_ordered_motion_alignment.py \
  --train-rows "$TRAIN_ROWS" \
  --train-cache "$TRAIN_CACHE" \
  --val-rows "$VAL_ROWS" \
  --val-cache "$VAL_CACHE" \
  --feature-key "$FEATURE_KEY" \
  --output-model "$MODEL" \
  --output-summary "$OUT_DIR/train_summary.json" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --max-train-rows "$MAX_TRAIN_ROWS" \
  --max-val-rows "$MAX_VAL_ROWS"

"$PYTHON" tools/score_ordered_motion_alignment.py \
  --model "$MODEL" \
  --rows "$VAL_ROWS" \
  --visual-cache "$VAL_CACHE" \
  --feature-key "$FEATURE_KEY" \
  --output-scores "$OUT_DIR/val_scores.jsonl" \
  --output-summary "$OUT_DIR/val_summary.json" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE"

"$PYTHON" tools/score_ordered_motion_alignment.py \
  --model "$MODEL" \
  --rows "$EVAL_ROWS" \
  --visual-cache "$EVAL_CACHE" \
  --feature-key "$FEATURE_KEY" \
  --output-scores "$OUT_DIR/eval_scores.jsonl" \
  --output-summary "$OUT_DIR/eval_summary.json" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --include-segment-ledger

"$PYTHON" tools/audit_ordered_motion_alignment.py \
  --model "$MODEL" \
  --rows "$EVAL_ROWS" \
  --visual-cache "$EVAL_CACHE" \
  --feature-key "$FEATURE_KEY" \
  --output-summary "$OUT_DIR/token_and_identity_audit.json" \
  --output-ledger "$OUT_DIR/failure_ledger.jsonl" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE"

if [[ -n "${VAL_PRIMARY:-}" && -n "${EVAL_PRIMARY:-}" && -n "${PRIMARY_KEY:-}" ]]; then
  "$PYTHON" tools/tune_fuse_ordered_motion.py \
    --val-primary "$VAL_PRIMARY" \
    --val-evidence "$OUT_DIR/val_scores.jsonl" \
    --eval-primary "$EVAL_PRIMARY" \
    --eval-evidence "$OUT_DIR/eval_scores.jsonl" \
    --primary-key "$PRIMARY_KEY" \
    --output-scores "$OUT_DIR/fused_eval_scores.jsonl" \
    --output-summary "$OUT_DIR/fusion_summary.json"
fi

if [[ "$RUN_RAW_FRAME_CONTROLS" == "1" ]]; then
  IMAGE_ROOT=${IMAGE_ROOT:?set IMAGE_ROOT when RUN_RAW_FRAME_CONTROLS=1}

  "$PYTHON" tools/make_temporal_control_rows.py \
    --input-rows "$EVAL_ROWS" \
    --output-rows "$OUT_DIR/eval_raw_reverse_rows.jsonl" \
    --output-summary "$OUT_DIR/eval_raw_reverse_rows_summary.json" \
    --control reverse \
    --seed "$SEED"
  "$PYTHON" tools/make_temporal_control_rows.py \
    --input-rows "$EVAL_ROWS" \
    --output-rows "$OUT_DIR/eval_raw_shuffle_rows.jsonl" \
    --output-summary "$OUT_DIR/eval_raw_shuffle_rows_summary.json" \
    --control shuffle \
    --seed "$SEED"

  "$PYTHON" tools/extract_vjepa_video_features.py \
    --index "$OUT_DIR/eval_raw_reverse_rows.jsonl" \
    --image-root "$IMAGE_ROOT" \
    --model-name "$VJEPA_MODEL" \
    --output "$OUT_DIR/eval_raw_reverse_vjepa.pt" \
    --token-summary-size 16 \
    --batch-size "$VJEPA_BATCH_SIZE" \
    --device "$DEVICE"
  "$PYTHON" tools/extract_vjepa_video_features.py \
    --index "$OUT_DIR/eval_raw_shuffle_rows.jsonl" \
    --image-root "$IMAGE_ROOT" \
    --model-name "$VJEPA_MODEL" \
    --output "$OUT_DIR/eval_raw_shuffle_vjepa.pt" \
    --token-summary-size 16 \
    --batch-size "$VJEPA_BATCH_SIZE" \
    --device "$DEVICE"

  "$PYTHON" tools/score_ordered_motion_alignment.py \
    --model "$MODEL" \
    --rows "$EVAL_ROWS" \
    --visual-cache "$OUT_DIR/eval_raw_reverse_vjepa.pt" \
    --output-scores "$OUT_DIR/eval_raw_reverse_scores.jsonl" \
    --output-summary "$OUT_DIR/eval_raw_reverse_summary.json" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"
  "$PYTHON" tools/score_ordered_motion_alignment.py \
    --model "$MODEL" \
    --rows "$EVAL_ROWS" \
    --visual-cache "$OUT_DIR/eval_raw_shuffle_vjepa.pt" \
    --output-scores "$OUT_DIR/eval_raw_shuffle_scores.jsonl" \
    --output-summary "$OUT_DIR/eval_raw_shuffle_summary.json" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"

  "$PYTHON" tools/compare_ordered_motion_scores.py \
    --scores "normal=$OUT_DIR/eval_scores.jsonl" \
    --scores "raw_reverse=$OUT_DIR/eval_raw_reverse_scores.jsonl" \
    --scores "raw_shuffle=$OUT_DIR/eval_raw_shuffle_scores.jsonl" \
    --output-summary "$OUT_DIR/raw_frame_control_comparison.json"
fi

(
  cd "$OUT_DIR"
  sha256sum ordered_motion_alignment.pt *.json *.jsonl > sha256.txt
)

run_state=complete
echo "Ordered motion audit complete: $RESULT_ARCHIVE"
