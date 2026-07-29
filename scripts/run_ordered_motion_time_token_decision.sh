#!/usr/bin/env bash
set -Eeuo pipefail

# Engineering decision only. The same 200-group protocol must not be reported
# as formal drive-disjoint evidence.
TRAIN_ROWS=${TRAIN_ROWS:?set TRAIN_ROWS}
VAL_ROWS=${VAL_ROWS:?set VAL_ROWS}
EVAL_ROWS=${EVAL_ROWS:?set EVAL_ROWS}
TRAIN_CACHE=${TRAIN_CACHE:?set TRAIN_CACHE}
VAL_CACHE=${VAL_CACHE:?set VAL_CACHE}
EVAL_CACHE=${EVAL_CACHE:?set EVAL_CACHE}
VAL_PRIMARY=${VAL_PRIMARY:?set VAL_PRIMARY}
EVAL_PRIMARY=${EVAL_PRIMARY:?set EVAL_PRIMARY}
PRIMARY_KEY=${PRIMARY_KEY:?set PRIMARY_KEY}

PYTHON=${PYTHON:-python}
OUT_ROOT=${OUT_ROOT:-work/ordered_motion_time_token_decision}
SEEDS=${SEEDS:-20260728,20260729,20260730}
RESULT_ARCHIVE="${OUT_ROOT%/}_results.tar.gz"
STATUS="$OUT_ROOT/run_status.json"

mkdir -p "$OUT_ROOT"
start_epoch=$(date +%s)
run_state=failed

finish() {
  exit_code=$?
  end_epoch=$(date +%s)
  printf '{"status":"%s","exit_code":%d,"elapsed_seconds":%d}\n' \
    "$run_state" "$exit_code" "$((end_epoch-start_epoch))" > "$STATUS"
}
trap finish EXIT

IFS=',' read -r -a seed_values <<< "$SEEDS"
if [[ "${#seed_values[@]}" -lt 3 ]]; then
  echo "SEEDS must contain at least three comma-separated values" >&2
  exit 2
fi

SHARED_CACHE_DIR="$OUT_ROOT/shared_time_cache"
mkdir -p "$SHARED_CACHE_DIR"
prepare_time_cache() {
  split=$1
  input_cache=$2
  output_cache="$SHARED_CACHE_DIR/${split}_time_vjepa.pt"
  "$PYTHON" tools/migrate_vjepa_time_tokens.py \
    --input-cache "$input_cache" \
    --output-cache "$output_cache" \
    --output-summary "$SHARED_CACHE_DIR/${split}_time_cache_summary.json" \
    --source-key "${LEGACY_FEATURE_KEY:-x_tokens}" \
    --output-key x_time_tokens \
    --tubelet-size "${TUBELET_SIZE:-2}"
}
prepare_time_cache train "$TRAIN_CACHE"
prepare_time_cache validation "$VAL_CACHE"
prepare_time_cache evaluation "$EVAL_CACHE"

summary_args=()
for raw_seed in "${seed_values[@]}"; do
  seed=$(echo "$raw_seed" | xargs)
  if [[ -z "$seed" ]]; then
    echo "SEEDS contains an empty value" >&2
    exit 2
  fi
  run_dir="$OUT_ROOT/seed_$seed"
  env \
    TRAIN_ROWS="$TRAIN_ROWS" \
    VAL_ROWS="$VAL_ROWS" \
    EVAL_ROWS="$EVAL_ROWS" \
    TRAIN_CACHE="$SHARED_CACHE_DIR/train_time_vjepa.pt" \
    VAL_CACHE="$SHARED_CACHE_DIR/validation_time_vjepa.pt" \
    EVAL_CACHE="$SHARED_CACHE_DIR/evaluation_time_vjepa.pt" \
    VAL_PRIMARY="$VAL_PRIMARY" \
    EVAL_PRIMARY="$EVAL_PRIMARY" \
    PRIMARY_KEY="$PRIMARY_KEY" \
    FEATURE_KEY=x_time_tokens \
    SEED="$seed" \
    OUT_DIR="$run_dir" \
    PYTHON="$PYTHON" \
    DEVICE="${DEVICE:-cuda}" \
    EPOCHS="${EPOCHS:-40}" \
    PATIENCE="${PATIENCE:-6}" \
    BATCH_SIZE="${BATCH_SIZE:-128}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}" \
    TUBELET_SIZE="${TUBELET_SIZE:-2}" \
    TIME_TOKENS_ALREADY_PREPARED=1 \
    REQUIRE_STRICT_SPLIT_DISJOINT=0 \
    RUN_RAW_FRAME_CONTROLS="${RUN_RAW_FRAME_CONTROLS:-0}" \
    bash scripts/run_ordered_motion_alignment_audit.sh
  summary_args+=(--run "seed_$seed=$run_dir")
done

"$PYTHON" tools/summarize_ordered_motion_decision.py \
  "${summary_args[@]}" \
  --output-summary "$OUT_ROOT/multi_seed_decision_summary.json"

tar -czf "$RESULT_ARCHIVE" \
  -C "$OUT_ROOT" \
  multi_seed_decision_summary.json \
  $(find "$OUT_ROOT" -maxdepth 2 -type f \
      \( -name 'fusion_summary.json' \
      -o -name 'fused_control_audit.json' \
      -o -name 'token_and_identity_audit.json' \
      -o -name 'split_independence_audit.json' \
      -o -name 'train_summary.json' \
      -o -name 'eval_summary.json' \
      -o -name '*_time_cache_summary.json' \
      -o -name 'run_status.json' \) \
      -printf '%P\n')

run_state=complete
echo "Ordered-motion time-token decision complete: $RESULT_ARCHIVE"
