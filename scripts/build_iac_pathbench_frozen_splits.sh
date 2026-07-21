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

INPUT="${INPUT:-indices_navsim_future/consistency_val.jsonl}"
OUT_DIR="${OUT_DIR:-indices_navsim_future/diagnostics}"
MAX_GROUPS="${MAX_GROUPS:-1000}"
HOLDOUT_START="${HOLDOUT_START:-1000}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" tools/build_low_iou_subset.py \
  --input "$INPUT" \
  --output "$OUT_DIR/consistency_val_low_iou_g${MAX_GROUPS}.jsonl" \
  --report "$OUT_DIR/consistency_val_low_iou_g${MAX_GROUPS}.json" \
  --max-groups "$MAX_GROUPS" \
  --start-rank 0

"$PYTHON_BIN" tools/build_low_iou_subset.py \
  --input "$INPUT" \
  --output "$OUT_DIR/consistency_val_low_iou_g${MAX_GROUPS}_holdout_rank${HOLDOUT_START}_$((HOLDOUT_START + MAX_GROUPS - 1)).jsonl" \
  --report "$OUT_DIR/consistency_val_low_iou_g${MAX_GROUPS}_holdout_rank${HOLDOUT_START}_$((HOLDOUT_START + MAX_GROUPS - 1)).json" \
  --max-groups "$MAX_GROUPS" \
  --start-rank "$HOLDOUT_START"

echo "[IAC] frozen split build complete: $OUT_DIR"
