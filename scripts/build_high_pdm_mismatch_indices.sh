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

INDEX_DIR="${INDEX_DIR:-indices_navsim_future}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_separated_heads_official_pdms_hardneg_vnext}"
MIN_PDM="${MIN_PDM:-0.85}"
PER_GROUP="${PER_GROUP:-1}"
MAX_GROUPS="${MAX_GROUPS:-0}"
MIN_MEAN_L2="${MIN_MEAN_L2:-1.0}"
MIN_ENDPOINT_L2="${MIN_ENDPOINT_L2:-2.0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-128}"

mkdir -p "$WORK_DIR"

"$PYTHON_BIN" -m py_compile tools/add_high_pdm_mismatch_negatives.py

build_split() {
  local split="$1"
  local input="$INDEX_DIR/consistency_${split}_official_pdms.jsonl"
  local output="$INDEX_DIR/consistency_${split}_official_pdms_highpdm_mismatch.jsonl"
  local summary="$WORK_DIR/highpdm_mismatch_${split}_summary.json"
  if [[ ! -s "$input" ]]; then
    echo "[IAC] Missing official PDMS index: $input" >&2
    exit 1
  fi
  echo "[IAC] Building high-PDM mismatch split=${split}"
  "$PYTHON_BIN" tools/add_high_pdm_mismatch_negatives.py \
    --input "$input" \
    --output "$output" \
    --summary "$summary" \
    --min-pdm "$MIN_PDM" \
    --per-group "$PER_GROUP" \
    --max-groups "$MAX_GROUPS" \
    --min-mean-l2 "$MIN_MEAN_L2" \
    --min-endpoint-l2 "$MIN_ENDPOINT_L2" \
    --max-attempts "$MAX_ATTEMPTS"
}

build_split train
build_split val

echo "[IAC] High-PDM mismatch indices ready."
