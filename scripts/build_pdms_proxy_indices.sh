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
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_structured_rules_pdms_vnext}"
mkdir -p "$INDEX_DIR" "$WORK_DIR"

"$PYTHON_BIN" -m py_compile tools/add_pdms_proxy_scores.py

"$PYTHON_BIN" tools/add_pdms_proxy_scores.py \
  --input "$INDEX_DIR/consistency_train.jsonl" \
  --output "$INDEX_DIR/consistency_train_pdms_proxy.jsonl" \
  --summary "$WORK_DIR/pdms_proxy_train_summary.json"

"$PYTHON_BIN" tools/add_pdms_proxy_scores.py \
  --input "$INDEX_DIR/consistency_val.jsonl" \
  --output "$INDEX_DIR/consistency_val_pdms_proxy.jsonl" \
  --summary "$WORK_DIR/pdms_proxy_val_summary.json"
