#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_multisolution_official_pdms_vnext}"
BUILD_PID_FILE="${BUILD_PID_FILE:-$WORK_DIR/build_official_pdms_indices.pid}"
TRAIN_LOG="${TRAIN_LOG:-$WORK_DIR/train_official_after_build.log}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$HOME/miniforge3/envs/drivingworld/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniforge3/envs/drivingworld/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

mkdir -p "$WORK_DIR"

echo "[IAC] Waiting for official PDM index build"
if [[ -f "$BUILD_PID_FILE" ]]; then
  build_pid="$(cat "$BUILD_PID_FILE")"
  while ps -p "$build_pid" >/dev/null 2>&1; do
    sleep 60
  done
fi

train_index="indices_navsim_future/consistency_train_official_pdms.jsonl"
val_index="indices_navsim_future/consistency_val_official_pdms.jsonl"
train_src="indices_navsim_future/consistency_train.jsonl"
val_src="indices_navsim_future/consistency_val.jsonl"

for path in "$train_index" "$val_index"; do
  if [[ ! -s "$path" ]]; then
    echo "[IAC] Missing built index: $path" >&2
    exit 1
  fi
done

train_lines="$(wc -l < "$train_index")"
val_lines="$(wc -l < "$val_index")"
train_src_lines="$(wc -l < "$train_src")"
val_src_lines="$(wc -l < "$val_src")"
if [[ "$train_lines" != "$train_src_lines" || "$val_lines" != "$val_src_lines" ]]; then
  echo "[IAC] Built official PDM index line counts do not match source indices" >&2
  echo "[IAC] train $train_lines/$train_src_lines val $val_lines/$val_src_lines" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

work_dir = Path(os.environ.get(
    "WORK_DIR",
    "work_dirs/iac_navsim_future_dinov2_multisolution_official_pdms_vnext",
))
for path in [work_dir / "official_pdms_train_summary.json", work_dir / "official_pdms_val_summary.json"]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures = data.get("failure_counts") or {}
    if failures:
        raise SystemExit(f"{path} has official PDM failures: {failures}")
print("[IAC] Official PDM summaries validated")
PY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi -L | wc -l)"
  else
    gpu_count=1
  fi
  if [[ "$gpu_count" -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((gpu_count - 1)))"
  else
    CUDA_VISIBLE_DEVICES="0"
  fi
  export CUDA_VISIBLE_DEVICES
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra gpu_items <<< "$CUDA_VISIBLE_DEVICES"
  export NPROC_PER_NODE="${#gpu_items[@]}"
fi

export EPOCHS="${EPOCHS:-80}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
export MAX_VAL_STEPS="${MAX_VAL_STEPS:-400}"
export PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-128}"

echo "[IAC] Starting official PDM training"
echo "[IAC] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=$NPROC_PER_NODE"
./scripts/run_multisolution_official_pdms_vnext.sh 2>&1 | tee "$TRAIN_LOG"
