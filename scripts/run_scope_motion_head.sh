#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_scope_motion_head.py}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_scope_motion_head}"
GPUS="${GPUS:-1}"

if [[ "$GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$GPUS" train_scope_motion_head.py \
    --config "$CONFIG" --work-dir "$WORK_DIR" "$@"
else
  python train_scope_motion_head.py \
    --config "$CONFIG" --work-dir "$WORK_DIR" "$@"
fi
