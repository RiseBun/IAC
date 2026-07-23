#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${IAC_ROOT:-}" ]]; then
  ROOT="$IAC_ROOT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

export CONFIG="${CONFIG:-configs/train_navsim_future_rgbdiff_motion_gtperception_temporal_contrast_probe.py}"
export WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_rgbdiff_motion_gtperception_temporal_contrast_probe}"

scripts/run_rgbdiff_motion_temporal_contrast_probe.sh
