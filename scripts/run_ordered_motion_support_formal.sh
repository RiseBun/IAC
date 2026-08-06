#!/usr/bin/env bash
set -Eeuo pipefail

# Formal 4s ordered-motion evaluation. The split audit runs before any GPU work
# so an unverifiable dataset cannot produce a reportable result.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=${BASE:-/mnt/slurmfs-4090node1/homes/zchen897/IAC}
PKG=${PKG:-/mnt/slurmfs-4090node1/homes/zchen897/IAC-ordered-motion}
TRAIN_WORK=${TRAIN_WORK:-/mnt/slurmfs-4090node1/homes/zchen897/IAC-ordered-motion/work/ordered_motion_4s_scene_stratified_v2}
WORK=${WORK:-/mnt/slurmfs-4090node1/homes/zchen897/IAC-ordered-motion/work/ordered_motion_4s_image_disjoint_v3}
RUN_OUT=${RUN_OUT:-$WORK/formal_support_v1}
PYBIN=${PYBIN:-$HOME/miniforge3/envs/drivingworld/bin/python}
MODEL=${MODEL:-$TRAIN_WORK/speed_rank_seed_20260805/ordered_motion_speed_rank.pt}
DEVICE=${DEVICE:-cuda}

mkdir -p "$RUN_OUT"

"$PYBIN" "$ROOT/audit/audit_formal_splits.py" \
  --split "train=$TRAIN_WORK/train_rows.jsonl" \
  --split "val=$WORK/val_rows.jsonl" \
  --split "eval=$WORK/eval_rows.jsonl" \
  --horizon 4s --require-formal-ready \
  --output-summary "$RUN_OUT/formal_split_audit.json"

"$PYBIN" "$PKG/tools/score_ordered_motion_alignment.py" \
  --model "$MODEL" --rows "$WORK/val_rows.jsonl" \
  --visual-cache "$WORK/val_vjepa.pt" --feature-key x_time_tokens \
  --output-scores "$RUN_OUT/val_segment_scores.jsonl" \
  --output-summary "$RUN_OUT/val_segment_summary.json" \
  --device "$DEVICE" --batch-size 64 --include-segment-ledger

"$PYBIN" "$ROOT/ordered_motion/calibrate_ordered_motion_support.py" \
  --scores "$RUN_OUT/val_segment_scores.jsonl" \
  --output-config "$RUN_OUT/ordered_motion_support_config.json" \
  --min-supported-precision 0.95 --min-unsupported-precision 0.95 \
  --min-unsupported-precision-lower-bound 0.95 --confidence-z 1.96

"$PYBIN" "$PKG/tools/score_ordered_motion_alignment.py" \
  --model "$MODEL" --rows "$WORK/eval_rows.jsonl" \
  --visual-cache "$WORK/eval_vjepa.pt" --feature-key x_time_tokens \
  --output-scores "$RUN_OUT/eval_segment_scores.jsonl" \
  --output-summary "$RUN_OUT/eval_segment_summary.json" \
  --device "$DEVICE" --batch-size 64 --include-segment-ledger

"$PYBIN" "$ROOT/ordered_motion/score_ordered_motion_support.py" \
  --scores "$RUN_OUT/eval_segment_scores.jsonl" \
  --config "$RUN_OUT/ordered_motion_support_config.json" \
  --output-scores "$RUN_OUT/eval_support.jsonl" \
  --output-summary "$RUN_OUT/eval_support_summary.json"

"$PYBIN" "$ROOT/audit/audit_ordered_motion_support.py" \
  --scores "$RUN_OUT/eval_support.jsonl" \
  --output-summary "$RUN_OUT/eval_support_label_audit.json"

printf '{"status":"complete","formal_evidence_ready":true}\n' \
  > "$RUN_OUT/run_status.json"
