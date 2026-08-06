#!/usr/bin/env bash
set -euo pipefail

ROWS=${ROWS:-work/eval_rows.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-/path/to/images}
BASE_SCORES=${BASE_SCORES:-work/base_scores.jsonl}
AUX_SCORES=${AUX_SCORES:-work/aux_scores.jsonl}
OUT_DIR=${OUT_DIR:-work/mainline}

mkdir -p "$OUT_DIR"

python pipeline/extract_vjepa_video_features.py \
  --index "$ROWS" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUT_DIR/vjepa.pt" \
  --token-summary-size 16

python pipeline/score_acceptability_calibrator.py \
  --model models/iac_acceptability_calibrator.pt \
  --primary-scores "$BASE_SCORES" \
  --aux "$AUX_SCORES" \
  --output-scores "$OUT_DIR/v3_scores.jsonl" \
  --output-summary "$OUT_DIR/v3_summary.json"

python pipeline/score_visual_mismatch_gate.py \
  --model models/clean_vjepa_traj_gate.pt \
  --rows "$OUT_DIR/v3_scores.jsonl" \
  --visual-cache "$OUT_DIR/vjepa.pt" \
  --visual-cache-key x_tokens \
  --output-scores "$OUT_DIR/clean_gate_scores.jsonl"

python pipeline/fuse_v3_clean_gate.py \
  --v3-scores "$OUT_DIR/v3_scores.jsonl" \
  --gate-scores "$OUT_DIR/clean_gate_scores.jsonl" \
  --output-scores "$OUT_DIR/fused_scores.jsonl" \
  --output-summary "$OUT_DIR/fused_summary.json" \
  --beta 0.15 \
  --threshold 0

python pipeline/score_iac_confidence.py \
  --primary-scores "$OUT_DIR/fused_scores.jsonl" \
  --score-key v3_clean_gate_fused_rank_score \
  --margin-space raw \
  --match-margin 0.2 \
  --mismatch-margin -0.5 \
  --confidence-temperature 0.2 \
  --output-groups "$OUT_DIR/confidence_groups.jsonl" \
  --output-summary "$OUT_DIR/confidence_summary.json"
