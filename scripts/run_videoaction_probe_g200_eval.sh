#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${IAC_ROOT:-}" ]]; then
  ROOT="$IAC_ROOT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniforge3/envs/drivingworld/bin/python}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]]; then
  export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-${TORCH_HOME}/hub/facebookresearch_dinov2_main}"
fi

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_supported_set_v3_videoaction_probe_vnext.py}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_supported_set_v3_videoaction_probe_vnext}"
CKPT="${CKPT:-$WORK_DIR/checkpoints/latest.pth}"
BENCH_MAX_GROUPS="${BENCH_MAX_GROUPS:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PATH_ALPHA="${PATH_ALPHA:-0.2}"
VIDEO_ALPHA="${VIDEO_ALPHA:-0.2}"

if [[ ! -s "$CKPT" ]]; then
  echo "[IAC] Missing checkpoint: $CKPT" >&2
  exit 1
fi

epoch="$("$PYTHON_BIN" - <<'PY'
import os
import torch
ck = torch.load(os.environ["CKPT"], map_location="cpu")
print(int(ck.get("epoch", 0)))
PY
)"
EVAL_ROOT="${EVAL_ROOT:-$WORK_DIR/g200_video_probe_epoch_${epoch}}"
mkdir -p "$EVAL_ROOT"
export EVAL_ROOT

"$PYTHON_BIN" -m py_compile \
  benchmark_wam.py \
  tools/audit_iac_ambiguity.py \
  tools/fuse_iac_score_jsonl.py \
  "$CONFIG"

run_score() {
  local split="$1"
  local score_key="$2"
  local input="$3"
  local out_dir="$EVAL_ROOT/${split}_${score_key}"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" benchmark_wam.py \
    --input "$input" \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --model-kind dinov2 \
    --output-dir "$out_dir" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-groups "$BENCH_MAX_GROUPS" \
    --consistency-score-key "$score_key" \
    --path-causal-metrics \
    --trajectory-specific-causal-metrics \
    --wrong-path-selection mask_iou \
    2>&1 | tee "$EVAL_ROOT/${split}_${score_key}.log"
  "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
    --scores "$out_dir/wam_iac_scores.jsonl" \
    --output "$EVAL_ROOT/${split}_${score_key}_ambiguity.json" \
    --per-sample-output "$EVAL_ROOT/${split}_${score_key}_ambiguity_groups.jsonl"
}

audit_scores() {
  local split="$1"
  local label="$2"
  local scores="$3"
  "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
    --scores "$scores" \
    --output "$EVAL_ROOT/${split}_${label}_ambiguity.json" \
    --per-sample-output "$EVAL_ROOT/${split}_${label}_ambiguity_groups.jsonl"
}

for split in regular low_iou holdout; do
  case "$split" in
    regular) INPUT="indices_navsim_future/consistency_val.jsonl" ;;
    low_iou) INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl" ;;
    holdout) INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl" ;;
  esac

  run_score "$split" consistency_logit "$INPUT"
  run_score "$split" path_evidence_logit "$INPUT"
  run_score "$split" video_action_match_logit "$INPUT"

  primary="$EVAL_ROOT/${split}_consistency_logit/wam_iac_scores.jsonl"
  path="$EVAL_ROOT/${split}_path_evidence_logit/wam_iac_scores.jsonl"
  video="$EVAL_ROOT/${split}_video_action_match_logit/wam_iac_scores.jsonl"

  fused_cp="$EVAL_ROOT/${split}_fused_consistency_path_scores.jsonl"
  "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
    --primary-scores "$primary" \
    --aux "$path:$PATH_ALPHA" \
    --label "consistency_path_${PATH_ALPHA}" \
    --output-scores "$fused_cp"
  audit_scores "$split" "fused_consistency_path" "$fused_cp"

  fused_cpv="$EVAL_ROOT/${split}_fused_consistency_path_video_scores.jsonl"
  "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
    --primary-scores "$primary" \
    --aux "$path:$PATH_ALPHA" \
    --aux "$video:$VIDEO_ALPHA" \
    --label "consistency_path_${PATH_ALPHA}_video_${VIDEO_ALPHA}" \
    --output-scores "$fused_cpv"
  audit_scores "$split" "fused_consistency_path_video" "$fused_cpv"
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"])
rows = []
for split in ["regular", "low_iou", "holdout"]:
    for scorer in [
        "consistency_logit",
        "path_evidence_logit",
        "video_action_match_logit",
        "fused_consistency_path",
        "fused_consistency_path_video",
    ]:
        path = root / f"{split}_{scorer}_ambiguity.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        miss = data.get("miss_breakdown") or {}
        rows.append({
            "split": split,
            "scorer": scorer,
            "hard_top1": data.get("hard_top1"),
            "ambiguity_adjusted_top1": data.get("ambiguity_adjusted_top1"),
            "likely_model_error": miss.get("likely_model_error"),
            "ambiguous_accept": miss.get("ambiguous_accept"),
            "evidence_supported_miss": miss.get("evidence_supported_miss"),
        })
(root / "g200_video_probe_summary.json").write_text(
    json.dumps({"eval_root": str(root), "rows": rows}, indent=2),
    encoding="utf-8",
)
for row in rows:
    print(row)
PY
