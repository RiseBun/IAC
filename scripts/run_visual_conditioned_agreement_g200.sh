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

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_supported_set_listwise_vnext.py}"
BASE_WORK_DIR="${BASE_WORK_DIR:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_vnext}"
CKPT="${CKPT:-$BASE_WORK_DIR/checkpoints/latest.pth}"
SOURCE_EVAL_ROOT="${SOURCE_EVAL_ROOT:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_grouped_recovered_set_k12_short_vnext/g200_grouped_recovered_set_k12}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_visual_conditioned_agreement_g200_vnext}"
GEOM_ALPHA="${GEOM_ALPHA:-0.3}"
VISUAL_ALPHAS="${VISUAL_ALPHAS:-0.1,0.2,0.3}"

FEATURE_DIR="$WORK_DIR/features"
SCORER_DIR="$WORK_DIR/scorer_from_regular"
mkdir -p "$FEATURE_DIR" "$SCORER_DIR"
export SCORER_DIR

"$PYTHON_BIN" -m py_compile \
  tools/extract_recovered_path_features.py \
  tools/train_visual_conditioned_agreement_scorer.py \
  tools/audit_iac_ambiguity.py \
  tools/fuse_iac_score_jsonl.py \
  "$CONFIG"

for split in regular low_iou holdout; do
  rows="$SOURCE_EVAL_ROOT/${split}_recovered_set_rows.jsonl"
  cache="$FEATURE_DIR/${split}_visual_motion_rich.pt"
  if [[ ! -s "$rows" ]]; then
    echo "[IAC] Missing recovered rows: $rows" >&2
    exit 1
  fi
  if [[ ! -s "$cache" ]]; then
    "$PYTHON_BIN" tools/extract_recovered_path_features.py \
      --config "$CONFIG" \
      --checkpoint "$CKPT" \
      --index "$rows" \
      --output "$cache" \
      --model-kind dinov2 \
      --input-mode motion_rich \
      --batch-size 32 \
      --num-workers 1 \
      --log-every 20
  fi
done

eval_args=()
for split in regular low_iou holdout; do
  eval_args+=(
    --eval "$split=$SOURCE_EVAL_ROOT/${split}_recovered_set_rows.jsonl,$FEATURE_DIR/${split}_visual_motion_rich.pt,$SCORER_DIR/${split}_visual_conditioned_scores.jsonl"
  )
done

"$PYTHON_BIN" tools/train_visual_conditioned_agreement_scorer.py \
  --train-rows "$SOURCE_EVAL_ROOT/regular_recovered_set_rows.jsonl" \
  --train-visual-cache "$FEATURE_DIR/regular_visual_motion_rich.pt" \
  --output-dir "$SCORER_DIR" \
  --steps "${SCORER_STEPS:-400}" \
  --visual-hidden-dim "${VISUAL_HIDDEN_DIM:-32}" \
  --hidden-dim "${SCORER_HIDDEN_DIM:-64}" \
  --dropout "${SCORER_DROPOUT:-0.20}" \
  "${eval_args[@]}"

for split in regular low_iou holdout; do
  visual="$SCORER_DIR/${split}_visual_conditioned_scores.jsonl"
  "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
    --scores "$visual" \
    --output "$SCORER_DIR/${split}_visual_conditioned_ambiguity.json" \
    --per-sample-output "$SCORER_DIR/${split}_visual_conditioned_ambiguity_groups.jsonl"

  geom="$SOURCE_EVAL_ROOT/${split}_fused_consistency_path_recovered_a${GEOM_ALPHA}_scores.jsonl"
  if [[ -s "$geom" ]]; then
    IFS=',' read -ra alpha_items <<< "$VISUAL_ALPHAS"
    for alpha in "${alpha_items[@]}"; do
      alpha="$(echo "$alpha" | xargs)"
      [[ -n "$alpha" ]] || continue
      fused="$SCORER_DIR/${split}_grouped_visual_a${alpha}_scores.jsonl"
      "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
        --primary-scores "$geom" \
        --aux "$visual:$alpha" \
        --label "grouped_recovered_${GEOM_ALPHA}_visual_${alpha}" \
        --output-scores "$fused"
      "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
        --scores "$fused" \
        --output "$SCORER_DIR/${split}_grouped_visual_a${alpha}_ambiguity.json" \
        --per-sample-output "$SCORER_DIR/${split}_grouped_visual_a${alpha}_ambiguity_groups.jsonl"
    done
  fi
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ.get("SCORER_DIR", "work_dirs/iac_navsim_future_dinov2_visual_conditioned_agreement_g200_vnext/scorer_from_regular"))
hard = {"image_swap", "time_shift", "time_shift_future", "time_shift_past", "high_pdm_image_mismatch"}

def load_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def source(row):
    for key in ("source_type", "action_type", "wam_name", "sample_type", "wam"):
        if row.get(key) is not None:
            return str(row[key])
    return "unknown"

def positive(row):
    if row.get("consistency_label") is not None:
        return float(row["consistency_label"]) > 0.5
    if row.get("label") is not None:
        return float(row["label"]) > 0.5
    return source(row) == "gt_pos"

def group_id(row):
    return row.get("group_id") or row.get("anchor_id") or row.get("sample_id")

def hard_above_gt(path):
    rows = load_rows(path)
    groups = defaultdict(list)
    for row in rows:
        groups[str(group_id(row))].append(row)
    vals = []
    for items in groups.values():
        positives = [row for row in items if positive(row)]
        if not positives:
            continue
        gt_score = float(positives[0]["iac_consistency"])
        vals.append(float(any(source(row) in hard and float(row["iac_consistency"]) > gt_score for row in items)))
    return sum(vals) / len(vals) if vals else None

summary = []
for path in sorted(root.glob("*_ambiguity.json")):
    stem = path.name[:-len("_ambiguity.json")]
    split = "holdout" if stem.startswith("holdout_") else ("low_iou" if stem.startswith("low_iou_") else "regular")
    scorer = stem[len(split) + 1:]
    data = json.loads(path.read_text(encoding="utf-8"))
    scores = root / f"{stem}_scores.jsonl"
    summary.append({
        "split": split,
        "scorer": scorer,
        "hard_top1": data.get("hard_top1"),
        "ambiguity_adjusted_top1": data.get("ambiguity_adjusted_top1"),
        "hard_mismatch_above_gt_group_rate": hard_above_gt(scores) if scores.exists() else None,
    })
out = {"scorer_root": str(root), "rows": summary}
(root / "visual_conditioned_g200_summary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for row in summary:
    print(row)
PY
