#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${IAC_ROOT:-}" ]]; then
  ROOT="$IAC_ROOT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniforge3/envs/drivingworld/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${FALLBACK_PYTHON_BIN:-python3}"
fi
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

VJEPA_MODEL="${VJEPA_MODEL:-facebook/vjepa2-vitl-fpc64-256}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs}"
SOURCE_EVAL_ROOT="${SOURCE_EVAL_ROOT:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_recovered_set_vnext/g200_recovered_set_k8}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_vjepa2_visual_conditioned_agreement_g200}"
VIDEO_MODE="${VIDEO_MODE:-history_future}"
NUM_FRAMES="${NUM_FRAMES:-64}"
VJEPA_POOLING="${VJEPA_POOLING:-mean_std_diff}"
VJEPA_BATCH_SIZE="${VJEPA_BATCH_SIZE:-1}"
MAX_GROUPS="${MAX_GROUPS:-0}"
SCORER_STEPS="${SCORER_STEPS:-400}"
GEOM_ALPHA="${GEOM_ALPHA:-0.4}"
VISUAL_ALPHAS="${VISUAL_ALPHAS:-0.05,0.1,0.2,0.3}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
HF_DEPS_DIR="${HF_DEPS_DIR:-work_dirs/vjepa2_hf_deps}"

if [[ -d "$HF_DEPS_DIR" ]]; then
  export PYTHONPATH="$ROOT/$HF_DEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

FEATURE_DIR="$WORK_DIR/features"
SCORER_DIR="$WORK_DIR/scorer_from_regular"
mkdir -p "$FEATURE_DIR" "$SCORER_DIR"
export SCORER_DIR

"$PYTHON_BIN" -m py_compile \
  tools/extract_vjepa_video_features.py \
  tools/train_visual_conditioned_agreement_scorer.py \
  tools/audit_iac_ambiguity.py \
  tools/fuse_iac_score_jsonl.py

trust_args=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  trust_args+=(--trust-remote-code)
fi

for split in regular low_iou holdout; do
  rows="$SOURCE_EVAL_ROOT/${split}_recovered_set_rows.jsonl"
  cache="$FEATURE_DIR/${split}_vjepa.pt"
  if [[ ! -s "$rows" ]]; then
    echo "[IAC] Missing recovered rows: $rows" >&2
    exit 1
  fi
  if [[ ! -s "$cache" ]]; then
    "$PYTHON_BIN" tools/extract_vjepa_video_features.py \
      --index "$rows" \
      --image-root "$IMAGE_ROOT" \
      --output "$cache" \
      --model-name "$VJEPA_MODEL" \
      --video-mode "$VIDEO_MODE" \
      --num-frames "$NUM_FRAMES" \
      --pooling "$VJEPA_POOLING" \
      --batch-size "$VJEPA_BATCH_SIZE" \
      --max-groups "$MAX_GROUPS" \
      --log-every 20 \
      "${trust_args[@]}"
  fi
done

eval_args=()
for split in regular low_iou holdout; do
  eval_args+=(
    --eval "$split=$SOURCE_EVAL_ROOT/${split}_recovered_set_rows.jsonl,$FEATURE_DIR/${split}_vjepa.pt,$SCORER_DIR/${split}_vjepa_visual_conditioned_scores.jsonl"
  )
done

"$PYTHON_BIN" tools/train_visual_conditioned_agreement_scorer.py \
  --train-rows "$SOURCE_EVAL_ROOT/regular_recovered_set_rows.jsonl" \
  --train-visual-cache "$FEATURE_DIR/regular_vjepa.pt" \
  --output-dir "$SCORER_DIR" \
  --steps "$SCORER_STEPS" \
  --visual-hidden-dim "${VISUAL_HIDDEN_DIM:-64}" \
  --hidden-dim "${SCORER_HIDDEN_DIM:-128}" \
  --dropout "${SCORER_DROPOUT:-0.20}" \
  "${eval_args[@]}"

for split in regular low_iou holdout; do
  visual="$SCORER_DIR/${split}_vjepa_visual_conditioned_scores.jsonl"
  "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
    --scores "$visual" \
    --output "$SCORER_DIR/${split}_vjepa_visual_conditioned_ambiguity.json" \
    --per-sample-output "$SCORER_DIR/${split}_vjepa_visual_conditioned_ambiguity_groups.jsonl"

  if [[ "$MAX_GROUPS" != "0" ]]; then
    continue
  fi
  geom="$SOURCE_EVAL_ROOT/${split}_fused_consistency_path_recovered_a${GEOM_ALPHA}_scores.jsonl"
  if [[ -s "$geom" ]]; then
    IFS=',' read -ra alpha_items <<< "$VISUAL_ALPHAS"
    for alpha in "${alpha_items[@]}"; do
      alpha="$(echo "$alpha" | xargs)"
      [[ -n "$alpha" ]] || continue
      fused="$SCORER_DIR/${split}_grouped_vjepa_a${alpha}_scores.jsonl"
      "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
        --primary-scores "$geom" \
        --aux "$visual:$alpha" \
        --label "grouped_recovered_${GEOM_ALPHA}_vjepa_${alpha}" \
        --output-scores "$fused"
      "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
        --scores "$fused" \
        --output "$SCORER_DIR/${split}_grouped_vjepa_a${alpha}_ambiguity.json" \
        --per-sample-output "$SCORER_DIR/${split}_grouped_vjepa_a${alpha}_ambiguity_groups.jsonl"
    done
  fi
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ.get("SCORER_DIR", "work_dirs/iac_vjepa2_visual_conditioned_agreement_g200/scorer_from_regular"))
hard = {"image_swap", "time_shift", "time_shift_future", "time_shift_past", "high_pdm_image_mismatch", "traj_swap", "reverse_traj"}

def load_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

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
    if not path.exists():
        return None
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
        "hard_mismatch_above_gt_group_rate": hard_above_gt(scores),
    })
out = {"scorer_root": str(root), "rows": summary}
(root / "vjepa_visual_conditioned_g200_summary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for row in summary:
    print(row, flush=True)
PY
