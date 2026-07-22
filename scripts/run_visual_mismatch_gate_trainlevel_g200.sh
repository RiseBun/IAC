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
TRAIN_INDEX="${TRAIN_INDEX:-indices_navsim_future/consistency_train_official_pdms_highpdm_mismatch.jsonl}"
GROUPED_PROBE="${GROUPED_PROBE:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_grouped_recovered_set_k12_short_vnext/recovered_path_set_probe_grouped_k12/recovered_path_set_probe.pt}"
SOURCE_G200_ROOT="${SOURCE_G200_ROOT:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_grouped_recovered_set_k12_short_vnext/g200_grouped_recovered_set_k12}"
G200_VISUAL_FEATURE_DIR="${G200_VISUAL_FEATURE_DIR:-work_dirs/iac_navsim_future_dinov2_visual_conditioned_agreement_g200_vnext/features}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_visual_mismatch_gate_trainlevel_vnext}"

TRAIN_MAX_GROUPS="${TRAIN_MAX_GROUPS:-1200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PATH_ALPHA="${PATH_ALPHA:-0.2}"
GEOM_ALPHAS="${GEOM_ALPHAS:-0.2,0.3,0.4}"
GATE_ALPHAS="${GATE_ALPHAS:-0.03,0.05,0.1,0.2}"
PENALTY_THRESHOLDS="${PENALTY_THRESHOLDS:-0.2,0.3,0.4,0.5}"
PENALTY_WEIGHTS="${PENALTY_WEIGHTS:-0.05,0.1,0.2,0.5}"
SCORER_STEPS="${SCORER_STEPS:-500}"
SUPPORTED_SOURCES="${SUPPORTED_SOURCES:-perturb_speed,perturb_lateral,perturb_heading}"
UNKNOWN_SOURCES="${UNKNOWN_SOURCES:-perturb_speed,perturb_lateral,perturb_heading}"
HARD_SOURCES="${HARD_SOURCES:-image_swap,time_shift_future,high_pdm_image_mismatch}"
MIN_SUPPORTED_QUALITY="${MIN_SUPPORTED_QUALITY:-0.90}"
UNKNOWN_WEIGHT="${UNKNOWN_WEIGHT:-0.10}"
UNKNOWN_MARGIN="${UNKNOWN_MARGIN:-1.0}"
LOSS_KIND="${LOSS_KIND:-margin}"
SUPPORTED_MARGIN="${SUPPORTED_MARGIN:-1.0}"
HARD_MARGIN="${HARD_MARGIN:-1.0}"
LOGIT_L2_WEIGHT="${LOGIT_L2_WEIGHT:-0.001}"
STANDARDIZE_CLIP="${STANDARDIZE_CLIP:-5.0}"
TRAIN_CAUSAL_METRICS="${TRAIN_CAUSAL_METRICS:-0}"

SCORE_ROOT="$WORK_DIR/train_sample_scores_g${TRAIN_MAX_GROUPS}"
FEATURE_DIR="$WORK_DIR/features"
GATE_DIR="$WORK_DIR/mismatch_gate_from_train_g${TRAIN_MAX_GROUPS}"
mkdir -p "$SCORE_ROOT" "$FEATURE_DIR" "$GATE_DIR"
export GATE_DIR

"$PYTHON_BIN" -m py_compile \
  benchmark_wam.py \
  tools/eval_recovered_path_set_agreement.py \
  tools/extract_recovered_path_features.py \
  tools/train_visual_mismatch_gate_scorer.py \
  tools/apply_visual_mismatch_penalty.py \
  tools/fuse_iac_score_jsonl.py \
  tools/audit_iac_ambiguity.py \
  "$CONFIG"

if [[ ! -s "$GROUPED_PROBE" ]]; then
  echo "[IAC] Missing grouped recovered probe: $GROUPED_PROBE" >&2
  exit 1
fi

run_score() {
  local score_key="$1"
  local out_dir="$SCORE_ROOT/${score_key}"
  mkdir -p "$out_dir"
  if [[ ! -s "$out_dir/wam_iac_scores.jsonl" ]]; then
    args=(
      "$PYTHON_BIN" benchmark_wam.py
      --input "$TRAIN_INDEX" \
      --checkpoint "$CKPT" \
      --config "$CONFIG" \
      --model-kind dinov2 \
      --output-dir "$out_dir" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-groups "$TRAIN_MAX_GROUPS" \
      --consistency-score-key "$score_key"
    )
    if [[ "$TRAIN_CAUSAL_METRICS" == "1" ]]; then
      args+=(
        --path-causal-metrics
        --trajectory-specific-causal-metrics
        --wrong-path-selection mask_iou
      )
    fi
    "${args[@]}" 2>&1 | tee "$SCORE_ROOT/${score_key}.log"
  fi
}

run_score consistency_logit
run_score path_evidence_logit

train_cp="$SCORE_ROOT/train_fused_consistency_path_scores.jsonl"
if [[ ! -s "$train_cp" ]]; then
  "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
    --primary-scores "$SCORE_ROOT/consistency_logit/wam_iac_scores.jsonl" \
    --aux "$SCORE_ROOT/path_evidence_logit/wam_iac_scores.jsonl:$PATH_ALPHA" \
    --label "train_consistency_path_${PATH_ALPHA}" \
    --output-scores "$train_cp"
fi

train_recovered_rows="$SCORE_ROOT/train_recovered_set_rows.jsonl"
train_recovered_scores="$SCORE_ROOT/train_recovered_set_scores.jsonl"
if [[ ! -s "$train_recovered_rows" ]]; then
  "$PYTHON_BIN" tools/eval_recovered_path_set_agreement.py \
    --scores "$train_cp" \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --probe "$GROUPED_PROBE" \
    --model-kind dinov2 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --score-key iac_consistency \
    --recover-mode row_future \
    --conformal-quantile 0.8 \
    --output-summary "$SCORE_ROOT/train_recovered_set_summary.json" \
    --output-per-group "$SCORE_ROOT/train_recovered_set_groups.jsonl" \
    --output-scored-rows "$train_recovered_rows" \
    --output-agreement-scores "$train_recovered_scores" \
    2>&1 | tee "$SCORE_ROOT/train_recovered_set.log"
fi

train_visual_cache="$FEATURE_DIR/train_visual_motion_rich.pt"
if [[ ! -s "$train_visual_cache" ]]; then
  "$PYTHON_BIN" tools/extract_recovered_path_features.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --index "$train_recovered_rows" \
    --output "$train_visual_cache" \
    --model-kind dinov2 \
    --input-mode motion_rich \
    --batch-size "$FEATURE_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --log-every 50
fi

eval_args=()
for split in regular low_iou holdout; do
  rows="$SOURCE_G200_ROOT/${split}_recovered_set_rows.jsonl"
  cache="$G200_VISUAL_FEATURE_DIR/${split}_visual_motion_rich.pt"
  if [[ ! -s "$rows" ]]; then
    echo "[IAC] Missing g200 recovered rows: $rows" >&2
    exit 1
  fi
  if [[ ! -s "$cache" ]]; then
    mkdir -p "$G200_VISUAL_FEATURE_DIR"
    "$PYTHON_BIN" tools/extract_recovered_path_features.py \
      --config "$CONFIG" \
      --checkpoint "$CKPT" \
      --index "$rows" \
      --output "$cache" \
      --model-kind dinov2 \
      --input-mode motion_rich \
      --batch-size "$FEATURE_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --log-every 20
  fi
  eval_args+=(
    --eval "$split=$rows,$cache,$GATE_DIR/${split}_visual_mismatch_gate_scores.jsonl"
  )
done

"$PYTHON_BIN" tools/train_visual_mismatch_gate_scorer.py \
  --train-rows "$train_recovered_rows" \
  --train-visual-cache "$train_visual_cache" \
  --output-dir "$GATE_DIR" \
  --steps "$SCORER_STEPS" \
  --supported-sources "$SUPPORTED_SOURCES" \
  --unknown-sources "$UNKNOWN_SOURCES" \
  --hard-sources "$HARD_SOURCES" \
  --min-supported-quality "$MIN_SUPPORTED_QUALITY" \
  --unknown-weight "$UNKNOWN_WEIGHT" \
  --unknown-margin "$UNKNOWN_MARGIN" \
  --loss-kind "$LOSS_KIND" \
  --supported-margin "$SUPPORTED_MARGIN" \
  --hard-margin "$HARD_MARGIN" \
  --logit-l2-weight "$LOGIT_L2_WEIGHT" \
  --standardize-clip "$STANDARDIZE_CLIP" \
  "${eval_args[@]}"

for split in regular low_iou holdout; do
  gate="$GATE_DIR/${split}_visual_mismatch_gate_scores.jsonl"
  "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
    --scores "$gate" \
    --output "$GATE_DIR/${split}_visual_mismatch_gate_ambiguity.json" \
    --per-sample-output "$GATE_DIR/${split}_visual_mismatch_gate_ambiguity_groups.jsonl"

  IFS=',' read -ra geom_items <<< "$GEOM_ALPHAS"
  for geom_alpha in "${geom_items[@]}"; do
    geom_alpha="$(echo "$geom_alpha" | xargs)"
    [[ -n "$geom_alpha" ]] || continue
    geom="$SOURCE_G200_ROOT/${split}_fused_consistency_path_recovered_a${geom_alpha}_scores.jsonl"
    [[ -s "$geom" ]] || continue
    IFS=',' read -ra gate_items <<< "$GATE_ALPHAS"
    for gate_alpha in "${gate_items[@]}"; do
      gate_alpha="$(echo "$gate_alpha" | xargs)"
      [[ -n "$gate_alpha" ]] || continue
      fused="$GATE_DIR/${split}_geom_a${geom_alpha}_gate_a${gate_alpha}_scores.jsonl"
      "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
        --primary-scores "$geom" \
        --aux "$gate:$gate_alpha" \
        --label "grouped_geom_${geom_alpha}_mismatch_gate_${gate_alpha}" \
        --output-scores "$fused"
      "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
        --scores "$fused" \
        --output "$GATE_DIR/${split}_geom_a${geom_alpha}_gate_a${gate_alpha}_ambiguity.json" \
        --per-sample-output "$GATE_DIR/${split}_geom_a${geom_alpha}_gate_a${gate_alpha}_ambiguity_groups.jsonl"
    done

    IFS=',' read -ra threshold_items <<< "$PENALTY_THRESHOLDS"
    for threshold in "${threshold_items[@]}"; do
      threshold="$(echo "$threshold" | xargs)"
      [[ -n "$threshold" ]] || continue
      IFS=',' read -ra weight_items <<< "$PENALTY_WEIGHTS"
      for weight in "${weight_items[@]}"; do
        weight="$(echo "$weight" | xargs)"
        [[ -n "$weight" ]] || continue
        penalized="$GATE_DIR/${split}_geom_a${geom_alpha}_penalty_w${weight}_t${threshold}_scores.jsonl"
        "$PYTHON_BIN" tools/apply_visual_mismatch_penalty.py \
          --primary-scores "$geom" \
          --gate-scores "$gate" \
          --weight "$weight" \
          --threshold "$threshold" \
          --label "grouped_geom_${geom_alpha}_visual_mismatch_penalty_w${weight}_t${threshold}" \
          --output-scores "$penalized"
        "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
          --scores "$penalized" \
          --output "$GATE_DIR/${split}_geom_a${geom_alpha}_penalty_w${weight}_t${threshold}_ambiguity.json" \
          --per-sample-output "$GATE_DIR/${split}_geom_a${geom_alpha}_penalty_w${weight}_t${threshold}_ambiguity_groups.jsonl"
      done
    done
  done
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["GATE_DIR"])
hard = {"image_swap", "time_shift", "time_shift_future", "time_shift_past", "high_pdm_image_mismatch"}

def rows(path):
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

def gid(row):
    return row.get("group_id") or row.get("anchor_id") or row.get("sample_id")

def hard_above(path):
    grouped = defaultdict(list)
    for row in rows(path):
        grouped[str(gid(row))].append(row)
    vals = []
    for items in grouped.values():
        pos = [row for row in items if positive(row)]
        if not pos:
            continue
        gt = float(pos[0]["iac_consistency"])
        vals.append(float(any(source(row) in hard and float(row["iac_consistency"]) > gt for row in items)))
    return sum(vals) / len(vals) if vals else None

summary = []
for path in sorted(root.glob("*_ambiguity.json")):
    stem = path.name[:-len("_ambiguity.json")]
    split = "holdout" if stem.startswith("holdout_") else ("low_iou" if stem.startswith("low_iou_") else "regular")
    scorer = stem[len(split) + 1:]
    data = json.loads(path.read_text(encoding="utf-8"))
    score_path = root / f"{stem}_scores.jsonl"
    summary.append({
        "split": split,
        "scorer": scorer,
        "hard_top1": data.get("hard_top1"),
        "ambiguity_adjusted_top1": data.get("ambiguity_adjusted_top1"),
        "hard_mismatch_above_gt_group_rate": hard_above(score_path) if score_path.exists() else None,
    })
out = {"gate_root": str(root), "rows": summary}
(root / "visual_mismatch_gate_g200_summary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for item in summary:
    print(item)
PY
