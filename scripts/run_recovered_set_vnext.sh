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
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_recovered_set_vnext}"

TRAIN_INDEX="${TRAIN_INDEX:-indices_navsim_future/consistency_train_official_pdms_highpdm_mismatch.jsonl}"
VAL_INDEX="${VAL_INDEX:-indices_navsim_future/consistency_val_official_pdms_highpdm_mismatch.jsonl}"
SUPPORTED_SOURCES="${SUPPORTED_SOURCES:-perturb_speed,perturb_lateral,perturb_heading}"
SUPPORTED_MIN_QUALITY="${SUPPORTED_MIN_QUALITY:-0.76}"

INPUT_MODE="${INPUT_MODE:-motion_rich}"
NUM_MODES="${NUM_MODES:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
PROBE_EPOCHS="${PROBE_EPOCHS:-80}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-256}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_TRAIN_FEATURES="${MAX_TRAIN_FEATURES:-20000}"
MAX_VAL_FEATURES="${MAX_VAL_FEATURES:-4000}"
BENCH_MAX_GROUPS="${BENCH_MAX_GROUPS:-200}"
PATH_ALPHA="${PATH_ALPHA:-0.2}"
RECOVERED_ALPHAS="${RECOVERED_ALPHAS:-0.1,0.2,0.4}"
CONFORMAL_QUANTILE="${CONFORMAL_QUANTILE:-0.8}"
RECOVER_MODE="${RECOVER_MODE:-group_gt_future}"
RUN_GATE_CALIBRATOR="${RUN_GATE_CALIBRATOR:-1}"
GATE_TRAIN_SPLIT="${GATE_TRAIN_SPLIT:-regular}"
GATE_STEPS="${GATE_STEPS:-300}"
GATE_HIDDEN_DIM="${GATE_HIDDEN_DIM:-8}"

mkdir -p "$WORK_DIR"

if [[ ! -s "$CKPT" ]]; then
  echo "[IAC] Missing checkpoint: $CKPT" >&2
  exit 1
fi

"$PYTHON_BIN" -m py_compile \
  benchmark_wam.py \
  tools/extract_recovered_path_features.py \
  tools/train_recovered_path_set_probe_from_features.py \
  tools/eval_recovered_path_set_agreement.py \
  tools/fuse_iac_score_jsonl.py \
  tools/train_iac_recovered_gate_calibrator.py \
  tools/audit_iac_ambiguity.py \
  "$CONFIG"

FEATURE_DIR="$WORK_DIR/features"
PROBE_DIR="$WORK_DIR/recovered_path_set_probe_k${NUM_MODES}"
EVAL_ROOT="${EVAL_ROOT:-$WORK_DIR/g200_recovered_set_k${NUM_MODES}}"
mkdir -p "$FEATURE_DIR" "$PROBE_DIR" "$EVAL_ROOT"
export EVAL_ROOT

train_cache="$FEATURE_DIR/train_supported_${INPUT_MODE}.pt"
val_cache="$FEATURE_DIR/val_supported_${INPUT_MODE}.pt"
probe_path="$PROBE_DIR/recovered_path_set_probe.pt"

if [[ ! -s "$train_cache" ]]; then
  "$PYTHON_BIN" tools/extract_recovered_path_features.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --index "$TRAIN_INDEX" \
    --output "$train_cache" \
    --model-kind dinov2 \
    --input-mode "$INPUT_MODE" \
    --positive-only \
    --supported-sources "$SUPPORTED_SOURCES" \
    --min-quality "$SUPPORTED_MIN_QUALITY" \
    --max-samples "$MAX_TRAIN_FEATURES" \
    --shuffle \
    --seed 20260722 \
    --batch-size "$FEATURE_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
fi

if [[ ! -s "$val_cache" ]]; then
  "$PYTHON_BIN" tools/extract_recovered_path_features.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --index "$VAL_INDEX" \
    --output "$val_cache" \
    --model-kind dinov2 \
    --input-mode "$INPUT_MODE" \
    --positive-only \
    --supported-sources "$SUPPORTED_SOURCES" \
    --min-quality "$SUPPORTED_MIN_QUALITY" \
    --max-samples "$MAX_VAL_FEATURES" \
    --shuffle \
    --seed 20260723 \
    --batch-size "$FEATURE_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
fi

if [[ ! -s "$probe_path" ]]; then
  "$PYTHON_BIN" tools/train_recovered_path_set_probe_from_features.py \
    --train-cache "$train_cache" \
    --val-cache "$val_cache" \
    --output-dir "$PROBE_DIR" \
    --num-modes "$NUM_MODES" \
    --hidden-dim "$HIDDEN_DIM" \
    --epochs "$PROBE_EPOCHS" \
    --batch-size "$PROBE_BATCH_SIZE"
fi

run_score() {
  local split="$1"
  local score_key="$2"
  local input="$3"
  local out_dir="$EVAL_ROOT/${split}_${score_key}"
  mkdir -p "$out_dir"
  if [[ ! -s "$out_dir/wam_iac_scores.jsonl" ]]; then
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
  fi
  if [[ ! -s "$EVAL_ROOT/${split}_${score_key}_ambiguity.json" ]]; then
    "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
      --scores "$out_dir/wam_iac_scores.jsonl" \
      --output "$EVAL_ROOT/${split}_${score_key}_ambiguity.json" \
      --per-sample-output "$EVAL_ROOT/${split}_${score_key}_ambiguity_groups.jsonl"
  fi
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

run_recovered() {
  local split="$1"
  local primary_scores="$2"
  if [[ ! -s "$EVAL_ROOT/${split}_recovered_set_scores.jsonl" ]]; then
    "$PYTHON_BIN" tools/eval_recovered_path_set_agreement.py \
      --scores "$primary_scores" \
      --config "$CONFIG" \
      --checkpoint "$CKPT" \
      --probe "$probe_path" \
      --model-kind dinov2 \
      --batch-size "$EVAL_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --score-key iac_consistency \
      --recover-mode "$RECOVER_MODE" \
      --conformal-quantile "$CONFORMAL_QUANTILE" \
      --output-summary "$EVAL_ROOT/${split}_recovered_set_summary.json" \
      --output-per-group "$EVAL_ROOT/${split}_recovered_set_groups.jsonl" \
      --output-scored-rows "$EVAL_ROOT/${split}_recovered_set_rows.jsonl" \
      --output-agreement-scores "$EVAL_ROOT/${split}_recovered_set_scores.jsonl" \
      2>&1 | tee "$EVAL_ROOT/${split}_recovered_set.log"
  fi
  audit_scores "$split" "recovered_set_agreement" "$EVAL_ROOT/${split}_recovered_set_scores.jsonl"
}

for split in regular low_iou holdout; do
  case "$split" in
    regular) INPUT="indices_navsim_future/consistency_val.jsonl" ;;
    low_iou) INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl" ;;
    holdout) INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl" ;;
  esac

  run_score "$split" consistency_logit "$INPUT"
  run_score "$split" path_evidence_logit "$INPUT"

  primary="$EVAL_ROOT/${split}_consistency_logit/wam_iac_scores.jsonl"
  path="$EVAL_ROOT/${split}_path_evidence_logit/wam_iac_scores.jsonl"
  recovered="$EVAL_ROOT/${split}_recovered_set_scores.jsonl"

  run_recovered "$split" "$primary"

  fused_cp="$EVAL_ROOT/${split}_fused_consistency_path_scores.jsonl"
  "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
    --primary-scores "$primary" \
    --aux "$path:$PATH_ALPHA" \
    --label "consistency_path_${PATH_ALPHA}" \
    --output-scores "$fused_cp"
  audit_scores "$split" "fused_consistency_path" "$fused_cp"

  IFS=',' read -ra alpha_items <<< "$RECOVERED_ALPHAS"
  for alpha in "${alpha_items[@]}"; do
    alpha="$(echo "$alpha" | xargs)"
    [[ -n "$alpha" ]] || continue
    fused_cpr="$EVAL_ROOT/${split}_fused_consistency_path_recovered_a${alpha}_scores.jsonl"
    "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
      --primary-scores "$primary" \
      --aux "$path:$PATH_ALPHA" \
      --aux "$recovered:$alpha" \
      --label "consistency_path_${PATH_ALPHA}_recovered_${alpha}" \
      --output-scores "$fused_cpr"
    audit_scores "$split" "fused_consistency_path_recovered_a${alpha}" "$fused_cpr"
  done
done

if [[ "$RUN_GATE_CALIBRATOR" == "1" ]]; then
  gate_dir="$EVAL_ROOT/learned_recovered_gate_from_${GATE_TRAIN_SPLIT}"
  gate_args=(
    --train-main "$EVAL_ROOT/${GATE_TRAIN_SPLIT}_consistency_logit/wam_iac_scores.jsonl"
    --train-path "$EVAL_ROOT/${GATE_TRAIN_SPLIT}_path_evidence_logit/wam_iac_scores.jsonl"
    --train-recovered "$EVAL_ROOT/${GATE_TRAIN_SPLIT}_recovered_set_scores.jsonl"
    --output-dir "$gate_dir"
    --steps "$GATE_STEPS"
    --hidden-dim "$GATE_HIDDEN_DIM"
  )
  for split in regular low_iou holdout; do
    gate_args+=(
      --eval "$split=$EVAL_ROOT/${split}_consistency_logit/wam_iac_scores.jsonl,$EVAL_ROOT/${split}_path_evidence_logit/wam_iac_scores.jsonl,$EVAL_ROOT/${split}_recovered_set_scores.jsonl"
    )
  done
  "$PYTHON_BIN" tools/train_iac_recovered_gate_calibrator.py "${gate_args[@]}"
  for split in regular low_iou holdout; do
    audit_scores "$split" "learned_recovered_gate_from_${GATE_TRAIN_SPLIT}" "$gate_dir/${split}_scores.jsonl"
  done
fi

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"])
rows = []
for path in sorted(root.glob("*_ambiguity.json")):
    stem = path.name[:-len("_ambiguity.json")]
    split = "holdout" if stem.startswith("holdout_") else ("low_iou" if stem.startswith("low_iou_") else "regular")
    scorer = stem[len(split) + 1:]
    data = json.loads(path.read_text(encoding="utf-8"))
    miss = data.get("miss_breakdown") or {}
    rec_summary = root / f"{split}_recovered_set_summary.json"
    high_pdm = None
    if rec_summary.exists():
        high_pdm = json.loads(rec_summary.read_text(encoding="utf-8")).get("high_pdm_mismatch_above_gt_rate")
    rows.append({
        "split": split,
        "scorer": scorer,
        "hard_top1": data.get("hard_top1"),
        "ambiguity_adjusted_top1": data.get("ambiguity_adjusted_top1"),
        "likely_model_error": miss.get("likely_model_error"),
        "ambiguous_accept": miss.get("ambiguous_accept"),
        "evidence_supported_miss": miss.get("evidence_supported_miss"),
        "high_pdm_mismatch_above_gt_rate": high_pdm,
    })
summary = {"eval_root": str(root), "rows": rows}
(root / "g200_recovered_set_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for row in rows:
    print(row)
PY
