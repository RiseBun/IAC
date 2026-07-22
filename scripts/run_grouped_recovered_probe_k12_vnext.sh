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
SOURCE_EVAL_ROOT="${SOURCE_EVAL_ROOT:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_recovered_set_row_future_k12_20k_vnext/g200_recovered_set_k12}"
FEATURE_DIR="${FEATURE_DIR:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_recovered_set_vnext/features}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_grouped_recovered_set_k12_vnext}"
TRAIN_NEGATIVE_INDEX="${TRAIN_NEGATIVE_INDEX:-indices_navsim_future/consistency_train_official_pdms_highpdm_mismatch.jsonl}"
VAL_NEGATIVE_INDEX="${VAL_NEGATIVE_INDEX:-indices_navsim_future/consistency_val_official_pdms_highpdm_mismatch.jsonl}"

NUM_MODES="${NUM_MODES:-12}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
PROBE_EPOCHS="${PROBE_EPOCHS:-35}"
PROBE_PATIENCE="${PROBE_PATIENCE:-8}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-64}"
EXCLUSION_LOSS_WEIGHT="${EXCLUSION_LOSS_WEIGHT:-0.0}"
EXCLUSION_MARGIN="${EXCLUSION_MARGIN:-2.0}"
MAX_NEGATIVES_PER_GROUP="${MAX_NEGATIVES_PER_GROUP:-8}"
HARD_NEGATIVE_SOURCES="${HARD_NEGATIVE_SOURCES:-image_swap,time_shift,time_shift_future,time_shift_past,traj_swap,reverse,reverse_traj,high_pdm_image_mismatch}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-1}"
RECOVERED_ALPHAS="${RECOVERED_ALPHAS:-0.2,0.3,0.4}"
CONFORMAL_QUANTILE="${CONFORMAL_QUANTILE:-0.8}"
RECOVER_MODE="${RECOVER_MODE:-row_future}"

mkdir -p "$WORK_DIR"

"$PYTHON_BIN" -m py_compile \
  tools/train_recovered_path_set_probe_grouped_from_features.py \
  tools/eval_recovered_path_set_agreement.py \
  tools/fuse_iac_score_jsonl.py \
  tools/audit_iac_ambiguity.py \
  tools/audit_recovered_set_failure_modes.py \
  "$CONFIG"

train_cache="$FEATURE_DIR/train_supported_motion_rich.pt"
val_cache="$FEATURE_DIR/val_supported_motion_rich.pt"
probe_dir="$WORK_DIR/recovered_path_set_probe_grouped_k${NUM_MODES}"
probe_path="$probe_dir/recovered_path_set_probe.pt"
eval_root="$WORK_DIR/g200_grouped_recovered_set_k${NUM_MODES}"
mkdir -p "$probe_dir" "$eval_root"
export GROUPED_EVAL_ROOT="$eval_root"

if [[ ! -s "$train_cache" || ! -s "$val_cache" ]]; then
  echo "[IAC] Missing feature cache under $FEATURE_DIR" >&2
  exit 1
fi

if [[ ! -s "$probe_path" ]]; then
  "$PYTHON_BIN" tools/train_recovered_path_set_probe_grouped_from_features.py \
    --train-cache "$train_cache" \
    --val-cache "$val_cache" \
    --output-dir "$probe_dir" \
    --num-modes "$NUM_MODES" \
    --hidden-dim "$HIDDEN_DIM" \
    --epochs "$PROBE_EPOCHS" \
    --batch-size "$PROBE_BATCH_SIZE" \
    --patience "$PROBE_PATIENCE" \
    --train-negative-index "$TRAIN_NEGATIVE_INDEX" \
    --val-negative-index "$VAL_NEGATIVE_INDEX" \
    --hard-negative-sources "$HARD_NEGATIVE_SOURCES" \
    --max-negatives-per-group "$MAX_NEGATIVES_PER_GROUP" \
    --exclusion-loss-weight "$EXCLUSION_LOSS_WEIGHT" \
    --exclusion-margin "$EXCLUSION_MARGIN"
fi

audit_scores() {
  local split="$1"
  local label="$2"
  local scores="$3"
  "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
    --scores "$scores" \
    --output "$eval_root/${split}_${label}_ambiguity.json" \
    --per-sample-output "$eval_root/${split}_${label}_ambiguity_groups.jsonl"
}

for split in regular low_iou holdout; do
  primary="$SOURCE_EVAL_ROOT/${split}_fused_consistency_path_scores.jsonl"
  recovered="$eval_root/${split}_recovered_set_scores.jsonl"
  if [[ ! -s "$primary" ]]; then
    echo "[IAC] Missing fused consistency+path scores for split=$split under $SOURCE_EVAL_ROOT" >&2
    exit 1
  fi
  if [[ ! -s "$recovered" ]]; then
    {
      "$PYTHON_BIN" tools/eval_recovered_path_set_agreement.py \
        --scores "$primary" \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --probe "$probe_path" \
        --model-kind dinov2 \
        --batch-size "$EVAL_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --score-key iac_consistency \
        --recover-mode "$RECOVER_MODE" \
        --conformal-quantile "$CONFORMAL_QUANTILE" \
        --output-summary "$eval_root/${split}_recovered_set_summary.json" \
        --output-per-group "$eval_root/${split}_recovered_set_groups.jsonl" \
        --output-scored-rows "$eval_root/${split}_recovered_set_rows.jsonl" \
        --output-agreement-scores "$recovered"
    } 2>&1 | tee "$eval_root/${split}_recovered_set.log"
  fi
  audit_scores "$split" "recovered_set_agreement" "$recovered"

  fused_cp="$eval_root/${split}_fused_consistency_path_scores.jsonl"
  cp "$primary" "$fused_cp"
  audit_scores "$split" "fused_consistency_path" "$fused_cp"

  IFS=',' read -ra alpha_items <<< "$RECOVERED_ALPHAS"
  for alpha in "${alpha_items[@]}"; do
    alpha="$(echo "$alpha" | xargs)"
    [[ -n "$alpha" ]] || continue
    fused_cpr="$eval_root/${split}_fused_consistency_path_recovered_a${alpha}_scores.jsonl"
    "$PYTHON_BIN" tools/fuse_iac_score_jsonl.py \
      --primary-scores "$fused_cp" \
      --aux "$recovered:$alpha" \
      --label "consistency_path_grouped_recovered_${alpha}" \
      --output-scores "$fused_cpr"
    audit_scores "$split" "fused_consistency_path_recovered_a${alpha}" "$fused_cpr"
  done
done

"$PYTHON_BIN" tools/audit_recovered_set_failure_modes.py \
  --eval-root "$eval_root" \
  --recovered-alpha "0.3" \
  --output-summary "$eval_root/recovered_set_failure_audit_a0.3.json" \
  --output-groups "$eval_root/recovered_set_failure_audit_a0.3_groups.jsonl"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

eval_root = Path(os.environ["GROUPED_EVAL_ROOT"])
rows = []
for path in sorted(eval_root.glob("*_ambiguity.json")):
    stem = path.name[:-len("_ambiguity.json")]
    split = "holdout" if stem.startswith("holdout_") else ("low_iou" if stem.startswith("low_iou_") else "regular")
    scorer = stem[len(split) + 1:]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "split": split,
        "scorer": scorer,
        "hard_top1": data.get("hard_top1"),
        "ambiguity_adjusted_top1": data.get("ambiguity_adjusted_top1"),
    })
summary = {"eval_root": str(eval_root), "rows": rows}
(eval_root / "g200_grouped_recovered_set_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for row in rows:
    print(row)
PY
