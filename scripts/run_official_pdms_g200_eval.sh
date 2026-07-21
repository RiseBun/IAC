#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${IAC_ROOT:-}" ]]; then
  ROOT="$IAC_ROOT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$HOME/miniforge3/envs/drivingworld/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniforge3/envs/drivingworld/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -d "${TORCH_HOME}/hub/facebookresearch_dinov2_main" ]]; then
  export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-${TORCH_HOME}/hub/facebookresearch_dinov2_main}"
fi

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_multisolution_official_pdms_vnext.py}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_multisolution_official_pdms_vnext}"
CKPT="${CKPT:-$WORK_DIR/checkpoints/latest.pth}"
BENCH_MAX_GROUPS="${BENCH_MAX_GROUPS:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-1}"
export CKPT

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

EVAL_ROOT="${EVAL_ROOT:-$WORK_DIR/g200_eval_epoch_${epoch}}"
mkdir -p "$EVAL_ROOT"
export EVAL_ROOT

"$PYTHON_BIN" -m py_compile benchmark_wam.py tools/audit_iac_ambiguity.py "$CONFIG"

for split in regular low_iou holdout; do
  case "$split" in
    regular)
      INPUT="indices_navsim_future/consistency_val.jsonl"
      ;;
    low_iou)
      INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl"
      ;;
    holdout)
      INPUT="indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl"
      ;;
  esac

  main_dir="$EVAL_ROOT/${split}_main"
  path_dir="$EVAL_ROOT/${split}_path_head"
  mkdir -p "$main_dir" "$path_dir"

  "$PYTHON_BIN" benchmark_wam.py \
    --input "$INPUT" \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --model-kind dinov2 \
    --output-dir "$main_dir" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-groups "$BENCH_MAX_GROUPS" \
    --consistency-score-key consistency_logit \
    --path-causal-metrics \
    --trajectory-specific-causal-metrics \
    --wrong-path-selection mask_iou \
    2>&1 | tee "$EVAL_ROOT/${split}_main.log"

  "$PYTHON_BIN" benchmark_wam.py \
    --input "$INPUT" \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --model-kind dinov2 \
    --output-dir "$path_dir" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-groups "$BENCH_MAX_GROUPS" \
    --consistency-score-key path_evidence_logit \
    --path-causal-metrics \
    --trajectory-specific-causal-metrics \
    --wrong-path-selection mask_iou \
    2>&1 | tee "$EVAL_ROOT/${split}_path_head.log"

  "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
    --scores "$main_dir/wam_iac_scores.jsonl" \
    --output "$EVAL_ROOT/${split}_consistency_ambiguity.json" \
    --per-sample-output "$EVAL_ROOT/${split}_consistency_ambiguity_groups.jsonl"

  for alpha in 0.2 0.5; do
    alpha_tag="${alpha/./p}"
    "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
      --primary-scores "$main_dir/wam_iac_scores.jsonl" \
      --aux-scores "$path_dir/wam_iac_scores.jsonl" \
      --alpha "$alpha" \
      --output "$EVAL_ROOT/${split}_fused_alpha_${alpha_tag}_ambiguity.json" \
      --per-sample-output "$EVAL_ROOT/${split}_fused_alpha_${alpha_tag}_ambiguity_groups.jsonl"
  done
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EVAL_ROOT"])
rows = []
for split in ["regular", "low_iou", "holdout"]:
    for scorer, suffix in [
        ("consistency", "consistency_ambiguity"),
        ("fused alpha=0.2", "fused_alpha_0p2_ambiguity"),
        ("fused alpha=0.5", "fused_alpha_0p5_ambiguity"),
    ]:
        path = root / f"{split}_{suffix}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "split": split,
            "scorer": scorer,
            "hard_top1": data.get("hard_top1"),
            "ambiguity_adjusted_top1": data.get("ambiguity_adjusted_top1"),
            "likely_model_error": (data.get("miss_breakdown") or {}).get("likely_model_error"),
            "ambiguous_accept": (data.get("miss_breakdown") or {}).get("ambiguous_accept"),
            "evidence_supported_miss": (data.get("miss_breakdown") or {}).get("evidence_supported_miss"),
        })
summary = {"eval_root": str(root), "rows": rows}
(root / "g200_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for row in rows:
    print(row)
PY

if [[ "${SUBMIT_CHAIN_AFTER_EVAL:-0}" == "1" ]]; then
  sbatch \
    --partition="${SBATCH_PARTITION:-GPU}" \
    --nodelist="${SBATCH_NODELIST:-4090node1}" \
    --gres="${SBATCH_GRES:-gpu:4}" \
    --cpus-per-task="${SBATCH_CPUS_PER_TASK:-16}" \
    --job-name="${CHAIN_JOB_NAME:-iac_epchain}" \
    --output="$ROOT/$WORK_DIR/slurm_epochchain_%j.log" \
    --chdir="$ROOT" \
    --export="ALL,IAC_ROOT=$ROOT" \
    "$ROOT/scripts/run_official_pdms_epoch_chain_step.sh" \
    | tee -a "$WORK_DIR/slurm_epoch_chain_submit.log"
fi
