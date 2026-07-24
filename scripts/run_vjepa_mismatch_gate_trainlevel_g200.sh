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

CONFIG="${CONFIG:-configs/train_navsim_future_dinov2_supported_set_listwise_vnext.py}"
BASE_WORK_DIR="${BASE_WORK_DIR:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_vnext}"
CKPT="${CKPT:-$BASE_WORK_DIR/checkpoints/latest.pth}"
TRAIN_INDEX="${TRAIN_INDEX:-indices_navsim_future/consistency_train_official_pdms_highpdm_mismatch.jsonl}"
GROUPED_PROBE="${GROUPED_PROBE:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_recovered_set_vnext/recovered_path_set_probe_k8/recovered_path_set_probe.pt}"
SOURCE_G200_ROOT="${SOURCE_G200_ROOT:-work_dirs/iac_navsim_future_dinov2_supported_set_listwise_recovered_set_vnext/g200_recovered_set_k8}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_vjepa2_mismatch_gate_trainlevel_g200}"

VJEPA_MODEL="${VJEPA_MODEL:-facebook/vjepa2-vitl-fpc64-256}"
VIDEO_MODE="${VIDEO_MODE:-history_future}"
NUM_FRAMES="${NUM_FRAMES:-8}"
VJEPA_POOLING="${VJEPA_POOLING:-mean_std_diff}"
VJEPA_BATCH_SIZE="${VJEPA_BATCH_SIZE:-1}"
TRAIN_FEATURE_SHARDS="${TRAIN_FEATURE_SHARDS:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
HF_DEPS_DIR="${HF_DEPS_DIR:-work_dirs/vjepa2_hf_deps}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -d "$HF_DEPS_DIR" ]]; then
  export PYTHONPATH="$ROOT/$HF_DEPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

TRAIN_MAX_GROUPS="${TRAIN_MAX_GROUPS:-1200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-1}"
PATH_ALPHA="${PATH_ALPHA:-0.2}"
GEOM_ALPHAS="${GEOM_ALPHAS:-0.4}"
PENALTY_THRESHOLDS="${PENALTY_THRESHOLDS:-0.2,0.3,0.4,0.5,0.6}"
PENALTY_WEIGHTS="${PENALTY_WEIGHTS:-0.02,0.05,0.1,0.2,0.35,0.5}"
SCORER_STEPS="${SCORER_STEPS:-700}"
SUPPORTED_SOURCES="${SUPPORTED_SOURCES:-perturb_speed,perturb_lateral,perturb_heading}"
UNKNOWN_SOURCES="${UNKNOWN_SOURCES:-perturb_speed,perturb_lateral,perturb_heading}"
HARD_SOURCES="${HARD_SOURCES:-image_swap,time_shift_future,high_pdm_image_mismatch}"
MIN_SUPPORTED_QUALITY="${MIN_SUPPORTED_QUALITY:-0.90}"
UNKNOWN_WEIGHT="${UNKNOWN_WEIGHT:-0.25}"
UNKNOWN_MARGIN="${UNKNOWN_MARGIN:-0.75}"
LOSS_KIND="${LOSS_KIND:-margin}"
SUPPORTED_MARGIN="${SUPPORTED_MARGIN:-0.75}"
HARD_MARGIN="${HARD_MARGIN:-0.75}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.75}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-1.0}"
LOGIT_L2_WEIGHT="${LOGIT_L2_WEIGHT:-0.01}"
STANDARDIZE_CLIP="${STANDARDIZE_CLIP:-5.0}"
INTERACTION_KIND="${INTERACTION_KIND:-bilinear}"

SCORE_ROOT="$WORK_DIR/train_sample_scores_g${TRAIN_MAX_GROUPS}"
FEATURE_DIR="$WORK_DIR/vjepa_features"
GATE_DIR="$WORK_DIR/mismatch_gate_from_train_g${TRAIN_MAX_GROUPS}"
mkdir -p "$SCORE_ROOT" "$FEATURE_DIR" "$GATE_DIR"
export GATE_DIR

"$PYTHON_BIN" -m py_compile \
  benchmark_wam.py \
  tools/eval_recovered_path_set_agreement.py \
  tools/extract_vjepa_video_features.py \
  tools/merge_feature_caches.py \
  tools/train_visual_mismatch_gate_scorer.py \
  tools/apply_visual_mismatch_penalty.py \
  tools/audit_iac_ambiguity.py \
  tools/fuse_iac_score_jsonl.py \
  "$CONFIG"

if [[ ! -s "$GROUPED_PROBE" ]]; then
  echo "[IAC] Missing grouped recovered probe: $GROUPED_PROBE" >&2
  exit 1
fi

trust_args=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  trust_args+=(--trust-remote-code)
fi

run_score() {
  local score_key="$1"
  local out_dir="$SCORE_ROOT/${score_key}"
  mkdir -p "$out_dir"
  if [[ ! -s "$out_dir/wam_iac_scores.jsonl" ]]; then
    "$PYTHON_BIN" benchmark_wam.py \
      --input "$TRAIN_INDEX" \
      --checkpoint "$CKPT" \
      --config "$CONFIG" \
      --model-kind dinov2 \
      --output-dir "$out_dir" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-groups "$TRAIN_MAX_GROUPS" \
      --consistency-score-key "$score_key" \
      2>&1 | tee "$SCORE_ROOT/${score_key}.log"
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

train_visual_cache="$FEATURE_DIR/train_vjepa.pt"
if [[ ! -s "$train_visual_cache" ]]; then
  if [[ "$TRAIN_FEATURE_SHARDS" -gt 1 ]]; then
    shard_dir="$FEATURE_DIR/train_vjepa_shards"
    mkdir -p "$shard_dir"
    "$PYTHON_BIN" - "$train_recovered_rows" "$shard_dir" "$TRAIN_FEATURE_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

rows_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
num_shards = int(sys.argv[3])
rows = [line for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
for shard in range(num_shards):
    path = out_dir / f"train_recovered_set_rows_shard{shard:02d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(shard, len(rows), num_shards):
            handle.write(rows[idx] + "\n")
    print(json.dumps({"shard": shard, "rows": sum(1 for idx in range(shard, len(rows), num_shards)), "path": str(path)}), flush=True)
PY
    shard_caches=()
    for ((shard=0; shard<TRAIN_FEATURE_SHARDS; shard++)); do
      shard_name="$(printf 'shard%02d' "$shard")"
      shard_rows="$shard_dir/train_recovered_set_rows_${shard_name}.jsonl"
      shard_cache="$shard_dir/train_vjepa_${shard_name}.pt"
      shard_caches+=("$shard_cache")
      if [[ ! -s "$shard_cache" ]]; then
        "$PYTHON_BIN" tools/extract_vjepa_video_features.py \
          --index "$shard_rows" \
          --image-root "$IMAGE_ROOT" \
          --output "$shard_cache" \
          --model-name "$VJEPA_MODEL" \
          --video-mode "$VIDEO_MODE" \
          --num-frames "$NUM_FRAMES" \
          --pooling "$VJEPA_POOLING" \
          --batch-size "$VJEPA_BATCH_SIZE" \
          --log-every 50 \
          "${trust_args[@]}"
      fi
    done
    "$PYTHON_BIN" tools/merge_feature_caches.py \
      --inputs "${shard_caches[@]}" \
      --output "$train_visual_cache"
  else
    "$PYTHON_BIN" tools/extract_vjepa_video_features.py \
      --index "$train_recovered_rows" \
      --image-root "$IMAGE_ROOT" \
      --output "$train_visual_cache" \
      --model-name "$VJEPA_MODEL" \
      --video-mode "$VIDEO_MODE" \
      --num-frames "$NUM_FRAMES" \
      --pooling "$VJEPA_POOLING" \
      --batch-size "$VJEPA_BATCH_SIZE" \
      --log-every 50 \
      "${trust_args[@]}"
  fi
fi

eval_args=()
for split in regular low_iou holdout; do
  rows="$SOURCE_G200_ROOT/${split}_recovered_set_rows.jsonl"
  cache="$FEATURE_DIR/${split}_vjepa.pt"
  if [[ ! -s "$rows" ]]; then
    echo "[IAC] Missing g200 recovered rows: $rows" >&2
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
      --log-every 20 \
      "${trust_args[@]}"
  fi
  eval_args+=(--eval "$split=$rows,$cache,$GATE_DIR/${split}_vjepa_mismatch_gate_scores.jsonl")
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
  --pairwise-weight "$PAIRWISE_WEIGHT" \
  --pairwise-margin "$PAIRWISE_MARGIN" \
  --interaction-kind "$INTERACTION_KIND" \
  --logit-l2-weight "$LOGIT_L2_WEIGHT" \
  --standardize-clip "$STANDARDIZE_CLIP" \
  "${eval_args[@]}"

for split in regular low_iou holdout; do
  gate="$GATE_DIR/${split}_vjepa_mismatch_gate_scores.jsonl"
  "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
    --scores "$gate" \
    --output "$GATE_DIR/${split}_vjepa_mismatch_gate_ambiguity.json" \
    --per-sample-output "$GATE_DIR/${split}_vjepa_mismatch_gate_ambiguity_groups.jsonl"

  IFS=',' read -ra geom_items <<< "$GEOM_ALPHAS"
  for geom_alpha in "${geom_items[@]}"; do
    geom_alpha="$(echo "$geom_alpha" | xargs)"
    [[ -n "$geom_alpha" ]] || continue
    geom="$SOURCE_G200_ROOT/${split}_fused_consistency_path_recovered_a${geom_alpha}_scores.jsonl"
    [[ -s "$geom" ]] || continue
    IFS=',' read -ra threshold_items <<< "$PENALTY_THRESHOLDS"
    for threshold in "${threshold_items[@]}"; do
      threshold="$(echo "$threshold" | xargs)"
      [[ -n "$threshold" ]] || continue
      IFS=',' read -ra weight_items <<< "$PENALTY_WEIGHTS"
      for weight in "${weight_items[@]}"; do
        weight="$(echo "$weight" | xargs)"
        [[ -n "$weight" ]] || continue
        penalized="$GATE_DIR/${split}_geom_a${geom_alpha}_vjepa_penalty_w${weight}_t${threshold}_scores.jsonl"
        "$PYTHON_BIN" tools/apply_visual_mismatch_penalty.py \
          --primary-scores "$geom" \
          --gate-scores "$gate" \
          --weight "$weight" \
          --threshold "$threshold" \
          --label "grouped_geom_${geom_alpha}_vjepa_mismatch_penalty_w${weight}_t${threshold}" \
          --output-scores "$penalized"
        "$PYTHON_BIN" tools/audit_iac_ambiguity.py \
          --scores "$penalized" \
          --output "$GATE_DIR/${split}_geom_a${geom_alpha}_vjepa_penalty_w${weight}_t${threshold}_ambiguity.json" \
          --per-sample-output "$GATE_DIR/${split}_geom_a${geom_alpha}_vjepa_penalty_w${weight}_t${threshold}_ambiguity_groups.jsonl"
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
hard = {"image_swap", "time_shift", "time_shift_future", "time_shift_past", "high_pdm_image_mismatch", "traj_swap", "reverse_traj"}
near = {"perturb_speed", "perturb_lateral", "perturb_heading"}

def rows(path):
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

def gid(row):
    return row.get("group_id") or row.get("anchor_id") or row.get("sample_id")

def source_rates(path):
    grouped = defaultdict(list)
    for row in rows(path):
        grouped[str(gid(row))].append(row)
    hard_vals = []
    near_vals = []
    for items in grouped.values():
        pos = [row for row in items if positive(row)]
        if not pos:
            continue
        gt = float(pos[0]["iac_consistency"])
        hard_vals.append(float(any(source(row) in hard and float(row["iac_consistency"]) > gt for row in items)))
        near_vals.append(float(any(source(row) in near and float(row["iac_consistency"]) > gt for row in items)))
    return {
        "hard_mismatch_above_gt_group_rate": sum(hard_vals) / len(hard_vals) if hard_vals else None,
        "near_perturb_above_gt_group_rate": sum(near_vals) / len(near_vals) if near_vals else None,
    }

summary = []
for path in sorted(root.glob("*_ambiguity.json")):
    stem = path.name[:-len("_ambiguity.json")]
    split = "holdout" if stem.startswith("holdout_") else ("low_iou" if stem.startswith("low_iou_") else "regular")
    scorer = stem[len(split) + 1:]
    data = json.loads(path.read_text(encoding="utf-8"))
    score_path = root / f"{stem}_scores.jsonl"
    item = {
        "split": split,
        "scorer": scorer,
        "hard_top1": data.get("hard_top1"),
        "ambiguity_adjusted_top1": data.get("ambiguity_adjusted_top1"),
    }
    if score_path.exists():
        item.update(source_rates(score_path))
    summary.append(item)

out = {"gate_root": str(root), "rows": summary}
(root / "vjepa_mismatch_gate_trainlevel_g200_summary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for item in summary:
    print(item, flush=True)
PY
