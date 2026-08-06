#!/usr/bin/env bash
set -Eeuo pipefail

# Reliable 4s ordered-motion pilot. Every extractor invocation is short enough
# to survive scheduler limits, then shards are merged without losing time tokens.
BASE=${BASE:-/mnt/slurmfs-4090node1/homes/zchen897/IAC}
PKG=${PKG:-$BASE/.codex-tmp/ordered_motion_portable_20260729b/pkg}
OUT=${OUT:-/mnt/slurmfs-4090node1/homes/zchen897/IAC-ordered-motion/work/ordered_motion_4s_pilot}
PYBIN=${PYBIN:-$HOME/miniforge3/envs/drivingworld/bin/python}
DEPS=${DEPS:-$BASE/work_dirs/vjepa2_hf_deps}
IMAGE_ROOT=${IMAGE_ROOT:-/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs}
GROUPS_PER_SHARD=${GROUPS_PER_SHARD:-30}

export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

prepare_shards() {
  local split=$1
  local rows="$OUT/${split}_rows.jsonl"
  local shard_dir="$OUT/${split}_shards"
  rm -rf "$shard_dir"
  mkdir -p "$shard_dir"
  "$PYBIN" - "$rows" "$shard_dir" "$GROUPS_PER_SHARD" <<'PY'
import json, sys
from collections import OrderedDict
from pathlib import Path

rows_path, shard_dir = map(Path, sys.argv[1:3])
groups_per_shard = int(sys.argv[3])
groups = OrderedDict()
with rows_path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        group_id = str(row["sample_id"]).rsplit("__", 1)[0]
        groups.setdefault(group_id, []).append(row)
for group_id, rows in groups.items():
    if len(rows) != 7:
        raise ValueError(f"{group_id} has {len(rows)} rows, expected 7")
items = list(groups.items())
for index in range(0, len(items), groups_per_shard):
    path = shard_dir / f"rows_{index // groups_per_shard:03d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for _, group_rows in items[index : index + groups_per_shard]:
            for row in group_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"groups": len(items), "shards": (len(items) + groups_per_shard - 1) // groups_per_shard}))
PY
}

extract_shards() {
  local split=$1
  local shard_dir="$OUT/${split}_shards"
  local cache_dir="$OUT/${split}_cache_shards"
  rm -rf "$cache_dir"
  mkdir -p "$cache_dir"
  local pids=()
  local slot=0
  local row_file
  for row_file in "$shard_dir"/rows_*.jsonl; do
    local stem
    stem=$(basename "$row_file" .jsonl)
    CUDA_VISIBLE_DEVICES=$slot "$PYBIN" "$PKG/tools/extract_vjepa_video_features.py" \
      --index "$row_file" --image-root "$IMAGE_ROOT" \
      --model-name facebook/vjepa2-vitl-fpc64-256 \
      --output "$cache_dir/${stem}_vjepa.pt" \
      --video-mode history_future --history-num-frames 4 --future-num-frames 8 \
      --num-frames 64 --token-summary-size 16 --batch-size 1 --device cuda --log-every 20 \
      > "$cache_dir/${stem}.log" 2>&1 &
    pids+=("$!")
    slot=$((slot + 1))
    if [ "$slot" -eq 4 ]; then
      local pid
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
      slot=0
    fi
  done
  local pid
  for pid in "${pids[@]}"; do wait "$pid"; done
}

merge_shards() {
  local split=$1
  local cache_dir="$OUT/${split}_cache_shards"
  "$PYBIN" - "$OUT/${split}_vjepa.pt" "$cache_dir"/rows_*_vjepa.pt <<'PY'
import json, sys
from pathlib import Path
import torch

output = Path(sys.argv[1])
caches = [torch.load(path, map_location="cpu", weights_only=False) for path in sys.argv[2:]]
required = ("x", "x_time_tokens", "x_tokens", "y")
if not caches or any(any(key not in cache for key in required) for cache in caches):
    raise ValueError("missing required feature tensors")
metadata = dict(caches[0]["metadata"])
if any(cache["metadata"].get("future_num_frames") != 8 for cache in caches):
    raise ValueError("refusing to merge a non-4s feature cache")
out = {
    "x": torch.cat([cache["x"].float() for cache in caches], dim=0),
    "x_time_tokens": torch.cat([cache["x_time_tokens"].float() for cache in caches], dim=0),
    "x_tokens": torch.cat([cache["x_tokens"].float() for cache in caches], dim=0),
    "y": torch.cat([cache["y"].float() for cache in caches], dim=0),
    "sample_id": sum((list(cache["sample_id"]) for cache in caches), []),
    "group_id": sum((list(cache["group_id"]) for cache in caches), []),
    "source_type": sum((list(cache["source_type"]) for cache in caches), []),
    "metadata": metadata,
}
metadata.update({"rows": len(out["sample_id"]), "num_shards": len(caches), "future_num_frames": 8})
torch.save(out, output)
print(json.dumps({"output": str(output), "rows": metadata["rows"], "future_num_frames": metadata["future_num_frames"], "x_time_tokens": list(out["x_time_tokens"].shape)}))
PY
}

extract_split() {
  prepare_shards "$1"
  extract_shards "$1"
  merge_shards "$1"
}

extract_split train
extract_split val
extract_split eval

cd "$PKG"
"$PYBIN" tools/train_ordered_motion_alignment.py \
  --train-rows "$OUT/train_rows.jsonl" --train-cache "$OUT/train_vjepa.pt" \
  --val-rows "$OUT/val_rows.jsonl" --val-cache "$OUT/val_vjepa.pt" \
  --feature-key x_time_tokens --output-model "$OUT/ordered_motion_alignment.pt" \
  --output-summary "$OUT/train_summary.json" --device cuda --seed 20260801 \
  --epochs 40 --patience 6 --batch-size 32

"$PYBIN" tools/score_ordered_motion_alignment.py \
  --model "$OUT/ordered_motion_alignment.pt" --rows "$OUT/val_rows.jsonl" \
  --visual-cache "$OUT/val_vjepa.pt" --feature-key x_time_tokens \
  --output-scores "$OUT/val_scores.jsonl" --output-summary "$OUT/val_summary.json" \
  --device cuda --batch-size 64

"$PYBIN" tools/score_ordered_motion_alignment.py \
  --model "$OUT/ordered_motion_alignment.pt" --rows "$OUT/eval_rows.jsonl" \
  --visual-cache "$OUT/eval_vjepa.pt" --feature-key x_time_tokens \
  --output-scores "$OUT/eval_scores.jsonl" --output-summary "$OUT/eval_summary.json" \
  --device cuda --batch-size 64 --include-segment-ledger

"$PYBIN" tools/audit_ordered_motion_alignment.py \
  --model "$OUT/ordered_motion_alignment.pt" --rows "$OUT/eval_rows.jsonl" \
  --visual-cache "$OUT/eval_vjepa.pt" --feature-key x_time_tokens \
  --output-summary "$OUT/token_and_identity_audit.json" \
  --output-ledger "$OUT/failure_ledger.jsonl" --device cuda --seed 20260801 --batch-size 64

printf '{"status":"complete"}\n' > "$OUT/run_status.json"
