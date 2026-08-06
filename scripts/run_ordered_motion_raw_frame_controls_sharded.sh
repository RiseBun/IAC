#!/usr/bin/env bash
set -Eeuo pipefail

# Re-extract raw-frame temporal controls for a completed ordered-motion run.
BASE=${BASE:-/mnt/slurmfs-4090node1/homes/zchen897/IAC}
PKG=${PKG:-$BASE/.codex-tmp/ordered_motion_portable_20260729b/pkg}
OUT=${OUT:?set OUT to a completed ordered-motion experiment directory}
PYBIN=${PYBIN:-$HOME/miniforge3/envs/drivingworld/bin/python}
DEPS=${DEPS:-$BASE/work_dirs/vjepa2_hf_deps}
IMAGE_ROOT=${IMAGE_ROOT:-/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_sensor_blobs}
GROUPS_PER_SHARD=${GROUPS_PER_SHARD:-30}

export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

test -f "$OUT/ordered_motion_alignment.pt"
test -f "$OUT/eval_rows.jsonl"
test -f "$OUT/eval_vjepa.pt"

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
for index, start in enumerate(range(0, len(groups), groups_per_shard)):
    path = shard_dir / f"rows_{index:03d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for group_rows in list(groups.values())[start : start + groups_per_shard]:
            for row in group_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

extract_and_merge() {
  local split=$1
  local shard_dir="$OUT/${split}_shards"
  local cache_dir="$OUT/${split}_cache_shards"
  rm -rf "$cache_dir"
  mkdir -p "$cache_dir"
  local pids=() slot=0 row_file
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
      pids=(); slot=0
    fi
  done
  local pid
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PYBIN" - "$OUT/${split}_vjepa.pt" "$cache_dir"/rows_*_vjepa.pt <<'PY'
import sys
from pathlib import Path
import torch

output = Path(sys.argv[1])
caches = [torch.load(path, map_location="cpu", weights_only=False) for path in sys.argv[2:]]
if not caches or any(cache["metadata"].get("future_num_frames") != 8 for cache in caches):
    raise ValueError("raw control is not a 4+8 cache")
out = {key: torch.cat([cache[key].float() for cache in caches], dim=0)
       for key in ("x", "x_time_tokens", "x_tokens", "y")}
out["sample_id"] = sum((list(cache["sample_id"]) for cache in caches), [])
out["group_id"] = sum((list(cache["group_id"]) for cache in caches), [])
out["source_type"] = sum((list(cache["source_type"]) for cache in caches), [])
out["metadata"] = dict(caches[0]["metadata"])
out["metadata"].update({"rows": len(out["sample_id"]), "num_shards": len(caches), "future_num_frames": 8})
torch.save(out, output)
PY
}

cd "$PKG"
for control in reverse shuffle; do
  split="eval_raw_${control}"
  "$PYBIN" tools/make_temporal_control_rows.py \
    --input-rows "$OUT/eval_rows.jsonl" --output-rows "$OUT/${split}_rows.jsonl" \
    --output-summary "$OUT/${split}_rows_summary.json" --control "$control" --seed 20260802
  prepare_shards "$split"
  extract_and_merge "$split"
  "$PYBIN" tools/score_ordered_motion_alignment.py \
    --model "$OUT/ordered_motion_alignment.pt" --rows "$OUT/eval_rows.jsonl" \
    --visual-cache "$OUT/${split}_vjepa.pt" --feature-key x_time_tokens \
    --output-scores "$OUT/${split}_scores.jsonl" \
    --output-summary "$OUT/${split}_summary.json" --device cuda --batch-size 64
done

"$PYBIN" tools/compare_ordered_motion_scores.py \
  --scores "normal=$OUT/eval_scores.jsonl" \
  --scores "raw_reverse=$OUT/eval_raw_reverse_scores.jsonl" \
  --scores "raw_shuffle=$OUT/eval_raw_shuffle_scores.jsonl" \
  --output-summary "$OUT/raw_frame_control_comparison.json"
