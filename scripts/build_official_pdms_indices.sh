#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$HOME/miniforge3/envs/drivingworld/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniforge3/envs/drivingworld/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/third_party/navsim_runtime_deps${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${ROOT}/third_party/navsim_official${PYTHONPATH:+:${PYTHONPATH}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/slurmfs-3090node1_msp/public_data/download/navtrain}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/slurmfs-3090node3_msp/public_data/nuplan/dataset/maps}"

INDEX_DIR="${INDEX_DIR:-indices_navsim_future}"
WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_multisolution_official_pdms_vnext}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-/mnt/slurmfs-3090node1_msp/public_data/download/navtrain/trainval_navsim_logs/trainval}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-work_dirs/navsim_official_metric_cache_iac}"
MAX_GROUPS="${MAX_GROUPS:-0}"
TRAFFIC_AGENTS="${TRAFFIC_AGENTS:-log_replay}"
SAVE_METRIC_CACHE="${SAVE_METRIC_CACHE:-0}"
OFFICIAL_PDMS_WORKERS="${OFFICIAL_PDMS_WORKERS:-1}"
SCORE_SOURCES="${SCORE_SOURCES:-gt_pos,perturb_speed,perturb_lateral,perturb_heading}"

mkdir -p "$INDEX_DIR" "$WORK_DIR"

"$PYTHON_BIN" -m py_compile tools/add_navsim_official_pdms_scores.py

cache_flag=()
if [[ "$SAVE_METRIC_CACHE" == "1" ]]; then
  cache_flag+=(--save-metric-cache --load-metric-cache)
fi

summarize_output() {
  local output="$1"
  local summary="$2"
  OUT_PATH="$output" SUMMARY_PATH="$summary" OFFICIAL_PDMS_WORKERS="$OFFICIAL_PDMS_WORKERS" "$PYTHON_BIN" - <<'PY'
import json
import os
from collections import Counter, defaultdict

out_path = os.environ["OUT_PATH"]
summary_path = os.environ["SUMMARY_PATH"]
sources = Counter()
scored = Counter()
failures = Counter()
groups = set()
tokens = set()
scores = []
by_source = defaultdict(list)

with open(out_path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        source = str(row.get("source_type") or row.get("sample_type") or "unknown")
        group = str(row.get("group_id") or row.get("anchor_id") or row.get("sample_id"))
        token = row.get("navsim_token") or row.get("token") or row.get("sample_token")
        sources[source] += 1
        groups.add(group)
        if token:
            tokens.add(str(token))
        score = row.get("official_pdm_score")
        if score is not None:
            score = float(score)
            scores.append(score)
            by_source[source].append(score)
            scored[source] += 1
        error = row.get("official_pdm_error")
        if error:
            failures[str(error).split(":", 1)[0]] += 1

def avg(values):
    return sum(values) / len(values) if values else None

summary = {
    "rows": sum(sources.values()),
    "groups": len(groups),
    "tokens": len(tokens),
    "source_counts": dict(sources),
    "scored_counts": dict(scored),
    "skipped_or_unscored_counts": {
        key: int(sources[key] - scored.get(key, 0))
        for key in sorted(sources)
        if sources[key] - scored.get(key, 0) > 0
    },
    "failure_counts": dict(failures),
    "mean_official_pdm_score": avg(scores),
    "mean_official_pdm_by_source": {
        key: avg(values) for key, values in sorted(by_source.items())
    },
    "parallel_shards": int(os.environ.get("OFFICIAL_PDMS_WORKERS", "1")),
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

build_split() {
  local split="$1"
  local input="$INDEX_DIR/consistency_${split}.jsonl"
  local output="$INDEX_DIR/consistency_${split}_official_pdms.jsonl"
  local summary="$WORK_DIR/official_pdms_${split}_summary.json"

  if [[ "$OFFICIAL_PDMS_WORKERS" -le 1 ]]; then
    "$PYTHON_BIN" tools/add_navsim_official_pdms_scores.py \
      --input "$input" \
      --output "$output" \
      --summary "$summary" \
      --navsim-log-path "$NAVSIM_LOG_PATH" \
      --metric-cache-path "$METRIC_CACHE_PATH" \
      --traffic-agents "$TRAFFIC_AGENTS" \
      --max-groups "$MAX_GROUPS" \
      --score-sources "$SCORE_SOURCES" \
      "${cache_flag[@]}"
    return
  fi

  echo "[IAC] Building official PDM ${split} with ${OFFICIAL_PDMS_WORKERS} shards"
  rm -f "$output" "$output".shard*.jsonl "$summary".shard*.json "$summary".shard*.log

  local pids=()
  for shard in $(seq 0 $((OFFICIAL_PDMS_WORKERS - 1))); do
    "$PYTHON_BIN" tools/add_navsim_official_pdms_scores.py \
      --input "$input" \
      --output "$output.shard${shard}.jsonl" \
      --summary "$summary.shard${shard}.json" \
      --navsim-log-path "$NAVSIM_LOG_PATH" \
      --metric-cache-path "$METRIC_CACHE_PATH" \
      --traffic-agents "$TRAFFIC_AGENTS" \
      --max-groups "$MAX_GROUPS" \
      --num-shards "$OFFICIAL_PDMS_WORKERS" \
      --shard-index "$shard" \
      --score-sources "$SCORE_SOURCES" \
      "${cache_flag[@]}" \
      > "$summary.shard${shard}.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[IAC] Official PDM ${split} shard ${idx} failed; tail follows:" >&2
      tail -80 "$summary.shard${idx}.log" >&2 || true
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    return 1
  fi

  : > "$output"
  for shard in $(seq 0 $((OFFICIAL_PDMS_WORKERS - 1))); do
    cat "$output.shard${shard}.jsonl" >> "$output"
  done
  summarize_output "$output" "$summary"
}

build_split train
build_split val
