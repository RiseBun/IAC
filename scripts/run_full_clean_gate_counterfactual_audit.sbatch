#!/usr/bin/env bash
#SBATCH --job-name=iac_cf_audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_mainline_counterfactual_audit_20260728/slurm_%j.log

set -euo pipefail

ROOT=/mnt/slurmfs-4090node1/homes/zchen897/IAC
FULL="$ROOT/work_dirs/iac_mainline_full_eval_20260727"
OUT="$ROOT/work_dirs/iac_mainline_counterfactual_audit_20260728"
PYTHON_BIN=/mnt/slurmfs-4090node1/homes/zchen897/miniforge3/envs/drivingworld/bin/python

mkdir -p "$OUT"
cd "$ROOT"

"$PYTHON_BIN" "$FULL/min_repo_tools/tools/audit_clean_gate_counterfactual.py" \
  --model "$FULL/min_repo_tools/models/clean_vjepa_traj_gate.pt" \
  --dataset "shard_0=$FULL/v3/regular_shards/shard_0/calibrated_scores.jsonl,$FULL/vjepa/regular_shards/shard_0_vjepa.pt" \
  --dataset "shard_1=$FULL/v3/regular_shards/shard_1/calibrated_scores.jsonl,$FULL/vjepa/regular_shards/shard_1_vjepa.pt" \
  --dataset "shard_2=$FULL/v3/regular_shards/shard_2/calibrated_scores.jsonl,$FULL/vjepa/regular_shards/shard_2_vjepa.pt" \
  --dataset "shard_3=$FULL/v3/regular_shards/shard_3/calibrated_scores.jsonl,$FULL/vjepa/regular_shards/shard_3_vjepa.pt" \
  --output-summary "$OUT/full_regular_summary.json" \
  --device cuda \
  --batch-size 512 \
  --seed 20260728
