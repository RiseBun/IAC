#!/usr/bin/env bash
#SBATCH --job-name=iac_gate_lofo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_mainline_counterfactual_audit_20260728/lofo_slurm_%j.log

set -euo pipefail

ROOT=/mnt/slurmfs-4090node1/homes/zchen897/IAC
FULL="$ROOT/work_dirs/iac_mainline_full_eval_20260727"
OUT="$ROOT/work_dirs/iac_mainline_counterfactual_audit_20260728/lofo_full"
PYTHON_BIN=/mnt/slurmfs-4090node1/homes/zchen897/miniforge3/envs/drivingworld/bin/python

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$OUT"
cd "$ROOT"

"$PYTHON_BIN" "$FULL/min_repo_tools/tools/run_clean_gate_lofo_audit.py" \
  --train-rows "$ROOT/work_dirs/iac_vjepa2_trajxattn_gate_g200_fast200/train_sample_scores_g200/train_recovered_set_scores.jsonl" \
  --train-visual-cache "$ROOT/work_dirs/iac_vjepa2_trajxattn_gate_g200_fast200/vjepa_features/train_vjepa.pt" \
  --baseline-model "$FULL/min_repo_tools/models/clean_vjepa_traj_gate.pt" \
  --dataset "shard_0=$FULL/v3/regular_shards/shard_0/calibrated_scores.jsonl,$FULL/vjepa/regular_shards/shard_0_vjepa.pt" \
  --dataset "shard_1=$FULL/v3/regular_shards/shard_1/calibrated_scores.jsonl,$FULL/vjepa/regular_shards/shard_1_vjepa.pt" \
  --dataset "shard_2=$FULL/v3/regular_shards/shard_2/calibrated_scores.jsonl,$FULL/vjepa/regular_shards/shard_2_vjepa.pt" \
  --dataset "shard_3=$FULL/v3/regular_shards/shard_3/calibrated_scores.jsonl,$FULL/vjepa/regular_shards/shard_3_vjepa.pt" \
  --output-dir "$OUT" \
  --output-summary "$OUT/summary.json" \
  --heldout-sources "image_swap,time_shift_future,traj_swap,high_pdm_image_mismatch" \
  --seeds "101,202,303" \
  --steps 800 \
  --log-every 800 \
  --batch-size 512 \
  --num-threads 8 \
  --device cuda
