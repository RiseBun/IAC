#!/bin/bash
#SBATCH -J iac_speed_rank
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH -o /mnt/slurmfs-4090node1/homes/zchen897/IAC-ordered-motion/slurm_speed_rank_%A_%a.out
#SBATCH -e /mnt/slurmfs-4090node1/homes/zchen897/IAC-ordered-motion/slurm_speed_rank_%A_%a.err

set -Eeuo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate drivingworld

BASE=/mnt/slurmfs-4090node1/homes/zchen897/IAC-ordered-motion/work/ordered_motion_4s_scene_stratified_v2
PKG=/mnt/slurmfs-4090node1/homes/zchen897/IAC/.codex-tmp/ordered_motion_portable_20260729b/pkg
PYBIN="$HOME/miniforge3/envs/drivingworld/bin/python"
SEEDS=(20260804 20260805)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT="$BASE/speed_rank_seed_${SEED}"

export PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

mkdir -p "$OUT"
cd "$PKG"

"$PYBIN" tools/train_ordered_motion_speed_rank.py \
  --init-model "$BASE/ordered_motion_alignment.pt" \
  --train-rows "$BASE/train_rows.jsonl" \
  --train-cache "$BASE/train_vjepa.pt" \
  --val-rows "$BASE/val_rows.jsonl" \
  --val-cache "$BASE/val_vjepa.pt" \
  --output-model "$OUT/ordered_motion_speed_rank.pt" \
  --output-summary "$OUT/training_summary.json" \
  --margin 0.25 \
  --lambda-speed-rank 0.50 \
  --selection-metric speed_pairwise \
  --epochs 40 \
  --patience 8 \
  --batch-size 64 \
  --num-workers 2 \
  --seed "$SEED"

"$PYBIN" tools/score_ordered_motion_alignment.py \
  --model "$OUT/ordered_motion_speed_rank.pt" \
  --rows "$BASE/eval_rows.jsonl" \
  --visual-cache "$BASE/eval_vjepa.pt" \
  --output-scores "$OUT/eval_scores.jsonl" \
  --output-summary "$OUT/eval_summary.json" \
  --device cuda \
  --batch-size 64
