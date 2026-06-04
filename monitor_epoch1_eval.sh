#!/usr/bin/env bash
set -u

cd /root/autodl-tmp/IAC

run="v5_minimal_full_e5_b96_amp_epochckpt_fromscratch"
ckpt="work_dirs/${run}/checkpoints/epoch_1.pth"
trainlog="logs/${run}.out"
evallog="logs/${run}_epoch1_eval.out"

echo "MONITOR_STARTED $(date '+%F %T')"
while [ ! -f "$ckpt" ]; do
  last_line="$(tail -n 1 "$trainlog" 2>/dev/null || true)"
  echo "WAITING $(date '+%F %T') ${last_line}"
  sleep 30
done

echo "FOUND_EPOCH1 $(date '+%F %T') $ckpt"
CUDA_VISIBLE_DEVICES=0 \
TORCH_HOME=/root/.cache/torch \
DINOV2_HUB_DIR=/root/.cache/torch/hub/facebookresearch_dinov2_main \
NUPLAN_DATA_ROOT=/root/autodl-tmp \
NUPLAN_DB_ROOT=/root/autodl-tmp/data/cache/mini \
NUPLAN_INDEX_ROOT=/root/autodl-tmp/nuplan/indices_v4 \
/root/miniconda3/bin/python eval_dinov2_critic.py \
  --checkpoint "$ckpt" \
  --config configs/train_dinov2_v5_minimal.py \
  --split val \
  --batch-size 32 \
  --max-samples 4096 \
  --eval-ranking \
  --max-ranking-groups 512 \
  --output-prefix eval_epoch1_4096_rank512 \
  > "$evallog" 2>&1

status=$?
echo "EVAL_EXIT=${status} $(date '+%F %T')"
tail -n 80 "$evallog" 2>/dev/null || true
exit "$status"
