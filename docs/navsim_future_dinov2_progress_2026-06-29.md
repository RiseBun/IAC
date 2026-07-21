# NAVSIM strict-future DINOv2 progress, 2026-06-29

## What changed

- Added `configs/train_navsim_future_dinov2.py`.
- Added `scripts/run_navsim_future_dinov2.sh`.
- Added `.gitattributes` to force LF line endings for shell scripts.
- Uploaded the DINOv2 strict-future config and script to:
  `/mnt/slurmfs-4090node1/homes/zchen897/IAC`.

## Runtime

- Server default SSH shell only exposes `/usr/bin/python3`, without torch.
- Working DINOv2 runtime:
  `/mnt/slurmfs-4090node1/homes/zchen897/miniforge3/envs/drivingworld/bin/python`
- `drivingworld` has Python 3.10 and CUDA torch, which is needed because the
  current torch.hub DINOv2 code uses Python 3.10 union annotations.
- DINOv2 weights were downloaded and cached under:
  `/mnt/slurmfs-4090node1/homes/zchen897/.cache/torch`.

## Completed remote run

Work dir:
`work_dirs/iac_navsim_future_dinov2_quick`

Training command used:

```bash
PYTHON_BIN=/mnt/slurmfs-4090node1/homes/zchen897/miniforge3/envs/drivingworld/bin/python \
CUDA_VISIBLE_DEVICES=0 \
WORK_DIR=work_dirs/iac_navsim_future_dinov2_quick \
MAX_TRAIN_STEPS=2000 \
MAX_VAL_STEPS=500 \
MAX_EVAL_SAMPLES=4096 \
BATCH_SIZE=4 \
EVAL_BATCH_SIZE=8 \
NUM_WORKERS=2 \
PREFLIGHT_SAMPLES=256 \
scripts/run_navsim_future_dinov2.sh
```

Training completed and saved:

- `work_dirs/iac_navsim_future_dinov2_quick/checkpoints/best.pth`
- `work_dirs/iac_navsim_future_dinov2_quick/checkpoints/latest.pth`
- `work_dirs/iac_navsim_future_dinov2_quick/checkpoints/epoch_1.pth`

## Follow-up benchmark

The first script run stopped after training because the shell script had CRLF
line endings, which broke Bash line continuations. The root cause was fixed by
adding `.gitattributes` and resyncing the script as LF.

Detached benchmark sessions started before SSH became unreachable:

- `iac_future_dinov2_bench`: 4096 validation samples on GPU 0.
- `iac_future_dinov2_bench1k`: 1024 validation samples on GPU 1.

Expected result paths:

- `work_dirs/iac_navsim_future_dinov2_quick/benchmark_val_1024/wam_iac_summary.json`
- `work_dirs/iac_navsim_future_dinov2_quick/benchmark_val_1024/score_analysis.json`
- `work_dirs/iac_navsim_future_dinov2_quick/benchmark_val_4096/wam_iac_summary.json`
- `work_dirs/iac_navsim_future_dinov2_quick/benchmark_val_4096/score_analysis.json`

## Current blocker

At about 2026-06-29 21:30 Asia/Shanghai, local network access to
`10.120.17.131` dropped:

- ping: 100% timeout
- SSH port 22: timeout
- local VPN-related adapters: disconnected

This is a network/routing blocker, not a model/training failure. Resume by
reconnecting to the server and reading the benchmark summaries above.
