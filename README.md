# IAC

Image-Action Consistency (IAC) is a benchmark and scorer for evaluating whether a driving action or trajectory is consistent with future visual evidence. The primary target is World Action Model (WAM) evaluation: given a history clip, a candidate action/trajectory, and generated future frames, IAC reports whether the image evolution matches the action and whether the trajectory is kinematically reasonable.

IAC is an auxiliary metric. It does not generate trajectories, replace a planner, or replace closed-loop simulation. It provides a simulator-free and label-free signal that can be used alongside FID/FVD, PDMS, driving score, and task-specific metrics.

## What It Scores

Each sample contains:

- `history_images`: past camera frames.
- `future_images`: logged or WAM-generated future frames.
- `ego_state`: ego velocity, yaw, acceleration, and yaw-rate features.
- `candidate_traj`: candidate future trajectory.

The critic outputs:

- `iac_consistency`: whether the trajectory matches the future visual evolution.
- `iac_validity`: whether the trajectory is context-free kinematically feasible.

The key benchmark question is:

```text
Given action A, did the WAM generate future images that actually reflect A?
```

## Repository Layout

```text
configs/train_consistency_mini.py   IAC training config
tools/build_consistency_index.py    nuPlan DB + camera images -> IAC JSONL index
train.py                            IAC critic training
eval_critic.py                      scorer evaluation on IAC validation data
stress_test_iac.py                  shortcut/stress probes
benchmark_wam.py                    WAM output benchmark entry point
data_paths.py                       environment-aware data path defaults
```

Generated data, checkpoints, logs, local indices, camera archives, and extracted dataset files are intentionally ignored by git.

## Data Paths

By default this project expects the current AutoDL-style layout:

```text
/root/autodl-tmp/data/cache/mini
/root/autodl-tmp/nuplan-v1.1_mini_camera_0
/root/autodl-tmp/nuplan-v1.1_mini_camera_1
```

You can override paths with:

```bash
export NUPLAN_DATA_ROOT=/path/to/data-root
export NUPLAN_DB_ROOT=/path/to/data/cache/mini
export NUPLAN_INDEX_ROOT=/path/to/IAC/indices
export NUPLAN_CAMERA_ROOTS="/path/to/camera_0:/path/to/camera_1"
```

## Build IAC Training Data

```bash
python tools/build_consistency_index.py \
  --db-root "$NUPLAN_DB_ROOT" \
  --image-roots /path/to/nuplan-v1.1_mini_camera_0 /path/to/nuplan-v1.1_mini_camera_1 \
  --output-dir indices
```

Smoke test:

```bash
python tools/build_consistency_index.py \
  --max-scenes 2 \
  --max-samples-per-scene 20 \
  --output-dir indices_smoke
```

The index contains positive samples and multiple self-supervised negatives:

- `gt_pos`
- `traj_swap`
- `image_swap`
- `time_shift_future`
- `perturb_lateral`
- `perturb_heading`
- `perturb_speed`

## Train The IAC Critic

Single-node two-GPU example:

```bash
mkdir -p work_dirs/iac_5epoch_2gpu

PYTHONUNBUFFERED=1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29606 \
  train.py \
  --config configs/train_consistency_mini.py \
  --work-dir work_dirs/iac_5epoch_2gpu \
  --epochs 5 \
  --batch-size 8 \
  --num-workers 4 \
  --preflight-samples 256 \
  2>&1 | tee work_dirs/iac_5epoch_2gpu/train.log
```

Training writes normal checkpoints only after a complete epoch and validation:

```text
work_dirs/<run>/checkpoints/latest.pth
work_dirs/<run>/checkpoints/best.pth
```

Interrupted or failed runs are saved separately as `interrupted_epoch_*.pth` or `error_epoch_*.pth` so they are not confused with valid scorer checkpoints.

## Evaluate The Scorer

```bash
python eval_critic.py \
  --checkpoint work_dirs/iac_5epoch_2gpu/checkpoints/best.pth \
  --split val \
  --batch-size 32 \
  --eval-ranking
```

Important metrics:

- positive recall on `gt_pos`
- AUROC / PR-AUC
- ECE calibration
- per-negative-type recall
- graded perturbation curves
- ranking metrics

Accuracy alone is not sufficient because the dataset is intentionally negative-heavy.

## Stress Test The Scorer

```bash
python stress_test_iac.py \
  --checkpoint work_dirs/iac_5epoch_2gpu/checkpoints/best.pth \
  --split val \
  --max-samples 256
```

Stress tests include future-frame reversal, trajectory mirroring, future-image corruption, noise, and trajectory shuffling. A reliable scorer should lower consistency scores under these interventions.

## Benchmark WAM Outputs

`benchmark_wam.py` is the benchmark entry point for WAM-generated future frames.

Input JSONL format:

```json
{
  "wam_name": "my_wam",
  "group_id": "scene_or_anchor_id",
  "history_images": ["history_0.jpg", "history_1.jpg", "history_2.jpg", "history_3.jpg"],
  "future_images": ["wam_future_0.jpg", "wam_future_1.jpg", "wam_future_2.jpg", "wam_future_3.jpg"],
  "ego_state": [0.0, 0.0, 0.0, 0.0, 0.0],
  "candidate_traj": [[0.0, 0.0, 0.0]],
  "action_type": "gt_or_perturbed",
  "consistency_label": 1,
  "validity_label": 1
}
```

Run:

```bash
python benchmark_wam.py \
  --input path/to/wam_outputs.jsonl \
  --checkpoint work_dirs/iac_5epoch_2gpu/checkpoints/best.pth \
  --output-dir work_dirs/wam_benchmark/my_wam
```

Outputs:

```text
wam_iac_scores.jsonl
wam_iac_summary.json
```

The summary reports overall IAC scores, per-WAM scores, per-action-type scores, ranking metrics, and graded perturbation curves.

## Current Scope

This repository currently targets a nuPlan-mini IAC-WAM benchmark prototype. Multi-dataset evaluation, DriveCritic/ACT-Bench/Vista adapters, human study protocols, and PDMS/CARLA correlation analysis are future extensions.

