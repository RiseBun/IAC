# Dual-Head Path Evidence Plan 2026-07-06

## Core Question

The unresolved scientific issue is not whether the current IAC score is useful.
It is whether the consistency score is driven by future-image evidence on the
candidate path, rather than by trajectory geometry or generic road-region
shortcuts.

The previous low-IoU g200 result showed the gap clearly:

- `fullgroup_strong` keeps good decision performance and is path-vs-sky grounded.
- On low-IoU counterfactual groups, its exclusive exact-path signal is weak or
  negative.
- Non-residual path-conditioned fusion produced positive exact-path evidence,
  but damaged decision performance.
- Residual path-conditioned fusion preserved decision performance, but the path
  signal was too weak.

## New Direction

Use a dual-head critic:

- `consistency_logit`: the inherited global decision score.
- `path_evidence_logit`: an independent candidate-path evidence score.

This separates two jobs that were previously fighting each other:

1. Keep ranking/decision performance stable.
2. Prove that a candidate-specific future-image path signal exists.

The path head is a scientific certificate first. It should be evaluated
separately on low-IoU counterfactual groups before being mixed into the final
decision score.

## Implemented Changes

- `train_dinov2_v5_minimal.py`
  - Added optional `dinov2.use_path_evidence_head`.
  - Outputs `path_evidence_logit` when path-conditioned evidence is enabled.
  - Keeps `path_evidence_logit` out of `consistency_logit` unless
    `dinov2.mix_path_evidence_into_consistency=True`.
  - Added `trainable_parameter_prefixes` support so a continuation run can
    freeze the global critic and train only the path evidence branch.

- `train.py`
  - Added `path_grounding_score_key` and
    `trajectory_specific_grounding_score_key`.
  - Added `lambda_path_evidence_consistency`.
  - Path/trajectory-specific grounding can now train `path_evidence_logit`
    directly.
  - If a requested non-default score key is missing, training raises an error
    instead of silently falling back to `consistency_logit`.

- `benchmark_wam.py`
  - Added `--consistency-score-key`.
  - Existing metrics still default to `consistency_logit`.
  - Passing `--consistency-score-key path_evidence_logit` evaluates the path
    evidence head independently, including path-vs-sky and low-IoU
    trajectory-specific causal metrics.

- `configs/train_navsim_future_dinov2_path_evidence_head.py`
  - New config for the dual-head continuation.
  - Inherits from the current strong global critic config.
  - Enables path-conditioned evidence and independent path evidence head.
  - Sets `path_residual_mix=0.0` and does not mix path evidence into the main
    decision score.
  - Freezes everything except:
    - `path_conditioned_traj_proj`
    - `path_conditioned_fusion`
    - `path_evidence_head`

## Server Status

Checked server `/mnt/slurmfs-4090node1/homes/zchen897/IAC`.

- No current user training/eval processes were running.
- GPUs were idle at check time.
- Code was synced to the server.
- Remote `py_compile` and config import passed in the `drivingworld` env.

## Next Experiment

Use only three GPUs, excluding GPU 0:

```bash
cd /mnt/slurmfs-4090node1/homes/zchen897/IAC
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate drivingworld

CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
  train_dinov2_v5_minimal.py \
  --config configs/train_navsim_future_dinov2_path_evidence_head.py \
  --resume-from work_dirs/iac_navsim_future_dinov2_trajspecific_fullgroup_strong_3gpu_400/checkpoints/latest.pth \
  --work-dir work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400 \
  --max-train-steps 400
```

Then evaluate two score keys:

1. Main decision score:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_wam.py \
  --input indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl \
  --checkpoint work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400/checkpoints/latest.pth \
  --config configs/train_navsim_future_dinov2_path_evidence_head.py \
  --output-dir work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400/eval_low_iou_g200_main \
  --batch-size 16 \
  --max-groups 200 \
  --path-causal-metrics \
  --trajectory-specific-causal-metrics \
  --wrong-path-selection mask_iou \
  --path-trajectory-mode positions \
  --path-projection-mode fixed
```

2. Independent path evidence score:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_wam.py \
  --input indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl \
  --checkpoint work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400/checkpoints/latest.pth \
  --config configs/train_navsim_future_dinov2_path_evidence_head.py \
  --output-dir work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400/eval_low_iou_g200_path_head \
  --batch-size 16 \
  --max-groups 200 \
  --consistency-score-key path_evidence_logit \
  --path-causal-metrics \
  --trajectory-specific-causal-metrics \
  --wrong-path-selection mask_iou \
  --path-trajectory-mode positions \
  --path-projection-mode fixed
```

## Success Criterion

This run is successful only if the path head improves the low-IoU exact-path
certificate:

- `positive_rows.mean_candidate_minus_wrong_exclusive_delta > 0`
- `positive_rows.candidate_exclusive_delta_gt_wrong_fraction > 0.60`

Main decision performance should be checked separately on `consistency_logit`.
If the path head passes but the main score is unchanged, that is still a
scientific success: it proves candidate-specific future-image evidence exists.

## Result: 3GPU 400-Step V2

Run:

- `work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400_v2`
- Trained from `fullgroup_strong_3gpu_400/checkpoints/latest.pth`
- GPUs: `CUDA_VISIBLE_DEVICES=1,2,3`
- Training: 400 steps, validation capped to 80 steps
- Checkpoint: `checkpoints/latest.pth`

Server was idle after completion.

Low-IoU g200 evaluation:

| Score key | Top1 | MRR | Path-sky | Positive exact-exclusive delta | Positive exact-exclusive win frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| `consistency_logit` | 0.335 | 0.612 | 0.02735 | -0.00062 | 0.455 |
| `path_evidence_logit` | 0.110 | 0.401 | 0.08524 | +0.04268 | 0.915 |

Interpretation:

- The main decision head stayed effectively unchanged on low-IoU g200. This is
  expected because the continuation froze the global critic and did not mix the
  path evidence head into `consistency_logit`.
- The independent path head strongly passed the exact-path certificate on
  positive low-IoU rows.
- This directly addresses the previous failure mode: the learned evidence is no
  longer just generic road/path sensitivity. It is candidate-specific path
  evidence under low projected-path IoU.

Next step:

Do not replace the main score with `path_evidence_logit`; its ranking quality is
poor by design. Instead, calibrate a small score combination:

```text
final_score = consistency_logit + alpha * path_evidence_logit
```

Search only a small alpha grid on validation, for example:

```text
alpha in {0.02, 0.05, 0.10, 0.15, 0.20}
```

Decision rule:

- Keep low-IoU top1/MRR close to the main score.
- Preserve positive exact-exclusive delta above zero.
- Prefer the smallest alpha that moves the exact-path certificate while not
  damaging ranking.

## Fused Score Sweep

Added reproducible tool:

- `tools/sweep_iac_fused_scores.py`

It fuses primary and auxiliary benchmark JSONL files in logit space:

```text
fused_logit = logit(consistency_score) + alpha * logit(path_evidence_score)
```

It also recomputes all masked-score delta fields after fusion, so
path-vs-sky and exact-path metrics reflect the fused score rather than stale
main-score deltas.

Outputs:

- `work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400_v2/fused_alpha_low_iou_g200.json`
- `work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400_v2/fused_alpha_regular_g200.json`

Low-IoU g200 fused sweep:

| Alpha | Top1 | MRR | Path-sky | Positive exact-exclusive delta | Positive exact-exclusive win frac |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.335 | 0.612 | 0.02735 | -0.00062 | 0.455 |
| 0.20 | 0.335 | 0.613 | 0.02667 | +0.00561 | 0.615 |
| 0.30 | 0.350 | 0.619 | 0.02516 | +0.00720 | 0.680 |
| 0.50 | 0.360 | 0.624 | 0.02113 | +0.00838 | 0.750 |
| 1.00 | 0.345 | 0.616 | 0.01124 | +0.00628 | 0.840 |

Regular g200 fused sweep:

| Alpha | Top1 | MRR | Balanced | Path-sky | Positive exact-exclusive delta | Positive exact-exclusive win frac |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.405 | 0.665 | 0.714 | 0.01684 | +0.00281 | 0.590 |
| 0.20 | 0.410 | 0.671 | 0.713 | 0.02146 | +0.00380 | 0.630 |
| 0.30 | 0.405 | 0.668 | 0.715 | 0.02173 | +0.00398 | 0.635 |
| 0.50 | 0.415 | 0.675 | 0.718 | 0.01998 | +0.00391 | 0.645 |
| 1.00 | 0.415 | 0.675 | 0.716 | 0.01189 | +0.00271 | 0.650 |

Recommendation:

- Use `alpha=0.5` as the current fused-score candidate.
- It improves low-IoU ranking and exact-path evidence at the same time:
  - Top1: `0.335 -> 0.360`
  - Positive exact-exclusive delta: `-0.00062 -> +0.00838`
  - Positive exact-exclusive win frac: `0.455 -> 0.750`
- It also improves regular g200:
  - Top1: `0.405 -> 0.415`
  - MRR: `0.665 -> 0.675`
  - Balanced accuracy: `0.714 -> 0.718`

This is the first configuration that simultaneously keeps decision quality and
adds a positive low-IoU exact-path certificate.

## Holdout Low-IoU Check

To avoid over-claiming from the same low-IoU top-200 groups used to choose
`alpha`, a disjoint holdout slice was built:

- Tool update: `tools/build_low_iou_subset.py --start-rank`
- Holdout input:
  `indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl`
- Selection: skip the first 200 eligible low-IoU groups, then take the next
  200 groups.
- Output rows: 1400
- This holdout is harder than regular g200, but less extreme than the top-200
  low-IoU tuning slice.

Holdout raw heads:

| Score key | Top1 | MRR | Path-sky | Positive exact-exclusive delta | Positive exact-exclusive win frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| `consistency_logit` | 0.420 | 0.661 | 0.02307 | +0.00321 | 0.585 |
| `path_evidence_logit` | 0.070 | 0.384 | 0.08736 | +0.03215 | 0.825 |

Holdout fused sweep:

| Alpha | Top1 | MRR | Balanced | Path-sky | Positive exact-exclusive delta | Positive exact-exclusive win frac |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.420 | 0.661 | 0.732 | 0.02307 | +0.00321 | 0.585 |
| 0.20 | 0.430 | 0.669 | 0.731 | 0.02422 | +0.00747 | 0.690 |
| 0.30 | 0.425 | 0.667 | 0.731 | 0.02336 | +0.00842 | 0.715 |
| 0.50 | 0.425 | 0.668 | 0.733 | 0.02022 | +0.00881 | 0.750 |
| 1.00 | 0.420 | 0.666 | 0.729 | 0.01119 | +0.00611 | 0.770 |

Interpretation:

- `alpha=0.5` was selected on the earlier low-IoU and regular g200 analysis,
  but it still improves the disjoint holdout slice.
- Holdout top1 improves from `0.420` to `0.425`.
- Holdout balanced accuracy improves from `0.732` to `0.733`.
- Holdout positive exact-exclusive win fraction improves from `0.585` to
  `0.750`.

This makes the benchmark claim stronger: the fused score is not merely tuned to
the first low-IoU top-200 groups.
