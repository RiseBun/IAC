# IAC-PathBench Protocol 2026-07-06

## Purpose

IAC-PathBench tests whether an image-action consistency score is grounded in
candidate-specific future-image path evidence, rather than relying mainly on
trajectory geometry or generic road-region shortcuts.

## Splits

Use three fixed validation slices:

1. `regular_g200`
   - Source: `indices_navsim_future/consistency_val.jsonl`
   - Selection: first 200 complete candidate groups
   - Purpose: preserve ordinary ranking and binary score quality.

2. `low_iou_g200_tune`
   - Source:
     `indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl`
   - Selection: lowest positive-vs-negative projected path IoU groups.
   - Purpose: select a small fusion alpha.

3. `low_iou_g200_holdout`
   - Source:
     `indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl`
   - Selection: skip first 200 eligible low-IoU groups, then take next 200.
   - Purpose: verify that alpha and exact-path evidence generalize beyond the
     tuning slice.

## Scores

Report three score variants:

1. `global`
   - `consistency_logit`

2. `path_evidence`
   - `path_evidence_logit`

3. `fused`
   - `global + alpha * path_evidence`
   - Current holdout default alpha: `0.2`
   - Stronger certificate diagnostic alpha: `0.3`
   - Fusion is done in logit space.

## Metrics

Decision quality:

- `top1_hit_rate`
- `MRR`
- `best_balanced_accuracy`

Future-image grounding:

- `mean_path_minus_sky_delta`
- `path_delta_gt_sky_fraction`

Exact candidate-path grounding:

- `positive_rows.mean_candidate_minus_wrong_exclusive_delta`
- `positive_rows.candidate_exclusive_delta_gt_wrong_fraction`
- `mean_path_mask_iou`

Uncertainty:

- Bootstrap confidence intervals resample candidate groups, not individual rows.
- Tool: `tools/bootstrap_iac_pathbench.py`
- Default: 1000 bootstrap samples.

## Current Passing Criteria

A fused score passes IAC-PathBench if:

- Regular g200 top1 and MRR do not decrease relative to `global`.
- Holdout low-IoU positive exact-exclusive delta is positive.
- Holdout low-IoU positive exact-exclusive win fraction point estimate is at
  least `0.70`.
- Holdout low-IoU path-vs-sky remains positive.

The current `alpha=0.5` fused score passes:

- Regular g200:
  - Top1: `0.405 -> 0.415`
  - MRR: `0.665 -> 0.675`
  - Balanced: `0.714 -> 0.718`

- Holdout low-IoU g200:
  - Top1: `0.420 -> 0.425`
  - MRR: `0.661 -> 0.668`
  - Balanced: `0.732 -> 0.733`
  - Positive exact-exclusive delta: `+0.00321 -> +0.00881`
  - Positive exact-exclusive win fraction: `0.585 -> 0.750`

## Bootstrap Results

Bootstrap configuration:

- Resampling unit: candidate group
- Samples: 1000
- Seed: 897

Holdout low-IoU g200:

| Score | Metric | Point | 95% CI |
| --- | --- | ---: | ---: |
| global | Top1 | 0.420 | [0.350, 0.485] |
| global | MRR | 0.661 | [0.619, 0.701] |
| global | Positive exact-exclusive delta | +0.00321 | [+0.00015, +0.00632] |
| global | Positive exact-exclusive win frac | 0.585 | [0.515, 0.650] |
| fused alpha=0.5 | Top1 | 0.425 | [0.360, 0.490] |
| fused alpha=0.5 | MRR | 0.668 | [0.629, 0.705] |
| fused alpha=0.5 | Positive exact-exclusive delta | +0.00881 | [+0.00663, +0.01088] |
| fused alpha=0.5 | Positive exact-exclusive win frac | 0.750 | [0.685, 0.810] |

Regular g200:

| Score | Metric | Point | 95% CI |
| --- | --- | ---: | ---: |
| global | Top1 | 0.405 | [0.330, 0.475] |
| global | MRR | 0.665 | [0.625, 0.705] |
| global | Positive exact-exclusive delta | +0.00281 | [+0.00139, +0.00440] |
| global | Positive exact-exclusive win frac | 0.590 | [0.520, 0.660] |
| fused alpha=0.5 | Top1 | 0.415 | [0.345, 0.485] |
| fused alpha=0.5 | MRR | 0.675 | [0.634, 0.715] |
| fused alpha=0.5 | Positive exact-exclusive delta | +0.00391 | [+0.00258, +0.00530] |
| fused alpha=0.5 | Positive exact-exclusive win frac | 0.645 | [0.580, 0.710] |

Statistical interpretation:

- The exact-path delta is robustly positive on both regular and holdout
  low-IoU splits.
- The fused score raises the exact-path delta with a positive lower CI bound.
- Ranking gains are directionally positive but have overlapping confidence
  intervals, so ranking improvement should be described as a secondary benefit,
  not the main scientific claim.
- Holdout fused exact-path win fraction has point estimate `0.750`, with 95% CI
  lower bound `0.685`; this supports the claim but is not yet a strict
  lower-bound-above-0.70 guarantee.

## Remaining Weaknesses

This is a strong internal benchmark, not yet final public benchmark.

Still needed:

- A true frozen test split not used for architecture, alpha, or metric design.
- More WAM/generator families.
- Path projection validation with better camera geometry or segmentation support.
- A single command that regenerates the full benchmark report.
- A formally frozen protocol with the four categories:
  `hit / ambiguous_accept / evidence_supported_miss / likely_model_error`.
- A blind alpha procedure where tuning and holdout reporting are fully
  separated.
- A cross-backbone validation pass so the protocol is not only proven on one
  model family.

## Next Step Plan

1. Freeze the protocol categories and score interpretation.
2. Validate one additional backbone/checkpoint family under the same protocol.
3. Keep alpha search on the tuning slice only.
4. Publish a fixed report with top1, MRR, exact-path delta, path-minus-sky
   delta, ambiguity-adjusted top1, likely-model-error fraction, and bootstrap
   95% CI.

## 2026-07-18 Maturity Update

The first maturity pass is now recorded in
`docs/iac_pathbench_maturity_2026-07-18.md`.

Completed:

- data coverage audit for regular val, low-IoU g200, and holdout low-IoU g200
- frozen v2 protocol validation tool
- CNN 3k cross-backbone run under the same protocol
- DINOv2 vNext fused `alpha=0.2` formal summaries and bootstrap CI

Current default report line is DINOv2 vNext fused `alpha=0.2`. Hard top1 remains
secondary; the headline evidence is exact-path delta, path-minus-sky delta,
ambiguity-adjusted top1, and likely-model-error fraction.
