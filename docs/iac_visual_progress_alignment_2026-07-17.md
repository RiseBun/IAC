# IAC Visual-Only Progress Alignment

Date: 2026-07-17

## Core Problem

The remaining failure mode is not ordinary classification error. Many misses are
speed, lateral, or heading near-neighbors whose rendered future is visually close
to the GT path. A single future image can support a set of plausible trajectories.

The benchmark therefore should not force a unique GT when the image evidence is
ambiguous. But it still must reject candidates whose motion progress is not
supported by the future image, especially time-shifted futures, trajectory swaps,
and visually clear negatives.

## First-Principles Requirement

The consistency score should answer:

> Does the future image show progress compatible with this candidate path?

It should not answer:

> Does the candidate trajectory geometry look like a typical GT trajectory?

Therefore the auxiliary signal must be image-driven. If the progress head reads
from a fused feature that already contains the candidate trajectory, it can learn
a trajectory geometry shortcut. The current implementation avoids this by
predicting visual progress only from:

- historical image feature
- future image feature
- future-minus-history visual delta

The predicted visual progress is then compared against candidate trajectory
progress with a group-wise rank loss.

## Implemented Changes

- Added a visual-only `progress_alignment_head` to `DINOv2ConsistencyCritic`.
- Exposed `progress_alignment_value` in model outputs when enabled.
- Added `_trajectory_progress_value` and `_progress_alignment_rank_loss`.
- Added training loss term `lambda_progress_alignment * progress_alignment_loss`.
- Added config `configs/train_navsim_future_dinov2_progress_alignment.py`.
- Added server runner `scripts/run_dinov2_progress_alignment.sh`.
- Added train/val logging for `prog_align` and `val_prog_align`.

## Loss Design

For each group:

- Compute image progress from visual features only.
- Compute candidate trajectory progress from the raw candidate trajectory.
- Compute alignment error: `abs(image_progress - trajectory_progress)`.
- Encourage GT to have lower alignment error than negatives.

Hard negatives:

- `image_swap`
- `time_shift_future`
- `traj_swap`
- `reverse_traj`

Near-neighbor negatives:

- `perturb_speed`
- `perturb_lateral`
- `perturb_heading`

Hard negatives get the stronger margin. Near-neighbors get a small margin and low
weight, because many are genuinely visually ambiguous.

## What Would Count As Progress

This is useful only if it improves the true scientific errors, not merely hard
top1.

Primary checks:

- `clear_negative_supported_error` should decrease.
- `unsupported_gt_error` should not increase.
- `exact_path_win_fraction` should stay at or above the current level.
- `visual_support_set_accuracy` should not drop.
- Hard top1 may improve, but it is secondary.

Current reference on holdout low-IoU g200:

- hard top1: `0.760`
- visual support set accuracy: `0.915`
- exact-path win fraction: `0.805`
- true model error fraction: `0.085`
- error split:
  - visually indistinguishable near miss: `31 / 200`
  - unsupported GT error: `8 / 200`
  - clear negative supported error: `4 / 200`
  - clear negative rejected but ranked: `5 / 200`

## Next Experiment

Run a conservative one-epoch fine-tune from the current best ambiguity-aware
DINOv2 checkpoint:

- enable `use_progress_alignment_head`
- set `lambda_progress_alignment = 0.15`
- use `final_displacement / 40.0` as the first progress target
- use hard margin `0.05`
- use near margin `0.005`
- use near weight `0.05`

Then evaluate regular g200, low-IoU g200, and holdout low-IoU g200 with the v3.2
support-set protocol.

The experiment succeeds only if it reduces true errors while preserving
path-grounded evidence. If it only raises hard top1 by using trajectory geometry,
it fails the scientific goal.

## Post-Training Observation

The first 20-step smoke fine-tune was not enough by itself:

- regular g200 hard top1: `0.430`
- regular g200 exact-path delta: `-0.00056`
- low-IoU g200 hard top1: `0.360`
- low-IoU g200 exact-path delta: `-0.01570`
- holdout low-IoU g200 hard top1: `0.375`
- holdout low-IoU g200 exact-path delta: `-0.00419`

This means the auxiliary loss alone is not the end state.

The useful signal came from the new scorer fusion:

- holdout 20-group baseline (`beta=0.0`):
  - top1 `0.40`
  - MRR `0.6375`
  - exact-path win fraction `0.45`
  - likely model error fraction `0.25`
- holdout 20-group fused (`beta=0.5`):
  - top1 `0.45`
  - MRR `0.6708`
  - exact-path win fraction `0.55`
  - likely model error fraction `0.0909`

With path-causal metrics on the same 20-group slice:

- baseline (`beta=0.0`):
  - top1 `0.40`
  - MRR `0.6375`
  - exact-path win fraction `0.45`
  - exact-path delta `-0.00128`
  - path-minus-sky delta `0.07008`
- fused (`beta=0.5`):
  - top1 `0.45`
  - MRR `0.6708`
  - exact-path win fraction `0.55`
  - exact-path delta `-0.00073`
  - path-minus-sky delta `0.06790`

So the next move is not more blind training. It is:

1. keep the progress head as an auxiliary visual constraint,
2. use `progress_alignment_value` in scorer fusion,
3. sweep `beta` on the disjoint holdout slice,
4. then only lock in the setting that improves both ranking and exact-path
   evidence.

## Holdout Result

On the full disjoint holdout low-IoU g200, the same pattern held:

- `beta=0.0`
  - top1 `0.375`
  - MRR `0.6378`
  - exact-path win fraction `0.44`
  - exact-path delta `-0.00419`
  - ambiguity-adjusted top1 `0.645`
- `beta=0.5`
  - top1 `0.41`
  - MRR `0.6566`
  - exact-path win fraction `0.45`
  - exact-path delta `-0.00422`
  - ambiguity-adjusted top1 `0.65`

Interpretation:

- progress fusion is a real ranking improvement;
- it is not yet the missing causal fix for exact-path evidence;
- exact-path still needs a stronger image-conditioned path mechanism, not just a
  scalar progress penalty.

## Local Verification

Completed locally:

- `py_compile` passed for training, benchmark, audit, and progress config.
- `load_config` successfully reads the new progress alignment config.
- Mock rank-loss test produces group pairs and nonzero loss.

Not completed locally:

- Bash syntax check could not run because the local Windows Bash service returned
  `E_ACCESSDENIED`. This is an environment permission issue.
- Server smoke/fine-tune could not start because SSH to `10.120.17.131:22`
  timed out. This matches the earlier network/routing blocker recorded on
  2026-06-29.

First command after server access recovers:

```bash
cd /mnt/slurmfs-4090node1/homes/zchen897/IAC
CUDA_VISIBLE_DEVICES=1,2,3 \
MAX_TRAIN_STEPS=200 \
MAX_VAL_STEPS=100 \
scripts/run_dinov2_progress_alignment.sh
```

## Path Evidence + Progress Head Update

After server access recovered, I trained a very small visual-only
`progress_alignment_head` on top of the existing path-evidence checkpoint
(`work_dirs/iac_navsim_future_dinov2_path_evidence_head_3gpu_400_v2/checkpoints/latest.pth`).
The intent was narrow: keep the path certificate intact, then see whether a
light progress prior can improve ranking without reintroducing a trajectory
geometry shortcut.

Training stayed constrained to the new head only:

- trainable params: `769 / 24,119,830`
- frozen backbone and existing path-evidence branch
- progress alignment mode: `final_displacement`
- progress loss weight: `0.15`

Observed checkpoint behavior:

- regular g200:
  - `top1 = 0.12`
  - `mrr = 0.4134`
  - `exact_path_win_fraction = 0.64`
  - `exact_path_delta = +0.01547`
  - `path_minus_sky_delta = +0.08149`
  - `ambiguity_adjusted_top1 = 0.33`
  - `likely_model_error_fraction = 0.25`
- low-IoU g200:
  - `top1 = 0.095`
  - `mrr = 0.3876`
  - `exact_path_win_fraction = 0.855`
  - `exact_path_delta = +0.03149`
  - `path_minus_sky_delta = +0.07407`
  - `ambiguity_adjusted_top1 = 0.37`
  - `likely_model_error_fraction = 0.105`
- holdout low-IoU g200:
  - `top1 = 0.07`
  - `mrr = 0.3884`
  - `exact_path_win_fraction = 0.695`
  - `exact_path_delta = +0.01790`
  - `path_minus_sky_delta = +0.07503`
  - `ambiguity_adjusted_top1 = 0.375`
  - `likely_model_error_fraction = 0.188`

Interpretation:

1. The path certificate still stays positive. The model remains path-grounded.
2. The progress head alone does not solve the ranking problem. It is too weak to
   turn the scorer into a strong benchmark ranker.
3. The holdout exact-path signal is positive, but weaker than the earlier
   path-evidence-fusion setting. So this is a calibration aid, not the core fix.

The next useful direction is not more scalar fusion tuning. The next step is to
make the evidence head itself more expressive about path shape, then keep
progress as an auxiliary prior.

## vNext Protocol

The vNext implementation now formalizes the protocol in code:

- benchmark summary exposes `raw_miss_fraction` and the four formal categories
  `hit / ambiguous_accept / evidence_supported_miss / likely_model_error`
- new config:
  `configs/train_navsim_future_dinov2_path_evidence_vnext.py`
- new runner:
  `scripts/run_path_evidence_vnext.sh`

The intended evaluation split is now explicit:

1. `consistency_logit` for the main decision surface
2. `path_evidence_logit` for the scientific certificate
3. `progress_alignment_value` only as an auxiliary prior, never as the main
   causal explanation

## vNext Fusion Sweep

The post-training fusion sweep on the vNext checkpoint shows the expected
trade-off more clearly:

- `alpha=0.0` keeps the main score strongest but leaves exact-path delta near
  zero on holdout.
- `alpha=0.2` is the best balanced setting so far on holdout:
  - `top1 = 0.42`
  - `MRR = 0.672`
  - `exact_path_delta = +0.00992`
  - `path_minus_sky_delta = +0.10248`
  - `ambiguity_adjusted_top1 = 0.775`
- `alpha=0.3` pushes the certificate harder:
  - `top1 = 0.405`
  - `MRR = 0.66325`
  - `exact_path_delta = +0.01359`
  - `ambiguity_adjusted_top1 = 0.775`
- `alpha=0.5` maximizes exact-path evidence more aggressively, but ranking
  starts to give up too much:
  - `top1 = 0.4`
  - `MRR = 0.65867`
  - `exact_path_delta = +0.01820`
  - `ambiguity_adjusted_top1 = 0.79`

Recommended current setting:

- use `alpha = 0.2` as the default fused scorer on holdout-style evaluation
- use `alpha = 0.3` as the stronger certificate-oriented diagnostic
- keep `path_evidence_logit` separate for scientific reporting

## Next Step Plan

The benchmark is usable now, but not yet frozen as a final public standard. The
next step is to lock the protocol and prove that it still works beyond the
current model family.

1. Freeze the protocol
   - Lock the four formal categories:
     `hit / ambiguous_accept / evidence_supported_miss / likely_model_error`
   - Lock the current fused reporting convention:
     `consistency_logit` as the main decision surface, `path_evidence_logit` as
     the scientific certificate, and `progress_alignment_value` only as an
     auxiliary prior.
   - Stop changing the category rules or score interpretation inside the same
     benchmark version.

2. Run generalization validation
   - Evaluate at least one different backbone family or checkpoint family.
   - Keep the benchmark protocol unchanged and only swap the model family.
   - Check whether the same exact-path and ambiguity trends still hold.

3. Separate alpha selection from holdout reporting
   - Select `alpha` only on the tuning slice.
   - Never use the holdout slice to search `alpha`.
   - Report the holdout result only once the fusion weight is frozen.

4. Produce the formal report
   - Fix the final output table to include:
     `top1 / MRR / exact_path_delta / path_minus_sky_delta /
     ambiguity_adjusted_top1 / likely_model_error_fraction / 95% CI`
   - Report bootstrap CIs at the candidate-group level.
   - Keep hard top1 as a secondary comparison, not the headline claim.

## Formal Report Shape

The next report should read like a benchmark paper appendix, not like a running
lab note.

Minimum sections:

- protocol
- model family
- frozen alpha selection
- holdout evaluation
- ambiguity breakdown
- causal grounding checks
- bootstrap confidence intervals

Minimum tables:

- per-split summary table
- fused-vs-global comparison table
- error-category breakdown table
- CI table for the holdout split

This is the point where the project becomes a benchmark candidate rather than
just a promising experiment log.
