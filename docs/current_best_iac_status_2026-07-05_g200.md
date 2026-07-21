# IAC Current Best Status - 2026-07-05 g200

## One-Sentence State

The current main model gives stable evidence that the consistency score depends on future-image driving-path evidence rather than sky/background occlusion. The exact trajectory-specific proof is still incomplete: positive rows move in the right direction, but the full-set trajectory-specific contrast is diluted by highly overlapping candidate paths.

## Current Main Model

- Model: `iac_navsim_future_dinov2_trajspecific_fullgroup_strong_3gpu_400`
- Checkpoint: `work_dirs/iac_navsim_future_dinov2_trajspecific_fullgroup_strong_3gpu_400/checkpoints/latest.pth`
- Config: `configs/train_navsim_future_dinov2_pathgrounded_strong.py`
- Eval dir: `work_dirs/iac_navsim_future_dinov2_trajspecific_fullgroup_strong_3gpu_400/traj_specific_causal_g200_fixedpos_maxwrong_exclusive_cfgroot_skipmissing`
- GPU constraint respected: training uses at most 3 GPUs; this g200 diagnostic used 1 GPU.

## Evaluation Protocol

- 200 candidate groups, 1400 rows, 7 candidates per group.
- Path projection uses `candidate_traj` as ego-frame future positions.
- Projection mode is fixed-meter scale, not per-trajectory normalization.
- Path mask and sky mask are area-matched.
- Trajectory-specific metric compares candidate path masking with same-group wrong-path masking.
- Exclusive metric uses candidate-only vs wrong-only pixels with equal area.
- Image-root audit: 85708/85708 rows available, 0 dropped.

## Decision Performance

Same-set threshold sweep:

- Best threshold: `0.1053`
- Balanced accuracy: `0.7142`
- Recall: `0.7950`
- TNR: `0.6333`
- Precision: `0.2654`

Group-disjoint calibration:

- Selected threshold: `0.1058`
- Held-out groups: 111 groups, 777 rows
- Held-out balanced accuracy: `0.7027`
- Recall: `0.7568`
- TNR: `0.6486`
- Precision: `0.2642`

Ranking:

- Top-1 hit rate: `0.405`
- MRR: `0.6652`
- NDCG@3: `0.7317`
- NDCG@5: `0.7509`

Interpretation: default threshold `0.5` is invalid for this score scale. Use a calibrated threshold near `0.106`.

## Causal Evidence

Path-vs-sky causal check:

- Mean path delta: `0.0312`
- Mean sky delta: `0.0144`
- Path minus sky: `0.0168`
- Path delta > sky delta fraction: `0.7886`
- Path mask area: `0.0776`
- Sky mask area: `0.0776`
- Path grounded: `true`

This is the strongest result. Because the sky control is area-matched, the effect is not explained by generic occlusion size. The score is more sensitive to the projected driving path than to an equally sized top-image control region.

Trajectory-specific check, all rows:

- Candidate path delta: `0.03119`
- Wrong path delta: `0.03112`
- Candidate minus wrong: `0.00008`
- Exclusive candidate minus wrong: `0.000004`
- Mean path IoU: `0.7919`
- Exclusive mask area: `0.0051`

All-row result is not sufficient to claim fully trajectory-specific grounding. The structural reason is that most negative candidates share nearly the same road corridor, so candidate and wrong masks overlap heavily.

Trajectory-specific check, positive rows only:

- Candidate path delta: `0.04938`
- Wrong path delta: `0.04687`
- Candidate minus wrong: `0.00251`
- Exclusive candidate delta: `0.00886`
- Exclusive wrong delta: `0.00614`
- Exclusive candidate minus wrong: `0.00272`
- Candidate exclusive > wrong exclusive fraction: `0.57`
- Mean path IoU: `0.4567`

Positive rows show the desired direction: masking the true candidate path hurts more than masking a wrong same-group path. The margin is still small, so the honest conclusion is: trajectory-specific evidence exists, but it is not yet strong enough as a final scientific proof.

## What This Proves

1. The score is not mainly driven by sky/background artifacts.
2. The score is materially driven by future-image path evidence.
3. Full candidate-group training is necessary; small batches weakened the causal signal.
4. Calibrated thresholding is required; raw `0.5` threshold collapses recall.

## What It Does Not Fully Prove Yet

It does not yet fully prove that the score is dominated by the exact candidate trajectory rather than a generic road corridor or geometry-correlated shortcut. The positive-row exclusive metric is promising, but the all-row exclusive metric remains near zero because wrong and candidate masks overlap too much.

## Follow-Up Diagnostic

I added a `--wrong-path-selection mask_iou` diagnostic to test whether the weak trajectory-specific result was caused by poor same-group wrong-path selection. On g100, mask-IoU selection barely changed the contrast:

- Distance-selected positive IoU: `0.4591`; positive exclusive diff: `0.00237`
- Mask-IoU-selected positive IoU: `0.4484`; positive exclusive diff: `0.00209`

This means the current candidate groups themselves are too geometrically overlapping in image space. The next step should be a low-IoU validation subset or synthetic counterfactual trajectories, not just a stronger same-group selector.

## Low-IoU Subset Result

I built a dedicated low-IoU validation subset from all 12244 validation candidate groups:

- Selected groups: `200`
- Selected rows: `1400`
- Mean selected positive-vs-wrong path IoU: `0.1806`
- Median selected IoU: `0.1818`
- Mean exclusive fraction: `0.1685`

On this subset, the current best `fullgroup_strong` model becomes an important negative control:

- Path-vs-sky becomes stronger: path minus sky `0.02735`, fraction `0.8714`
- Positive path IoU drops to `0.1660`
- Positive exclusive candidate-minus-wrong becomes negative: `-0.00062`

Interpretation: the model is genuinely future-path grounded, but when the correct and wrong paths are clearly separated, it does not yet prefer the exact candidate path. That is the central remaining scientific gap.

## Mask-IoU Exclusive Training Result

I trained `iac_navsim_future_dinov2_trajspecific_maskiou_exclusive_3gpu_400` with:

- `trajectory_specific_wrong_selection = "mask_iou"`
- `trajectory_specific_grounding_exclusive = true`
- `lambda_trajectory_specific_grounding = 0.22`
- 3 GPUs, batch size 7, 400 steps

This did not solve the gap:

- Regular g100 positive exclusive diff fell from `0.00237` to `0.00162`
- Low-IoU g200 positive exclusive diff fell from `-0.00062` to `-0.00235`
- Low-IoU path-vs-sky stayed strong: path minus sky `0.02388`, fraction `0.8729`

Interpretation: stronger sampling/loss is not enough. The architecture lacks an explicit path-conditioned visual evidence channel, so it can learn generic road/path evidence but not robustly bind that evidence to the exact candidate trajectory.

## Next Scientific Step

The next breakthrough should be structural, not another scalar loss tweak:

1. Add a path-conditioned visual evidence branch that extracts future-image features inside the candidate projected path mask.
2. Fuse that path evidence with `z_traj_cons`, not only with global `z_fut`.
3. Keep `fullgroup_strong` as the current main model until a new model improves both calibrated decision performance and low-IoU exact-path metrics.
4. Continue reporting the system as two claims: calibrated consistency decision plus causal path-grounding certificate.
