# Path-Grounded IAC Plan, 2026-07-01

## Core Concern

Plain image-action consistency can be fooled by visual shortcuts.

A critic may lower consistency because the sky, lighting, texture, or image
quality changed, not because the future image no longer supports the candidate
trajectory.

So the key scientific question is:

- does the consistency score depend more on the trajectory-relevant future path
  region than on irrelevant background?

## Current Implementation

`benchmark_wam.py` now supports:

```bash
--path-causal-metrics
```

For each sample, it evaluates three scores:

- original consistency score
- consistency after masking the projected future path corridor
- consistency after masking an equal-area sky/background control region

The projected path corridor is a lightweight image-space approximation from
`candidate_traj`. It is not yet a calibrated camera projection; it is a stable
diagnostic ROI for testing path sensitivity.

`tools/analyze_wam_scores.py` now adds:

```json
"iworld_style_diagnostics": {
  "path_grounding_causal_test": ...
}
```

## Decision Rule

The model is more path-grounded if:

```text
score(original) - score(path_masked)
>
score(original) - score(sky_masked)
```

for most samples, with matched mask area.

This does not prove full semantic path understanding, but it rejects the
weakest shortcut explanation: that consistency is driven mainly by unrelated
background pixels.

## Current Source-Aware 512 Result

Path-causal benchmark:

```text
work_dirs/iac_navsim_future_dinov2_sourceaware_smoke512/path_causal_512
```

Result:

```text
mean_path_delta          = 0.0557
mean_sky_delta           = 0.0344
mean_path_minus_sky      = 0.0213
path_delta_gt_sky_frac   = 0.7422
path_mask_fraction       = 0.1518
sky_mask_fraction        = 0.1514
is_path_grounded         = true
```

Interpretation:

- the current best model is more sensitive to future path-region evidence than
  to equal-area sky/background masking
- this is evidence against pure background shortcut
- geometry perturbation false positives are still not solved, so this is a
  diagnostic gain, not a final method claim

## Implemented Training Pressure

`train.py` now supports path-grounded consistency training:

```python
lambda_path_grounding
path_grounding_margin
path_grounding_sky_weight
path_grounding_path_width
path_grounding_sky_ratio
path_grounding_positive_only
```

For positive samples, the loss enforces:

```text
score(original) - score(path_masked) >= margin
score(sky_masked) ~= score(original)
```

This is intentionally positive-only by default. Negative samples are already
inconsistent, so forcing path masking to further reduce their scores would not
answer the causal question.

The DINOv2 runner `train_dinov2_v5_minimal.py` uses the same training epoch
function and now prints the active path-grounding weight plus epoch-level
`path_ground_loss`.

Config:

```text
configs/train_navsim_future_dinov2_pathgrounded.py
```

Smoke run:

```text
work_dirs/iac_navsim_future_dinov2_pathgrounded_logcheck
Path grounding  = 0.12
path_ground_loss = 0.0053
```

Path-causal smoke:

```text
work_dirs/iac_navsim_future_dinov2_pathgrounded_smoke/path_causal_128
mean_path_delta        = 0.0822
mean_sky_delta         = 0.0450
mean_path_minus_sky    = 0.0372
path_delta_gt_sky_frac = 0.7422
is_path_grounded       = true
```

This is only a smoke-scale sanity check. The next decision requires a larger
run and comparison against the previous source-aware checkpoint using the same
path-causal sample count.

## Next Method Step

Run a real continuation and compare:

- normal IAC metrics: balanced accuracy, precision, recall, TNR
- path-causal metrics: path-minus-sky delta and path greater than sky fraction
- geometry shortcut metrics: trajectory-family and perturbation-family TNR

The likely next model contribution is:

```text
Path-Grounded Image-Action Consistency
```

not just a stronger black-box critic.
