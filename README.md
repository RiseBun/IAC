# IAC: Image-Action / Image-Trajectory Consistency

IAC is a benchmark and modeling pipeline for judging whether a candidate ego trajectory is supported by a generated future image.

The core question is not:

> Is this candidate exactly the GT trajectory?

It is:

> Given history images, a future image, and a candidate trajectory, is this trajectory inside the trajectory set supported by that future image?

This distinction matters because driving futures are multi-solution. A non-GT trajectory can still be visually supported if it is a small same-scene speed, lateral, or heading perturbation. Conversely, a trajectory can be physically reasonable and high-PDMS but still not match this particular future image.

## Current Main Result

The current trusted result is the grouped recovered-set pipeline, evaluated on g200 with ambiguity-adjusted top1:

| split | CP baseline | grouped recovered-set best | gain |
|---|---:|---:|---:|
| regular | 0.750 | 0.830 | +0.080 |
| low_iou | 0.705 | 0.800 | +0.095 |
| holdout | 0.730 | 0.780 | +0.050 |

This is the current main line:

```text
future image -> recover K supported paths -> compare candidate with recovered set
```

The most important finding is that recovering a supported set is more faithful to the problem than forcing a single GT ranking target.

## Current Pipeline

1. Train an image-trajectory consistency model with supported-set/listwise supervision.
2. Fuse consistency and path evidence as the CP baseline.
3. Train a grouped recovered-set probe from frozen visual features.
4. Recover K candidate supported paths from the future image.
5. Score each candidate by agreement with the recovered K-set.
6. Evaluate with ambiguity-adjusted top1, not hard GT top1.

Important files:

- `configs/train_navsim_future_dinov2_supported_set_listwise_vnext.py`
- `scripts/run_grouped_recovered_probe_k12_vnext.sh`
- `tools/train_recovered_path_set_probe_grouped_from_features.py`
- `tools/eval_recovered_path_set_agreement.py`
- `tools/extract_recovered_path_features.py`

## Supervision Design

The model no longer treats GT as the only positive.

```text
positive:
  gt_pos

soft positive:
  high-PDMS / high-EPDMS same-scene perturb_speed
  high-PDMS / high-EPDMS same-scene perturb_lateral
  high-PDMS / high-EPDMS same-scene perturb_heading

unknown:
  medium-quality same-scene perturbations

hard negative:
  image_swap
  time_shift_future
  traj_swap
  reverse_traj
  high_pdm_image_mismatch
```

Unknown rows are masked out of BCE/listwise objectives when their label is ambiguous.

## Why Hard GT Top1 Is Not Enough

Hard top1 asks whether GT is ranked first. That is too strict for this task because multiple candidates may be visually valid.

The main metric is:

```text
ambiguity-adjusted top1
```

It accepts GT or visually reasonable same-scene perturbations when the future image plausibly supports them.

We also track:

- hard mismatch above GT group rate
- low_iou and holdout performance
- source-wise calibration
- hard top1 as a secondary diagnostic only

## Visual-Time Specificity Findings

The recovered-set model improved ranking but still had a failure mode:

| split | hard mismatch above GT |
|---|---:|
| regular | 40.0% |
| low_iou | 29.0% |
| holdout | 39.5% |

This means the model learned:

```text
this future image roughly implies this motion shape
```

but not always:

```text
this specific future image at this specific time supports this exact candidate set
```

That is why visual-conditioned agreement/gate experiments were added.

## Visual Mismatch Gate Status

A visual-conditioned scorer can detect visual-time mismatch, but it is not yet a stable global ranker.

Recent calibrated-gate experiments:

1. BCE gate learned specificity but became over-saturated.
2. Three-class labels helped, but BCE still produced unreliable probabilities.
3. Margin loss without clipping exposed train/eval feature distribution drift.
4. Margin loss plus standardized feature clipping fixed the numerical explosion.

Best calibrated margin+clip results:

| split | best calibrated result | hard mismatch above GT |
|---|---:|---:|
| regular | 0.830 | 0.02 |
| low_iou | 0.790 | 0.04 |
| holdout | 0.785 | 0.03 |

Conclusion:

```text
The gate is useful as a diagnostic and conservative penalty candidate,
but it does not yet promote over grouped recovered-set because low_iou
still drops below 0.800.
```

Relevant files:

- `tools/train_visual_mismatch_gate_scorer.py`
- `tools/apply_visual_mismatch_penalty.py`
- `scripts/run_visual_mismatch_gate_trainlevel_g200.sh`

## Current Decision

Do not promote the visual gate as the main scorer yet.

Trusted main result remains:

| split | trusted result |
|---|---:|
| regular | 0.830 |
| low_iou | 0.800 |
| holdout | 0.780 |

Promotion criteria for the next method:

- regular >= 0.830
- low_iou >= 0.800
- holdout >= 0.800
- hard mismatch above GT < 25%
- source calibration must not show many GT=0 or time_shift_future=1 cases

## Next Step

The shortest next experiment is:

```text
train the margin+clip visual gate on more train groups
```

Use the same calibrated objective:

```text
supported:
  gt_pos + high-quality same-scene perturbations

unknown:
  near perturbations inside a neutral logit band

hard negative:
  image_swap / time_shift_future / high_pdm_image_mismatch
```

If a larger train set still cannot improve low_iou/holdout, the visual-time contrast should move into the main representation training instead of staying as a post-hoc gate.

## Repository Boundary

This repo contains code, configs, and evaluation tools. It does not include:

- raw nuPlan data
- checkpoints
- `work_dirs`
- logs
- cache files

