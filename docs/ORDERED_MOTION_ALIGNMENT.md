# Ordered Motion Alignment: corrected time-token decision

## Status and claim boundary

This is an opt-in failure-analysis module for the trusted IAC mainline. It does
not alter the default IAC scorer and is not, by itself, a paper novelty claim.

The first real 200-group run showed a small fused improvement, but the
end-to-end order control failed. A subsequent code audit found the reason that
must be tested first: legacy `x_tokens` are equal chunks of the flattened
V-JEPA `(T,H,W)` patch grid, not pure time tokens. With the returned 8-frame,
tubelet-2 cache, 16 legacy chunks represent four time slices with four spatial
chunks per slice.

The corrected implementation restores explicit time before learning:

```text
V-JEPA flattened patches [B, T*H*W, D]
  -> reshape [B,T,H,W,D]
  -> spatial mean over H,W
  -> true time tokens [B,T,D]
```

Existing equal-chunk caches can be migrated without rerunning the backbone when
their metadata records `num_frames`. For the returned 8-frame cache, averaging
each consecutive group of four legacy chunks exactly reconstructs the four
time-wise spatial means.

## What the head tests

The visual estimator remains candidate-blind:

```text
true V-JEPA time tokens
  -> local monotonic segment-to-time alignment
  -> per-segment visual motion estimate and uncertainty

candidate trajectory
  -> deterministic longitudinal, lateral, heading and curvature targets

visual estimate vs candidate targets
  -> non-negative, additive mismatch ledger
```

The candidate trajectory and source label are never visual-network inputs.
Source labels are used only to select GT supervision in training, tune fusion
on validation, and report metrics.

## Fast three-seed engineering decision

Run this first on the already returned regular/low-IoU/holdout rows and caches.
It migrates each cache once, shares the corrected compact caches across three
seeds, tunes fusion on validation only, and applies the identical selected
fusion rule to every order and identity control.

```bash
nohup env \
  TRAIN_ROWS=/path/to/regular/calibrated_scores.jsonl \
  TRAIN_CACHE=/path/to/regular_vjepa.pt \
  VAL_ROWS=/path/to/low_iou/calibrated_scores.jsonl \
  VAL_CACHE=/path/to/low_iou_vjepa.pt \
  EVAL_ROWS=/path/to/holdout/calibrated_scores.jsonl \
  EVAL_CACHE=/path/to/holdout_vjepa.pt \
  VAL_PRIMARY=/path/to/low_iou/calibrated_scores.jsonl \
  EVAL_PRIMARY=/path/to/holdout/calibrated_scores.jsonl \
  PRIMARY_KEY=iac_acceptability_calibrated \
  OUT_ROOT=work/ordered_motion_time_token_decision \
  SEEDS=20260728,20260729,20260730 \
  bash scripts/run_ordered_motion_time_token_decision.sh \
  > work/ordered_motion_time_token_decision.nohup.log 2>&1 &
```

This job does not download a model, install packages, shut down the server, or
change IAC defaults. Return:

- `work/ordered_motion_time_token_decision_results.tar.gz`;
- `work/ordered_motion_time_token_decision.nohup.log`.

The main result is
`multi_seed_decision_summary.json`. The corrected head advances only if:

1. normal MRR beats every order control in at least two of three seeds and has
   a positive mean margin over the strongest order control;
2. normal MRR beats both identity derangements in every seed;
3. mean acceptable Top-1 and MRR do not decrease;
4. mean speed and time-shift pairwise results do not decrease.

This rerun is an engineering decision, not formal evidence, because the
previous validation and holdout sets share scenes and exact images.

## Formal run after the gate passes

Only after the corrected head passes the above gate should it be rerun on
train/validation/evaluation rows separated by log or drive. Use the base script
with a hard disjointness check:

```bash
nohup env \
  TRAIN_ROWS=/path/to/drive_disjoint_train.jsonl \
  TRAIN_CACHE=/path/to/train_vjepa.pt \
  VAL_ROWS=/path/to/drive_disjoint_validation.jsonl \
  VAL_CACHE=/path/to/validation_vjepa.pt \
  EVAL_ROWS=/path/to/drive_disjoint_test.jsonl \
  EVAL_CACHE=/path/to/test_vjepa.pt \
  VAL_PRIMARY=/path/to/validation_iac_scores.jsonl \
  EVAL_PRIMARY=/path/to/test_iac_scores.jsonl \
  PRIMARY_KEY=iac_acceptability_calibrated \
  REQUIRE_STRICT_SPLIT_DISJOINT=1 \
  OUT_DIR=work/ordered_motion_drive_disjoint \
  bash scripts/run_ordered_motion_alignment_audit.sh \
  > work/ordered_motion_drive_disjoint.nohup.log 2>&1 &
```

If a cache is absent, the base script can extract it from local images by also
setting `IMAGE_ROOT` and `VJEPA_MODEL`. New extraction always writes
shape-aware `x_time_tokens`; `x_tokens` is retained only for old-gate
compatibility and is marked as flattened spatiotemporal chunks in metadata.

## Controls and outputs

The audit reports:

- normal, reversed and permuted visual time;
- normal, reversed and permuted trajectory segments;
- within-group candidate Sattolo derangement;
- cross-group visual derangement;
- optional raw-frame reverse/shuffle re-extraction;
- split overlap by sample, group, scene/log and exact image;
- per-candidate longitudinal/lateral/heading/path-shape ledger;
- validation-selected fusion applied unchanged to all controls.

The small result archive excludes large V-JEPA caches. The source caches are
never modified.

## Scientific decision

Passing the corrected order and identity gates would justify a larger,
drive-disjoint experiment. It still would not prove that speed
counterfactuals are solved.

If the order gate fails again, stop scaling this generic token-alignment head.
The next method should explicitly predict the image motion induced by each
candidate under camera geometry and compare that prediction with observed
flow or point tracks. That candidate-induced physical residual, rather than a
larger generic head, is the stronger remaining novelty direction.
