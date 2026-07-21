# NAVSIM Future Results, 2026-06-29

## What Was Fixed

- Built `indices_navsim_future` with `future_image_policy=future`.
- Audited train/val indexes: positive history/future exact overlap is `0.0`.
- Added `tools/audit_consistency_index.py` so fallback indexes cannot be reported as strict future-frame IAC by accident.
- Fixed `benchmark_wam.py` so NAVSIM `sample_id=anchor__source_type` rows are grouped back into anchor candidate sets for ranking.
- Added `tools/analyze_wam_scores.py` for confusion metrics and highest-confidence errors.

## Runs

| run | samples | consistency AUC | recall | TNR | pos mean | neg mean | validity AUC | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iac_navsim_future_cnn_3k` | 4096 | 0.4915 | 0.0000 | 1.0000 | 0.2544 | 0.2546 | 0.9976 | all-negative consistency head |
| `iac_navsim_future_balanced_3k` | 2048 | 0.5138 | 1.0000 | 0.0000 | 0.5360 | 0.5358 | 0.9984 | large positive weight flips to all-positive |
| `iac_navsim_future_stratified_1k5` | 2048 | 0.5170 | 1.0000 | 0.0045 | 0.5130 | 0.5125 | 0.9981 | balanced sampler still not visually separable |

## Interpretation

The original issue was real: old NAVSIM compatibility indexes could replay
history frames, which is not a valid future-image IAC setup. The new
`indices_navsim_future` index fixes that data invariant.

After the data fix, the lightweight CNN critic does not learn useful
image-action consistency on NAVSIM future frames. The consistency score is
nearly constant for positives and negatives across all tested weighting and
sampling schemes. Accuracy alone is misleading because the validation stream is
about one positive to six negatives.

The validity head is stable and useful: all runs keep validity AUC near `0.998`.

## Next Shortest Path

Do not spend more time tuning global BCE weights for the CNN baseline. The
evidence says the visual consistency representation is the bottleneck.

Next run should use a stronger visual backbone, preferably the existing DINOv2
path, on `indices_navsim_future`, with:

- strict index audit before training;
- explicit benchmark score analysis after training;
- ranking grouped by inferred NAVSIM anchor id;
- smaller, resumable training/eval stages because this server's image I/O is slow.
