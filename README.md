# IAC Mainline

IAC scores whether a WAM-generated future image sequence is consistent with a
given candidate action trajectory. The current trusted mainline is deliberately
small:

1. `v3` acceptability calibrator: learns the task metric, where GT and visually
   plausible mild action perturbations are acceptable, while image/time/traj
   swaps are hard mismatches.
2. Clean V-JEPA2 trajectory gate: uses frozen V-JEPA2 future-video tokens plus
   the candidate trajectory, with scalar side features zeroed, to detect
   visual-action mismatch evidence.
3. Conservative fusion and confidence: v3 remains the main ranker; the clean
   gate only penalizes candidates that are visually less compatible inside the
   same group. Low-margin cases are reported as `ambiguous`, not forced into a
   false binary decision.

## Current Result

Using 200 grouped cases per split:

| Split | v3 acceptable / hard | v3 + clean gate acceptable / hard | Confidence verdict |
| --- | ---: | ---: | --- |
| regular | 0.970 / 0.030 | 0.990 / 0.010 | 189 match, 11 ambiguous, 0 mismatch |
| low_iou | 0.985 / 0.015 | 1.000 / 0.000 | 200 match |
| holdout | 0.985 / 0.015 | 1.000 / 0.000 | 198 match, 2 ambiguous |

Fusion used `beta=0.15`, `threshold=0`, then confidence used raw margins with
`match_margin=0.2`, `mismatch_margin=-0.5`, `temperature=0.2`.

## Files

- `models/iac_acceptability_calibrator.pt`: trained v3 calibrator.
- `models/clean_vjepa_traj_gate.pt`: clean V-JEPA2 trajectory gate.
- `tools/extract_vjepa_video_features.py`: frozen V-JEPA2 feature extraction.
- `tools/score_acceptability_calibrator.py`: apply v3 calibrator.
- `tools/score_visual_mismatch_gate.py`: apply clean trajectory gate.
- `tools/fuse_v3_clean_gate.py`: conservative v3 + gate fusion.
- `tools/score_iac_confidence.py`: group-level match / ambiguous / mismatch verdicts.
- `tools/train_iac_acceptability_calibrator.py`: retrain v3 calibrator.
- `tools/train_visual_mismatch_gate_scorer.py`: retrain the clean gate.

## Minimal Run

```bash
python tools/extract_vjepa_video_features.py \
  --index work/eval_rows.jsonl \
  --image-root /path/to/images \
  --output work/eval_vjepa.pt \
  --token-summary-size 16

python tools/score_acceptability_calibrator.py \
  --model models/iac_acceptability_calibrator.pt \
  --primary-scores work/base_scores.jsonl \
  --aux work/aux_scores.jsonl \
  --output-scores work/v3_scores.jsonl \
  --output-summary work/v3_summary.json

python tools/score_visual_mismatch_gate.py \
  --model models/clean_vjepa_traj_gate.pt \
  --rows work/v3_scores.jsonl \
  --visual-cache work/eval_vjepa.pt \
  --visual-cache-key x_tokens \
  --output-scores work/clean_gate_scores.jsonl

python tools/fuse_v3_clean_gate.py \
  --v3-scores work/v3_scores.jsonl \
  --gate-scores work/clean_gate_scores.jsonl \
  --output-scores work/fused_scores.jsonl \
  --output-summary work/fused_summary.json \
  --beta 0.15 \
  --threshold 0

python tools/score_iac_confidence.py \
  --primary-scores work/fused_scores.jsonl \
  --score-key v3_clean_gate_fused_rank_score \
  --margin-space raw \
  --match-margin 0.2 \
  --mismatch-margin -0.5 \
  --confidence-temperature 0.2 \
  --output-groups work/confidence_groups.jsonl \
  --output-summary work/confidence_summary.json
```

Input JSONL rows must contain stable `group_id`, `sample_id`, `source_type`,
`candidate_traj`, and the score fields used by the v3 calibrator. V-JEPA
extraction also needs `history_images` and `future_images`.

## Why This Mainline

The original need is not BEV reconstruction or trajectory prediction by itself.
The need is to judge: "If this action really happened, would the generated
future images look like this?" Mild speed, heading, or lateral differences can
be visually indistinguishable in front-view video, so the evaluator reports
confidence and ambiguity instead of pretending every group has only one correct
answer.
