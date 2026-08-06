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
4. Ordered-motion support: aggregates candidate-blind, segment-level visual
   motion residuals into `supported`, `unsupported`, or
   `insufficient_evidence`. Missing visibility or excessive uncertainty causes
   abstention instead of being counted as agreement.

## Current Result

Using 200 grouped cases per split:

| Split | v3 acceptable / hard | v3 + clean gate acceptable / hard | Confidence verdict |
| --- | ---: | ---: | --- |
| regular | 0.970 / 0.030 | 0.990 / 0.010 | 189 match, 11 ambiguous, 0 mismatch |
| low_iou | 0.985 / 0.015 | 1.000 / 0.000 | 200 match |
| holdout | 0.985 / 0.015 | 1.000 / 0.000 | 198 match, 2 ambiguous |

Fusion used `beta=0.15`, `threshold=0`, then confidence used raw margins with
`match_margin=0.2`, `mismatch_margin=-0.5`, `temperature=0.2`.

The audited 4s ordered-motion run uses 500 / 105 / 105 groups with zero scene
or image overlap. On the repaired evaluation split, the frozen 20260805 model
keeps `0.9905` acceptable top-1 and `0.0095` hard-mismatch top-1. A validation-
frozen Wilson gate makes 105 unsupported decisions at `0.9619` precision and 7
supported decisions at `1.0` empirical precision; 623 / 735 rows abstain, so
total decision coverage is `0.1524`. The positive tail is exploratory: only six
validation examples support its threshold and its 95% Wilson lower bound is
`0.610`.

See [`pipeline/README.md`](pipeline/README.md) for the full step-by-step strategy of the mainline (v3 → clean gate → fuse → confidence) and the current metric table with per-step design rationale. [`ARCHITECTURE.md`](ARCHITECTURE.md) is the higher-level module map.

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║   INPUT  Each JSONL row = one candidate trajectory in a group of same-scene rivals   ║
║          ┌────────────────────────────────────────────────────────────────────────┐  ║
║          │ group_id · sample_id · source_type · candidate_traj                    │  ║
║          │ history_images · future_images                                         │  ║
║          │ upstream scalar scores (iac_consistency, recovered_set_*, ...)         │  ║
║          └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                      ║
║   Each group contains:                                                               ║
║     acceptable  = { gt_pos, perturb_speed, perturb_lateral, perturb_heading }        ║
║                    (visually near-indistinguishable in front-view video)             ║
║     hard        = { image_swap, time_shift_future, traj_swap, reverse_traj,          ║
║                     high_pdm_image_mismatch }                                        ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1  V-JEPA2 visual feature extraction         [candidate-blind: never sees traj] │
│ ──────────────────────────────────────────────────────────────────────────────────── │
│                                                                                      │
│     history_images ─┐                                                                │
│                     ├──► resample to 64 frames ──► facebook/vjepa2-vitl-fpc64-256    │
│     future_images ──┘                              (frozen, eval, inference_mode)    │
│                                                    │                                 │
│                                                    ├──► pool → x         (pooled)    │
│                                                    │                                 │
│                                                    └──► 16-chunk token-mean          │
│                                                            → x_tokens [B, 16, 1024]  │
│                                                                                      │
│     script:  pipeline/extract_vjepa_video_features.py                                │
│     output:  work/eval_vjepa.pt   { x_tokens, x, sample_id, group_id, ... }          │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │  x_tokens  (used only in Step 3)
                                       │
                                       ▼
   ┌───── two independent evidence branches, run in parallel ─────┐
   │                                                              │
   ▼                                                              ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│ STEP 2  v3 acceptability calibrator  │    │ STEP 3  Clean V-JEPA trajectory gate │
│ (scalar-side evidence)               │    │ (visual × trajectory evidence)       │
│ ──────────────────────────────────── │    │ ──────────────────────────────────── │
│                                      │    │                                      │
│  31 hand-crafted scalar features:    │    │  Inputs:                             │
│   • iac_consistency + its logit      │    │   visual  = x_tokens [16, 1024]      │
│     + diffs against aux scores       │    │   traj    = candidate_traj first 8   │
│   • recovered_set_agreement,         │    │             points, each 5-dim:      │
│     minade, topmode_ade,             │    │             [x, y, sin(h), cos(h),   │
│     best_mode_fde, heading_error,    │    │              cum_distance]           │
│     progress_error, path_iou,        │    │   scalar  = [0.0]   (ZEROED, so gate │
│     supported                        │    │             cannot look at v3 side)  │
│   • path_minus_sky_delta,            │    │                                      │
│     candidate_minus_wrong_*_delta    │    │  Model  MismatchGate (traj_cross_    │
│   • 10 traj geom features            │    │         attention):                  │
│     (endpoint, path len, directness, │    │    visual_proj: 1024 → 32            │
│      step stats, heading change, ...)│    │    traj_proj  :    5 → 32            │
│                                      │    │    MHA(query=traj_proj,              │
│  Model  Calibrator:                  │    │        key/value=visual_proj), 4 heads│
│    Linear(31 → 16)                   │    │    fuse [attn.mean, q.mean,          │
│    LayerNorm → ReLU → Dropout        │    │          attn*q, |attn-q|, scalar=0] │
│    Linear(16  → 1)                   │    │    → Linear → ReLU → Linear(1)       │
│  (intentionally tiny to prevent      │    │                                      │
│   source-label leakage)              │    │  Loss (margin, not BCE):             │
│                                      │    │    ReLU(m⁺ - pos_logits)             │
│  Loss  BCE + 0.35 × pairwise-margin  │    │    + ReLU(neg_logits + m⁻)           │
│  Weights: gt_pos=1.0, other          │    │    + group softplus pairwise         │
│    acceptable=0.85, traj_swap /      │    │    + w · ReLU(|unknown|-m_u)         │
│    time_shift=1.2, other hard=1.0    │    │                                      │
│                                      │    │  Kept small in scope: gate is        │
│  Optim  AdamW lr=5e-3, wd=1e-3,      │    │  independent second evidence, not    │
│         2000 steps                   │    │  a replacement ranker.               │
│                                      │    │                                      │
│  script pipeline/score_acceptability │    │  script pipeline/score_visual_       │
│         _calibrator.py               │    │         mismatch_gate.py             │
│  model  models/iac_acceptability_    │    │  model  models/clean_vjepa_traj_     │
│         calibrator.pt (31d, h=16)    │    │         gate.pt                      │
│                                      │    │                                      │
│  writes  iac_acceptability_          │    │  writes  visual_non_mismatch_logit   │
│          calibrated ∈ (0,1)          │    │          visual_non_mismatch         │
└──────────────────────────────────────┘    └──────────────────────────────────────┘
                    │                                          │
                    │  v3_score                     gate_logit │
                    │                                          │
                    └──────────────┐        ┌──────────────────┘
                                   ▼        ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │ STEP 4  Conservative in-group fusion                                │
        │ ─────────────────────────────────────────────────────────────────── │
        │                                                                     │
        │   For each group, independently:                                    │
        │                                                                     │
        │     group_max_gate = max(gate_logit) over this group's rows         │
        │     penalty        = max(0, group_max_gate − gate_logit − threshold)│
        │     fused_score    = v3_score − beta × penalty                      │
        │                                                                     │
        │   Frozen:  beta = 0.15    threshold = 0                             │
        │                                                                     │
        │   Why max(0, …)?  Gate can ONLY subtract from v3, never boost.      │
        │   The best-visual row in a group gets penalty=0, keeping v3 intact. │
        │   Every worse-visual row is pulled down proportionally to how much  │
        │   worse it looks. Gate is a veto knob, not a ranker.                │
        │                                                                     │
        │   script pipeline/fuse_v3_clean_gate.py                             │
        │   writes v3_clean_gate_fused_rank_score                             │
        └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       │  fused_score (per row)
                                       ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │ STEP 5  Multi-solution acceptance + asymmetric margin verdict       │
        │ ─────────────────────────────────────────────────────────────────── │
        │                                                                     │
        │   Per group:                                                        │
        │     best_accept = argmax(score) over rows with source ∈ acceptable  │
        │     best_bad    = argmax(score) over rows with source ∈ hard        │
        │                       (fallback: non-acceptable if hard is empty)   │
        │     margin      = best_accept.score − best_bad.score       (raw)    │
        │                                                                     │
        │        margin ≥ +0.20   ─────►  verdict = match                     │
        │        margin ≤ −0.50   ─────►  verdict = mismatch                  │
        │        else             ─────►  verdict = ambiguous                 │
        │                                                                     │
        │     decision_confidence = sigmoid(|margin| / 0.20)                  │
        │     match_confidence    = sigmoid( margin  / 0.20)                  │
        │                                                                     │
        │   Asymmetric on purpose:                                            │
        │     +0.20  → cheap to grant "match" (downstream still gates)        │
        │     −0.50  → expensive to grant "mismatch" (higher damage from      │
        │              false negative), require strong evidence.              │
        │                                                                     │
        │   Multi-solution: gt_pos AND same-scene perturbations are ALL       │
        │   valid; we do not force a fake single-winner per group.            │
        │                                                                     │
        │   script pipeline/score_iac_confidence.py                           │
        │   writes  verdict ∈ { match, ambiguous, mismatch }                  │
        │           decision_confidence, match_confidence, accept_margin, ... │
        └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼

              OUTPUT  work/confidence_groups.jsonl   (per-group verdicts)
                      work/confidence_summary.json  (verdict_counts, top-1 rates,
                                                     margin quartiles, etc.)


┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Parallel sibling channel (not part of the v3+gate mainline, joined at report time):  │
│                                                                                      │
│   ordered_motion_segment_ledger  ──►  ordered_motion/                                │
│                                       ├─ calibrate_ordered_motion_support.py (val)   │
│                                       ├─ score_ordered_motion_support.py     (eval)  │
│                                       └─ ordered_motion_support.py           (lib)   │
│                                                                                      │
│   Emits per-row: supported / unsupported / insufficient_evidence                     │
│   (visibility + uncertainty gate; abstains rather than agreeing)                     │
└──────────────────────────────────────────────────────────────────────────────────────┘


        LEGEND     ─►    data flow                 [Sx]     step Sx of the mainline
                   ┌─┐   module boundary            frozen  values from mainline_manifest
```

### Metrics (frozen mainline, 200 groups per split)

| Split | acceptable_top1 | hard_mismatch_top1 | match | ambiguous | mismatch |
| :-- | --: | --: | --: | --: | --: |
| **regular** | 0.990 | 0.010 | 189 | 11 | 0 |
| **low_iou** | 1.000 | 0.000 | 200 | 0 |  0 |
| **holdout** | 1.000 | 0.000 | 198 |  2 |  0 |

Zero `mismatch` verdicts across 600 groups: no false negatives. 13 `ambiguous` groups are visually undecidable cases held out of the binary decision, not silently miscounted as agreement.

## Files

- `models/iac_acceptability_calibrator.pt`: trained v3 calibrator.
- `models/clean_vjepa_traj_gate.pt`: clean V-JEPA2 trajectory gate.
- `pipeline/extract_vjepa_video_features.py`: frozen V-JEPA2 feature extraction.
- `pipeline/score_acceptability_calibrator.py`: apply v3 calibrator.
- `pipeline/score_visual_mismatch_gate.py`: apply clean trajectory gate.
- `pipeline/fuse_v3_clean_gate.py`: conservative v3 + gate fusion.
- `pipeline/score_iac_confidence.py`: group-level match / ambiguous / mismatch verdicts.
- `training/train_iac_acceptability_calibrator.py`: retrain v3 calibrator.
- `training/train_visual_mismatch_gate_scorer.py`: retrain the clean gate.
- `ordered_motion/ordered_motion_support.py`: visibility-aware aggregation and
  three-state support decisions.
- `audit/audit_formal_splits.py`: fail-closed group, scene, image, and horizon
  split audit.
- `ordered_motion/calibrate_ordered_motion_support.py`: freeze support
  thresholds using validation labels only.
- `ordered_motion/score_ordered_motion_support.py`: apply the frozen thresholds
  without reading labels.
- `audit/audit_ordered_motion_support.py`: report decision-tail precision after
  inference without modifying thresholds.
- `scripts/run_ordered_motion_support_formal.sh`: audited 4s scoring entrypoint.

## Minimal Run

```bash
python pipeline/extract_vjepa_video_features.py \
  --index work/eval_rows.jsonl \
  --image-root /path/to/images \
  --output work/eval_vjepa.pt \
  --token-summary-size 16

python pipeline/score_acceptability_calibrator.py \
  --model models/iac_acceptability_calibrator.pt \
  --primary-scores work/base_scores.jsonl \
  --aux work/aux_scores.jsonl \
  --output-scores work/v3_scores.jsonl \
  --output-summary work/v3_summary.json

python pipeline/score_visual_mismatch_gate.py \
  --model models/clean_vjepa_traj_gate.pt \
  --rows work/v3_scores.jsonl \
  --visual-cache work/eval_vjepa.pt \
  --visual-cache-key x_tokens \
  --output-scores work/clean_gate_scores.jsonl

python pipeline/fuse_v3_clean_gate.py \
  --v3-scores work/v3_scores.jsonl \
  --gate-scores work/clean_gate_scores.jsonl \
  --output-scores work/fused_scores.jsonl \
  --output-summary work/fused_summary.json \
  --beta 0.15 \
  --threshold 0

python pipeline/score_iac_confidence.py \
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

## Ordered-Motion Formal Run

The upstream ordered-motion scorer must write
`ordered_motion_segment_ledger` by using `--include-segment-ledger`. A formal
run then has three ordered stages:

```bash
python audit/audit_formal_splits.py \
  --split train=work/train_rows.jsonl \
  --split val=work/val_rows.jsonl \
  --split eval=work/eval_rows.jsonl \
  --horizon 4s --require-formal-ready \
  --output-summary work/formal_split_audit.json

python ordered_motion/calibrate_ordered_motion_support.py \
  --scores work/val_segment_scores.jsonl \
  --output-config work/ordered_motion_support_config.json \
  --min-supported-precision 0.95 \
  --min-unsupported-precision 0.95 \
  --min-unsupported-precision-lower-bound 0.95

python ordered_motion/score_ordered_motion_support.py \
  --scores work/eval_segment_scores.jsonl \
  --config work/ordered_motion_support_config.json \
  --output-scores work/eval_support.jsonl \
  --output-summary work/eval_support_summary.json
```

`source_type` is used only to calibrate thresholds on validation data. It is
not read by inference. Missing scene identity is an audit failure, not evidence
that two splits are disjoint. The one-command server entrypoint is:

```bash
WORK=/path/to/ordered_motion_4s_pilot \
PKG=/path/to/ordered_motion_package \
bash scripts/run_ordered_motion_support_formal.sh
```

## Why This Mainline

The original need is not BEV reconstruction or trajectory prediction by itself.
The need is to judge: "If this action really happened, would the generated
future images look like this?" Mild speed, heading, or lateral differences can
be visually indistinguishable in front-view video, so the evaluator reports
confidence and ambiguity instead of pretending every group has only one correct
answer.
