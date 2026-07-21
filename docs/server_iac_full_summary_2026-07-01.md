# IAC Server Project Full Summary, 2026-07-01

## 1. What this project is trying to solve

IAC is not a planner and not a world model.
Its role is to evaluate a World Action Model (WAM).

The core question is:

- given `history images`
- given a `candidate trajectory` or action
- given `future images` from a WAM

can we judge:

- whether the future images support the action (`consistency`)
- whether the action itself is physically and semantically reasonable (`validity`)

This matters because current WAM evaluation is split:

- visual prediction is often judged by pixel or perceptual metrics
- action quality is often judged only by downstream success

That split misses the real WAM premise:

- actions should be grounded in visual anticipation
- visual anticipation should constrain action choice

So IAC is being developed as a simulator-free, annotation-free, plug-and-play benchmark for this coupling.

## 2. Current project framing

The current server-side project has converged to the following framing:

- main line: use IAC as a WAM benchmark
- method line: strengthen the critic so it better distinguishes supported futures from unsupported futures
- analysis line: use probing and layer-wise diagnostics to understand where the signal actually lives

The project is no longer being treated as "build a bigger critic".
It is being treated as:

- define a meaningful WAM evaluation problem
- build a reasonably strong evaluator
- explain what the evaluator is actually using

## 3. Data and benchmark setup

### 3.1 Datasets

Two dataset directions were discussed during the project:

- `NAVSIM`
- `nuPlan`

The current active experiments on the server are primarily on the strict-future `NAVSIM` setup.

The key point is not just which dataset is used, but how the index is built.

### 3.2 Strict-future requirement

The project discovered an early failure mode:

- some old setups could accidentally use history-tail replay or misaligned frames as "future evidence"

That makes the benchmark invalid.

So a major early effort was:

- build a strict-future NAVSIM index
- audit the index
- guarantee that positive rows use real future frames

This is one of the most important pieces of the project, because if this is wrong, all later training and reporting are misleading.

Relevant files on the server/local repo:

- [README.md](/C:/Users/LPN19/Desktop/iac/IAC/README.md)
- [index_audit.md](/C:/Users/LPN19/Desktop/iac/IAC/docs/index_audit.md)
- [evaluation_protocol_2026-07-01.md](/C:/Users/LPN19/Desktop/iac/IAC/docs/evaluation_protocol_2026-07-01.md)

### 3.3 Labels and sample construction

IAC does not rely on manual annotation for "image and action are consistent".
Instead it constructs positives and negatives automatically.

Positive rows:

- real history frames
- real future frames
- real ego future trajectory

Negative rows are built by perturbation or swapping, such as:

- `traj_swap`
- `image_swap`
- `time_shift_future`
- `perturb_lateral`
- `perturb_heading`
- `perturb_speed`
- sometimes reverse-style negatives

This is important because the benchmark is meant to scale and stay objective.

## 4. Model pipeline: what the current best line actually is

The current best line is not a full multilayer DINO method.
It is a stronger single-layer evaluator with extra training structure.

The active pipeline is:

- frozen `DINOv2 ViT-S/14`
- single visual layer, currently layer 11
- separate encoding of `history images` and `future images`
- MLP encoding of `candidate trajectory`
- MLP encoding of `ego state`
- fusion into:
  - `z_shared` for consistency-style reasoning
  - `z_validity` for validity-style reasoning
- two main outputs:
  - `consistency`
  - `validity`

In addition, the model includes:

- auxiliary heads for speed / steering / progress / temporal consistency
- future evidence branch
- hierarchical consistency branch

Relevant implementation files:

- [train.py](/C:/Users/LPN19/Desktop/iac/IAC/train.py)
- [train_dinov2_v5_minimal.py](/C:/Users/LPN19/Desktop/iac/IAC/train_dinov2_v5_minimal.py)
- [train_navsim_future_dinov2_evidence.py](/C:/Users/LPN19/Desktop/iac/IAC/configs/train_navsim_future_dinov2_evidence.py)
- [train_navsim_future_dinov2_evidence_recallboost.py](/C:/Users/LPN19/Desktop/iac/IAC/configs/train_navsim_future_dinov2_evidence_recallboost.py)

## 5. Why DINOv2 was introduced

The older visual baselines were not strong enough.
The project moved to DINOv2 because:

- it provides stronger visual semantics than the earlier lightweight visual encoders
- it is easy to freeze and reuse
- it makes it possible to ask whether future visual evidence is genuinely useful

The current choice is conservative:

- `DINOv2 ViT-S/14`
- frozen backbone
- single layer

This was intentional.
The goal was not to win by backbone scale first.
The goal was to test whether future visual evidence can improve evaluation at all.

## 6. Architectural additions beyond the plain single-head critic

### 6.1 Future evidence injection

The project added a `future_consistency_evidence` branch so future frames are not just passively concatenated.

Purpose:

- make future-image evidence explicitly affect the consistency decision
- test whether the model becomes more sensitive to future-frame changes

This branch is conceptually important because the benchmark should answer:

- is the trajectory really supported by the future images?

not just:

- is the trajectory generally plausible?

### 6.2 Hierarchical consistency

The project also added a more structured consistency decomposition:

- `physics_support`
- `action_support`
- `future_support`
- fused into `consistency_fuse`

Purpose:

- separate physical plausibility from future-image support
- make the consistency decision less monolithic
- give better internal states for probing

### 6.3 Ranking-based learning

The project moved away from pure pointwise BCE.

It now uses:

- BCE-style consistency and validity supervision
- group ranking loss over grouped candidates
- future evidence auxiliary supervision
- hierarchical auxiliary supervision

The ranking part is critical because the benchmark is not only about absolute classification.
It is also about:

- does the model place the true supported candidate above the unsupported candidates?

## 7. The major conceptual shift in training

One of the most important discoveries in this project was:

- selecting checkpoints by `val_loss` is not aligned with the benchmark goal

Why:

- a model can get decent validation loss while collapsing to conservative consistency outputs
- this can create very high apparent accuracy under a bad threshold
- but recall becomes zero, and the model is useless as a WAM evaluator

So the project changed checkpoint selection from:

- `val_loss`

to consistency-aware criteria such as:

- `c_score_gap`
- `c_balanced_acc`
- later precision/tnr-aware variants

This change matters as much as the model changes.
It changed what the training process tries to preserve as "best".

## 8. Probing and diagnosis: what we learned about where the signal lives

This project did not stop at benchmark numbers.
It also extracted internal features and trained linear probes on them.

Probed states included:

- `hist_seq`
- `fut_seq`
- `z_traj_cons`
- `z_traj_val`
- `z_shared`
- `z_validity`
- future-evidence-related states

Tools added for this:

- [extract_probe_features.py](/C:/Users/LPN19/Desktop/iac/IAC/tools/extract_probe_features.py)
- [train_layer_probes.py](/C:/Users/LPN19/Desktop/iac/IAC/tools/train_layer_probes.py)

### 8.1 Main probe conclusion

The strongest physics / geometry signal does **not** first appear in the image sequence layers.
It appears first in the trajectory branches.

Most readable layers for physics-like quantities were:

- `z_traj_val`
- `z_traj_cons`

Especially strong for:

- `mean_speed`
- `heading_change`
- `final_disp`
- `lateral_abs`

### 8.2 Consistency signal location

Consistency was mainly readable from:

- `z_shared`

not from:

- `hist_seq`
- `fut_seq`

Interpretation:

- current consistency reasoning still mostly happens at the fusion stage
- future image evidence has not cleanly migrated into early visual layers

### 8.3 Consequence for the "multilayer" idea

This changed the project's understanding of "multilayer".

Important conclusion:

- the meaningful "layers" for IAC are currently more like internal WAM/critic reasoning layers
- not just DINO backbone layers

So backbone multi-layer fusion is not the main story.
The more useful story is:

- where do physics signals emerge?
- where does consistency become readable?
- does future evidence move earlier or stay late?

## 9. Main experiment timeline and outcomes

Below is the high-signal summary of the major runs that matter now.

### 9.1 Stable evidence baseline

Workdir:

- `work_dirs/iac_navsim_future_dinov2_evidence_ddp4_resume`

Best 2048 benchmark result:

- consistency balanced accuracy: `0.6089`
- consistency precision: `0.1722`
- consistency recall: `0.8352`
- consistency F1: `0.2855`
- consistency TNR: `0.3825`
- validity accuracy: `0.9868`

What it showed:

- future-evidence line is viable
- but the model still had poor calibration
- default threshold behavior could collapse into all-negative predictions

### 9.2 Hierarchical smoke

Workdir:

- `work_dirs/iac_navsim_future_dinov2_hierarchical_smoke`

1024 benchmark:

- consistency balanced accuracy: `0.5961`
- precision: `0.1600`
- recall: `0.9481`
- F1: `0.2738`
- TNR: `0.2441`

What it showed:

- hierarchical chain can prevent total all-negative collapse
- but can also create many false positives

### 9.3 Metric-aware checkpoint line

Workdir:

- `work_dirs/iac_navsim_future_dinov2_metric_ddp4_continue`

2048 benchmark:

- consistency balanced accuracy: `0.6517`
- precision: `0.1953`
- recall: `0.8278`
- F1: `0.3161`
- TNR: `0.4755`
- validity accuracy: `0.9888`

This was the first major real step forward.

Why it improved:

- not mainly because the model got huge
- but because the training / checkpoint selection finally aligned with benchmark usefulness

This run proved that the evaluator can beat the earlier baseline when selected by the right criterion.

### 9.4 Hard-negative continuation

Workdir:

- `work_dirs/iac_navsim_future_dinov2_hardneg_ddp4_continue`

2048 benchmark:

- consistency balanced accuracy: `0.6604`
- precision: `0.1921`
- recall: `0.9084`
- F1: `0.3171`
- TNR: `0.4124`
- validity accuracy: `0.9893`

This is the current best formal 2048 result on the server.

Interpretation:

- it beat the previous best in balanced accuracy and F1
- but not in the most satisfying way
- the gain came mostly from pushing recall higher
- precision did not really improve

So this is a real improvement, but not yet the kind of improvement we actually want.

### 9.5 Precision-first smoke

Workdir:

- `work_dirs/iac_navsim_future_dinov2_precision_smoke`

128 benchmark:

- consistency balanced accuracy: `0.6624`
- precision: `0.1892`
- recall: `0.5833`
- F1: `0.2857`
- TNR: `0.7414`

This run came from making hard-negative constraints stronger and changing checkpoint selection to prefer precision/TNR more explicitly.

Interpretation:

- it did move the model into a more conservative regime
- TNR increased
- but overall sample-size smoke quality was worse than the previous hard-negative smoke

So this direction is not wrong, but the current strength setting is too aggressive.

## 10. What the current best model really is

Current best formal result on the server:

- checkpoint:
  `~/IAC/work_dirs/iac_navsim_future_dinov2_hardneg_ddp4_continue/checkpoints/latest.pth`
- benchmark:
  `~/IAC/work_dirs/iac_navsim_future_dinov2_hardneg_ddp4_continue/benchmark_val_2048/score_analysis.json`

What this model is:

- single-layer frozen DINOv2 visual backbone
- future evidence injection
- hierarchical consistency branch
- ranking-based training
- hard-negative auxiliary pressure
- consistency-aware checkpoint selection

What it is not:

- not a full multilayer DINO solution
- not a learned world model
- not a planner
- not yet a fully satisfying high-precision evaluator

## 11. What has gone wrong along the way

Several failure modes were discovered.

### 11.1 Invalid future-image setup

If positive rows do not use true future frames, the benchmark is conceptually wrong.
This was fixed by strict-future index building and auditing.

### 11.2 Default-threshold illusion

A model could look good under naive accuracy, because almost everything is negative.
But consistency recall could be zero.

This was one of the most important lessons in the project.

### 11.3 Shortcut risk

Probe results suggest:

- geometry priors are strong
- trajectory branches are very readable
- future-image branches are comparatively weak

This means the evaluator may still partly rely on trajectory plausibility instead of real future evidence.

### 11.4 External kill / unstable long continuation

One continuation run was killed by external `SIGKILL` during training.
This did not destroy the experiment, because an earlier valid best checkpoint was already saved.
But it does mean long uninterrupted runs on the server are not always reliable.

## 12. What the current numbers mean

Current best official-like result:

- consistency balanced accuracy around `0.66`
- consistency precision around `0.19`
- consistency recall around `0.91`
- consistency F1 around `0.317`

This should be interpreted carefully.

Good news:

- the model is clearly stronger than the older evidence baseline
- the benchmark setup is no longer trivial or broken
- the evaluator has real discriminative ability

Bad news:

- precision is still low
- false positives are still too many
- future evidence is not yet dominating early representation

So the evaluator is useful, but still not strong enough to be the final story.

## 13. Current best understanding of the project

At this point, the project can support the following claims:

### 13.1 What we can say with confidence

- IAC is a meaningful benchmark direction for WAM evaluation
- strict-future setup is essential and is now in place
- frozen single-layer DINOv2 is a workable evaluator backbone
- ranking-aware and benchmark-aware selection improves results materially
- trajectory branches carry strong physics information
- consistency is still mainly realized in fused latent states

### 13.2 What we cannot yet honestly claim

- that future image evidence is already the dominant source of consistency judgment
- that multilayer DINO fusion is the core innovation
- that the current evaluator is high precision or close to solved
- that we have fully answered whether images are truly necessary in the strongest causal sense

## 14. What is currently worth discussing with collaborators

This is the part most relevant for group reflection.

### 14.1 Is the main research contribution benchmark-first or method-first?

There are two possible stories:

- benchmark-first:
  define a simulator-free, label-free, plug-and-play WAM evaluator
- method-first:
  propose a stronger consistency critic with future evidence and hierarchical constraints

The current codebase and results support the first story more strongly than the second.

### 14.2 What should "multilayer" mean in this project?

Current evidence suggests:

- meaningful layer analysis should focus on critic/WAM internal states
- not simply on DINO backbone depth

This matters because it changes what counts as a real innovation.

### 14.3 What do we actually need next?

The central unresolved issue is not backbone size.
It is:

- how to reduce false positives without losing all recall
- how to force future evidence to matter more directly
- how to make the evaluator causally sensitive to future perturbation

### 14.4 Should we continue optimizing the current critic, or pivot to richer diagnostics?

A reasonable split now would be:

- keep one stable best benchmark line
- continue one controlled training line for modest gains
- invest more into diagnostic and causal analysis

This may produce a better paper story than endlessly tuning one critic.

## 15. Concrete next-step options

If the team wants to keep pushing the current line, the most sensible next options are:

### Option A: Source-specific negative shaping

Instead of one global hard-negative pressure:

- assign stronger penalties to specific high-confusion source types
- for example `time_shift_future`, `traj_swap`, `perturb_speed`

Reason:

- the current false positives are probably not uniform
- global negative squeezing seems too blunt

### Option B: Better calibration without changing representation too much

- preserve the current best latent representation
- learn a better calibrated decision surface for consistency
- possibly with source-aware calibration or held-out calibration

Reason:

- current score separation exists
- but the model does not cleanly map separation to robust precision

### Option C: Stronger causal future-evidence tests

- explicit future perturbation sensitivity reports
- compare score movement under image perturbation versus trajectory perturbation
- test whether the model's decision is truly caused by future evidence

Reason:

- this would directly strengthen the benchmark story

### Option D: Cleaner WAM-level probing instead of more backbone complexity

- expand probe datasets
- compare image-layer vs trajectory-layer vs fusion-layer sensitivity
- produce explanation artifacts per model run

Reason:

- this aligns with what the project has actually learned so far

## 16. Recommended current baseline to keep

If the team wants one benchmark result to use as the current best mainline, keep:

- `work_dirs/iac_navsim_future_dinov2_hardneg_ddp4_continue`

Why:

- it has the best current 2048 balanced accuracy
- it finished cleanly
- it includes the newer training improvements

If the team wants the cleanest conceptual comparison point, keep alongside it:

- `work_dirs/iac_navsim_future_dinov2_metric_ddp4_continue`

Why:

- it is the clearest example of the checkpoint-selection improvement
- it is slightly simpler to explain than the hard-negative continuation

## 17. Bottom-line summary

The project has already achieved something meaningful:

- it turned IAC from a loose critic idea into a benchmark pipeline with audited data, reproducible training, formal benchmark outputs, and diagnostic tools
- it demonstrated that benchmark-aware training and selection can materially improve evaluator quality
- it showed that internal trajectory/fusion states currently carry most of the useful signal
- it produced a stronger current best result than earlier baselines

But the project has not yet solved the hardest problem:

- getting future-image evidence to produce a strong, precise, low-shortcut consistency judgment

So the honest summary is:

- benchmark direction: strong and meaningful
- critic quality: improving, but still not strong enough
- scientific insight: already useful
- final method claim: not settled yet

That is exactly the right moment to pause, summarize, and bring in other people to think with us.
