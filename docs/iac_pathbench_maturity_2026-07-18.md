# IAC-PathBench 成熟化进展 - 2026-07-18

## 结论

IAC-PathBench 现在已经从“单模型实验日志”推进到“benchmark candidate”：

- 数据覆盖已审计：三个核心 split 都是严格 future-frame，无 history/future 泄漏，并覆盖全部 7 类 GT/反事实来源。
- 协议已冻结：正式使用 `hit / ambiguous_accept / evidence_supported_miss / likely_model_error` 四类错误解释，hard top1 只作为辅助指标。
- 跨模型验证已开始：CNN backbone 在同协议下几乎没有 exact-path/path-minus-sky 信号，DINOv2 vNext 明显更强，说明协议有区分力。

还不能称为最终 public benchmark，因为仍缺更大的 frozen test split 和更多模型族，但现在已经具备正式 benchmark 的骨架。

## 冻结协议

当前冻结口径：

- 主协议：`IAC-PathBench v2 ambiguity-aware`
- 默认融合：`consistency_logit + 0.2 * path_evidence_logit`，logit-space fusion
- 主指标：
  - `exact_path_delta`
  - `path_minus_sky_delta`
  - `ambiguity_adjusted_top1`
  - `likely_model_error_fraction`
- 辅助指标：
  - hard `top1`
  - `MRR`
  - `best_balanced_accuracy`
- 正式分类：
  - `hit`
  - `ambiguous_accept`
  - `evidence_supported_miss`
  - `likely_model_error`

新增协议校验工具：

```bash
python tools/validate_iac_pathbench_protocol.py path/to/wam_iac_summary.json
```

这一步保证 summary 里必须包含冻结类别、主指标、辅助指标和 diagnostic 字段。

## 数据覆盖

新增覆盖审计能力在：

```bash
python tools/audit_consistency_index.py \
  indices_navsim_future/consistency_val.jsonl \
  indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl \
  indices_navsim_future/diagnostics/consistency_val_low_iou_g200_holdout_rank200_399.jsonl \
  --json-out work_dirs/iac_pathbench_maturity_2026_07_18/data_coverage_audit.json \
  --fail-positive-exact-overlap 0.01 \
  --fail-positive-any-overlap 0.05 \
  --fail-missing-required-source
```

服务器审计结果：

| split | rows | groups | group size | required sources | positive overlap |
|---|---:|---:|---:|---|---:|
| regular val | 85708 | 12244 | 7 | all present | 0.0000 |
| low-IoU g200 | 1400 | 200 | 7 | all present | 0.0000 |
| holdout low-IoU g200 | 1400 | 200 | 7 | all present | 0.0000 |

每个 g200 split 都包含：

- `gt_pos`: 200
- `image_swap`: 200
- `time_shift_future`: 200
- `traj_swap`: 200
- `perturb_lateral`: 200
- `perturb_heading`: 200
- `perturb_speed`: 200

这说明当前问题不是“某类反事实没覆盖”，而是 benchmark 规模还小、模型族还少。

## 跨模型验证

新增跨模型对照工具：

```bash
python tools/compare_iac_pathbench_models.py \
  --summary label_a=path/to/a/wam_iac_summary.json \
  --summary label_b=path/to/b/wam_iac_summary.json \
  --markdown-out comparison.md
```

新增服务器 runner：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
RUN_CNN=1 \
RUN_DINOV2=0 \
BOOTSTRAP_SAMPLES=500 \
scripts/run_iac_pathbench_cross_backbone.sh
```

本轮已跑 CNN 3k checkpoint 作为不同 backbone 对照：

- checkpoint: `work_dirs/iac_navsim_future_cnn_3k/checkpoints/best.pth`
- output: `work_dirs/iac_pathbench_cross_backbone_2026_07_18/cnn_3k/`

### Holdout Low-IoU g200

| model | hard top1 | MRR | exact-path delta | path-minus-sky | ambiguity-adjusted top1 | likely model error | raw miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN 3k | 0.035 | 0.307 | +0.0000 | +0.0003 | 0.140 | 0.466 | 0.965 |
| DINOv2 vNext main | 0.425 | 0.672 | +0.0001 | +0.0898 | 0.745 | 0.287 | 0.575 |
| DINOv2 vNext fused alpha=0.2 | 0.420 | 0.672 | +0.0099 | +0.1025 | 0.775 | 0.164 | 0.580 |

fused alpha=0.2 holdout bootstrap 95% CI:

- hard top1: `0.420`, CI `[0.352, 0.485]`
- MRR: `0.672`, CI `[0.634, 0.710]`
- exact-path delta: `+0.0099`, CI `[+0.0021, +0.0166]`
- path-minus-sky delta: `+0.1025`, CI `[+0.0940, +0.1100]`
- ambiguity-adjusted top1: `0.775`, CI `[0.725, 0.828]`
- likely model error fraction: `0.164`, CI `[0.100, 0.229]`

### Regular g200

| model | hard top1 | MRR | exact-path delta | path-minus-sky | ambiguity-adjusted top1 | likely model error | raw miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN 3k | 0.030 | 0.305 | -0.0000 | +0.0003 | 0.140 | 0.443 | 0.970 |
| DINOv2 vNext main | 0.400 | 0.668 | -0.0037 | +0.0558 | 0.590 | 0.342 | 0.600 |
| DINOv2 vNext fused alpha=0.2 | 0.405 | 0.673 | +0.0049 | +0.0852 | 0.680 | 0.252 | 0.595 |

## 科学解释

这次跨 backbone 验证很关键：

1. CNN 不是简单“分数低一点”，而是几乎没有路径因果信号。
   - exact-path delta 约等于 0。
   - path-minus-sky delta 约等于 0。
   - hard top1 接近 0。
2. DINOv2 main 能做 ranking，但 exact-path delta 在 holdout 上接近 0。
3. DINOv2 fused alpha=0.2 保住 ranking，同时显著增强 path-grounded 证书。

所以 benchmark 的主张应该是：

> IAC-PathBench 评估 future image 是否支持 candidate path；它能区分只有粗糙图像一致性能力的模型和真正具备路径证据敏感性的模型。

## 还差什么

成熟 benchmark 的下一步不是继续调一个模型，而是补外部有效性：

1. 扩大 frozen test
   - 当前 g200 可以作为快速报告 split。
   - 论文级 benchmark 应增加 g1000 或更大的 frozen test。

2. 再加一个强模型族
   - CNN 作为弱 backbone 对照已经足够证明协议有区分力。
   - 还需要一个强 backbone/checkpoint family，例如 DINOv2 多层、ViT-B/14、或者一个不同训练目标的 strong checkpoint。

3. 固定 alpha 盲测
   - `alpha=0.2` 只能由 tune split 选择。
   - 后续 holdout/test 只报告一次，不再 sweep。

4. 出正式报告
   - 每个 split 固定输出：
     `top1 / MRR / exact_path_delta / path_minus_sky_delta /
     ambiguity_adjusted_top1 / likely_model_error_fraction / 95% CI`
   - 每个模型族必须先通过 `validate_iac_pathbench_protocol.py`。

## 服务器产物

- data coverage:
  `work_dirs/iac_pathbench_maturity_2026_07_18/data_coverage_audit.json`
- CNN cross-backbone:
  `work_dirs/iac_pathbench_cross_backbone_2026_07_18/cnn_3k/`
- DINOv2 fused alpha=0.2:
  `work_dirs/iac_pathbench_maturity_2026_07_18/dinov2_vnext_fused_alpha0p2/`
- final comparisons:
  `work_dirs/iac_pathbench_maturity_2026_07_18/cross_model_holdout_final_comparison.md`
  `work_dirs/iac_pathbench_maturity_2026_07_18/cross_model_regular_final_comparison.md`

## Next step

The current judge is good enough to distinguish CNN from DINOv2, but the raw
consistency surface is still too weak to be the only scientific certificate.
The next implementation step is:

1. Train a second-stage fused judge that lightly injects `path_evidence_logit`
   into `consistency_logit`.
2. Validate that fused judge on the frozen g1000 low-IoU splits, not only g200.
3. Keep the protocol frozen while comparing multiple backbone families.

## 2026-07-18 follow-up execution plan

The four current blockers are handled as separate tracks:

- Evidence head too weak:
  run `configs/train_navsim_future_dinov2_path_evidence_stage1_strong_vnext.py`.
  This freezes the main critic and trains segmented path evidence with stronger
  exact-path and candidate-vs-wrong-path pressure.
- Main score not stably using evidence:
  run `configs/train_navsim_future_dinov2_path_evidence_fused_vnext.py` from the
  stage-1 checkpoint. This lightly mixes `path_evidence_logit` into
  `consistency_logit` and fine-tunes only the small judge heads.
- Larger frozen validation:
  run `scripts/run_iac_pathbench_g1000.sh` on low-IoU g1000 and holdout g1000,
  reporting both raw and path-evidence scores plus fused sweeps.
- More backbone families:
  keep CNN 3k as the weak-control family and add at least one stronger DINOv2
  checkpoint family under the same frozen protocol.

Reproducible entrypoint:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 scripts/run_path_evidence_two_stage_vnext.sh
```

## Two-stage result snapshot

Stage 1 evidence-only strengthening was run from
`work_dirs/iac_navsim_future_dinov2_path_evidence_vnext/checkpoints/latest.pth`
for one controlled epoch:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
RUN_STAGE1=1 RUN_STAGE2=0 RUN_EVAL=0 \
STAGE1_EPOCHS=9 STAGE1_MAX_TRAIN_STEPS=400 STAGE1_MAX_VAL_STEPS=100 \
scripts/run_path_evidence_two_stage_vnext.sh
```

Stage 1 `path_evidence_logit` g200 results:

| split | top1 | MRR | exact-path delta | path-minus-sky | ambiguity-adjusted top1 | likely model error |
|---|---:|---:|---:|---:|---:|---:|
| regular | 0.080 | 0.373 | +0.0178 | +0.0872 | 0.230 | 0.332 |
| low-IoU | 0.090 | 0.391 | +0.0364 | +0.0799 | 0.325 | 0.099 |
| holdout low-IoU | 0.090 | 0.381 | +0.0223 | +0.0812 | 0.325 | 0.209 |

This confirms that the certificate head became more path-sensitive, but it is
still not a good standalone ranker.

Stage 2 fused-judge training was then run from the stage-1 `latest.pth` for one
controlled epoch. Raw `consistency_logit` did not materially improve on its own,
but logit-space fusion with the stronger evidence head improved exact-path
evidence while mostly preserving ranking:

| split | alpha | top1 | MRR | exact-path delta | path-minus-sky | exact-path win frac |
|---|---:|---:|---:|---:|---:|---:|
| regular | 0.2 | 0.415 | 0.677 | +0.0044 | +0.0808 | 0.500 |
| low-IoU | 0.2 | 0.345 | 0.616 | +0.0103 | +0.1015 | 0.545 |
| holdout low-IoU | 0.1 | 0.430 | 0.676 | +0.0047 | +0.0955 | 0.510 |
| holdout low-IoU | 0.2 | 0.425 | 0.675 | +0.0086 | +0.0979 | 0.570 |
| holdout low-IoU | 0.5 | 0.400 | 0.659 | +0.0158 | +0.0894 | 0.630 |

Current interpretation: the evidence head can be strengthened, and the main
score can use it through controlled fusion, but aggressive fusion trades ranking
for stronger causal evidence. The next decisive check is the frozen g1000 run:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
RUN_CNN=0 RUN_DINOV2=1 RUN_BOOTSTRAP=0 BENCH_MAX_GROUPS=1000 \
OUT_ROOT=work_dirs/iac_pathbench_g1000_stage2_2026_07_18 \
DINO_CONFIG=configs/train_navsim_future_dinov2_path_evidence_fused_vnext.py \
DINO_CKPT=work_dirs/iac_navsim_future_dinov2_path_evidence_fused_vnext/checkpoints/best.pth \
scripts/run_iac_pathbench_g1000.sh
```
