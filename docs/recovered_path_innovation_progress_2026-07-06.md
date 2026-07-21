# Recovered-Path Innovation Progress - 2026-07-06

## 一句话结论

我们已经验证了一个新的方向：**future image 特征可以恢复出粗略未来路径**，但“单一路径点估计”不足以在 speed/lateral/heading 近邻里选唯一 GT。因此下一步创新不应是继续调 consistency calibrator，而应升级为 **多模态 recovered-path confidence set**：从 future image 预测一组可能路径/置信区域，再判断 candidate 是否落在支持集合内。

## 本轮做了什么

新增 5 个工具：

- `tools/eval_future_geometry_recovery.py`
  - 审计模型内置 `future_traj_geometry_pred` 是否能解释当前错误。
- `tools/train_recovered_path_probe.py`
  - 端到端训练 recovered-path probe。
- `tools/extract_recovered_path_features.py`
  - 缓存 frozen DINOv2 visual features，避免反复跑大模型。
- `tools/train_recovered_path_probe_from_features.py`
  - 基于缓存特征快速训练不同 recovered-path heads。
- `tools/eval_recovered_path_agreement.py`
  - 做 recover-then-compare：
    - `row_future`: 每个 row 从自己的 future image 恢复 path。
    - `group_gt_future`: 全组用 GT future 恢复 path，只用于近邻歧义诊断。

## 三个方向的实验结果

### 1. 内置 future geometry head 不够用

在 holdout low-IoU g200 上：

- current hard top1: 0.705
- future_geometry_top1: 0.030
- gt_geometry_error_lt_current_winner_frac: 0.160

结论：旧的 8 维 `future_traj_geometry_pred` 不是解决方案。它不是专门为 recovered path 训练的，不能作为 benchmark 主证据。

### 2. 新 recovered-path probe 有实质学习信号

用 frozen DINOv2 motion-rich features 训练 4000 个正样本，val 1000 个正样本：

- best val ADE: 2.189
- FDE: 3.879
- ADE p50: 1.584
- ADE p90: 4.800
- ADE p95: 6.401

对比 smoke probe：

- smoke ADE: 6.309
- cached MLP ADE: 2.189

结论：future image 特征里确实有可恢复路径信息。这是一个可作为论文创新点继续做大的方向。

### 3. 单一路径点估计不能直接替代 benchmark ranking

在 holdout low-IoU g200，row-wise recover-then-compare：

- current hard top1: 0.705
- recovered_path_top1: 0.185
- mean GT recovered ADE: 2.922
- mean current winner recovered ADE: 2.993
- GT ADE < current winner ADE fraction: 0.175
- ambiguity radius, q=0.90: 5.961
- mean ambiguity set size: 4.795

支持分类：

- hit: 141
- recovered_ambiguous_near_miss: 37
- recovered_prefers_gt: 16
- recovered_prefers_winner_or_error: 6

结论：点估计 path 可以解释一部分错误，但太“平均化”，不能在近邻中稳定选唯一 GT。

## 关键发现

按 source 看 recovered-path ADE：

- `gt_pos`: mean 2.922
- `perturb_lateral`: mean 3.051
- `perturb_speed`: mean 4.010
- `perturb_heading`: mean 2.931
- `traj_swap`: mean 12.625

这说明 recovered-path probe 对 `traj_swap` 这种明显错误非常敏感，但对 `heading/lateral` 近邻不够分辨。这个结果非常合理，也正好解释了为什么 hard top1 难：很多近邻在 future image 里本来就不强可观测。

## 对科学问题的回答

当前证据支持：

1. consistency/path evidence 不是纯 trajectory geometry shortcut。
   - exact-path delta 在 holdout 上稳定为正。
   - recovered-path probe 不输入 candidate trajectory，也能从 future image 特征恢复粗路径。

2. hard top1 低的根因不是模型完全不懂图像。
   - recovered probe 能强烈排斥 `traj_swap`。
   - 剩余主要难点是 speed/lateral/heading 近邻。

3. 单一 GT hard ranking 不是合理主指标。
   - 近邻轨迹往往都落入 recovered-path confidence region。
   - 应使用 ambiguity-aware / confidence-set evaluation。

## 下一步真正有创新的方案

### A. Multi-Modal Recovered Path Set

不要预测一条 path，预测 K 条 path 或 path distribution：

- 输出 K=6/8 个 recovered trajectories。
- 用 minADE(candidate, recovered_set) 评估支持性。
- ambiguity set = candidates whose minADE <= conformal radius。

这对应 trajectory forecasting 的 minADE@K，也对应 world model benchmark 的 recover-then-compare。

### B. Path Heatmap Evidence

不要只回归轨迹点，预测 BEV/path heatmap：

- future image features -> path occupancy heatmap。
- candidate trajectory 投影到 heatmap 上取平均支持值。
- 这样比点估计更适合处理天空/路面小差异和视觉模糊。

### C. Source-Agnostic Conformal Ambiguity

不再用 `perturb_speed/lateral/heading` 手写规则：

- 在 validation positive rows 上校准 recovered-path error radius。
- candidate 如果落入 confidence set，则是 image-supported。
- winner 和 GT 同时被支持则判 ambiguity。
- winner 不被支持但胜出则判 model error。

这会让 benchmark 从“人工近邻规则”升级到“数据校准的可支持集合”。

### D. Hybrid Scoring

最终 score 不应只靠 consistency logit：

```text
score = IAC consistency
      + exact-path evidence
      + recovered-path support(candidate | future image)
```

但 recovered-path support 必须是 set/heatmap，不是当前单一路径点估计。

## 当前判断

最有价值的创新路线是：

> IAC-PathBench v3 = exact-path causal masking + recover-then-compare + conformal ambiguity set.

这比继续追 hard top1 更科学，也更接近 ACT-Bench / ReSim / iWorld-Bench 的方向。

## 服务器结果位置

- Feature cache:
  - `/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/recovered_path_feature_cache_4k`
- Best recovered-path probe:
  - `/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/recovered_path_probe_mlp2048_shape`
- Built-in geometry audit:
  - `/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/future_geometry_recovery_holdout_listwise_strict_summary.json`
