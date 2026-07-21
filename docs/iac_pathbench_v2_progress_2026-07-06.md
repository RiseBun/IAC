# IAC-PathBench v2 Progress - 2026-07-06

## 一句话结论

今天推进的核心不是继续训练大模型，而是把评测/校准目标从“逐样本二分类”推进到“组内排序 + 歧义感知”。结果证明：IAC 的 exact-path 证据在 holdout 上稳定为正，但 hard top1 与科学主张不是同一个目标；listwise/tie-aware 版本能把 holdout ambiguity-adjusted top1 提到 0.930，并把 likely model error fraction 降到 0.102，同时保持 exact-path delta 95% CI 下界大于 0。

## 当前 Pipeline

1. DINOv2 IAC critic 输入 history images、future images、ego state、candidate trajectory。
2. 输出两类分数：
   - `iac_consistency`: future image 是否支持 candidate trajectory。
   - `path_evidence_head`: 更偏向路径区域的辅助证据。
3. `benchmark_wam.py` 在每个 candidate group 内排序，并计算 causal mask 指标：
   - path mask delta。
   - sky mask delta。
   - candidate-vs-wrong exact path delta。
4. IAC-PathBench v2 不再只看 hard top1，而是同时报告：
   - exact path win fraction / delta。
   - path-minus-sky delta。
   - ambiguity-adjusted top1。
   - hard top1 / MRR。
   - likely model error fraction。

## 本轮新增

新增工具：

- `tools/train_iac_pathbench_v2_listwise_calibrator.py`
  - 用 low-IoU g200 训练一个轻量 listwise calibrator。
  - 训练目标是 candidate group 内 softmax 排序，不是逐行 BCE。
  - GT 是主目标；`perturb_speed/lateral/heading` 近邻可拿少量 soft target；`image_swap/time_shift/traj_swap/reverse_traj` 保持 hard negative。
  - 默认打分特征不使用 `source_type`，避免 benchmark-construction leakage。

- `tools/bootstrap_iac_pathbench_v2.py`
  - 按 candidate group bootstrap。
  - 输出 v2 主指标的 95% CI，包括 ambiguity-adjusted top1 和 likely model error fraction。

## 关键结果

### Holdout low-IoU g200

| 方法 | hard top1 | ambiguity-adjusted top1 | MRR | exact-path win | exact-path delta | likely model error |
|---|---:|---:|---:|---:|---:|---:|
| pointwise nopw calibrator | 0.725 | 0.765 | 0.825 | 0.805 | +0.0328 | 0.491 |
| listwise strict calibrator | 0.705 | 0.930 | 0.822 | 0.805 | +0.0328 | 0.102 |

Listwise strict 的 holdout 95% CI：

- hard top1: 0.705, CI [0.635, 0.775]
- ambiguity-adjusted top1: 0.930, CI [0.895, 0.965]
- exact-path delta: +0.0328, CI [+0.0268, +0.0387]
- likely model error fraction: 0.102, CI [0.033, 0.180]

Pointwise nopw 的 holdout 95% CI：

- hard top1: 0.725, CI [0.660, 0.785]
- ambiguity-adjusted top1: 0.765, CI [0.705, 0.820]
- exact-path delta: +0.0328, CI [+0.0268, +0.0387]
- likely model error fraction: 0.491, CI [0.363, 0.618]

## 怎么解释

第一性原理上，我们要判断的是：future image 是否支持 candidate path。这个问题不等价于“在多个视觉几乎不可区分的速度/横向/航向近邻里强迫选唯一 GT”。

当前数据说明：

1. exact-path evidence 是稳定的。
   - holdout exact-path win fraction = 0.805。
   - exact-path delta CI 下界 > 0。
   - 这支持“score 被 future image 中路径区域驱动”，不是纯 trajectory geometry shortcut。

2. hard top1 仍有限，不是因为模型完全看不懂图像。
   - 大量剩余 miss 是 `perturb_speed/lateral/heading` 近邻。
   - listwise 后 close miss rate 仍很高，说明很多错例其实是小分差近邻。

3. pointwise 和 listwise 对应两个不同目标。
   - pointwise 更擅长把 GT 拉到第一，所以 hard top1 最高。
   - listwise/tie-aware 更符合 v2 benchmark 科学主张：把真正明显错误排低，把不可判别近邻归为 ambiguity。

## 当前能不能作为 benchmark

可以，但应该明确称为 **IAC-PathBench v2 ambiguity-aware benchmark**，不能再把 hard top1 当唯一核心指标。

可成立的主张：

- 该 benchmark 能评估 future image 是否支持 candidate path。
- 该 benchmark 能区分明显错误 future/trajectory 与视觉合理近邻。
- exact-path causal masking 提供了路径区域驱动证据。

还不能过度声称：

- 不能声称模型已经能完美恢复唯一 GT trajectory。
- 不能用 hard top1 单独代表 benchmark 成败。
- 不能说所有 speed/lateral/heading 近邻都是真歧义；还需要 inverse-motion/recover-then-compare 进一步证明。

## 下一步

最短路径不是继续调线性校准，而是补强 exact-path binding：

1. 做 recover-then-compare diagnostic。
   - 从 future frames / path evidence map 恢复一条 ego path proxy。
   - 比较 GT、winner、near-neighbor 与 recovered path 的距离。
   - 目标是把“真歧义”与“模型没分清”分开。

2. 做 failure stratification。
   - 按 straight / turn / lane change / low-speed / high-curvature 分组。
   - 看 hard top1 低到底集中在哪类运动。

3. 如果 recover proxy 证明近邻并非真歧义，再训练一个小型 inverse-motion head。
   - 输入 future visual/path evidence features。
   - 输出 recovered trajectory 或 path heatmap。
   - 用它作为第二路 exact-path binding score，而不是继续加大 DINOv2 critic。

## 文件与结果位置

- 新工具：
  - `tools/train_iac_pathbench_v2_listwise_calibrator.py`
  - `tools/bootstrap_iac_pathbench_v2.py`
- 服务器结果：
  - `/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/listwise_pathbench_v2_calibrator_strict_near005_margin02`
  - `/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/bootstrap_v2_list_strict_holdout.json`
- 本地同步结果：
  - `docs/iac_pathbench_v2_progress_2026-07-06_assets/`
