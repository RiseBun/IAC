# IAC-PathBench v3 Recovered-Set Progress - 2026-07-06

## 一句话结论

把 recovered path 从单一路径升级为 K-path set 是有效的：val minADE 从单路径 probe 的 2.189 降到 1.266；把 recovered-set support 加入 hybrid listwise scorer 后，holdout ambiguity-adjusted top1 从 0.930 提到 0.950，likely model error fraction 从 0.102 降到 0.070，同时 exact-path delta 仍保持 +0.0328 且 95% CI 下界大于 0。

## 本轮新增

新增工具：

- `tools/train_recovered_path_set_probe_from_features.py`
  - 从 frozen DINOv2 features 训练 K 条 recovered paths。
  - 使用 minADE@K / winner-take-best objective。
  - 修复了第一版无界 diversity loss 发散问题，改为 hinge diversity penalty。

- `tools/eval_recovered_path_set_agreement.py`
  - 对每个 row 从自己的 future image 恢复 K 条 path。
  - 用 `minADE(candidate, recovered_set)` 判断 candidate 是否被 future image 支持。
  - 输出 conformal support set、source 分布和 per-row recovered-set agreement。

## K-Path Set Probe 效果

训练数据：

- frozen DINOv2 motion-rich features
- train positive rows: 4000
- val positive rows: 1000
- K = 6

结果：

| Probe | Val Metric | Value |
|---|---:|---:|
| single-path MLP | ADE mean | 2.189 |
| K-path set | minADE mean | 1.266 |
| K-path set | minFDE mean | 2.386 |
| K-path set | minADE p50 | 0.805 |
| K-path set | minADE p90 | 2.756 |
| K-path set | minADE p95 | 4.217 |

解释：K-path set 明显缓解了“平均路径”问题，证明 future image 特征中存在多模态路径信息。

## Holdout Recovered-Set Diagnostic

在 holdout low-IoU g200 上，用 q=0.8 conformal radius：

- ambiguity radius: 2.869
- GT supported fraction: 0.800
- current winner supported fraction: 0.780
- mean ambiguity set size: 4.125

q sweep：

| q | radius | GT supported | winner supported | set size |
|---:|---:|---:|---:|---:|
| 0.5 | 1.148 | 0.500 | 0.475 | 2.160 |
| 0.6 | 1.394 | 0.600 | 0.560 | 2.670 |
| 0.7 | 1.965 | 0.700 | 0.675 | 3.385 |
| 0.8 | 2.869 | 0.800 | 0.780 | 4.125 |
| 0.9 | 4.232 | 0.900 | 0.880 | 4.900 |

Source-level minADE:

| Source | mean minADE | p50 | p90 |
|---|---:|---:|---:|
| gt_pos | 1.937 | 1.145 | 4.231 |
| perturb_heading | 1.939 | 1.144 | 4.299 |
| perturb_lateral | 2.106 | 1.390 | 4.206 |
| perturb_speed | 2.697 | 1.584 | 6.426 |
| time_shift_future | 2.865 | 1.974 | 6.631 |
| image_swap | 4.458 | 4.049 | 8.319 |
| traj_swap | 9.946 | 9.166 | 15.195 |

关键解释：

- `traj_swap` 被 recovered-set 明显排开，说明 recovered-set 对明显错误轨迹有强信号。
- `gt_pos` 和 `perturb_heading` 几乎重合，说明 heading 近邻在图像上确实弱可分。
- `perturb_lateral/speed` 介于二者之间，部分可分但不是稳定唯一可分。

## Hybrid Scorer 效果

使用特征：

```text
bias
main_logit
exact_path_delta
path_minus_sky_delta
recovered_set_agreement
recovered_set_supported
```

训练 split：low-IoU g200  
评估 split：regular / low-IoU / holdout

### Holdout low-IoU g200

| Method | hard top1 | ambiguity-adjusted top1 | MRR | exact-path delta | likely model error |
|---|---:|---:|---:|---:|---:|
| listwise strict v2 | 0.705 | 0.930 | 0.822 | +0.0328 | 0.102 |
| v3 recovered-set hybrid | 0.715 | 0.950 | 0.825 | +0.0328 | 0.070 |
| pointwise nopw | 0.725 | 0.765 | 0.825 | +0.0328 | 0.491 |

v3 recovered-set hybrid holdout bootstrap 95% CI:

- hard top1: 0.715, CI [0.645, 0.780]
- ambiguity-adjusted top1: 0.950, CI [0.915, 0.980]
- MRR: 0.825, CI [0.781, 0.866]
- exact-path delta: +0.0328, CI [+0.0268, +0.0387]
- likely model error fraction: 0.070, CI [0.016, 0.140]

## 科学判断

这轮结果支持一个更强的 benchmark 版本：

> IAC-PathBench v3 should evaluate whether future image supports a candidate path through exact-path causal evidence plus recovered-path conformal support, not through unique-GT hard top1 alone.

现在可以更有底气地说：

1. Future image 里确实存在可恢复路径信息。
2. K-path set 比单一路径更符合真实多解场景。
3. Recovered-set support 能减少 likely model error，同时不破坏 exact-path delta。
4. Hard top1 仍不是最合理主指标，因为 heading/lateral/speed 近邻在 recovered-set 下本身经常都被支持。

## 下一步

1. 把 recovered-set support 正式加入 `benchmark_wam.py` summary。
2. 用 q=0.8 或 q=0.9 作为 v3 默认 conformal ambiguity setting。
   - q=0.8：更严格，set size 4.125。
   - q=0.9：更保守，GT coverage 0.9，set size 4.9。
3. 训练更大的 feature cache，例如 20k positives，检查 minADE 是否继续下降。
4. 做 path heatmap head，解决 K-path 仍然需要离散模式数量的问题。

## 服务器结果位置

- K-path set probe:
  - `/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/recovered_path_set_probe_k6_stable`
- v3 hybrid scorer:
  - `/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/listwise_pathbench_v3_recovered_set_hybrid`
- v3 bootstrap:
  - `/mnt/slurmfs-4090node1/homes/zchen897/IAC/work_dirs/iac_navsim_future_dinov2_ambiguity_aware_3gpu_400/bootstrap_v2_v3_recovered_set_hybrid_holdout.json`
