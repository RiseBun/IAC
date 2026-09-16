# 纵向视觉通道真实 NAVSIM 审计

状态：完成；不修改冻结视觉后端和阈值。

## 数据与问题

审计使用 95 条 source-disjoint NAVSIM logged-realized 序列，共 380 个一秒区间。
输入是已经保存的 Metric3Dv2-v2-S、静态对应、Reloc3r rotation 和 known-R PnP
结果，因此本次是离线重算，不重新拟合尺度，也不读取 WAM action 作为视觉输入。

本次回答：视觉层是否能读取粗粒度纵向进度、现有质量门是否有效，以及冻结容差是否
过宽。它不验证生成域的米制深度准确率。

## 总体结果

| 指标 | 结果 |
|---|---:|
| 序列 / 区间 | 95 / 380 |
| 几何可测区间 | 335 |
| score coverage | **88.2%** |
| MAE | 1.686 m |
| 中位绝对误差 | **0.865 m** |
| 距离排序 Spearman | **0.665** |
| 精确距离档位准确率 | **58.5%** |
| 相同或相邻档准确率 | **89.6%** |
| 冻结接受率（可测区间） | **90.1%**，source-bootstrap 95% CI `[85.2%,94.5%]` |
| 冻结 effective rate（全部区间） | 79.5% |

45 个弃权区间中，44 个由 inlier fraction 低于 0.10 触发。因而主要瓶颈不是
SegFormer 道路掩码本身，而是跨帧静态对应经过几何验证后缺少足够内点。

## 距离分层

| 真实档位 | 可测区间 | 精确档 | 相邻档 | 中位误差 |
|---|---:|---:|---:|---:|
| stop | 13 | 100.0% | 100.0% | 0.129 m |
| very short | 97 | 86.6% | 96.9% | 0.404 m |
| short | 129 | 58.1% | 96.9% | 0.887 m |
| medium | 72 | 27.8% | 76.4% | 2.159 m |
| long | 24 | 16.7% | 54.2% | 4.307 m |

视觉层在 stop/very-short/short 上有效，但对 medium/long 存在明显低估和档位压缩。
因此它只支持 ordinal progress 和宽容差一致性，不能支持精确米制距离或速度。

## 场景分层

| NAVSIM stratum | coverage | 精确档 | 相邻档 | 中位误差 |
|---|---:|---:|---:|---:|
| acceleration | 95.0% | 64.5% | 93.4% | 0.694 m |
| braking | 98.8% | 53.0% | 88.0% | 1.337 m |
| lateral turn | 77.8% | 61.4% | 93.6% | 0.710 m |
| stop | 100.0% | 100.0% | 100.0% | 0.048 m |
| straight cruise | 100.0% | 40.6% | 65.6% | 2.707 m |

转弯场景的主要问题是覆盖率；straight cruise 的主要问题是距离尺度。二者不能用同一
失败原因解释。

## 容差敏感性

| 规则 | 可测区间接受率 | 全部区间 effective rate |
|---|---:|---:|
| exact bin only | 58.5% | 51.6% |
| 1 m / 25%，不允许相邻档 | 68.7% | 60.5% |
| 2 m / 33% / 相邻一档 | 89.6% | 78.9% |
| **冻结：3 m / 50% / 相邻一档** | **90.1%** | **79.5%** |
| 4 m / 50% / 相邻一档 | 91.3% | 80.5% |

冻结规则相对 `2 m / 33% / 相邻一档` 仅提高 0.6 个百分点。当前高接受率主要来自
“相邻一档”，而不是 3 m/50% 容差放宽。因此无需重新选择容差，但论文必须同时报告
精确档位与相邻档结果，避免把 90.1% 描述成米制距离准确率。

## 质量门

将最小 inlier fraction 从 0.05 提高到 0.40，coverage 从 98.4% 降到 17.6%，
中位误差从 0.92 m 降到 0.25 m。冻结门槛 0.10 对应 88.2% coverage 和 0.865 m
中位误差，是合理的 risk--coverage 折中；继续收紧会主要损失覆盖率。

## 决策

1. 保留 `Metric3D + static matches + Reloc3r rotation + known-R PnP`，不更换后端。
2. 冻结输出名称为 `coarse longitudinal progress`。
3. 正式报告必须包含 exact-bin、adjacent-bin、coverage、MAE、Spearman 和分层结果。
4. raw metre/speed 继续保持 diagnostic-only。
5. 下一步不再扩大同分布真实样本；应解决跨域问题：用同一 source pool 比较
   logged-real 与 WAM 生成帧的几何 coverage 和 failure reasons。

机器可读结果：`reports/longitudinal_real_audit_20260916.json`。
