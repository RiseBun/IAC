# IAC 视觉层与 Alignment Score

状态：冻结于 2026-09-15
协议：`iac-visual-output-layer-v1`、`iac-history-conditioned-as-v1.2`

## 1. 评测问题

AS 只回答一个问题：

> WAM 生成未来图像中可观察到的自车运动，是否与同一次输出的轨迹一致？

它不评判画面是否逼真，也不证明轨迹因果控制了视频生成。现实接近程度由 GS
评测；改变轨迹后画面是否响应由 RCS 评测。

## 2. 固定输入

每条样本必须包含：

```text
4 history frames: -1.5, -1.0, -0.5, 0.0 s
4 future frames:   1.0, 2.0, 3.0, 4.0 s
4 trajectory poses at the future timestamps, relative to t=0 ego
camera intrinsics and camera_to_ego
```

历史最后一帧与四张未来帧形成四个可观察的 1 秒区间。视觉分支只读取图像；
轨迹在视觉运动提取完成后才参与比较。

## 3. 冻结视觉层

```text
frames
  ├─ Reloc3r-512 ───────────────→ relative rotation / signed yaw
  └─ Metric3Dv2-v2-S depth
       + static feature matches
       + Reloc3r fixed rotation ─→ known-R PnP longitudinal motion
                                  ↓
                         geometry quality gate
```

- Reloc3r-512：读取左/直/右和累计转向方向；不使用其平移模长作为距离。
- Metric3Dv2-v2-S：为第一帧像素提供米制深度，静态跨帧对应经反投影后使用
  known-rotation PnP 求相机平移。
- SegFormer：只提供道路/动态区域辅助掩码。道路纹理不足，所以它不是唯一匹配区域，
  也不能单独否决全图静态背景估计。
- 质量门：`matches >= 40`、`inlier_fraction >= 0.10`、
  `median reprojection error <= 3 px`；不足时输出 `uncertain`。

视觉层正式输出：

```text
yaw: left / straight / right
progress: stop / very_short / short / medium / long / uncertain
raw yaw/distance, confidence, coverage, abstention reason
```

纵向结果采用宽容差：误差不超过 `max(3 m, 参考距离的 50%)`，或预测与参考
距离档位相同/相邻，即视为粗粒度一致。原始米制值只用于诊断。95 条真实 NAVSIM
序列（380 个区间）的独立审计见
[`reports/longitudinal_real_audit_20260916.md`](../reports/longitudinal_real_audit_20260916.md)。

## 4. AS 评分

条件诊断分保留用于定位问题：

- `AS-yaw`：明确转弯样本上的左右方向正确率；
- `AS-progress`：至少存在一个质量合格区间的样本上，逐样本等权的纵向接受率；
- `AS-composite`：旧的可观测分量平均，仅作诊断，不得作为主分数。

正式报告把“一致性”和“可测性”分开：

```text
P_cond = acceptable progress intervals / quality-scored progress intervals
Y_cond = correct yaw directions / visually scored applicable turns
AS_conditional = 100 × sqrt(P_cond × Y_cond)

C_progress = quality-scored progress intervals / all expected intervals
C_yaw      = visually scored turns / all applicable turns
AS_observability = sqrt(C_progress × C_yaw)

AS_deployment = AS_conditional × AS_observability
              = 100 × sqrt(P_eff × Y_eff)
```

`AS_conditional` 是主要一致性估计；两个 channel coverage 界定其证据范围。
`AS_deployment` 保留原 v1.1 单分数，表示一致性与可观测性的联合部署摘要，但禁止
脱离 conditional score、两个 coverage、状态计数和弃权原因单独报告。由于 progress
和 yaw 的统计单位不同，`AS_observability` 仅为摘要，不能替代两个原始 coverage。

## 5. 当前结果

视觉读取器先在真实 NAVSIM 上独立确认：Reloc3r-512 的三分类准确率为 91.7%，
明确转弯方向准确率为 97.5%（95% CI `[91.3%, 99.3%]`），重复帧控制 0/32
误报转弯。纵向真实审计的 score coverage 为 88.2%、中位绝对误差 0.865 m、
Spearman 0.665、精确档位准确率 58.5%、相邻档准确率 89.6%。因此纵向通道
冻结为 coarse ordinal progress，而非米制距离/速度。

| 条件 | N | progress conditional | progress coverage | yaw conditional | `AS_conditional` | deployment AS |
|---|---:|---:|---:|---:|---:|---:|
| NAVSIM logged-realized | 95 | 90.1% | 88.2% | 100.0% | **94.9** | 89.1 |
| DriveWAM lineage-fixed | 1490 | 48.5% | 48.4% | 93.5% | **67.4** | 46.8 |
| DriveWAM controlled-pair | 348 | 82.4% | 68.2% | 97.0% | **89.4** | 73.8 |
| WorldDrive | 10 | 90.5% | 52.5% | 100.0% | **95.1** | 68.9 |

DriveWAM 的 `AS-yaw=0.935` 表明生成帧中存在稳定可读的转向方向。纵向通道在
2884 个可测区间中接受 1399 个，即 conditional progress 为 48.5%；同时只有
2884/5960=48.4% 的应测区间通过几何门。因此 deployment AS 的 46.8 同时反映
纵向不一致和低可测性，不能被单独解释为 WAM 的纯一致性分数。

旧 DriveWAM 清单曾把 round-robin shard 当成连续 offset，造成历史与未来错配。
正式构建器现在从 WAM 实际消费的 source pickle 读取不可变 `metadata.source_key`，
lineage 审计已由失败恢复为通过。旧清单及其分数只能用作回归案例。

## 6. 可报告边界

可以报告：粗粒度纵向进度、左右转向方向、覆盖率、拒答原因和轨迹—图像观察性一致性。

不能报告：精确米制速度/距离、完整 SE(3) 轨迹、物理真实性，或“轨迹导致图像运动”。

## 7. 最小代码入口

| 文件 | 用途 |
|---|---|
| `configs/visual_output_layer_v1.json` | 视觉输出、容差和质量门 |
| `configs/as_history_conditioned_v1.json` | 4+4+4 输入与 AS 聚合协议 |
| `src/iac_new/frame_matching.py` | manifest、内参缩放和静态帧匹配 |
| `src/iac_new/longitudinal_probe.py` | road-plane / metric-depth PnP 几何 |
| `src/iac_new/visual_output_layer.py` | 冻结的粗粒度输出和接受规则 |
| `src/iac_new/history_conditioned_as.py` | 单样本与数据集 AS |
| `tools/run_metric3d_longitudinal_probe.py` | 一次生成四区间候选盲视觉 probe |
| `tools/run_history_conditioned_as.py` | 将视觉 probe 与轨迹独立连接并评分 |
| `reproduction/drivewam/build_ccfc_manifest.py` | DriveWAM source-key 安全连接 |
| `reports/as_release_20260915.json` | 紧凑机器可读结果 |

最小运行方式：

```text
python tools/run_metric3d_longitudinal_probe.py \
  --manifest benchmark_v3.jsonl --output visual_probe.json \
  --metric3d-root <Metric3D代码目录> --metric3d-checkpoint <权重> \
  --reloc3r-root <Reloc3r代码目录> --resolution 512

python tools/run_history_conditioned_as.py \
  --manifest benchmark_v3.jsonl --probe visual_probe.json \
  --mode wam --output as_result.json
```

模型权重、第三方仓库、NAVSIM 图像和逐样本私有 manifest 不提交到本仓库；运行器
通过显式参数接收它们，避免硬编码本机路径。
