# IAC 视觉层与 Alignment Score

状态：冻结于 2026-09-15
协议：`iac-visual-output-layer-v1`、`iac-history-conditioned-as-v1.1`

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
距离档位相同/相邻，即视为粗粒度一致。原始米制值只用于诊断。

## 4. AS 评分

条件诊断分保留用于定位问题：

- `AS-yaw`：明确转弯样本上的左右方向正确率；
- `AS-progress`：至少存在一个质量合格区间的样本上，逐样本等权的纵向接受率；
- `AS-composite`：旧的可观测分量平均，仅作诊断，不得作为主分数。

正式单分数把覆盖率纳入结果：

```text
P_eff = acceptable longitudinal intervals / (4 × declared rows)
Y_eff = correct yaw directions / all applicable turn rows
AS    = 100 × sqrt(P_eff × Y_eff)
```

不可测纵向区间和缺失视觉 yaw 在正式 AS 中不获得分数，但仍通过 coverage 和原因字段
区分“读不出来”与“读出但不一致”。几何平均保证转向或纵向任一通道失效时，另一通道
不能完全掩盖它。

## 5. 当前结果

视觉读取器先在真实 NAVSIM 上独立确认：Reloc3r-512 的三分类准确率为 91.7%，
明确转弯方向准确率为 97.5%（95% CI `[91.3%, 99.3%]`），重复帧控制 0/32
误报转弯。Metric3D known-R PnP 的早期真实 24 区间中，19 个通过冻结宽容差；
正式 AS 仍按全量 coverage 对纵向通道降分。

| 条件 | N | 有效纵向区间覆盖 | `P_eff` | `Y_eff` | `AS / 100` | 状态 |
|---|---:|---:|---:|---:|---:|---|
| NAVSIM logged-realized | 95 | 88.2% | 0.795 | 1.000 | **89.1** | 视觉读取参考上限 |
| DriveWAM lineage-fixed | 1490 | 48.4% | 0.235 | 0.935 | **46.8** | 正式 joint-output AS |
| DriveWAM controlled-pair | 348 | 68.2% | 0.562 | 0.970 | **73.8** | 观察性控制 |
| WorldDrive | 10 | 52.5% | 0.475 | 1.000 | **68.9** | 小样本探索，不作排名 |

DriveWAM 的 `AS-yaw=0.935` 表明生成帧中存在稳定可读的转向方向；最终 AS 较低的
主因是纵向运动只有 1399/5960 个应测区间同时可测且落入宽容差。

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
