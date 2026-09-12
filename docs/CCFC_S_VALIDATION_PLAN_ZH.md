# RCS-yaw 验证计划（v1，旧名 CCFC-S）

## 目标

验证结构版反事实指标 `RCS-yaw`（旧名 `CCFC-S`）是否能在不恢复米制轨迹的前提下，可靠识别同一
干预造成的“未来结构变化 ↔ native action 变化”。本计划验证的是 action–future
structural consistency，不直接声称 future-to-action 因果，也不用于 logged-GT
几何保真度排名。

## 固定测量

Step 1 使用冻结的 S1.3：候选盲 RAFT-Large、真实域可靠性门和
`horizontal_flow_center`。测量输出不得读取候选轨迹、logged future state 或 PDM
分数。结构差分按同一 `source_key` 的左右分支计算：

```text
ΔS_F(t) = S_F(left, t) − S_F(right, t)
```

原始差分用于方向；共同运动归一化差分只作幅度诊断。缺少最低共同 interval 的
pair 标记 `unavailable`，不零填充。

## 确认集要求

- 校准集、确认集按 twin 原子拆分，不共享 source、scene 或 exact image；
- 每个 twin 固定 history、command、时间轴、标定、模型版本、随机种子和 nuisance；
- 至少两个 WAM 分别报告，不把自然质量差异当成有效性证明；
- 目标统计单位为 twin/source，不是共享 source 的 interval。

## 必做控制

1. **正常顺序：** 保留左/右图像和对应 action，方向应与干预一致；
2. **顺序反转：** 只反转图像差分方向，准确率和 Spearman 应显著下降；
3. **身份错配：** 左图像配右 action、右图像配左 action，应接近随机或负相关；
4. **零差异：** 两侧结构描述相同，方向和 Spearman 按契约为 unavailable，不能被
   解释成模型失败或填成 0 分。

脚本：

```bash
PYTHONPATH=src:. python scripts/validate_counterfactual_flow_delta.py \
  --measurement <left-and-right-measurements.jsonl> \
  --manifest <paired-action-manifest.jsonl> \
  --descriptor horizontal_flow_center \
  --action-reference endpoint_column \
  --action-column 2 \
  --minimum-action-delta 0.01 \
  --output <ccfc-s-controls.json>
```

## Promotion gate

配置 [`configs/flow_structure_counterfactual_delta_v1.json`](../configs/flow_structure_counterfactual_delta_v1.json)
中预注册的门槛为：

- twin coverage ≥ `0.90`；
- 方向准确率 95% CI 下界 ≥ `0.75`；
- 至少两个 WAM；
- 通过正常顺序、倒序、错身份、零差异四类控制；
- 所有阈值在确认集前冻结。

coverage 和准确率必须并列报告，并按 stratum、模型和弃权原因拆开。pilot 只能用于
调试和估计方差，不能替代确认集。

## 解释边界

即使 RCS-yaw 通过，也只能得出：

> 在同一反事实干预下，WAM 预测未来的结构响应与 native action 响应一致。

若要声称 native action 是由 predicted future 产生，还必须进行 future-only
intervention 或 future-pathway ablation。若无法干预该路径，保持
`future-to-action = design only`。

## 当前 pilot（非确认结果）

现有 255 对 DriveWAM 左右分支可用于管线 smoke test，但不是 pure-speed twin，且
当前 `horizontal_flow_center` pilot 的覆盖和方向 CI 尚未达到 promotion gate。该结果
只证明控制脚本和输出契约可运行，不改变 S1.3 的冻结状态，也不把结构通道升级为
正式 primary。

需要与上述旧 pilot 区分的是，当前冻结 S1.3 的受控对照运行已经完成了 observer-level
复核：在相同的 174 个 source 上，DriveWAM 的 pair coverage 为 `174/174 = 100%`，
方向准确率为 `91/106 = 85.8%`，95% CI 为 `[0.780, 0.912]`，Spearman 为 `0.802`；
Epona 的 pair coverage 为 `170/174 = 97.7%`，方向准确率为 `97/105 = 92.4%`，
95% CI 为 `[0.857, 0.961]`，Spearman 为 `0.739`。两组的倒序和身份错配方向均
显著翻转，零差异按契约返回 `unavailable`。

这证明的是 S1.3 对已知视觉反事实变化的方向敏感性和 fail-closed 行为；它不是自然
WAM 质量排名，也不能替代尚未完成的独立 pure-speed twin 确认集。255 对旧 pilot 与
174 source 的受控运行使用了不同的产物和统计口径，不能混合计算。

第三个模型 DriveVA 的 10-twin pilot 还验证了模型专用图像几何 adapter 的必要性：
其历史帧为 `1920×1080`、生成帧为 `832×480`，必须显式声明
`direct_resize` 变换；同时，旧 manifest 的 `gt_candidate_id` 和相对
`future_frame_paths` 需要在派生评测 manifest 中修正。修正后 20/20 分支完成测量，
18/20 分支四秒全可用，10/10 twin 可评分；正常方向为 `7/10 = 70.0%`，95% CI
为 `[0.397, 0.892]`，倒序/身份错配为 `3/10`，零差异为 `unavailable`。由于 twin
数太少且 CI 很宽，这只是 adapter 与第三模型的 pilot，不满足正式 promotion gate。
原始 DriveVA manifest 不被覆盖，派生修正仅用于验证。

随后在 scene-disjoint 的 50-twin DriveVA holdout 上，用同一冻结 S1.3 和同一
`direct_resize` adapter 运行结构控制。该集合是 command-conditioned replication，
不是 pure-speed：pair coverage 为 `46/50 = 92.0%`，方向命中为 `20/38 = 52.6%`，
95% CI 为 `[0.373, 0.675]`，Spearman 为 `0.258`；倒序和身份错配均为 `18/38`
（Spearman `−0.258`），零差异按契约为 `unavailable`。因此 DriveVA 的 coverage
达到门槛，但结构 yaw 响应未达到方向和 CI 门槛。该结果是模型/数据条件下的真实
负结果：它不推翻 S1.3 在 DriveWAM/Epona 的受控 observer 验证，也不能把 DriveVA
自然质量排名解释成 CCFC-S 失败或成功。

## 当前能力矩阵（2026-09-11）

| 能力/数据条件 | twin coverage | 方向准确率（95% CI） | 结论 |
|---|---:|---:|---|
| DriveWAM，受控视觉反事实 | 100.0% | 85.8% `[0.780, 0.912]` | 通过 coverage/方向门；observer-level |
| Epona，受控视觉反事实 | 97.7% | 92.4% `[0.857, 0.961]` | 通过 coverage/方向门；observer-level |
| DriveVA，scene-disjoint command-conditioned holdout | 92.0% | 52.6% `[0.373, 0.675]` | coverage 通过，响应门失败 |
| 255 对旧 structural pilot | 74.1% | 77.9%（CI 未达 promotion gate） | 仅管线/pilot，不作确认结果 |
| pure-speed confirmation：Epona / DriveWAM | 94.9% / 100.0% | 76.8% `[0.642, 0.859]` / 48.8% `[0.383, 0.594]` | 方向/秩信号未跨模型稳定，progress 不晋级 |

这张表刻意把“测得到”与“响应方向正确”分开。DriveVA 的结果是一个模型级负
结果，说明 adapter 能解决输入尺寸和运行契约，但不能凭空产生稳定的动作响应。
pure-speed confirmation 进一步表明 progress 结构量在 Epona 上有方向性但在
DriveWAM 上接近随机，且两者的 Spearman 分别为 `-0.037` 与 `0.077`，因此不能
把 progress 晋级为正式 primary，也不能据此宣称 RCS-yaw 已经成为跨模型质量排名
指标。当前可以冻结的是候选盲的 S1.3 **动作—视觉响应测量器**；progress 保持
diagnostic，米制 MAS/CFAC 仍需独立 action-to-structure 校准。
