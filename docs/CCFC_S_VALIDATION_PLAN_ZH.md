# CCFC-S 验证计划（v1）

## 目标

验证结构版反事实指标 `CCFC-S` 是否能在不恢复米制轨迹的前提下，可靠识别同一
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

即使 CCFC-S 通过，也只能得出：

> 在同一反事实干预下，WAM 预测未来的结构响应与 native action 响应一致。

若要声称 native action 是由 predicted future 产生，还必须进行 future-only
intervention 或 future-pathway ablation。若无法干预该路径，保持
`future-to-action = design only`。

## 当前 pilot（非确认结果）

现有 255 对 DriveWAM 左右分支可用于管线 smoke test，但不是 pure-speed twin，且
当前 `horizontal_flow_center` pilot 的覆盖和方向 CI 尚未达到 promotion gate。该结果
只证明控制脚本和输出契约可运行，不改变 S1.3 的冻结状态，也不把结构通道升级为
正式 primary。
