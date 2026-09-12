# 指标优先的证据契约

Step1 不是一个必须服务所有指标的单一运动读取器。MAS、RCS、GS、FCS 要回答的问题不同，因此每个指标只消费自己的最小证据；缺证据就返回 `unavailable`，不得用其他指标的结果填补。

## 四个指标分别需要什么

| 指标 | 问题 | 主观测器 | 不能声称 |
|---|---|---|---|
| MAS | 单支未来视频是否支持它收到的 native action | 给定轨迹的视觉条件似然 | future-to-action 因果 |
| RCS | 同源动作干预是否产生预期的成对视觉响应 | 左右差分 + 正常/反转/零差异控制 | 单支动作读取 |
| GS | 生成未来是否接近外部真实未来 | 参考条件 grounding | 动作中介因果 |
| FCS | future intervention 是否改变独立执行成功率 | 独立 rollout + 成对干预 | 仅凭图像质量推出因果 |

## 现有方法如何挂接

- SE(2) 条件似然是 MAS/RCS 的主观测器；
- Step1.3 的 FOE、散度、旋度、结构置信度是 GS/输入质量诊断，不能默认改写 MAS/RCS；
- RAFT、SEA-RAFT、NeuFlow 和深度都是可替换的观测后端；
- FCS 由独立模拟器和 future-only intervention 完成，Step1 只提供必要的视觉证据，不替代干预实验。

## 计分状态

每个指标独立返回：`scored`、`weak` 或 `unavailable`。`unavailable` 不是 0 分，也不影响同一模型报告其余具备证据的指标。

契约实现见 [`src/iac_new/metric_evidence_contract.py`](../src/iac_new/metric_evidence_contract.py)，配置见 [`configs/metric_evidence_contract_v1.json`](../configs/metric_evidence_contract_v1.json)。
