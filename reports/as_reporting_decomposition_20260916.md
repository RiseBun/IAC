# AS v1.2：一致性与可观测性分解

## 决策

AS 不再用一个数字同时承担“图像与动作是否一致”和“视觉层是否读得出来”两种解释。
正式报告固定为：

```text
AS_conditional   = sqrt(progress conditional × yaw conditional)
AS_observability = sqrt(progress score coverage × yaw score coverage)
AS_deployment    = AS_conditional × AS_observability
```

`AS_conditional` 是主要一致性估计；两个原始 channel coverage 界定证据范围。
`AS_deployment` 保留 v1.1 的 coverage-aware 数值，只作为部署摘要，禁止单独报告。

## DriveWAM 分解

```text
progress conditional = 1399 / 2884 = 48.5%
progress coverage    = 2884 / 5960 = 48.4%
yaw conditional      = 1007 / 1077 = 93.5%
yaw coverage         = 100%

AS_conditional       = 67.4 / 100
AS_observability     = 69.6%
AS_deployment        = 46.8 / 100
```

因此，旧的 46.8 不能被解释为纯模型一致性：它同时包含纵向读数约 48.4% 的低覆盖。
但分解也没有把问题全部归因于视觉层，因为在可测纵向区间中只有 48.5% 被接受。

## 协议变化

- 不改变视觉后端、阈值、容差、输入和历史结果；
- 聚合器新增 conditional score、两个 channel coverage、状态计数和弃权原因；
- 历史 v1.1 报告保持不变，新解释写入独立 v1.2 报告；
- progress 采用 interval-level conditional estimate，置信区间仍必须按 source 重采样；
- progress 与 yaw 的统计单位不同，所以必须保留各自 coverage，不能只报告其几何平均。

下一项应验证纵向通道本身：扩大 source-disjoint NAVSIM 真实区间，并完成分层误差、
bin confusion、risk--coverage 和冻结容差敏感性分析。
