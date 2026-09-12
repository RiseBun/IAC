# 发布边界与剩余证据

当前 IAC 可以作为**条件式视觉—动作一致性/现实保真度标准**发布。这里的
“条件式”是协议设计，不是对模型的先验假设：WAM 不必证明自己一定由预测未来
驱动，只有实际提供且可审计的通道才计分。

## 已闭合的发布能力

| 通道 | 当前状态 | 可以声称 |
|---|---|---|
| MAS-yaw | Epona、DriveWAM 通过冻结门槛 | 单支视觉 yaw 结构与 native action 方向一致 |
| RCS-yaw | Epona、DriveWAM 通过冻结门槛 | 同源干预下视觉响应与动作响应一致 |
| GS | Epona、DriveWAM 通过冻结校准/确认 | 生成视觉结构与外部 logged future 的 grounding |
| 条件式 scorecard | 已实现并失败关闭 | 缺失通道为 `unavailable`，不当作 0 |

## 尚未闭合的证据

### Future-to-action mediation

需要同一 source 提供四个条件：baseline、future-only perturbation、pathway
blocked、fixed-action control。四个条件必须保持 history、command、seed 和
model revision 不变，并携带 future fingerprint 与 pathway state。通过后才能
使用“动作由预测未来产生”的因果措辞；MAS/RCS 不能替代它。

当前 scorer 还会强制每个条件携带同一个、只用 calibration sources 拟合并在确认
前冻结的 `action_normalization_fingerprint`/`action_normalization_scale`。只有
“可计算”不等于“可晋级”：少于 30 个 source 时报告仍可用于调试，但
`promotion.claim_enabled=false`，不能写入因果结论。这样可以防止一条或少数几条
干预样本被误读成 mediation 证据。

### FCS 跨模型

每个模型至少需要独立 simulator rollout、native action 注入证明、独立 realized
state 和明确的 task-label provenance。当前 DriveWAM 已有一份合规汇总，第二个
模型仍需单独提交，不能从 MAS/RCS 推断。

### 非 yaw 运动量

速度、前进距离、横向幅度、曲率和米制 SE(2) 仍是 diagnostic。要晋级，必须在
real-only calibration 后冻结 adapter，并在 source-disjoint 的至少两个模型上同时
满足 coverage、准确度和控制门槛；否则保持 `diagnostic_only`。

## 数据公开边界

GS 的执行代码和冻结尺度公开；logged future、NAVSIM/Waymo 图像和模型权重仍由
评测服务器保留。因此第三方可以复核协议、运行方式和缺失值策略，但不能仅凭
公开仓库重算当前私有参考表。

## 推荐实验顺序

1. 先完成第二模型 FCS rollout；
2. 再做至少一个 WAM 的 future-only/pathway-ablation mediation；
3. 只有非 yaw 通道出现跨模型、跨控制的稳定证据时，才考虑扩展 MAS/RCS。

在这些证据出现前，不应修改当前冻结 yaw 评分，也不应创建一个把不同能力压成
单一总分的排行榜。
