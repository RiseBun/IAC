# 发布边界与剩余证据

当前 IAC 的代码、协议和单项结果可以作为研究实现公开，但**完整三指标 benchmark
尚未通过发布门禁**。这里的“条件式”是协议设计，不是对模型的先验假设：WAM
不必证明自己一定由预测未来驱动，只有实际提供且可审计的通道才计分。

## 指标级现状

| 通道 | 当前状态 | 可以声称 |
|---|---|---|
| AS | DriveWAM 完整合同已冻结；跨模型确认待完成 | 单支视觉 yaw + 粗粒度 progress 与 native action 一致 |
| RCS-yaw | Epona、DriveWAM 通过冻结门槛 | 同源干预下视觉响应与动作响应一致 |
| GS | Epona 通过 identity-specificity；DriveWAM 仅作低 grounding 诊断 | Epona 上生成视觉结构与外部 logged future 的 grounding |
| 条件式 scorecard | 已实现并失败关闭 | 缺失通道为 `unavailable`，不当作 0 |

### 已实现但尚无证据的新通道（2026-09-12）

两个通道已实现并失败关闭，但都还没有跑出证据，因此状态是
`implemented_evidence_pending`，不改变任何当前分数。

**source 级联合分析** [`tools/analyse_joint_source_table.py`](../tools/analyse_joint_source_table.py)。
把 AS/RCS 的逐 source 分数与独立执行结果按 `source_key` 拼成一张表，报
source 级 bootstrap 的 Spearman、与随机置换的对照、以及一致性分数相对
chance（RCS 为 0.5）的偏离。这是当前**最便宜、也最可能推翻现有框架**的检查：
它不需要新图像、不需要新模型，只需要已有的逐 source 表。若一致性分数与任务
成功无关，那么先投资一致性通道就是错的。零结果的含义是“一致性通道对该 split
的任务成功没有信息”，不构成对某个模型通路依赖的反证。

**条件式 foresight-conditioned success** [`tools/score_conditional_foresight.py`](../tools/score_conditional_foresight.py)。
独立执行 scorer 报的是边际成功率，一个完全忽略自己预测视频、直接从 history 出动作的
模型拿到同样的分数；要让它成为 *foresight-conditioned*，必须报同 source 配对
的 `success(future_perturbed) - success(baseline)`，并同时跑
`pathway_blocked`（归因控制）与 `fixed_action`（特异性控制）。该 scorer 额外
强制一条 **dose-response** 监视：扰动若几乎没推动动作，平坦的 delta 是探针太弱
的证据，不是“模型忽略未来”的证据。这两件事必须分开报告，否则阴性结果无法解释。

## 尚未闭合的证据

### 完整 AS 的第二模型确认

DriveWAM 已具备完整 native-action provenance，并完成 yaw + 粗粒度 ordinal progress
AS。Epona 的旧 `mas_yaw_*` 结果只能支撑 AS-yaw 组件，WorldDrive 的同源样本仅
`n=10`，二者都不能充当完整 AS 的第二模型。最短补证路径是在一个具备原生动作、
同轴未来帧和至少 30 个 source-disjoint 样本的第二 WAM 上冻结复用同一视觉层和
阈值，不重新调参。

### GS 的第二个 identity-specific 模型

Epona 通过 identity-shuffle 控制；DriveWAM 的 GS 中位数 `0.191` 低于其
identity-shuffle 均值 `0.234`，因此它只能说明 grounding 低，不能证明评分具有
身份专一性。需要第二个 WAM 在同源外部真实未来上通过 identity-shuffle 与时间控制。

### Future-to-action mediation

需要同一 source 提供四个条件：baseline、future-only perturbation、pathway
blocked、fixed-action control。四个条件必须保持 history、command、seed 和
model revision 不变，并携带 future fingerprint 与 pathway state。通过后才能
使用“动作由预测未来产生”的因果措辞；AS/RCS 不能替代它。

对 DriveWAM 的代码检查已经定位了可行入口：其执行顺序是先生成未来帧、保留
共享 transformer cache，再生成 native action。但当前 `rollout_external_action`
暴露的是 **action→video** 条件干预，不是 future-only 干预；改变
`condition_chunk`、denoise seed 或 native action 不能直接当作 mediation。真正的
DriveWAM probe 必须只改 retained future pathway，并在相同干预下再运行 pathway
blocked 条件，然后才交给下方 scorer。

当前 scorer 还会强制每个条件携带同一个、只用 calibration sources 拟合并在确认
前冻结的 `action_normalization_fingerprint`/`action_normalization_scale`。只有
“可计算”不等于“可晋级”：少于 30 个 source 时报告仍可用于调试，但
`promotion.claim_enabled=false`；CLI 还必须显式提供 calibration source 清单，
否则即使有 30 个 source 也只是 `insufficient_evidence`。这样可以防止一条或
少数几条干预样本，或与校准集重叠的 source，被误读成 mediation 证据。

已有一个 WorldDrive 内部 future-latent permutation pilot：25 组中 11 组改变了
选择或轨迹，但只有 1 组达到预注册的 lateral/yaw 实质门槛。它缺少
pathway-blocked、fixed-action、source-disjoint 和至少 30 个实质 source，因此状态为
`pilot_unqualified`，不能提升 mediation 状态，也不能把 AS/RCS 改写为因果分数。
记录见 [`reports/future_to_action_worlddrive_pilot_20260912.json`](../reports/future_to_action_worlddrive_pilot_20260912.json)。

### 独立执行的跨模型审计

每个模型至少需要独立 simulator rollout、native action 注入证明、独立 realized
state 和明确的 task-label provenance。当前 DriveWAM 已有一份合规汇总，第二个
模型仍需单独提交，不能从 AS/RCS 推断。
仓库提供 [`tools/assess_cross_model_execution.py`](../tools/assess_cross_model_execution.py) 做
第二道审核：至少两个不同 `wam_model_id`、每个模型达到最小 scored rows，且每个
报告已经通过 native-action provenance 检查，才允许报告跨模型外部任务验证。现有
`closed_loop_recovered_20260829` 中缺少 action source 或标为 staging 的记录会被
明确拒绝，不会被计作第二模型。
当前逐项审计记录见 [`reports/independent_execution_cross_model_readiness_20260912.json`](../reports/independent_execution_cross_model_readiness_20260912.json)。

### AS 之外的连续运动量

AS 已冻结粗粒度 ordinal progress，但精确速度、米制前进距离、横向幅度、曲率和
完整 SE(2) 仍是 diagnostic。要晋级，必须在
real-only calibration 后冻结 adapter，并在 source-disjoint 的至少两个模型上同时
满足 coverage、准确度和控制门槛；否则保持 `diagnostic_only`。

### Metric depth 的边界

Metric3Dv2-v2-S 已进入 AS 的 progress 读取链，但它不直接输出自车路程。当前链路
仍需静态对应、Reloc3r 旋转和 known-R PnP，最后只晋级为粗粒度 ordinal progress。
真实 NAVSIM 上的读数器验证支撑这个粗粒度声明；它不支撑米制距离或绝对速度。

## 数据公开边界

GS 的执行代码和冻结尺度公开；logged future、NAVSIM/Waymo 图像和模型权重仍由
评测服务器保留。因此第三方可以复核协议、运行方式和缺失值策略，但不能仅凭
公开仓库重算当前私有参考表。

## 推荐实验顺序

1. **先补完整 AS 第二模型**：复用冻结视觉层和阈值，确认 native-action provenance；
2. **再补 GS 第二个 identity-specific 模型**：必须通过 identity-shuffle 控制；
3. source 级联合表作为效用审计：拼已有 RCS/AS 与独立执行结果；
4. 单独推进 mediation：使用 baseline vs future_perturbed 配对并做 dose-response，
   再补 pathway-blocked 与 fixed-action 两个条件；
5. 只有新增通道出现跨模型、跨控制的稳定证据时，才考虑扩展 AS/RCS。

在这些证据出现前，不应修改当前冻结 yaw 评分，也不应创建一个把不同能力压成
单一总分的排行榜。
