# IAC：WAM 联合评测框架（v1）

## 1. 研究问题

IAC 不把视频质量或任务成功率单独当作 WAM 正确性的证据，也不预设所有 WAM
都会使用预测未来来产生动作。它提供的是一个按能力条件化的测量标准：有证据就
报告相应分数，没有所需信息或干预就诚实报告 `unavailable`，而不是把缺失当成零分。
核心问题是：

> 在给定可复现输入和干预下，WAM 预测的未来是否与它生成/执行的动作保持可审计的一致性？

“米制轨迹是否精确”是几何保真度问题，不是联合性本身的必要条件。因此框架先
测 action–future consistency，再把绝对几何和任务成功作为独立轴。若某个 WAM 的
动作与未来没有可检测的一致性，评测器应报告低一致性或 `unavailable`（取决于
输入是否可测），而不是替模型补出一个“未来驱动”的结论。

### 1.1 条件性解释原则

每个 WAM、每种干预、每个指标都单独形成证据格子：

| 情形 | 协议输出 |
|---|---|
| 输入和探针有效，且达到质量门 | 正常分数 + coverage + CI |
| 输入可测但方向/响应不一致 | 低分，并保留反向/身份/零差异控制 |
| 图像、时间轴、干预或必要 GT 缺失 | `unavailable`，记录原因，不填零 |
| 缺少 native action 或 future visual 等硬准入字段 | `ineligible` |

因此 IAC 是“能够测到什么就报告什么”的评测标准，而不是要求所有 WAM
必须通过某个预设的 future-driven 假设。

### 1.2 条件分数的分母和解释

“条件性”不是把不可测样本偷偷删掉，而是把两个分母同时公开：

```text
conditional_score = metric aggregation over scored units only
score_coverage    = scored units / all declared units
```

每个指标必须逐单位保留以下状态：

| 状态 | 含义 | 是否进入分数 | 是否进入 coverage 分母 |
|---|---|---:|---:|
| `scored` | 输入、探针和质量门均通过 | 是 | 是 |
| `abstain` / `weak` | 尝试测量，但证据不足以安全读出 | 否 | 是 |
| `unavailable` | 模型没有该通道，或外部参考/干预不存在 | 否 | 是 |
| `missing` | 提交声称有该通道，但材料不完整 | 否 | 是 |
| `ineligible` | 违反硬准入或泄漏契约 | 不进入该提交 | 不进入 |

因此四种结果必须分开解释：高分高 coverage 表示广泛的条件证据；高分低
coverage 只表示一个狭窄可测子集；低分高 coverage 表示通道可观测但不一致；
`unavailable` 只表示“对此能力没有结论”，不是零分，也不是模型失败。报告至少
要包含 `conditional_score`、`score_coverage`、`status_counts`、逐层
`abstention_reasons` 和置信区间。任何指标都不得用插值、零填充或把 abstain
重命名为 fail 来提高表面覆盖率。

历史字段如 `pair_coverage`、`branch_coverage` 或 `source_coverage` 继续保留，
但它们只描述输入/配对是否存在；当方向死区、共同 interval 或质量门进一步筛掉
单位时，不能把它们冒充 `score_coverage`。两者必须并列报告。

## 2. 总体结构

```text
history + command
        ↓
      WAM
        ├──────────────→ native action P_A
        └──────────────→ generated future frames
                              ↓
                       Step 1 image observer
                              ↓
                       structure S_F(t)
                              ↓
               confidence / explained / abstain
                              ↓
               same-source counterfactual delta ΔS_F
                              ↓
                   MAS / RCS / future-to-action audit

native action P_A ───────────→ independent simulator ─→ FCS
logged future + GT-compatible channel ────────────────→ GS
```

每条分支都必须携带 source、时间轴、模型、标定、随机种子和 lineage。统计单位是
source/twin，而不是共享同一 source 的 interval。

## 3. Step 1：图像侧观测器

### 3.1 冻结通道

当前冻结的 S1.3 使用候选盲 RAFT-Large 光流、真实域可靠性校准和
`horizontal_flow_center` yaw 描述子。它只读取图像和相机标定，不读取候选轨迹、
logged future state 或 PDM 分数。

输出必须按 interval 给出：

```text
structure_value
confidence
input_available
status = explained | weak | abstain
abstention_reason
```

S1.3 已在两个 WAM 上验证为 yaw action-response 测量器；这不等于已验证自然模型
质量排名或 logged-GT 几何保真度。

### 3.2 结构通道

```text
S_F(t) = [horizontal_flow_center,
          vertical_expansion,
          divergence,
          curl,
          foe_x,
          median_flow_magnitude]
```

`horizontal_flow_center` 目前是 primary yaw；progress 相关量在独立 speed-swap
twin 通过验证前只能作 diagnostic。结构量不自动转换为米、米/秒或曲率。

## 4. 反事实差分

对同一个 source 的左右分支：

```text
C(t) = (S_L(t) + S_R(t)) / 2
D(t) = S_L(t) - S_R(t)
D_norm(t) = D(t) / max(abs(C(t)), epsilon)
```

方向使用原始 `D(t)`；归一化差分只用于跨场景幅度诊断。它不能使用当前拟合轨迹
来选择像素或 interval。

pair 只有在左右分支都拥有所需 channel 和最低共同 interval 时才可计分；否则
整个 pair 为 unavailable，不能零填充。

## 5. 指标定义

### 5.1 RCS-yaw（旧 CCFC-S）

```text
ΔS_F = S_F(left) - S_F(right)
ΔP_A = P_A(left) - P_A(right)
RCS-yaw = ordinal_consistency(ΔS_F, ΔP_A)
```

报告：方向准确率、pair coverage、Spearman/配对排序、时间持续性和四类控制。

它回答的是“预测未来和动作是否对同一个干预作出一致响应”。

### 5.2 MAS-yaw（旧 CFAC-S）

```text
MAS-yaw = alignment(S_F, yaw_direction(P_A))
```

当前冻结的 MAS-yaw 只使用 real-only 校准的 yaw 方向/死区适配器，不声称米制
轨迹重建。任何 lateral、纵向距离、速度和曲率映射都必须单独校准并通过独立
跨模型门槛，否则只能标为 diagnostic/unavailable。

### 5.3 Future-to-action audit

`ΔS_F ↔ ΔP_A` 不是路径级因果证明。要声称 action 由 future 驱动，必须做至少一个：

1. 固定 history 和 command，只改变 predicted future/future latent，观察 action 是否变化；
2. 屏蔽 future-to-action pathway，观察 action 和 RCS 是否下降；
3. 固定 predicted future，只改变 action pathway，作为反向特异性对照。

没有这一步，论文措辞只能使用“action-state consistency”，不能使用“future-caused
action generation”。

另外，四个 intervention condition 必须来自同一个明确的 `wam_model_id` 和同一个
native `action_source`。logged、oracle、proxy、candidate、staging 或评测端注入的
轨迹都不是 mediation 证据；缺少这些 provenance 时，该 source 直接
`unavailable`。这条门把“动作向量发生了变化”和“WAM 自己的 future-to-action
路径发生了变化”严格区分开。

### 5.4 GS 与 FCS

GS 继续使用 logged-GT-compatible 图像侧表示，衡量生成视觉未来是否接近外部
真实未来。它不能替代 RCS，也不单独证明 future-to-action 因果关系。

FCS 继续把 native action 放入独立模拟器，根据实际状态和任务标签评分；它不读取
WAM 生成视频。

## 6. 可靠性和弃权

每个 channel 单独决定 explained/weak/abstain。正式统计必须并列报告：

- conditional accuracy/alignment；
- branch/pair/twin coverage；
- 各 stratum coverage；
- 弃权原因；
- confidence interval 和 risk–coverage；
- identity、顺序反转、零差异等控制。

覆盖率不能通过事后放宽阈值获得；阈值和校准必须在独立校准集冻结，再用于确认集。

## 7. 当前状态

| 组件 | 状态 |
|---|---|
| S1.3 yaw action-response | frozen / validated on two WAMs |
| 结构差分 scorer | implemented / exploratory |
| RCS-yaw（旧 CCFC-S） | frozen / validated on two WAMs |
| pure-speed progress channel | cross-model confirmation completed; not promoted (no stable signal) |
| MAS-yaw（旧 CFAC-S） | frozen / validated on two WAMs |
| future-to-action mediation | WorldDrive pilot unqualified / formal four-condition confirmation pending |
| metric SE(2) reconstruction | diagnostic only |
| FAU | independent GT-compatible axis |
| FCS | validated on DriveWAM / cross-model confirmation pending |

## 8. 最终报告形式

在所有通道完成验证前，不压成单一总分，优先报告：

```text
{ MAS, RCS, GS, future-to-action mediation, FCS, coverage, abstention }
```

这样可以区分“视频看起来逼真”“动作与未来一致”“未来真正影响动作”和“实际任务
成功”，避免一个高分掩盖另一个通道失效。
