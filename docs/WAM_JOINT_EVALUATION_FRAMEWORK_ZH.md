# IAC：WAM 联合评测框架（v1）

## 1. 研究问题

IAC 不把视频质量或任务成功率单独当作 WAM 正确性的证据。核心问题是：

> WAM 预测的未来是否与它生成/执行的动作保持可审计的反事实一致性？

“米制轨迹是否精确”是几何保真度问题，不是联合性本身的必要条件。因此框架先
测 action–future consistency，再把绝对几何和任务成功作为独立轴。

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
                   CCFC-S / CFAC-S / F→A audit

native action P_A ───────────→ independent simulator ─→ FCS
logged future + GT-compatible channel ────────────────→ FAU
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

### 5.1 CCFC-S

```text
ΔS_F = S_F(left) - S_F(right)
ΔP_A = P_A(left) - P_A(right)
CCFC-S = ordinal_consistency(ΔS_F, ΔP_A)
```

报告：方向准确率、pair coverage、Spearman/配对排序、时间持续性和四类控制。

它回答的是“预测未来和动作是否对同一个干预作出一致响应”。

### 5.2 CFAC-S

```text
CFAC-S = alignment(S_F, structuralized(P_A))
```

这里需要一个只在校准集拟合、随后冻结的 action-to-structure 映射。没有该映射时，
CFAC-S 必须报告为未完成，不能把结构值直接与米制 action 做 MAE。

### 5.3 Future-to-action audit

`ΔS_F ↔ ΔP_A` 不是路径级因果证明。要声称 action 由 future 驱动，必须做至少一个：

1. 固定 history 和 command，只改变 predicted future/future latent，观察 action 是否变化；
2. 屏蔽 future-to-action pathway，观察 action 和 CCFC-S 是否下降；
3. 固定 predicted future，只改变 action pathway，作为反向特异性对照。

没有这一步，论文措辞只能使用“action-state consistency”，不能使用“future-caused
action generation”。

### 5.4 FAU 与 FCS

FAU 继续使用 logged-GT-compatible 图像侧表示，衡量预测未来和 native action 是否
接近真实未来。结构 ordinal channel 不能替代它。

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
| CCFC-S 框架 | specified / validation pending |
| pure-speed progress channel | pilot evidence only |
| CFAC-S calibration | not yet run |
| future-to-action mediation | design only |
| metric SE(2) reconstruction | diagnostic only |
| FAU | independent GT-compatible axis |
| FCS | independent simulator axis |

## 8. 最终报告形式

在所有通道完成验证前，不压成单一总分，优先报告：

```text
{ CCFC-S, CFAC-S, future-to-action, FAU, FCS, coverage, abstention }
```

这样可以区分“视频看起来逼真”“动作与未来一致”“未来真正影响动作”和“实际任务
成功”，避免一个高分掩盖另一个通道失效。
