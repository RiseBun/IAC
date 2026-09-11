# Step1.3 改进计划：从可测 yaw 到可靠的多通道运动观测

## 1. 当前问题的边界

S1.3 现在是候选盲的 RAFT-Large `horizontal_flow_center` yaw 响应测量器。
在受控视觉反事实上，DriveWAM/Epona 的 pair coverage 为 `100%/97.7%`，方向
准确率为 `85.8%/92.4%`；因此“是否对 yaw 干预产生方向响应”已经可测。

但它仍不是完整运动测量器：

1. 单目光流不能稳定分离 yaw 与 lateral translation，也不能提供可靠米制尺度；
2. 接近零的 flow delta 目前可能只反映估计噪声，现有 `1e-12` 非零判断不是
   可辨别性门；
3. 末端 action 与跨 interval flow 中位数存在潜在时间错配；
4. real-domain uncertainty gate 的概率含义不能自动外推到生成域；
5. pair coverage（至少两个共同 interval）不等于四个时刻全部可测；
6. `Delta S_F` 与 `Delta P_A` 证明一致性，不证明 predicted future 对 action 的
   因果中介作用。

pure-speed confirmation 已验证 progress 通道不能晋级：Epona 方向 `76.8%`
（CI 下界 `0.642`、ρ `-0.037`），DriveWAM 方向 `48.8%`（CI 下界 `0.383`、
ρ `0.077`）。因此不能通过添加一个未经验证的速度/距离通道来掩盖 yaw 通道的
范围限制。

## 2. 改进优先级

### P0：先改测量协议，不换模型

在同一 flow backbone 上增加三项候选盲规则：

- **interval 对齐**：将 action trajectory 的每个增量与对应时间 interval 的
  flow descriptor 配对，预先冻结 median、面积和符号持续性三个输出；不在确认集
  上择优。
- **噪声 deadband**：用真实帧重复运行、common-random zero-contrast 和同分支数值
  重跑估计 flow delta 的噪声分布；用 SNR/置信区间决定 `reliable / weak /
  unavailable`，不再用 `delta != 0`。
- **分层报告**：同时报告 interval、branch、pair、twin coverage，以及每一层的
  abstention reason。四时刻要求不能被“至少两个 interval”覆盖率替代。

P0 的目标是提高准确性和诚实性，不承诺提升 coverage；它不改变冻结 S1.3
primary，先作为 preregistered A/B。

#### P0-B 首轮结果（2026-09-11）

B 已在两个模型的 pure-speed confirmation 上以同一套实现跑完。它没有改变冻结
S1.3，也没有在确认集上估计 deadband；`0.005 px` 只是待独立噪声校准的 pilot
值。对照结果如下：

| 模型 | 冻结 v1（endpoint/整段聚合） | B：interval 对齐、无 deadband | B：0.005 px deadband |
|---|---:|---:|---:|
| Epona | 94.9% / 76.8% / ρ=-0.037 | 94.9% / 76.8% / ρ=-0.019 | 94.9% / 76.8% / ρ=-0.019 |
| DriveWAM | 100.0% / 48.8% / ρ=0.077 | 100.0% / 52.2% / ρ=0.088 | 95.3% / 52.9% / ρ=0.009 |

表中依次为 pair coverage / 方向准确率 / action-response Spearman。DriveWAM 的
小幅上升没有排除 0.5（95% Wilson CI 为 `[0.406, 0.635]`，deadband 后为
`[0.413, 0.641]`）；Epona 完全不变。deadband 从 `0.01 px` 开始反而持续损失
coverage 和方向，未观察到可迁移的去噪窗口。因此 B 的当前结论是**口径修正已
完成，但不是性能修复，也不能晋级 primary**。死区必须改由独立重复流噪声标定，
之后再决定是否保留为 weak/unavailable 门，而不能用确认集择优。

### P1：光流 backbone A/B 或小规模生成域适配

直接替换光流不是已知解。仓库已有 SEA-RAFT A/B：coverage 从 `74.1%` 降到
`42.4%`，方向准确率几乎不变，因此现成 SEA-RAFT 不能替代 RAFT-Large。

仍可测试 GMFlow/UniMatch 或生成域微调，但验收必须同时要求：

- pair coverage 不低于 `0.90`；
- 方向准确率 95% CI 下界不低于 `0.75`；
- reversed / identity / zero controls 不出现假阳性；
- 在 DriveWAM、Epona 以及一个未参与调参的模型上重复。

只提升 flow 端点误差、但降低 coverage 或没有控制区分力，不算成功。

### P2：深度辅助的运动分解

深度的作用是给每个像素一个尺度/3D 先验，从而减少 rotation–translation ambiguity，
再估计 lateral、forward progress 和 curvature。它**不能**修复：

- 生成图像本身没有稳定纹理或对应；
- 光流在生成域发生系统性伪流；
- 时间轴或分支身份错误；
- candidate-dependent 的采样/门控。

因此深度只能作为独立 diagnostic channel：

```text
flow + depth -> robust 3D/SE(2) fit -> lateral/progress/curvature
```

必须先在真实 logged 帧上校准，再在生成帧上独立验证；深度不确定时返回
`unavailable`，不能用单目深度强行填满 coverage。UniDepth 明确区分 metric 与
scale-agnostic monocular depth；Depth Anything V2 是稳健的相对深度候选，但二者
在本项目生成域的尺度和跨模型稳定性都必须实测，不能由论文 benchmark 外推。

### P3：多相机/立体几何

若目标最终是米制 lateral、距离和曲率，这是最有原则的结构改进；但它改变输入
条件和公开协议，成本最高。只有 P0/P1 证明结构流方向可靠、而 P2 在真实帧上仍
无法分离尺度时，才值得进入。

## 3. 推荐的新 Step1 结构

```text
输入帧 + 显式模型 adapter
        |
        +--> RAFT-Large / 可选第二 flow backend
        |       |
        |       +--> temporal alignment + calibrated SNR
        |       +--> horizontal_flow_center  (frozen yaw primary)
        |       +--> flow delta / controls   (CCFC-S diagnostic)
        |
        +--> optional depth adapter
                |
                +--> robust SE(2) / 3D motion (diagnostic only)

每个 channel 独立输出 reliable / weak / unavailable

### 3.1 前向视觉一致性（新主攻方向）

SE(2) 自由拟合回答的是“有没有某条轨迹能解释图像”，不是“给定轨迹是否
解释图像”。因此新增一个不做自由优化的实验协议：由给定的 `[x, y, yaw]`
轨迹和相机几何前向生成预期流场，再与视频实际流场逐 interval 比较。可用地面
平面模型，也可接入深度模型；两者都必须把投影无效区域从证据中排除并报告
`unavailable`，不能 zero-fill。

该协议输出 robust flow residual、方向余弦、inlier fraction 和 interval coverage，
代码位于 `src/iac_new/visual_consistency.py`，配置位于
`configs/forward_visual_consistency_v1.json`。它目前是 experimental diagnostic，
不是 S1.3 primary；阈值必须在 held-out real videos 上冻结，并通过 normal / reversed /
identity / zero controls 和至少两个 WAM 的 scene-disjoint confirmation。

#### 前向一致性首轮真实/生成对照（2026-09-11）

在同一批 Epona source 上，使用同一 RAFT、同一相机几何和同一道路 ROI：

| 输入 | interval residual 中位数 | direction cosine 中位数 | reliable interval fraction |
|---|---:|---:|---:|
| logged-real（59 source） | `2.43 px` | `0.633` | `5.1%` |
| generated（118 branches） | `5.35 px` | `0.356` | `4.0%` |

生成帧 residual 约为真实帧的 `2.2×`，方向余弦更低；这支持“生成视觉运动与给定
轨迹的前向后果偏离”的假设，但还不是最终几何保真度结论。可靠 interval fraction
较低，主要因为 4 秒前向投影的有效道路区域在部分时刻不足；这些 interval 被标为
`unavailable`，没有被 zero-fill。下一步必须在真实 held-out 集上校准 residual/support
阈值，并补 normal / reversed / identity controls 后才能形成正式分数。

#### 前向一致性控制结果（Epona pure-speed）

在同一批生成视频上做了 normal / swapped / zero 控制。zero 轨迹的 residual 明显更大
（`43.53 px`），说明该分数并非恒定背景量；但 swapped 轨迹（`4.55 px`）略优于
normal（`5.35 px`），说明当前地面平面前向模型尚未可靠恢复 pure-speed 的动作身份。
因此这个通道目前只能证明“视频存在运动且可被几何假设部分解释”，不能证明“运动由
给定分支动作正确诱导”。这也是正式接入 CCFC-S 前必须解决的控制条件。

进一步的 twin differential 控制显示：normal 的 `ΔF_video` 与 `ΔF_action` 方向余弦
为 `+0.416`，reversed 为 `−0.416`；但观测分支差中位数仅 `2.90 px`，轨迹诱导差异
约 `15.13 px`。因此当前最可信的读出是**方向/排序响应**，不是幅度或米制保真度：
normal 比 reversed 好（residual `13.17` 对 `17.43 px`），但 zero-difference 的
residual 仍最低（`2.90 px`）。该结果支持把 twin differential 作为结构性 CCFC-S
候选，同时明确禁止把它解释成绝对运动重建。

在 DriveWAM 上的同口径结果完全不同：normal/reversed residual 分别为 `14.36/14.26 px`，
direction cosine 仅 `+0.016/-0.016`，而视频分支差中位数只有 `0.049 px`（轨迹预期差异
`13.70 px`）。因此该方法还不能作为跨模型统一 primary；它能够诚实地区分“有方向但
幅度坍缩”（Epona）和“没有可检测反事实差”（DriveWAM），这本身是几何保真度诊断，
但不能替代跨模型的正式一致性分数。
```

S1.3 yaw 保持唯一冻结主通道；任何新通道都必须通过独立校准、双模型确认和控制
实验后才能晋级。这样既不把 SE(2) 的米制数字伪装成可靠真值，也不把 progress
的跨模型负结果隐藏掉。

## 4. 下一轮最小实验矩阵

| 臂 | 改动 | 目的 |
|---|---|---|
| A | 当前 RAFT-Large S1.3 | 基线 |
| B | A + interval alignment + SNR deadband | 检验协议问题 |
| C | SEA-RAFT/GMFlow drop-in | 检验 backbone 问题 |
| D | A + UniDepth/Depth Anything V2 的深度辅助分解 | 检验几何歧义 |

所有臂使用相同 source、相同 adapter、相同 controls；先在 logged-real 校准集
冻结规则，再在 scene-disjoint 的 DriveWAM/Epona confirmation 上报告。若 B 已达到
准确率门而 D 没有额外收益，不引入深度；若 D 只改善 lateral/尺度而不改善 yaw，
则保留为 diagnostic；若 C coverage 下降或 controls 失败，停止换 backbone。

## 5. 当前决策

- **不**直接把深度接入冻结 primary；
- **不**把 SEA-RAFT 作为现成替代；
- 先做 P0 的时间对齐和噪声校准，再做小规模 C/D A/B；
- 将“给定轨迹 → 预期视觉运动 → 生成流场残差”作为比自由 SE(2) 拟合更直接的
  consistency 主攻方向，但在校准和控制实验完成前不晋级；
- 只有新通道同时满足准确率、coverage、controls 和跨模型要求，才允许改变
  primary；否则维持 S1.3 yaw primary + SE(2)/progress diagnostic。
响应增益（观测 `ΔF` / 轨迹预期 `ΔF`）的 pair 中位数也明显分离：Epona 为 `0.232`
（IQR `0.112–0.378`），DriveWAM 仅 `0.0045`（IQR `0.0026–0.0081`）；normal
temporal persistence 分别为 `1.0` 和 `0.5`。因此可以
把“方向余弦 + response gain + zero/reversed controls”组合成一个**结构性几何保真度
诊断**：它不输出米制距离，但能区分分支是否响应、响应方向是否正确以及响应幅度是否
接近动作预期。它仍需第三模型和独立阈值校准后，才能进入正式 CCFC-S。

向量门限敏感性（`0.05–1.0 px`）保持同一结论：Epona 的 normal direction cosine
稳定在 `+0.41～+0.48`，reversed 始终为负；DriveWAM 接近零并随门限变号
（`+0.047` 到 `−0.083`）。Epona/DriveWAM 的 response gain 中位数分别稳定在
`0.232` 和 `0.0045`。这表明结构量的跨模型分离不是由单一阈值调出来的，可以
冻结为“方向 + 响应增益 + controls”的 CCFC-S 候选，但仍不能称为米制保真度。

第三模型 WorldDrive 的首个 5-twin command-0/command-2 pilot 已完成：normal/reversed
direction cosine 为 `+0.762/−0.762`，residual 为 `3.24/5.06 px`，normal response
gain 中位数 `0.956`（IQR `0.954–1.026`）。随后在独立的 eval25 池中追加 10 个
twin（20 个分支、10 个场景），得到 normal/reversed direction cosine
`+0.998/−0.998`、normal response gain `1.469`（IQR `1.140–2.284`），所有
normal twin 的方向余弦均为正，且正常相对反向的 residual margin 中位数为
`146.7 px`，normal temporal persistence 为 `1.0`。阈值 `0.05–1.0 px` 下结果不变。该扩展强化了“WorldDrive 的方向响应
可被结构量读出”的 pilot 证据，并与 Epona（方向正确、增益约 `0.232`）和 DriveWAM
（无可检测分支响应、增益约 `0.0045`）形成可解释的三模型梯度；但 WorldDrive
仍只有 10 个扩展 twin，95% 二项下界约 `0.692`，不能满足 `≥0.75` 的正式
promotion 门槛。统一数值已写入
`reports/ccfc_structure_forward_scorecard_20260911.json`，下一步仍是更大、场景
隔离的第三模型确认与独立真实视频阈值校准。

将首个 5-twin 池与 eval25 池按 `(source_key, branch_role)` 去重后得到 13 个
WorldDrive twin。 pooled normal direction cosine 仍为 `0.998`，response gain
为 `1.426`，但方向并非全场景稳定：12/13 个 twin 为正，scene 聚类为 10/11
个为正（95% 二项下界分别约 `0.640` 和 `0.587`）。因此 eval25 的 10/10
正方向不能作为正式 promotion 证据；该合并结果反而暴露了场景级稳定性缺口，
后续 confirmation 必须预先固定 scene-level 方向准确率和置信区间。

作为额外的跨模型 sanity check，DriveVA 的 10-twin 生成池也使用同一协议完成了
流提取和 normal/reversed/zero controls。其 normal/reversed direction cosine 为
`+0.129/−0.129`，normal response gain 为 `0.061`，normal temporal persistence 为
`0.786`，normal 与 reversed residual
仅相差约 `1.08 px`；阈值敏感性下方向余弦仍接近零。这不是协议失效，而是该模型在
这组 twin 上没有足够强的可读反事实响应，因此不能把 coverage 高误读成几何保真度。
它进一步说明候选量确实能把“强正确响应”（WorldDrive）与“弱/无响应”（DriveVA、
DriveWAM）分开。DriveVA 同样只有 10 个 twin，结果不用于正式 promotion。
对合并池进一步按 scene 聚类并做 20,000 次 bootstrap：11 个 scene 中 10 个方向为正，
scene-level 方向准确率为 `0.909`，95% 区间 `[0.727, 1.000]`；scene 聚合方向余弦
均值为 `0.837`，区间 `[0.637, 0.964]`。这一步避免把同一 scene 的多个窗口当成
独立证据，但也说明当前场景数不足以证明下界达到预注册的 `0.75`。完整计算记录在
`reports/forward_visual_worlddrive_combined_scene_bootstrap_20260911.json`；在新增
scene 前不改变 promotion 判据。
