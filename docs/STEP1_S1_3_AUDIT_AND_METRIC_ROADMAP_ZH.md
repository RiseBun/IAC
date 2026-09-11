# S1.3 审计与后续指标路线（2026-09-11）

## 当前冻结结论

S1.3 的冻结实现是候选盲的 `horizontal_flow_center` 结构测量器。它已经在
DriveWAM 和 Epona 的受控视觉反事实上通过 coverage 与方向门，但这证明的是
“动作—视觉响应可测”，不是自然 WAM 质量排名，也不是 logged-GT 米制保真度。

## 已发现、尚未改入冻结版的风险

### 1. 低信号方向被当成有效方向

当前方向判定只要求聚合 flow delta 非零（数值阈值约 `1e-12`），没有公开的
最小可辨别响应幅度。DriveVA 50-twin holdout 中，32 个 action 可比较 twin
的 `|flow_delta|` 中位数约 `0.00375`；其中 19/32 小于 `0.005`。这部分样本的
符号容易被 RAFT 误差、纹理伪流和分支配准扰动决定。

这不是把阈值调到 DriveVA 上的建议。正确做法是：在独立真实帧重复/扰动数据上
估计同一分支的噪声地板，预注册一个候选盲的 flow-response deadband，再在未见
确认集上报告 `explained / weak / unavailable`。默认冻结版保持现状，直到该
消融在至少两个模型上完成。

### 2. 末端 action 与全时域 flow 的时序错配

`score_flow_structure_pairs` 当前用末端 action yaw 差异，与共同 interval 上的
flow 中位数比较。对 DriveVA holdout 做只读的 interval 对齐诊断时，205 个可比
interval 的符号一致率约为 48%，与末端聚合得到的近随机结果一致。该结果不能
证明 DriveVA 没有响应，因为响应可能是早期建立、随后保持位姿偏移，或 action
与图像生成的时间响应有延迟。

下一轮应同时报告两种预注册量：

- 末端响应：保持当前 S1.3，便于版本连续性；
- 时序响应：将 action trajectory 插值/重采样到相同 interval，按固定规则计算
  中位数、面积或符号持续性，规则在确认集前冻结。

两者不能在同一确认集上择优后只报较高者。

### 3. 可靠性门是跨域校准，不是生成域真值

RAFT refinement uncertainty 的阈值由真实 NAVSIM 帧校准。生成帧的 coverage 很高
并不等于 uncertainty 仍然具有相同概率含义；因此生成域必须并列报告输入流支持、
结构置信度、方向结果和弃权原因，不能把 real-calibrated gate 当作生成域准确率
证明。

## 后续指标推进顺序

### A. CCFC-S（优先）

继续使用同一 source 的左右分支：

```text
Delta S_F = S_F(left) - S_F(right)
Delta P_A = P_A(left) - P_A(right)
CCFC-S = ordinal_consistency(Delta S_F, Delta P_A)
```

必须保留正常、倒序、身份错配、零差异控制，并按 twin 而非 interval 统计。当前
结构控制脚本已完成；正式升级还缺独立 pure-speed swap confirmation，且至少要有
两个模型。DriveVA 的 50-twin 结果只作为模型级负结果，不可替代确认集。

### B. CFAC-S（其次）

CFAC-S 需要单独的 action-to-structure 校准集，把 native action 映射为结构域
剖面，再在完全留出的确认集上比较 `S_F` 与该结构剖面。未完成校准前，CFAC-S
必须为 `unavailable`，不能把 flow 值直接当作米制 action 误差。

### C. Future-to-action mediation（最后）

CCFC-S 只能证明同一干预下的响应一致，不能证明 action 读取了 predicted future。
要做因果声明，必须固定 history 和 command，只干预 future/latent，或者做 future
pathway ablation，并观察 action 是否改变。该通道目前仍是 design-only。

## 决策规则

在 deadband 与时序诊断完成前，不修改 S1.3 的冻结分数，不把 DriveVA 的近随机
结果解释成全局协议失败；在 pure-speed 双模型确认完成前，不把 progress 或
CCFC-S 升级为正式 primary；在 CFAC-S 校准完成前，不报告结构域的正式 CFAC。

## 执行状态（2026-09-11）

- **pure-speed confirmation：未完成。** 当前可访问的 DriveWAM 255 对、DriveVA
  50-twin 和 WorldDrive swap 均不是满足“纯速度互换、同 history、标签互换、
  scene-disjoint calibration/confirmation”的确认集，因此不计入 progress promotion。
  正式输入必须显式声明 `intervention_type=pure_speed`、快慢身份、模型 ID、
  calibration/confirmation split 和 twin 原子 ID。
- **CFAC-S calibration：未完成。** 现有 `evaluate_cfac_fau.py` 只实现米制
  SE(2) CFAC/FAU join，不是结构域的 action→structure 校准。CFAC-S 在独立校准集
  拟合映射、冻结版本和 SHA 之前保持 `unavailable`。
- **当前对外状态：** S1.3 yaw action-response 可报告；progress 仍为
  `diagnostic_only`；CCFC-S 为框架就绪但确认待完成；CFAC-S 为校准待完成。

## 已准备的 pure-speed action roots

已在服务器生成验证专用 action roots：

`/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_runs/pure_speed_confirmation_roots_v2_20260911`

该集合来自 255 个去重 source，使用 temporal-fixed 的 8 点 action 轨迹，按 scene
hash 拆为 calibration `168` twins 和 confirmation `87` twins，共 `510` 个分支；
包含 acceleration `64`、lateral_turn
`104`、braking `52`、straight_cruise `29`、stop `6`，并声明
`model_ids=[drivewam_navsim,epona_nuplan]`。每个 twin 的 fast/slow 分支使用
`1.25×/0.75×` XY translation，yaw 逐点保持不变，history、nuisance 和 source
保持相同。当前状态为 `action_roots_ready_images_pending`：还没有把这组 roots
冒充成模型生成的 future images；必须分别用两个模型生成图像后，才可进入
CCFC-S 的 calibration/confirmation 运行。

此前的 `pure_speed_confirmation_roots_20260911` 使用了旧的 4 点 action 口径，
已被 v2 替代，不得与本确认集混用。
