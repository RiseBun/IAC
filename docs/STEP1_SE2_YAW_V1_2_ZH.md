# Step 1.2 冻结说明：SE(2) yaw 测量

> **时间契约勘误（2026-09-10）：本页旧 DriveWAM 生成端数字已撤回。**
> 旧输入把 4 帧历史和 8 帧未来拼成 12 帧，官方读取器却把第 0 帧当作当前时刻；
> 模型实际读取约 `[-1.5,-0.5,+0.5,+1.5,+2.5]s`，与以 `t=0` 为锚点的动作和
> 输出标签错位。真实帧 logged-GT 准确率不受影响。生成端 coverage、方向准确率、
> 下表已经替换为修复成 `current + 8 future` 后的 `temporal-fixed` 重跑结果；
> 真实帧 logged-GT 准确率不受影响。

## 决策

恢复连续地平面 SE(2) 作为 Step 1 主测量后端，但不恢复 GitHub 旧版的有效性逻辑和
字段策略。冻结版本为 `iac-step1-se2-yaw-v1.2`，状态为 `frozen_pilot`：

- primary measurement：两支都 `explained` 后的末点 yaw 响应；
- primary statistics：方向准确率和 Spearman；
- diagnostic：lateral、curvature、纵向距离/速度、旧 CFAC/FAU composite；
- Step 1-S：独立的流场结构交叉检查，不再是主后端。

## 与 GitHub 旧版的差异

对照基线为 `origin/main@2eaec6b`。

| 项目 | GitHub 旧版 | Step 1.2 |
|---|---|---|
| 内参尺寸 | 从已缩放文件尺寸推断 | manifest 必须显式给出 `intrinsics_source_size`，并与冻结的 `1920x1080` 一致 |
| 像素取样 | 全局按可靠性权重取 top-900 | 固定中远场、空间分层取样 |
| 无效投影 | 从损失分母删除；全空时返回常数 4.0 | 固定输入证据分母，无效候选投影承担上限代价 |
| 可观测性 | 只检查输入 ROI/FB/权重 | 同时检查拟合输出的逐 interval 投影支持 |
| 输出状态 | 顶层 `valid=True`，没有拟合质量状态 | `measurement_available` 与 `explained/weak/abstain` 分开报告 |
| 单分支计分 | 输入门通过即可 | 仅 `motion_explanation_status=explained` |
| 成对计分 | 没有双方解释门 | 左右两支都 `explained` |
| primary 字段 | lateral + yaw + curvature | yaw；其余字段仅 diagnostic |
| primary 统计 | 混合幅度 composite | yaw 方向准确率 + Spearman |

这些修改解决了两个不同问题。显式内参尺寸修复了生成图像 `448x256` 与标定
`1920x1080` 混用造成的几何错误；输出投影和拟合状态修复了“无有效投影仍返回
初始化轨迹并报告有效”的不诚实行为。空间分层减轻近场像素集中，但不是准确度保证。

数值上，GitHub 旧代码会把顶层 `valid` 固定写成 true，因此表面的覆盖接近 100%，
但这个数没有测量含义。后来把输出投影门补上、仍保留错误内参时，生成端只有约
`18.4%` explained、约 `20%` interval 投影支持；修正内参后分别变为 `75.7%` 和
`96.0%`。提升主要来自标定 bug 修复，不是放宽弃权门。

## 冻结结果

数据为同一批 255 个 source、左右 510 条 temporal-fixed DriveWAM 分支、四个未来
时刻。

| 层级 | 生成帧 | 真实帧 |
|---|---:|---:|
| interval 投影支持 | 1960/2040 = 96.1% | 1004/1020 = 98.4% |
| 单分支四时刻都可投影 | 459/510 = 90.0% | 243/255 = 95.3% |
| 单分支 explained | 380/510 = 74.5% | 217/255 = 85.1% |
| weak | 79/510 = 15.5% | 26/255 = 10.2% |
| abstain | 51/510 = 10.0% | 12/255 = 4.7% |

生成端成对口径：

| 量 | 结果 |
|---|---:|
| 两支四时刻都可投影 | 218/255 = 85.5% |
| 两支都 explained，正式 pair coverage | 171/255 = 67.1% |
| material yaw pair（native action 末点差 >= 0.01 rad） | 106 |
| yaw 方向准确率 | 87/106 = 82.1%，95% CI [73.7%, 88.2%] |
| yaw Spearman | 0.832，bootstrap 95% CI [0.731, 0.902] |

因此 `90.0%` 不是 CCFC 可计分覆盖率。它是单分支投影覆盖；正式的成对
`explained` 覆盖率是 `67.1%`。

## G 初始化器消融（Step 1.3 候选）

在不改变像素、光流、投影门、`minimum_fit_improvement=0.05` 或 primary 字段的
前提下，G 先用固定、候选无关的速度—曲率粗网格寻找可达起点，再运行原局部优化。
它只修复“局部优化从坏起点出发”的覆盖损失，不改变弃权定义。`temporal-fixed`
DriveWAM 全量结果为：

| 量 | Step 1.2 | G 候选 |
|---|---:|---:|
| 单分支 explained | 380/510 = 74.5% | 442/510 = 86.7% |
| 两支都 explained | 171/255 = 67.1% | 205/255 = 80.4% |
| material yaw pair | 106 | 116 |
| yaw 方向准确率 | 82.1% [73.7%, 88.2%] | 81.9% [73.9%, 87.8%] |
| yaw Spearman | 0.832 [0.731, 0.902] | 0.925 [0.861, 0.960] |

G 回收 34 对、丢失 0 个原可评分 pair；在原 106 个共同 material pair 上，方向
命中由 87 提到 88。代价是解码时间约为 Step 1.2 的 2.5 倍。

同一 255 条真实帧 logged-GT 验证中，G 的 `explained` 从 217/255 提到
231/255；yaw 读出比例中位数从 `0.916` 提到 `0.939`，Spearman 从 `0.918`
提到 `0.992`，方向准确率为 `120/122 = 98.4%`。因此当前证据支持 G 是真实的
初始化改进，而不是用错误拟合换取覆盖。但其生成端 pair coverage 仍只有 80.4%，
配置继续标为 `experimental_ablation`，不回写冻结 Step 1.2。

逐 interval 的候选无关诊断也说明 SE(2) 覆盖的剩余边界。只汇总左右两支同时
`explained` 的 interval，并在相同 interval 上比较 yaw 增量时，G 的覆盖—准确率为：

| 最少共同 explained interval | pair coverage | material yaw 方向准确率 | Spearman |
|---:|---:|---:|---:|
| 4/4 | 161/255 = 63.1% | 82/99 = 82.8% | 0.934 |
| >=3/4 | 207/255 = 81.2% | 101/119 = 84.9% | 0.951 |
| >=2/4 | 227/255 = 89.0% | 108/127 = 85.0% | 0.949 |
| >=1/4 | 240/255 = 94.1% | 110/129 = 85.3% | 0.949 |

`>=2/4` 已保持较强方向和排序，但覆盖仍低于 90%；`>=1/4` 虽越过覆盖门，单个
时刻不足以支撑稳定的四秒轨迹声明。因此该表只作为“可识别 interval 的 yaw 增量”
诊断，不替代严格 terminal-yaw primary，也不通过改变门槛追求过线。

上表的正式数值必须由归档脚本重算后再冻结；当前发布门仍使用整条轨迹的双方
`explained`，不允许把缺失 interval 插补为可测。高覆盖的方向/排序由独立的
S1.3 流场结构测量承担，SE(2)-G 保留为米制诊断层。

## “准确率”的边界

生成帧没有外部视觉 GT。这里的 `82.1%` 测的是生成视频的 yaw 变化是否与生成该
分支的 native action 同向；它证明条件动作与视觉响应一致，不证明该反事实未来与
现实 logged GT 一致。

真实帧准确率不读取 manifest 的 `gt_candidate_id`，而是沿
`lineage.source_sample` 打开 NAVSIM pickle，并逐条核对
`future_trajectory[*].pose == realized_future_ego_state[:3]`。255 个 source 中
217 条为 explained；采用预注册 material 阈值 x >= 1.0 m、|y| >= 0.3 m、
|yaw| >= 0.05 rad 后：

| 字段 | n | 末点读出比例中位数 | IQR | 方向准确率 | Spearman |
|---|---:|---:|---:|---:|---:|
| longitudinal | 217 | 0.565 | [0.314, 0.734] | 100% | 0.509 |
| lateral | 123 | 0.480 | [0.212, 0.646] | 95.1% | 0.926 |
| yaw | 117 | 0.916 | [0.695, 0.974] | 116/117 = 99.1% | 0.918 |

其中 lateral-turn 有 92 条 explained，91 条达到 lateral/yaw material 阈值；其
yaw 比例中位数为 `0.930`。因此“严格 logged-GT 子集只有 85 条且没有转弯”的旧
说法撤回。它来自把另一份 private manifest 的 `gt_candidate_id` 当成 logged GT。

旧 manifest 的该字段实际指向 `wam_action_head`。这不是命名问题，而是参考身份
错误：任何按字段读取的下游都会把模型输出当真值。Step 1.2 已将生成 manifest 的
`gt_candidate_id` 置空，并在协议层拒绝把动作头声明为 GT；只有显式标注为
`navsim_logged_realized` 或 `private_realized_gt` 的候选可通过冻结 reference
契约。生成帧的 action-alignment 与真实帧的 logged-GT accuracy 必须分开报告。

## 发布状态

预注册升级条件不变：pair coverage >= 90%，方向准确率 95% CI 下界 >= 0.75，
并在同一协议上分离至少两个 WAM。temporal-fixed 重跑的 coverage 为 67.1%，方向
CI 下界为 73.7%，两项均未通过；同时仍缺少同分布的第二个 WAM。因此不能标为已
验证 primary benchmark。

服务器归档：

```text
/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_runs/temporal_contract_fix_20260910/
  ccfc_manifest_validated.jsonl
  se2_v1_2/merged.jsonl
  se2_v1_2/counterfactual_alignment.json
```

旧 `kfix_full_20260910` 生成端 SHA 已被时间契约勘误取代；temporal-fixed 产物的
SHA-256 在本轮归档完成后记录。
