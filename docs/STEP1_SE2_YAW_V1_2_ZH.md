# Step 1.2 冻结说明：SE(2) yaw 测量

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

数据为同一批 255 个 source、左右 510 条 DriveWAM 分支、四个未来时刻。

| 层级 | 生成帧 | 真实帧 |
|---|---:|---:|
| interval 投影支持 | 1959/2040 = 96.0% | 1004/1020 = 98.4% |
| 单分支四时刻都可投影 | 452/510 = 88.6% | 243/255 = 95.3% |
| 单分支 explained | 386/510 = 75.7% | 217/255 = 85.1% |
| weak | 66/510 = 12.9% | 26/255 = 10.2% |
| abstain | 58/510 = 11.4% | 12/255 = 4.7% |

生成端成对口径：

| 量 | 结果 |
|---|---:|
| 两支四时刻都可投影 | 207/255 = 81.2% |
| 两支都 explained，正式 pair coverage | 175/255 = 68.6% |
| material yaw pair（native action 末点差 >= 0.01 rad） | 103 |
| yaw 方向准确率 | 90/103 = 87.4%，95% CI [79.6%, 92.5%] |
| yaw Spearman | 0.786，bootstrap 95% CI [0.656, 0.886] |

因此旧表的 `88.6%` 不是 CCFC 可计分覆盖率。它是单分支投影覆盖；正式的成对
`explained` 覆盖率是 `68.6%`。

## “准确率”的边界

生成帧没有外部视觉 GT。这里的 `87.4%` 测的是生成视频的 yaw 变化是否与生成该
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
并在同一协议上分离至少两个 WAM。当前只通过方向准确率；coverage 为 68.6%，
且只测了 DriveWAM。因此版本可以冻结并用于 pilot 报告，但不能标为已验证 primary
benchmark，也不能恢复 GitHub 旧版的历史 CFAC/CCFC/FAU 分数。

服务器归档：

```text
/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_runs/kfix_full_20260910/
  ccfc_step1_2_joined.jsonl
  ccfc_step1_2_yaw_primary.json
  real_step1_2_logged_gt_pickle_audit.json
```

冻结 config、生成 pair 结果和 logged-GT pickle 审计的 SHA-256 分别为
`9fba23839bfc229e2504823e26e763589379e0961183a99aefd24ffbdab5e5e4`、
`2cfc37749c255d550f1e7c004a400d12c250a6acbc991334cb86bbb30e627db3`、
`db63790efdf604aab181bd42f5fbe93050ffed45a5047e40aed23f008811cf5f`。
