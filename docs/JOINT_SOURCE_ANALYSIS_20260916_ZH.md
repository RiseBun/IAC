# Source-level joint audit（2026-09-16）

## 完成了什么

找回服务器保留的 1490-row、745-source lineage-fixed DriveWAM AS，以及 978 条独立 NAVSIM PDM rollout，接入本地 745-source RCS。补齐文档曾引用但实际不存在的 `analyse_joint_source_table.py`。

AS 在每个 source 内复用冻结 `history_conditioned_as.aggregate`，输出 coverage-aware `AS_overall`；RCS 使用正式命中条件：动作差实质、视觉响应非零、方向匹配，零动作/零视觉响应不计为命中。旧 `composite_mean` 与 ternary `yaw_match` 不是这两个正式分数。

## 当前探索性结果

| 关联 | 可计算 source | Spearman |
|---|---:|---:|
| AS-overall vs execution success | 574 | 0.123 |
| AS-overall vs execution task score | 574 | 0.149 |
| RCS-yaw vs execution success | 728 | 0.019 |
| RCS-yaw vs execution task score | 728 | 0.042 |

AS 总输入 745 source，其中直行等 source 没有适用的 yaw 通道，完整 AS 不能计算；缺失值保持缺失，不补零。common-10 仍保留为调试报告，不作正式证据。

## 为什么不能作正式有效性验证

动作指纹审计已经确认这个问题不是推测：728 个 source 有交集，但 FCS 执行 `drivewam_native` 分支，而 AS 使用左右命令分支；0/728 条轨迹达到 `1e-6` 精确一致，0/728 达到 `1e-2` 近似一致，最近分支的最大坐标误差中位数为 `12.41`、95 分位为 `39.69`。model revision 标识也不一致。结果见 [`action_alignment_drivewam_as_fcs_20260916.json`](../reports/action_alignment_drivewam_as_fcs_20260916.json)。

因此不能把弱相关解释为指标有效，也不能把近零相关解释为指标无效。尚未完成 bootstrap/置换和场景分层，但更优先的问题是样本/动作同一性，不应先给错配关联添加统计显著性。

## 最短下一步

直接对 AS 已通过 lineage 的 manifest 中左右分支动作运行独立 NAVSIM PDM rollout，保存与视觉分支一致的 source、branch、model revision、seed 和 action fingerprint。先用小样本验证动作指纹完全一致，再扩展；不需要重新生成视频，不需要更换视觉后端。

只有同动作配对闭合后，才能检验“一致性与执行质量是否有关”。即使最终关系弱，AS/RCS 作为一致性指标的定义也不自动失效；是否预测驾驶成功是另一个待验证主张。

产物：

- [`joint_source_analysis_drivewam_full_20260916.json`](../reports/joint_source_analysis_drivewam_full_20260916.json)
- [`analyse_joint_source_table.py`](../tools/analyse_joint_source_table.py)
