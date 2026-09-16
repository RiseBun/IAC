# Source-level joint audit（2026-09-16）

## 完成了什么

找回服务器保留的 1490-row、745-source lineage-fixed DriveWAM AS，以及新生成的 1456 条同动作独立 NAVSIM PDM rollout，接入本地 745-source RCS。补齐文档曾引用但实际不存在的 `analyse_joint_source_table.py`。

AS 在每个 source 内复用冻结 `history_conditioned_as.aggregate`，输出 coverage-aware `AS_overall`；RCS 使用正式命中条件：动作差实质、视觉响应非零、方向匹配，零动作/零视觉响应不计为命中。旧 `composite_mean` 与 ternary `yaw_match` 不是这两个正式分数。

## 同动作配对结果

全量 lineage-fixed manifest 的 1490 个分支中，1456 条 NAVSIM PDM rollout 成功，34 条因 metric cache 缺失而 unavailable。成功 rollout 的动作 fingerprint 与视觉 manifest 为 `1456/1456` 精确一致；FCS success rate 为 `55.08%`，Wilson 95% CI `[52.52%, 57.62%]`。

| 关联 | 可计算 source | Spearman | log-cluster bootstrap CI |
|---|---:|---:|---:|
| AS-overall vs execution success | 574 | 0.158 | log-cluster bootstrap CI `[0.077, 0.241]` |
| AS-overall vs execution task score | 574 | 0.168 | log-cluster bootstrap CI `[0.080, 0.249]` |
| RCS-yaw vs execution success | 728 | 0.086 | log-cluster bootstrap CI `[0.009, 0.159]` |
| RCS-yaw vs execution task score | 728 | 0.032 | log-cluster bootstrap CI `[-0.040, 0.106]` |

AS 总输入 745 source，其中直行等 source 没有适用的 yaw 通道，完整 AS 不能计算；缺失值保持缺失，不补零。common-10 仍保留为调试报告，不作正式证据。

## 为什么不能作正式有效性验证

旧的 native rollout 与 AS 的错配已经被修复；旧审计仍保留在 [`action_alignment_drivewam_as_fcs_20260916.json`](../reports/action_alignment_drivewam_as_fcs_20260916.json)，作为失败回归案例。当前正式 matched rollout 的审计结果见 [`action_alignment_drivewam_as_matched_20260916.json`](../reports/action_alignment_drivewam_as_matched_20260916.json)。

同动作配对后，AS 与执行 task score 呈小幅正相关，且 log-cluster bootstrap 区间不跨 0；RCS-yaw 与连续 task score 的区间跨 0。这个结果只能作为“指标与执行结果存在部分关联”的探索性证据，不能作为预测性能、因果性或统一质量排序。

## 最短下一步

下一步是固定 source/log 分层和预注册 bootstrap/置换检验，并把 matched rollout 的 branch、model revision、seed 元数据补齐到公开 evidence packet。不需要重新生成视频，也不需要更换视觉后端。即使最终关系弱，AS/RCS 作为一致性指标的定义也不自动失效；是否预测驾驶成功是另一个待验证主张。

产物：

- [`joint_source_analysis_drivewam_full_20260916.json`](../reports/joint_source_analysis_drivewam_full_20260916.json)
- [`joint_source_analysis_drivewam_matched_20260916.json`](../reports/joint_source_analysis_drivewam_matched_20260916.json)
- [`fcs_drivewam_as_matched_score_20260916.json`](../reports/fcs_drivewam_as_matched_score_20260916.json)
- [`analyse_joint_source_table.py`](../tools/analyse_joint_source_table.py)
