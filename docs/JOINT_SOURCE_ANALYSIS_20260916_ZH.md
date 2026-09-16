# Source-level joint analysis（2026-09-16）

## 目的

检查视觉一致性分数是否与独立闭环执行结果在同一 source 上共同变化。该分析只做描述性关联，不把相关性解释为预测能力或因果关系。

## 数据契约

- AS 使用当前 common-10 DriveWAM 结果；AS 分支先按 source 聚合；
- RCS 使用 lineage-fixed DriveWAM yaw pairs；
- FCS 使用 NAVSIM PDM 独立闭环 rollout，任务成功来自 `pdm_score`，不读取生成图像；
- source 通过 `source_key` 精确连接；缺失通道保持缺失，不补零；
- Spearman 只在两个通道都存在的 source 上计算。

## 结果

| 关联 | source 数 | Spearman | 解释边界 |
|---|---:|---:|---|
| AS vs FCS success | 10 | 0.326 | common-10 pilot，不能作正式相关性结论 |
| AS vs FCS task score | 10 | 0.035 | common-10 pilot，连续分数几乎无排序证据 |
| RCS-yaw vs FCS success | 728 | 0.061 | 现有 RCS/FCS source overlap 的描述性结果 |
| RCS-yaw vs FCS task score | 728 | 0.080 | 不能解释为无效，只能说明该 split 未见明显单调关系 |

## 决策

这一步没有证明“视觉一致性高就一定执行得好”。它也没有推翻视觉层：RCS/AS 测量的是生成视觉与动作的结构一致性，FCS 测量的是独立闭环任务结果，两者并非同一能力。

真正的缺口是 AS 与 FCS 尚未在至少 30 个 source 的同一正式公共池上对齐。下一步应扩大 common pool，并把 FCS rollout 的 source manifest 固定为该池；在此之前，不应把 joint correlation 写成主结果，也不应据此调整 AS/RCS 阈值。

机器可读结果：[`joint_source_analysis_drivewam_common10_20260916.json`](../reports/joint_source_analysis_drivewam_common10_20260916.json)。
