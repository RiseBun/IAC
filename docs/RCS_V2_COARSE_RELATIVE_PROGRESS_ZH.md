# RCS v2：粗粒度相对进度响应

## 冻结定义

RCS v2 复用冻结 AS 视觉层，测量同一历史下，动作干预增大时图像中实际表现出的运动是否同向增大。纵向通道不恢复精确速度或绝对距离，只比较同源动作进度最低和最高分支的相对视觉 progress。

正式输入合同：

- 同一组内使用同一 source、历史帧和 nuisance seed；
- WAM 对每个动作分支实际重新生成未来帧，且动作注入路径已验证；
- action endpoint 差至少 `0.5 m`；
- 视觉测量不读取候选动作；
- 正式统计中每个独立日志/场景最多贡献一个 counterfactual group。

单组命中条件为：

\[
(p_{high}^{act}-p_{low}^{act})(p_{high}^{vis}-p_{low}^{vis})>0
\]

动作端已按 low/high 排序，所以等价于 `visual(high) > visual(low)`；零视觉响应不给分。

正式门槛在看结果前冻结为：至少 30 个独立且动作合格的 source、coverage `>= 0.9`、endpoint accuracy `>= 0.75`、Wilson 95% CI 下界 `> 0.5`。相邻档位排序、完整多档单调性、细粒度速度和原始米制误差只作诊断。

## 独立 64 日志实验

64 个独立 NAVSIM mini 日志各取一个窗口并为 `stop / normal-progress` 分支重新生成，共 128 条视频。正式结果为：

| 指标 | 结果 |
|---|---:|
| 动作差合格的独立 source | 36 |
| 有视觉结果 | 35 |
| coverage | 35/36 = 97.2% |
| 端点方向命中 | 26/35 |
| `RCS-relative-progress` | **0.743** |
| Wilson 95% CI | **[0.579, 0.858]** |
| 晋级结果 | **未通过** |

该结果显著高于随机，说明 Epona 的图像确实包含较强的粗粒度纵向响应；但点估计比冻结的 `0.75` 门槛低 `0.007`，因此不能把 progress 通道宣布为已验证或正式晋级，也不事后移动阈值。

事后诊断显示失败主要集中在较弱干预，而不是视觉层大面积拒答：仅 1/36 组无视觉结果；动作差 `0.5–1.5 m`、`1.5–3 m`、`>=3 m` 的命中率分别为 `4/7=57.1%`、`8/11=72.7%`、`14/17=82.4%`。这说明当前瓶颈是 Epona 对中小纵向干预的图像响应不稳定。该分层不改变正式阈值与总分。

## 相关窗口 pilot 的纠正

此前 `n=40` 得到 `27/35 = 0.771`，但 40 个窗口全部来自同一个日志且互相重叠。它只能作为相关窗口 pilot，独立 source 数是 1，不能作为正式统计证据。评分器现已自动审计并将该报告标记为 `pilot`、`promotion_pass=false`。

## 综合分数

冻结公式仍为：

\[
RCS = 0.5\,RCS_{yaw} + 0.5\,RCS_{relative-progress}
\]

Epona 当前 `RCS-yaw=0.875`，独立日志实验的 `RCS-relative-progress=0.743`，数值合成为 `RCS=0.809`（80.9/100，bootstrap 95% CI `[0.718, 0.893]`）。由于 progress 通道未通过晋级门槛，这只是完整性诊断值，状态为 `not_promoted`，不是正式 RCS 成绩。

## 产物

- 冻结配置：[response_consistency_v2.json](../configs/response_consistency_v2.json)
- 独立日志结果：[rcs_v2_relative_progress_epona_multilog64_20260916.json](../reports/rcs_v2_relative_progress_epona_multilog64_20260916.json)
- 相关窗口 pilot：[rcs_v2_relative_progress_epona_n40_20260916.json](../reports/rcs_v2_relative_progress_epona_n40_20260916.json)
- 综合诊断：[rcs_v2_epona_composite_20260916.json](../reports/rcs_v2_epona_composite_20260916.json)

声明边界：该指标只验证大幅动作干预下的相对视觉运动响应，不声称精确米制距离、精确速度或完整轨迹因果控制。
