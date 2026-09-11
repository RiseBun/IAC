# 结构反事实指标：Step1 到 MAS / RCS / GS

## 目的

本协议的目标不是恢复所有未来帧的米制真值，而是测量 WAM 的预测未来与 native
action 是否在同一反事实干预下保持一致。它是图像侧观测协议，不宣称完成
logged-GT 几何保真度重建。

## 测量契约

对同一 `source_key` 的左右分支，分别计算候选盲结构描述 `S_F(t)`，再计算：

```text
C(t) = (S_F(left, t) + S_F(right, t)) / 2
D(t) = S_F(left, t) - S_F(right, t)
D_norm(t) = D(t) / max(abs(C(t)), epsilon)
```

方向使用原始 `D(t)`；归一化值只作跨场景幅度诊断。缺少共同有效 interval 的 pair
不填零，直接标为 unavailable。时间持续性报告共同 interval 中差分符号的一致程度。

## 下游映射

* `Response Consistency Score (RCS)`（旧名 `CCFC-S`）：比较 `Delta S_F` 与
  `Delta P_A` 的方向、排序和时间持续性。
* `Motion Alignment Score (MAS)`（旧名 `CFAC-S`）：在冻结的
  action-to-structure 标定后，比较单分支结构剖面与 action 结构剖面。
* `Grounding Score (GS)`（旧组件名 `FAU`）：仍需 logged-GT-compatible 的独立
  通道；本协议不提供米制 GT 误差。
* `FCS`：继续由 native action 和独立模拟器完成，不依赖本协议。

## 反事实边界

`Delta S_F` 与 `Delta P_A` 的一致性证明 action-state consistency，不单独证明
action decoder 实际读取了 predicted future。若要声称 future-to-action 的因果依赖，
还必须做 future-only intervention：固定 history 和 command，只改变 future/latent，
观察 action 是否改变；或做 future pathway ablation。

## 验收

progress 通道在独立 speed-swap twin 上达到以下条件前只能作 diagnostic：twin coverage
至少 0.90、方向准确率 95% CI 下界至少 0.75、至少两个 WAM、并通过正常顺序/倒序/
错身份/零差异控制。所有阈值必须在独立校准集冻结后再用于确认集。

## GS 跨模型 pilot（2026-09-11）

在同一 source 的生成分支与 NAVSIM logged future 之间做了 candidate-blind 结构对照，
并用随机身份置换作为负对照。Epona 的生成流幅度与 logged future 的分支级秩相关为
`rho=0.835`（逐区间 `rho=0.725`），水平流为 `rho=0.641`；随机置换的幅度相关
95% 上界为 `0.158`。DriveWAM 的对应幅度相关为 `rho=-0.042`（分支聚合
`rho=0.122`，未超过随机置换 95% 上界 `0.135`），生成幅度中位数为 `0.85 px`，
而 logged future 为 `47.9 px`。

该 pilot 说明 GS 有能力区分保留真实运动结构的模型与未保留的模型，但它仍不是冻结
标量：当前结果使用 confirmation-only 的原始结构比较。正式发布前必须在独立 calibration
split 冻结 descriptor 聚合、尺度和缺失策略，再在 untouched confirmation split 验收；GS
也不能单独被解释为 future-to-action 因果证明。

基于 real-only calibration 的候选聚合（`exp(-|delta|/scale)`，descriptor/interval/branch
均取中位数，至少 3 个有效 interval）在 confirmation 上得到：Epona 中位 GS `0.553`
（coverage `94.9%`），DriveWAM 中位 GS `0.191`（coverage `94.2%`）。Epona 的随机
身份置换均值为 `0.303`，DriveWAM 为 `0.234`，因此该候选分数同时显示模型差异和
负对照方向。不过 calibration 目前只冻结了真实域尺度，生成域 calibration split 尚未
独立完成；上述数值仍标记为 candidate，不进入正式排行榜。

随后用 115 个与 confirmation 不重合的 Epona source（230 个分支）完成了生成域
calibration 验证：候选 GS coverage `98.3%`、中位数 `0.503`，随机身份置换均值
`0.289`（95% 上界 `0.303`）。因此聚合规则在 source-disjoint 生成数据上保持稳定。
正式协议仍需将该规则写入冻结配置，并在 untouched confirmation 上按预注册门槛验收。

按 source/twin 原子做 20,000 次 bootstrap 后，Epona confirmation 的 GS 中位数为
`0.550`（95% CI `[0.487, 0.635]`），DriveWAM 为 `0.191`（`[0.158, 0.215]`），
两模型 paired 差值为 `0.314`（`[0.251, 0.340]`）。这些区间均按 source 聚类计算，
不是把左右分支当作独立样本；因此 GS candidate 已达到“可冻结候选”的证据门槛，但
正式发布仍需完成最终协议版本号和 SHA 固定。
