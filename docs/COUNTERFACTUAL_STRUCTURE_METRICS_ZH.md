# 结构反事实指标：Step1 到 CCFC-S / CFAC-S

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

* `CCFC-S`：比较 `Delta S_F` 与 `Delta P_A` 的方向、排序和时间持续性。
* `CFAC-S`：在冻结的 action-to-structure 标定后，比较单分支结构剖面与 action
  结构剖面。
* `FAU`：仍需 logged-GT-compatible 的独立通道；本协议不提供米制 GT 误差。
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
