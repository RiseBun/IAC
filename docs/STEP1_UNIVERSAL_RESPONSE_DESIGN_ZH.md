# Step1 Universal Response：面向多数 WAM 的实验设计

当前 S1.3 的 `horizontal_flow_center` 已经能在 Epona 和 DriveWAM 上验证
yaw 方向，但它不是完整运动测量器，也没有证明对多数架构都有效。本设计不把
它继续扩展成一个伪造的米制解码器，而是把 Step1 重定义为**反事实视觉响应层**。

## 核心问题

给定同一 history 和一对动作干预，生成未来是否产生了：

1. 与动作方向一致的视觉响应；
2. 在同一 source 内可区分的响应差异；
3. 跨时间持续而不是单帧噪声；
4. 在正常、反转、身份错配、零差异控制下表现出正确关系。

核心量是同源差分，而不是从光流恢复米制轨迹：

```text
Delta F = F_left - F_right
Delta F_normalized = Delta F / robust_scale(common_source_motion)
RCS = ordinal_agreement(Delta F_normalized, Delta native_action)
```

这种参数化保留反事实问题本身，并尽量抵消场景外观、绝对深度和共同运动。
它不输出米、速度或曲率；这些只有在独立通道通过晋级门后才能进入正式协议。

## 通道与晋级

| 通道 | 当前地位 | 说明 |
|---|---|---|
| `yaw_direction` | 已有 validated adapter | S1.3 的水平流中心 |
| `longitudinal_order` | candidate | 只允许做 pairwise 顺序，不能报告绝对速度 |
| `lateral_direction` | unvalidated | 必须先解决旋转/平移歧义 |
| `temporal_persistence` | candidate | 要求共同 interval 上符号持续 |

通道彼此独立晋级。任何通道必须同时满足：覆盖率至少 90%、方向准确率的
95% CI 下界至少 0.75、source-disjoint real-only calibration、四类控制以及
至少三个不同架构族。未通过的通道是 `diagnostic` 或 `unavailable`，不会被
拼进 MAS/RCS。

## MAS 与 RCS

`RCS` 是优先主线，因为同源左右差可以消除更多共同误差。`MAS` 只有在存在
冻结的基准/零动作响应，且动作到视觉结构映射通过 real-only 校准后才计分；否则
该分支是 `unavailable`，不能靠单支视频硬凑一个分数。

## 与因果性的边界

响应签名证明的是视觉—动作一致或反事实响应。它不证明动作由预测未来产生。要
使用 future-to-action 因果措辞，仍需独立的 future-only intervention、blocked
pathway 和 fixed-action control。

实验契约见 [`../configs/step1_universal_response_v1.json`](../configs/step1_universal_response_v1.json)，
其状态固定为 `experimental_contract_not_promoted`。
