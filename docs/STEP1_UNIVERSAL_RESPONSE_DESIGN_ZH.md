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
RCS = ordinal_agreement(Delta F, Delta native_action)
```

有符号差分本身会抵消同一 source 的加性共同运动；幅度归一化在证明对加性
共同运动不变之前只能作 diagnostic，不能进入正式分数。这个限制是最小证明中
主动发现并写入协议的，而不是隐藏在实现里。该参数化保留反事实问题本身，并尽量
抵消场景外观、绝对深度和共同运动。它不输出米、速度或曲率；这些只有在独立通道
通过晋级门后才能进入正式协议。

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

## 最小证明

[`../tools/prove_step1_universal_response.py`](../tools/prove_step1_universal_response.py)
在同一 pair scorer 上运行四类合成控制。当前证明结果是：正常方向 `1.0`、反转
方向 `0.0`、identity/zero 均为 `unavailable`，加性共同运动改变前后的有符号
差分误差约为 `3e-15`。这只证明协议的代数不变量和 fail-closed 行为，不证明
任何真实 WAM 有效；真实数据上的跨架构实验仍是下一步。

对已有 Epona、DriveWAM、DriveVA、WorldDrive twin-control 产物的真实数据桥接审计
已经完成，见 [`../reports/step1_universal_response_real_control_audit_20260912.json`](../reports/step1_universal_response_real_control_audit_20260912.json)。
四个模型都缺少同口径的 `identity_swap` 控制，因此 `promoted_model_count=0`，
新 Step1 尚未晋级。暂时忽略 identity 缺口时，WorldDrive 的小样本方向 CI 通过，
Epona/DriveWAM 的有效方向覆盖或 CI 仍不足，DriveVA 也未通过 CI；这正是需要
补原始 identity 控制和扩大 holdout 的证据，而不是可以发布的普适结果。

## 原始流四控制最小实验（2026-09-12）

随后对四个模型的归档原始 flow 数组重算了同一套控制，避免把旧报告中
“reversed 的负余弦”误当成控制通过。`normal` 要求正向命中；`reversed_action`
和 `identity_swap` 是负控制，要求正向假命中率不超过 0.25；`zero_contrast`
把两支观测置为共同中点，方向估计必须是 `unavailable`，不能填 0 或伪造方向。
结果见 [`../reports/step1_universal_response_raw_control_audit_20260912.json`](../reports/step1_universal_response_raw_control_audit_20260912.json)：

| 模型 | twin 数 | normal coverage | normal direction | 95% CI | reversed/identity 假阳性 | 模型门 |
|---|---:|---:|---:|---|---:|---|
| Epona | 59 | 47.5% | 85.7% | [71.4%, 96.4%] | 14.3% / 14.3% | 未通过 coverage/CI |
| DriveWAM | 86 | 62.8% | 48.1% | [35.2%, 61.1%] | 51.9% / 51.9% | 未通过 |
| DriveVA | 10 | 100.0% | 80.0% | [50.0%, 100.0%] | 20.0% / 20.0% | 未通过 CI |
| WorldDrive | 13 | 100.0% | 92.3% | [76.9%, 100.0%] | 7.7% / 7.7% | 仅 pilot 通过 |

因此当前只有 1 个模型通过全部门槛，远未达到至少 3 个架构的正式晋级条件。
这轮实验确认了原始数据和控制实现可统一重算，但没有证明新通道已经是普适
MAS/RCS；Epona、DriveWAM 的 coverage/方向问题仍是真实限制，DriveVA 的样本量
和 CI 仍不足。`zero_contrast` 的共同中点控制仅验证 fail-closed 行为，不把
它解释为模型没有响应。

## Step1 重构：共享证据层

当前结论不再是继续把一个 yaw 标量调成“通用指标”，而是让 Step1 输出四类
相互独立的证据：

1. `temporal_motion`：单支未来中的时序运动和持续性；
2. `counterfactual_response`：同源左右分支的有符号视觉差分；
3. `grounding`：生成 future 与外部 logged future 的结构对齐；
4. `reliability`：支持率、深度有效率和弃权原因。

MAS 消费单支 temporal/action evidence，RCS 消费 twin response evidence，GS
消费 grounding evidence；FCS 仍需要独立的 future-only intervention，不能由
Step1 单独推出。实现入口是
[`../src/iac_new/step1_evidence.py`](../src/iac_new/step1_evidence.py)，协议配置是
[`../configs/step1_evidence_layer_v1.json`](../configs/step1_evidence_layer_v1.json)。

深度在这一版中是可选 adapter，而不是硬依赖。它可以帮助判断几何有效性和
grounding 质量，但不能：按候选动作筛像素、把缺失的视觉响应补出来、或在没有
独立真实验证时宣称米制轨迹准确。这样可以先让没有深度输出的 WAM 参评，同时
保留深度作为后续 GS/结构质量的增强通道。

## 多通道 raw-flow 首轮结果

把四个模型的原始 flow 转成共享证据后，再按 source/twin 聚合，而不是把四个
interval 当成四个独立样本。结果见
[`../reports/step1_evidence_channel_audit_20260912.json`](../reports/step1_evidence_channel_audit_20260912.json)。

| 模型 | 通道 | pair coverage | 方向命中 | 95% CI | 解释 |
|---|---|---:|---:|---|---|
| Epona | yaw | 0% | — | — | action yaw 无左右差异 |
| Epona | lateral | 94.9% | 37.5% | [25.0%, 50.0%] | 描述子方向不稳定 |
| Epona | longitudinal | 94.9% | 60.7% | [48.2%, 73.2%] | 有序信号但不足 |
| DriveWAM | yaw | 95.3% | 36.6% | [26.8%, 47.6%] | 方向接近随机 |
| DriveWAM | lateral | 95.3% | 37.8% | [26.8%, 48.8%] | 方向接近随机 |
| DriveWAM | longitudinal | 95.3% | 39.0% | [29.3%, 48.8%] | 方向接近随机 |
| DriveVA | yaw | 100% | 60.0% | [30.0%, 90.0%] | 样本太小 |
| WorldDrive | yaw | 100% | 92.3% | [76.9%, 100%] | 仅此通道达到 pilot gate |

这轮结果说明“增加 lateral/纵向字段”本身不会自动解决 Step1；描述子必须和
动作分量建立独立的 real-only 映射，并通过负控制。当前只有 WorldDrive 的 yaw
通道达到 pilot 门，仍不能升级为普适 MAS/RCS。GS 的 grounding 通道尚未在这轮
raw-flow 桥接中计算，因为归档中没有同口径的 external reference profile；这
不是填零，而是 `unavailable`。
