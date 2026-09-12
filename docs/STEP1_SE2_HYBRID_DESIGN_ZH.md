# Step1：固定支持的轨迹条件视觉似然

## 为什么回到 SE(2)

Step1 的任务不是预测一个视频统计量，而是回答一个条件问题：给定动作/轨迹假设，生成视频中的光流是否支持它。SE(2) 的估计量正好表达这个问题；FoE、散度、旋度等结构量只能作为诊断，不能替代“这条给定轨迹是否解释图像”。

## 修正点

旧实现把候选轨迹的投影有效性混入证据分母。某条候选把像素投影出画面后，支持点变少，可能通过常数损失或零轨迹取得虚假的优势。

新通道在比较任何候选前先冻结支持集：`ROI ∩ observed-flow-validity`。随后分别报告：

- `fixed_support_fraction`：输入视频本身提供了多少固定证据；
- `projection_valid_fraction`：当前候选在这些固定证据上还能投影多少；
- `median_residual_px`、方向余弦和连续 `likelihood`；
- `scored / weak / unavailable` 状态。

投影不足只产生 `unavailable`，不填 0、不填 robust 上限，也不计作通过。

## MAS 与 RCS

- MAS：单支视频与该支动作轨迹的条件似然；用于判断“这支视频是否视觉上支持它收到的动作”。
- RCS：同一 source 的左右分支差分与左右动作差分的条件似然；同时报告反转动作和零差异控制。

两者都不重建米制轨迹，也不单独证明 future-to-action 因果关系；FCS/路径消融仍需独立实验。

## 当前四模型 pilot

详见 [`reports/step1_se2_hybrid_pilot_20260912.json`](../reports/step1_se2_hybrid_pilot_20260912.json)。结果是异质的：WorldDrive、Epona 的 RCS 有明显控制分离，DriveVA 较弱，DriveWAM 接近随机。因此这一版完成了正确估计量和诚实缺失值机制，但尚未达到正式发布门槛。
