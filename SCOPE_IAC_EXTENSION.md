# IAC：DINO 时序运动头与光流速度证据模块

这组代码把我们当前最容易交给 IAC 复现的两项工作整理成了独立、可选的模块。它不会改变仓库中原有训练入口和默认配置。

## 1. 这是否只是“给 DINO 加了一个头”

从工程角度可以这样简称，但更准确的描述是：

> 在冻结的 DINOv2 帧特征上增加一个**候选轨迹盲、显式使用时间顺序**的运动属性头；视觉端先独立估计运动，之后才用带不确定性的比较器检查候选轨迹是否与这些视觉证据一致。

这里有三个关键约束：

1. 运动头的 `forward` 只接受历史/未来图像的 DINO 特征，不能看到候选轨迹，避免从候选轨迹反推“视觉进度”。
2. 它不是只预测一个 progress 标量，而是预测 IAC 已定义的 36 维全局与分时段运动属性，包括纵向、横向、转向、速度变化与转向时机。
3. 它同时输出每个属性的不确定性。候选轨迹只在第二阶段进入比较器；不确定证据的惩罚较弱，明确矛盾的证据不能互相抵消。

因此，这个改动与 IAC 已有的标量 `progress_alignment_head`、确定性的 `motion_rule_visual_head` 兼容，但不是简单重复：它强化了时间建模、候选盲隔离和不确定性比较。训练仍复用 IAC 现有的分组排序、反事实样本、checkpoint 与评测代码。

## 2. 文件与接口

- `iac_extensions/dino_motion_head.py`：可独立复用的 DINO 时序运动头、比较器和异方差回归损失。
- `train_scope_motion_head.py`：对当前 `DINOv2ConsistencyCritic` 的薄封装，不复制 IAC 训练循环。
- `eval_scope_motion_head.py`：使用 IAC 原评测器读取新 checkpoint。
- `configs/train_navsim_future_dinov2_scope_motion_head.py`：默认 36 维、3 个未来时段的实验配置。
- `iac_extensions/flow_evidence.py`：DIS/Farneback 光流统计、速度目标、Ridge 速度头和一致性能量。
- `tools/flow_speed_head.py`：光流速度头的 `fit` / `apply` 命令行工具。
- `tests/test_scope_research_extensions.py`：模块形状、梯度、候选比较、光流和模型序列化测试。

DINO 运动头的主要输出为：

- `visual_motion_rule_pred`：图像独立预测的 36 维运动属性；
- `visual_motion_rule_logvar`：逐属性不确定性；
- `traj_motion_rule_target`：候选轨迹的对应属性；
- `scope_motion_energy`：越小表示视频越支持这条候选轨迹；
- `motion_rule_match_logit`：复用 IAC 现有一致性损失和最终分数融合的 logit。

## 3. 训练和评测 DINO 运动头

先沿用 IAC 配置中的数据索引路径。单卡训练：

```bash
python train_scope_motion_head.py \
  --config configs/train_navsim_future_dinov2_scope_motion_head.py \
  --work-dir work_dirs/iac_navsim_future_dinov2_scope_motion_head \
  --dinov2-freeze --amp
```

也可从现有最佳 checkpoint 小学习率继续训练：

```bash
python train_scope_motion_head.py \
  --config configs/train_navsim_future_dinov2_scope_motion_head.py \
  --resume-from /path/to/best.pth --epochs 1 --dinov2-freeze --amp
```

多卡可运行 `GPUS=4 scripts/run_scope_motion_head.sh`。正式评测：

```bash
python eval_scope_motion_head.py \
  --checkpoint work_dirs/iac_navsim_future_dinov2_scope_motion_head/checkpoints/best.pth \
  --config configs/train_navsim_future_dinov2_scope_motion_head.py \
  --split val --eval-ranking
```

配置只用 `consistency_label >= 0.999` 的样本监督视觉属性均值，避免让一张图同时学习多条 near-miss 候选的不同运动值；全部正负样本仍用于候选比较和分组排序。不确定性通过候选比较损失得到梯度。若 IAC 希望直接使用异方差回归，模块也提供 `uncertainty_weighted_motion_loss`，但本分支没有侵入式修改公共训练循环。

## 4. 拟合和应用光流速度证据

光流模块默认使用我们验证时保留的 DIS ultra-fast。它对 8 帧的 7 个相邻帧对提取全局、道路区域与 3×3 网格统计，然后仅用严格正样本拟合 6 个速度量。视觉预测完成后，候选轨迹才进入能量计算。

```bash
python tools/flow_speed_head.py fit \
  --train-index /path/to/train.jsonl \
  --val-index /path/to/val.jsonl \
  --image-root /data/navsim \
  --cache-dir /data/iac_cache/flow \
  --output work_dirs/flow_speed/flow_speed_head.npz \
  --workers 8
```

应用到任意 IAC JSONL 索引：

```bash
python tools/flow_speed_head.py apply \
  --index /path/to/test.jsonl \
  --image-root /data/navsim \
  --model work_dirs/flow_speed/flow_speed_head.npz \
  --cache-dir /data/iac_cache/flow \
  --output work_dirs/flow_speed/test_with_flow.jsonl \
  --workers 8
```

输出保留原行，并增加 `flow_speed_prediction`、`flow_speed_candidate` 和越小越好的 `flow_speed_energy`。如要与 IAC 分数融合，权重、中心和尺度必须只在 validation drives 上选择，不能用 test 调参；本工具因此不内置一个看似方便但会造成数据泄漏的固定融合权重。

## 5. 已有证据与需要重新验证的部分

这些数字来自我们先前独立、按 drive 划分的实验，不等于本分支新头已经在 IAC 全量训练完成：

- 严格 scene-disjoint 的 DINO 视觉进度 probe：PLCC 0.9087、SROCC 0.9042；在 speed perturb 与 trajectory swap 上的成对判断分别为 68.21% 和 82.38%，合计 75.30%。它证明冻结 DINO 特征中存在可用运动信息，也说明速度仍是较难的部分。
- DIS 光流的验证集平均速度 SROCC 为 0.813。加入既有主分数后，测试 Top-1 从 62.83% 到 63.05%，speed pairwise 从 73.78% 到 73.84%。增益很小，所以光流目前应定位为独立诊断/辅助证据，而不是主要 novelty。

本分支真正要由 IAC 复现验证的是：36 维候选盲时序头能否在不降低总体一致性 AUC、positive recall 与 hard Top-1 的前提下，减少 `perturb_speed`、`time_shift_future` 和 `traj_swap` 错误。建议同时报告主指标、按反事实类型的 pairwise accuracy，以及 `no_image` / 帧序打乱消融；只有完整视频明显优于这些控制，才能说明头确实使用了时序视觉证据。

## 6. 最小验证

```bash
python -m py_compile \
  iac_extensions/dino_motion_head.py iac_extensions/flow_evidence.py \
  train_scope_motion_head.py eval_scope_motion_head.py tools/flow_speed_head.py
python -m unittest discover -s tests -v
python train_scope_motion_head.py --help
python tools/flow_speed_head.py --help
```
