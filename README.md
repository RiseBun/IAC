# IAC

IAC（Image-Action Consistency）是一个面向自动驾驶 WAM 输出的一致性评测器。
它回答的不是“这条轨迹是不是 GT”，而是：

> 给定历史图像、未来图像和候选轨迹，这条轨迹是否被这段未来图像支持？

它不是 planner，也不是 world model。IAC 不生成图像或轨迹，只对已有 WAM 输出打分。

## 当前问题定义

我们现在把任务从 single-GT 监督改成 supported-set 监督：

- GT 仍然是正例，但不再是唯一正例
- 同场景高 PDMS / EPDMS 的 `perturb_speed`、`perturb_lateral`、`perturb_heading` 视为软正例
- 中等质量、语义不确定的同场景 perturb 视为 `unknown`
- `image_swap`、`time_shift_future`、`traj_swap`、`reverse_traj`、`high_pdm_image_mismatch` 视为明确负例

这一步的本质是把监督目标从“找 GT”改成“识别被未来图像支持的候选集合”。

## 当前模型

当前主线是 separated-head + learned fusion：

- `image_trajectory_consistency_head` 学图像-轨迹对应
- `trajectory_reasonableness_head` 学轨迹本身是否合理
- 最终 `consistency_logit` 通过内部 learned gate 融合两者

关键约束：

- 不把原始 PDMS 直接喂给最终一致性分数
- PDMS 只作为轨迹合理性辅助监督
- unknown 样本不参与 BCE 和 listwise ranking

## 当前监督

训练里现在同时看到三类信号：

```text
soft positive:
GT + 高 PDMS/EPDMS 的同场景 perturb

unknown:
中等质量、语义不确定的同场景 perturb

hard negative:
image_swap / time_shift_future / traj_swap / reverse_traj / high_pdm_image_mismatch
```

ranking 也从硬 GT 排序改成 soft/listwise。

## 最新验证信号

最近一次完整的 separated-head + official PDMS + hard mismatch 训练，已跑到 epoch 26，验证集信号如下：

- `val_loss = 1.4347`
- `val_c_bal = 0.6778`
- `val_c_recall = 0.5167`
- `val_c_gap = 0.2618`
- `val_reason_mae = 0.1827`
- `best val_iac_precision = 0.6326`

这说明：

- hard negative 没把模型压死，recall 已恢复
- `c_gap` 持续上升，区分度在变强
- reasonableness head 已经学到有效 PDMS 信号
- 低 IoU / 强不一致样本仍是主要难点

当前更近的短训分支是 `supported_set_listwise_vnext`，目标是验证“soft positive + unknown mask + hard mismatch”的监督几何是否比 GT-only 更稳。

## 如何评估

现在不再把“GT 必须排第一”当作唯一正确性标准。

更合理的评估是：

- support-aware / ambiguity-adjusted top1
- clear-negative rejection rate
- low_iou / holdout_low_iou 子集表现
- `val_c_gap`
- `val_reason_mae`

如果一个轨迹不是 GT，但被未来图像支持，也不该被当成错。

## 训练入口

- `configs/train_navsim_future_dinov2_separated_heads_official_pdms_hardneg_vnext.py`
- `configs/train_navsim_future_dinov2_supported_set_listwise_vnext.py`
- `scripts/run_separated_heads_official_pdms_hardneg_vnext.sh`
- `scripts/run_official_pdms_g200_eval.sh`
- `benchmark_wam.py`
- `train.py`
- `eval_critic.py`

## 数据构建

当前训练索引包含高 PDMS mismatch 样本，相关工具在：

- `tools/add_high_pdm_mismatch_negatives.py`
- `scripts/build_high_pdm_mismatch_indices.sh`

目标是显式制造“轨迹本身合理，但和这张未来图像不对应”的 hard negative。

## 评估输出

`benchmark_wam.py` 会输出：

- `wam_iac_scores.jsonl`
- `wam_iac_summary.json`

其中 summary 关注：

- overall mean
- 按 group / action 的分组结果
- ambiguity-aware ranking 指标
- graded perturbation 曲线

## 仓库边界

仓库只保留 IAC 主链路相关内容：

- 训练与评估脚本
- 索引构建与数据处理工具
- benchmark 主逻辑
- 主要配置文件

不包含：

- 原始 nuPlan 数据
- checkpoint
- `work_dirs`
- 日志
- 缓存文件

