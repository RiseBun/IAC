# IAC P0 改造后项目说明

## 1. 项目定位

改造后的 `/root/autodl-tmp/nuplan` 是一个面向 **Image-Action Consistency (IAC)** 的 nuPlan-mini 原型项目。它的目标不是训练一个 planner，也不是替代 closed-loop simulator，而是提供一个辅助评测维度，用来判断候选轨迹是否与未来视觉演化一致。

从 AAAI 目标来看，当前项目完成的是 **P0 可信度改造**：先在现有 nuPlan 原型上堵住主要 shortcut，修正 `Validity` 定义，补充 baseline、hard negative、诊断指标和 stress test。它还不是完整的 NAVSIM / Bench2Drive / nuScenes benchmark。

IAC 的基本输入为：

- `history_images`：历史图像序列。
- `future_images`：未来图像序列，作为客观视觉证据。
- `ego_state`：当前 ego 状态。
- `candidate_traj`：候选未来轨迹。

IAC 的核心输出为：

- `Consistency`：候选轨迹是否与未来视觉演化一致。
- `Validity`：候选轨迹自身是否满足 context-free 的运动学可行性。

## 2. 改造前后的核心区别

改造前，`Consistency` 和 `Validity` 共享同一个融合特征：

```text
history image + future image + ego + trajectory
        -> shared fusion
        -> consistency / validity
```

这个结构有一个明显风险：`Validity` 本应判断轨迹自身是否合理，但它也能看到图像，因此可能学习到场景相关 shortcut。

改造后，模型结构变为：

```text
Consistency:
history image + future image + ego + 前 4 步 trajectory
        -> shared fusion
        -> consistency score

Validity:
ego + 完整 8 步 trajectory
        -> validity fusion
        -> validity score
```

对应实现位于 `train.py` 的 `ConsistencyCriticModel`。

关键变化包括：

- `Consistency` 保留图像、ego、轨迹输入，用于判断 image-action consistency。
- `Consistency` 只使用与 `future_images` 时间窗对齐的前 4 步轨迹。
- `Validity` 改为独立分支，只看 `ego + trajectory`，不再接收图像特征。
- 完整 8 步轨迹仍用于 `Validity`，用于判断更长 horizon 的运动学可行性。
- 细粒度 heads 仍保留为输出接口，但默认 loss 权重设为 `0.0`，避免在没有独立标签时被误当成真实多维监督。

## 3. 时间窗对齐

当前配置中：

```python
future_num_frames = 4
candidate_traj_steps = 8
consistency_traj_steps = 4
future_step_time_s = 0.5
```

也就是说，未来图像约覆盖 2 秒，而完整轨迹约覆盖 4 秒。改造前，`Consistency` 会直接看到完整 8 步轨迹，这会导致后半段轨迹缺乏视觉证据支撑。

改造后：

- `Consistency` 只看前 `consistency_traj_steps=4` 步轨迹。
- `Validity` 继续使用完整 `candidate_traj_steps=8` 步轨迹。

这样可以保证 `Consistency` 的监督目标更清楚：它只判断有未来视觉证据覆盖的轨迹片段。

## 4. Baseline 审计能力

为了判断 critic 是否真的利用了未来视觉证据，项目现在支持以下 P0 baseline mode：

- `full`：完整 IAC-Critic。
- `no_image`：禁用历史/未来图像特征，只看 ego + trajectory。
- `ego_only`：只看 ego。
- `no_traj`：禁用 trajectory。
- `traj_only`：只看 trajectory。

训练时可以使用：

```bash
python train.py \
  --config configs/train_consistency_mini.py \
  --baseline-mode no_image
```

评估时也可以覆盖 checkpoint/config 中的模式：

```bash
python eval_critic.py \
  --checkpoint work_dirs/iac_full/checkpoints/best.pth \
  --baseline-mode ego_only
```

这些 baseline 的意义是：

- 如果 `no_image` 表现接近 `full`，说明图像证据贡献不足。
- 如果 `ego_only` 表现很高，说明 ego-state shortcut 严重。
- 如果 `traj_only` 表现很高，说明任务可能被轨迹形状或运动学规则解决。
- 如果 `full` 明显优于这些 baseline，IAC 才更可信。

## 5. 数据索引构建升级

核心脚本为：

```text
tools/build_consistency_index.py
```

改造后，索引构建不再只是简单生成正负样本，而是加入了更完整的诊断信息。

### 5.1 样本类型

正样本：

- `gt_pos`：历史图像、未来图像、ego 状态和真实轨迹匹配。

负样本：

- `traj_swap`：替换成其他 anchor 的轨迹。
- `image_swap`：替换成其他 anchor 的未来图像。
- `time_shift_future`：同一 scene 内 future images 时间错位。
- `perturb_lateral`：横向扰动轨迹。
- `perturb_heading`：航向扰动轨迹。
- `perturb_speed`：速度缩放扰动轨迹。

其中 `time_shift_future` 是本轮新增的 hard negative。它比跨 scene 的 `image_swap` 更难，因为它来自同一个 scene 内部，能减少模型只靠场景外观差异分类的 shortcut。

### 5.2 新增 metadata

索引样本现在会包含更多字段，例如：

- `negative_family`
- `validity_reason`
- `perturb_type`
- `perturb_magnitude`
- `perturb_level`
- `time_shift_anchor_steps`
- `time_shift_timestamp_us`

这些字段用于后续做 per-type evaluation、graded perturbation curve 和错误分析。

## 6. Validity 标签重定义

改造前，`validity_label` 主要由 `source_type` 简单决定，例如某些 swap 或 perturb 会被直接设为 0 或 1。这会造成目标混乱：模型可能学到“是否被扰动过”，而不是真正的运动学可行性。

改造后，`validity_label` 由 `compute_kinematic_validity()` 计算，只依赖：

- `ego_state`
- `candidate_traj`

判断规则包括：

- 轨迹是否为空或非数值。
- 横向位移是否过大。
- 航向变化是否过大。
- 是否存在过大的反向运动。
- 单步速度是否过大。
- 第一步速度是否与当前 ego speed 严重不连续。
- 加速度是否过大。
- 相邻 yaw step 是否过大。

输出包括：

- `validity_label`
- `validity_reason`

这样可以把两个任务拆清楚：

- `Consistency`：轨迹是否与未来视觉证据匹配。
- `Validity`：轨迹本身是否运动学合理。

## 7. 评估指标升级

核心脚本为：

```text
eval_critic.py
```

改造后，评估指标从原来的 accuracy / AUC 扩展为：

- Accuracy
- Precision
- Recall
- F1
- AUROC
- PR-AUC
- ECE
- TNR / FPR
- per-source-type metrics
- negative recall by type
- graded perturbation curve
- ranking metrics，可通过 `--eval-ranking` 启用

评估结果会保存为：

```text
eval_val_results.json
eval_val_summary.json
```

其中 `eval_val_results.json` 保存完整结果，`eval_val_summary.json` 更适合后续整理成论文表格。

### 7.1 为什么不只看 Accuracy

AAAI 目标中强调，binary accuracy 容易被 shortcut 拉高。比如：

- 跨 scene image swap 太容易。
- ego-state 与 trajectory 的局部不连续可能直接暴露负样本。
- 轨迹扰动幅度过大时，模型不需要理解图像也能分类。

因此改造后必须同时看：

- `AUROC / PR-AUC`：衡量排序和不平衡数据下的分类能力。
- `ECE`：衡量分数是否校准。
- `per-source-type recall`：看不同负样本类型是否均衡。
- `graded perturbation curve`：看扰动幅度增大时分数是否单调下降。

## 8. Stress Test

新增脚本：

```text
stress_test_iac.py
```

该脚本用于 cheap shortcut probes，覆盖：

- `future_reverse`：未来帧反序。
- `traj_mirror`：轨迹左右镜像。
- `future_black`：未来图像置黑。
- `future_white`：未来图像置白。
- `future_noise`：未来图像替换为噪声。
- `traj_shuffle`：轨迹步顺序打乱。

使用真实 checkpoint：

```bash
python stress_test_iac.py \
  --checkpoint work_dirs/iac_full/checkpoints/best.pth \
  --max-samples 128
```

如果当前环境没有图像数据，也可以运行 synthetic smoke：

```bash
python stress_test_iac.py \
  --synthetic-smoke \
  --max-samples 2
```

理想情况下，训练好的模型应当满足：

- future 反序后，`Consistency` 分数下降。
- 轨迹镜像后，`Consistency` 分数下降。
- future 图像被破坏后，`Consistency` 分数下降。
- trajectory shuffle 后，`Consistency` 分数下降。

如果这些扰动不影响分数，说明模型可能没有真正利用 future visual evidence。

## 9. 推荐实验流程

### 9.1 重建索引

在有 nuPlan mini DB 和 camera images 的环境中运行：

```bash
python tools/build_consistency_index.py \
  --output-dir indices
```

可选地限制场景数量做调试：

```bash
python tools/build_consistency_index.py \
  --output-dir indices \
  --max-scenes 2 \
  --max-samples-per-scene 20
```

### 9.2 训练完整模型

```bash
python train.py \
  --config configs/train_consistency_mini.py \
  --work-dir work_dirs/iac_full
```

### 9.3 训练 P0 Baselines

```bash
python train.py \
  --config configs/train_consistency_mini.py \
  --baseline-mode no_image \
  --work-dir work_dirs/iac_no_image

python train.py \
  --config configs/train_consistency_mini.py \
  --baseline-mode ego_only \
  --work-dir work_dirs/iac_ego_only

python train.py \
  --config configs/train_consistency_mini.py \
  --baseline-mode no_traj \
  --work-dir work_dirs/iac_no_traj

python train.py \
  --config configs/train_consistency_mini.py \
  --baseline-mode traj_only \
  --work-dir work_dirs/iac_traj_only
```

### 9.4 评估

```bash
python eval_critic.py \
  --checkpoint work_dirs/iac_full/checkpoints/best.pth \
  --eval-ranking
```

也可以对同一个 checkpoint 覆盖 baseline mode 做诊断：

```bash
python eval_critic.py \
  --checkpoint work_dirs/iac_full/checkpoints/best.pth \
  --baseline-mode no_image
```

### 9.5 Stress Test

```bash
python stress_test_iac.py \
  --checkpoint work_dirs/iac_full/checkpoints/best.pth \
  --max-samples 256
```

## 10. 当前已经验证的内容

本轮改造完成后，已通过以下检查：

```bash
python -m py_compile train.py eval_critic.py tools/build_consistency_index.py stress_test_iac.py verify_upgrade.py
```

```bash
python verify_upgrade.py
```

```bash
python stress_test_iac.py --synthetic-smoke --max-samples 2
```

同时，`ReadLints` 未发现 linter 错误。

需要注意的是，旧索引和旧 checkpoint 已不再作为有效实验输入。真实训练和评估需要先设置 `NUPLAN_DATA_ROOT`、`NUPLAN_DB_ROOT`、`NUPLAN_INDEX_ROOT`，再用当前 `tools/build_consistency_index.py` 重建 IAC 索引。

## 11. 当前仍未完成的 AAAI Benchmark 部分

本项目目前完成的是 P0 可信度改造，还不是完整 AAAI benchmark。尚未完成的部分包括：

- NAVSIM 数据适配与主 benchmark。
- Bench2Drive / CARLA 迁移。
- nuScenes / Waymo 跨域验证。
- 与 PDMS / EPDMS / Driving Score 的 Spearman / Kendall 相关性。
- 与 DriveCritic / ACT-Bench / Vista reward / WoTE 的正式对比。
- 真实 planner / WAM 输出接入。
- 小规模 human study。
- 完整错误分析和可解释性可视化。

## 12. 总结

改造后的 `/root/autodl-tmp/nuplan` 可以被视为一个 **P0-audited IAC-Critic 原型**。

它现在具备：

- 更清晰的 `Consistency` / `Validity` 任务拆分。
- 与未来图像时间窗对齐的 consistency 轨迹输入。
- context-free kinematic validity 标签。
- 更强的 hard negative，尤其是 `time_shift_future`。
- baseline mode，用于审计 ego / image / traj shortcut。
- 更完整的评估指标，包括 PR-AUC、ECE 和 per-type 诊断。
- stress test，用于证伪模型是否真正依赖未来视觉证据。

它的下一步不是继续堆模型结构，而是运行完整实验并回答以下问题：

- `full` 是否显著优于 `no_image` 和 `ego_only`？
- `time_shift_future`、`traj_swap`、`perturb_speed` 是否仍是弱项？
- graded perturbation 下，consistency score 是否随扰动幅度单调下降？
- stress test 中未来图像破坏和轨迹镜像是否显著降低分数？

只有这些问题得到正面回答，这个 nuPlan-mini 原型才有资格继续迁移到 NAVSIM，并进一步支撑 AAAI 论文中的 benchmark 叙事。
