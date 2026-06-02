# IAC：图像-动作一致性评测基准

IAC（Image-Action Consistency）是一个面向自动驾驶 World Action Model（WAM）的评测基准。它要回答的问题很简单：给定一段历史图像、一个动作或未来轨迹，以及 WAM 生成的未来图像，这些未来图像是否真的反映了这个动作？同时，这条轨迹本身是否符合基本运动学约束？

本项目不是 planner，也不是 world model。IAC 不负责生成图像或生成轨迹，只负责对已有的 WAM 输出进行打分。

## 我们的模型评测什么

IAC Critic 有两个输出头：

- `consistency`：判断“未来图像”和“候选轨迹/动作”是否一致。例如轨迹明显向左转，但未来画面仍像直行，这一项应该低分。
- `validity`：判断候选轨迹本身是否合理。例如速度突变、横向偏移过大、航向变化不连续，这一项应该低分。

最终用于 benchmark 的两个分数是：

- `iac_consistency`：图像-动作一致性分数，越高表示 WAM 生成的未来图像越符合给定动作。
- `iac_validity`：轨迹运动学合理性分数，越高表示轨迹本身越可行。

## 输入是什么

训练和评测时，每个样本包含四类输入：

- `history_images`：历史相机图像序列。
- `future_images`：未来图像序列。训练时通常来自 nuPlan 真实未来帧；评测 WAM 时来自 WAM 生成帧。
- `ego_state`：当前自车状态，例如速度、加速度、yaw、yaw rate 等。
- `candidate_traj`：待评测的未来轨迹，表示一个动作或动作序列。

WAM benchmark 的 JSONL 输入示例：

```json
{
  "wam_name": "my_wam",
  "group_id": "scene_001",
  "history_images": ["history_0.jpg", "history_1.jpg", "history_2.jpg", "history_3.jpg"],
  "future_images": ["wam_future_0.jpg", "wam_future_1.jpg", "wam_future_2.jpg", "wam_future_3.jpg"],
  "ego_state": [0.0, 0.0, 0.0, 0.0, 0.0],
  "candidate_traj": [[0.0, 0.0, 0.0], [1.0, 0.1, 0.02]],
  "action_type": "left_turn"
}
```

`consistency_label` 和 `validity_label` 是可选字段。如果输入里提供标签，脚本会额外计算 accuracy、recall、AUROC、PR-AUC 等监督评估指标；如果没有标签，脚本仍然会输出每个 WAM 的 IAC 分数。

## 输出是什么

运行 `benchmark_wam.py` 后会生成两个文件：

```text
work_dirs/wam_benchmark/<name>/
├── wam_iac_scores.jsonl
└── wam_iac_summary.json
```

`wam_iac_scores.jsonl` 是逐样本结果，包含：

- 样本 ID / 分组 ID。
- WAM 名称。
- `iac_consistency`。
- `iac_validity`。
- 可选的预测标签和原始 logit。

`wam_iac_summary.json` 是汇总结果，包含：

- overall 平均分。
- 按 WAM 分组的平均分。
- 按动作类型分组的平均分。
- 如果有同一场景下的多候选动作，会计算 ranking 指标。
- 如果有扰动强度字段，会计算 graded perturbation curve。

## 我们是怎么学习的

IAC 使用自监督/弱监督构造训练样本，不需要人工标注“图像和动作是否一致”。

正样本来自真实 nuPlan 片段：

- 历史图像是真实历史帧。
- 未来图像是真实未来帧。
- 候选轨迹是真实 ego 未来轨迹。
- 因此 `consistency_label=1`，`validity_label=1`。

负样本由真实片段自动构造：

- `traj_swap`：图像不变，换成别的场景或别的时刻的轨迹。
- `image_swap`：轨迹不变，换成别的未来图像。
- `time_shift_future`：使用时间错位的未来图像。
- `perturb_lateral`：横向扰动轨迹。
- `perturb_heading`：扰动航向角。
- `perturb_speed`：扰动速度/进度。

这些负样本让模型学习“图像变化应该和动作一致”，而不是只记住图像质量或轨迹平滑度。训练时使用二分类损失，`consistency` 头学习图像-轨迹匹配，`validity` 头学习轨迹本身是否符合运动学规则。

## 基本流程

构建 IAC 索引：

```bash
python tools/build_consistency_index.py \
  --db-root "$NUPLAN_DB_ROOT" \
  --image-roots /path/to/nuplan-v1.1_mini_camera_0 /path/to/nuplan-v1.1_mini_camera_1 \
  --output-dir indices
```

训练 IAC Critic：

```bash
PYTHONUNBUFFERED=1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29606 \
  train.py \
  --config configs/train_consistency_mini.py \
  --work-dir work_dirs/iac_5epoch_2gpu \
  --epochs 5 \
  --batch-size 8 \
  --num-workers 4 \
  --preflight-samples 256
```

评估 IAC Critic：

```bash
python eval_critic.py \
  --checkpoint work_dirs/iac_5epoch_2gpu/checkpoints/best.pth \
  --split val \
  --batch-size 32 \
  --eval-ranking
```

评测 WAM 输出：

```bash
python benchmark_wam.py \
  --input path/to/wam_outputs.jsonl \
  --checkpoint work_dirs/iac_5epoch_2gpu/checkpoints/best.pth \
  --output-dir work_dirs/wam_benchmark/my_wam
```

## 数据路径

默认按 AutoDL 当前环境查找：

```text
/root/autodl-tmp/data/cache/mini
/root/autodl-tmp/nuplan-v1.1_mini_camera_0
/root/autodl-tmp/nuplan-v1.1_mini_camera_1
```

也可以用环境变量覆盖：

```bash
export NUPLAN_DATA_ROOT=/path/to/data-root
export NUPLAN_DB_ROOT=/path/to/data/cache/mini
export NUPLAN_INDEX_ROOT=/path/to/IAC/indices
export NUPLAN_CAMERA_ROOTS="/path/to/camera_0:/path/to/camera_1"
```

## 仓库里保留什么

仓库只保留 IAC benchmark 主链路：

- `train.py`：训练 IAC Critic。
- `eval_critic.py`：验证 IAC Critic。
- `benchmark_wam.py`：评测 WAM 输出。
- `stress_test_iac.py`：检查模型是否依赖捷径。
- `tools/build_consistency_index.py`：构建训练/验证索引。
- `configs/train_consistency_mini.py`：默认训练配置。
- `data_paths.py`：路径配置。
- `scripts/dlc_train.sh`：训练脚本模板。

以下内容不进入仓库：

- nuPlan 原始数据和相机图像。
- 训练索引 JSONL。
- checkpoint。
- `work_dirs`。
- 训练日志。
- `__pycache__`。
- 本地 smoke/test 输出。

## DrivingWorld 是否有用

当前版本不保留 DrivingWorld 集成。原因是 IAC benchmark 的边界是“评测 WAM 输出”，而不是“在本仓库里运行某个 WAM 生成图像”。如果要评测 DrivingWorld，只需要先用 DrivingWorld 在外部生成未来图像和对应 manifest，再把 manifest 输入 `benchmark_wam.py`。这样 IAC 对 DrivingWorld、Drive-WM 或任何其它 WAM 都是同一个接口、同一套打分逻辑，更适合作为公平 benchmark。

## 借鉴 iWorld-Bench 的扩展

`benchmark_wam.py` 现在可选 3 个 iWorld-Bench 风格的交叉验证维度，对 critic 打分做正交补充：

| 模块 | 借鉴 iWorld-Bench 的 | 用法 | 关注风险 |
|------|---------------------|------|----------|
| `iac_video_metrics.py` | 4 个无参考视觉指标（Brightness Consistency / Color Temperature / Sharpness Retention / Image Quality via MUSIQ） | `benchmark_wam.py --visual-metrics` | 防止 critic 学会图像质量 shortcut |
| `iac_traj_metrics.py` | Trajectory Accuracy（Eq. 13）+ Tolerance（Eq. 14）+ Alignment（Eq. 17），用 OpenCV 必备矩阵 + LK 光流替代 VIPe | `benchmark_wam.py --geometric-metrics` | 几何硬评测，不依赖 critic |
| `iac_memory_metrics.py` | Memory Symmetry（Eq. 15-16）+ Loop-Closure Drift | `benchmark_wam.py --memory-metrics` | 测 WAM 是否能回到起点 |

索引构建也吸收了 2 个 iWorld-Bench 设计：

| 选项 | 来源 | 行为 |
|------|------|------|
| `tools/build_consistency_index.py --add-reverse-traj` | iWorld-Bench memory task | 加入 `reverse_traj` 负样本，用 `iac_memory_metrics.reverse_candidate_traj()` 严格反演 |
| `tools/build_consistency_index.py --quality-filter` | iWorld-Bench Algorithm 1 | 跑 Z-score 亮度异常 + 静止帧检测 + 时长过滤，丢弃可疑 anchor |

`iac_difficulty_sampler.py` 把样本按 D1–D4 难度分桶（iWorld-Bench §3.2.2 思想），可作为 train.py 训练采样器使用，平衡 D1 单轴扰动和 D2/D3 复合扰动的暴露频率——直接针对 v4 中 `perturb_speed` / `time_shift_future` 召回偏低的问题。

依赖说明：
- `iac_video_metrics.py` 可选 `musiq`、`scipy`；缺时回退到 BRISQUE-lite proxy，函数永不抛错。
- `iac_traj_metrics.py` 仅依赖 `opencv-python`（已在 nuPlan 栈中）。
- `iac_memory_metrics.py` 仅依赖 numpy。

## 借鉴的边界

不引入 iWorld-Bench 的以下部分：
- **4 个仿真器**（aerial_VLN / UAV_ON / Openfly / EmbodiedCity）—— 视角与自动驾驶不同
- **NeRF / 3D 重建指标**（DL3DV / RealEstate）—— 4 帧未来不需要
- **多视角统一编码**（UAV / UGV / Human / Robot）—— IAC 专注前向驾驶
- **GPT-4o 标注**（$28k / 330k 视频）—— nuPlan 本身有真值
- **3D Simulator 仿真数据**—— nuPlan 已是真实采集

借鉴的核心只有 3 件事：
1. **用无参考视觉指标做交叉验证**，不全押在 critic 上
2. **用对极几何反推轨迹做硬评测**，绕开 critic 偏置
3. **难度分级 + 数据精炼**，让 critic 学到更鲁棒的特征

