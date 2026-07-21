# Depth-Aware Path-Grounded IAC 项目总结与下一步计划

日期：2026-07-02

## 1. 一句话结论

当前 IAC 项目已经从“训练一个 consistency critic”推进到“证明 critic 是否真的看未来图像里的轨迹相关证据”。

`net_depth.py` 和 `config_depth.yaml` 的价值不是直接提升 IAC 分类分数，而是可以把当前较粗的 2D 路径遮挡升级成更可信的深度/几何感知路径因果测试，用来进一步排除 shortcut。

当前最重要的下一步是：

```text
从 path-grounded
升级到 trajectory-specific geometry-grounded
```

也就是证明：

```text
consistency score 不是只看天空/背景，也不是只看道路中心区域，
而是真的依赖 candidate trajectory 对应的未来图像几何证据。
```

## 2. IAC 项目的本质

IAC 不是 planner，也不是 world model。它是一个 WAM evaluator。

核心输入：

```text
history images
future images
candidate trajectory
ego state
```

核心输出：

```text
consistency: future images 是否支持 candidate trajectory
validity: candidate trajectory 本身是否合理
```

本项目真正要回答的问题是：

```text
一个 WAM 生成的未来图像，是否真的和它声称支持的动作/轨迹一致？
```

这比单独评价图像质量或轨迹质量更接近 WAM 的本质。

## 3. 当前最大的科学问题

当前最关键的问题不是“模型能不能分类”，而是：

```text
consistency score 是否真的由 future image 中和轨迹相关的变化驱动？
还是主要由 trajectory geometry shortcut / road-region shortcut / image artifact 驱动？
```

如果模型只学到：

```text
这条 trajectory 本身看起来合理
```

而不是：

```text
future image 支持这条 trajectory
```

那么它就不是一个真正的 image-action consistency evaluator。

因此当前科学目标是：

```text
证明 consistency 是 future-image-grounded，
并且进一步证明它是 candidate-trajectory-specific。
```

## 4. 当前已完成的主线

### 4.1 Strict-future NAVSIM benchmark

当前 IAC 主要实验使用 strict-future NAVSIM index。

关键要求：

```text
history images 不能泄漏 future
future images 必须是真未来
positive 必须是真实 future + 真实 ego trajectory
negative 必须是可控反事实
```

负样本来源包括：

```text
image_swap
time_shift_future
traj_swap
perturb_heading
perturb_lateral
perturb_speed
```

这使得 IAC 可以不依赖人工标注，而是通过自动反事实构造 consistency labels。

### 4.2 当前 DINOv2 consistency critic

当前主模型线：

```text
frozen DINOv2 ViT-S/14
history image encoder
future image encoder
candidate trajectory encoder
ego state encoder
fusion head
consistency / validity heads
```

额外训练结构：

```text
group ranking loss
hard negative pressure
future evidence branch
hierarchical consistency branch
source-aware negative weighting
```

当前较好的 source-aware checkpoint 大致指标：

```text
balanced accuracy ≈ 0.6864
precision ≈ 0.2688
recall ≈ 0.6410
TNR ≈ 0.7318
F1 ≈ 0.3788
```

这说明模型已经具备基本判别能力，但 geometry perturbation 仍然是主要难点。

### 4.3 Path-grounded causal diagnostic

当前已经在 `benchmark_wam.py` 中加入：

```bash
--path-causal-metrics
```

它对每个样本计算：

```text
score(original)
score(path_masked)
score(sky_masked)
```

判断规则：

```text
score(original) - score(path_masked)
>
score(original) - score(sky_masked)
```

当前 smoke 结果：

```text
mean_path_delta = 0.0822
mean_sky_delta = 0.0450
mean_path_minus_sky = 0.0372
path_delta_gt_sky_frac = 0.7422
is_path_grounded = true
```

解释：

```text
遮住候选路径区域后，consistency 平均下降 0.0822
遮住同面积天空/背景后，consistency 平均下降 0.0450
路径遮挡比天空遮挡多造成 0.0372 的下降
74.22% 样本里，遮路径比分数下降更大
```

这已经证明：

```text
模型不是只对任意图像变化敏感，
它更敏感于 future image 中靠近 candidate path 的区域。
```

但这还没完全证明：

```text
模型真的理解 candidate trajectory 和 future image 的几何对应关系。
```

因为当前 path mask 仍然是一个轻量 2D heuristic，而不是严格相机几何投影。

## 5. `net_depth.py` 的作用

`net_depth.py` 是一个 EFG evaluator entry，用于 MoGe-2 + sparse-LiDAR 对齐深度评估。

它的 pipeline：

```text
RGB image
→ MoGe-2 monocular depth
→ metric depth
→ sparse LiDAR affine alignment
→ calibrated depth
→ depth metrics evaluation
```

核心类：

```python
MoGe2AlignNet
```

它做的事情：

1. 加载 frozen MoGe-2。

```python
self.model = MoGeModel.from_pretrained(moge_cfg.pretrained)
self.model.eval()
self.model.requires_grad_(False)
```

2. 对原始 RGB 图像推理 metric depth。

```python
output = self.model.infer(
    image_tensor,
    num_tokens=self.num_tokens,
    force_projection=False,
    apply_mask=True,
    fov_x=fov_x,
    use_fp16=self.use_fp16,
)
depth_pred = output["depth"]
```

3. 从 OpenScene sparse LiDAR GT 恢复 GT depth。

```python
gt_depth = f_px / torch.clamp(gt_inv * orig_w, min=1e-3)
```

4. 用 sparse LiDAR 对 MoGe-2 depth 做后处理对齐。

支持两种模式：

```text
affine: depth_aligned = scale * depth_pred + shift
scale:  depth_aligned = scale * depth_pred
```

当前配置默认：

```yaml
align_mode: affine
```

5. 输出预测深度、GT 深度和有效 mask。

```python
{
  "pred_depth": depth_aligned,
  "gt_depth": gt_depth,
  "valid_mask": valid_mask
}
```

6. 自带深度指标 evaluator。

```python
OpenSceneDepthMetricsEvaluator
```

指标包括：

```text
d1
d2
d3
abs_rel
sq_rel
rmse
rmse_log
silog
log10
mae
```

也可以保存：

```text
.npy depth
depth visualization
RGB + sparse LiDAR overlay
```

## 6. `config_depth.yaml` 的作用

`config_depth.yaml` 是 depth evaluator 的运行配置。

任务类型：

```yaml
task: val
```

这说明当前是验证/推理任务，不是训练任务。

数据部分：

```yaml
dataset:
    type: OpenSceneEFGDataset
    parquet_name: openscene_trainval_shard0.parquet
    target_size: 1536
    max_eval_depth: 200.0
    min_eval_depth: 0.1
    save_npy: true
    save_vis: true
```

MoGe-2 配置：

```yaml
moge2:
    pretrained: /mnt/volumes/base-pi-lx-my/bjzhu/models/moge-2/moge-2-vitl/model.pt
    num_tokens: 2000
    align_mode: affine
    use_fp16: true
```

评估器配置：

```yaml
trainer:
    type: MetricAnythingTrainer
    evaluators:
        - OpenSceneDepthMetricsEvaluator
```

所以这两个文件当前不是 IAC 主训练代码，而是：

```text
深度估计 + sparse LiDAR 对齐 + 深度质量评估工具
```

## 7. 深度工具和 IAC 的关系

当前 IAC 的 path-grounding 做法是：

```text
candidate trajectory
→ 简单映射到图像下半部分路径 corridor
→ 遮挡 path ROI
→ 与同面积 sky ROI 比较
```

这个方法可以证明：

```text
模型更看路径附近区域，而不是天空/背景。
```

但别人仍然可以质疑：

```text
你遮的是图像下半部分道路区域，不一定是 candidate trajectory 真正经过的区域。
```

`net_depth.py` 的价值就是补这个缺口。

有 depth 后，我们可以把问题升级成：

```text
candidate trajectory 在 ego/world 几何中
future image 有 depth 或 pseudo-depth
mask 构造可以更接近真实 3D path evidence
```

这让 path mask 从：

```text
2D heuristic ROI
```

升级到：

```text
depth-aware / geometry-aware ROI
```

这是当前最有潜力的创新方向。

## 8. 当前风险

### 风险 1：当前 path mask 仍然是近似

当前 path ROI 是：

```text
candidate_traj → image polyline heuristic
```

不是严格相机投影。

可能被质疑为：

```text
只是图像下方道路区域
只是道路中心区域
不是 trajectory-specific path
```

### 风险 2：path-grounded 不等于 trajectory-specific

当前结果说明：

```text
路径区域比天空区域重要
```

但还没证明：

```text
左转轨迹对应左侧路径
右转轨迹对应右侧路径
快慢轨迹对应不同未来位置
```

### 风险 3：geometry perturbation 还没彻底解决

当前模型对 image_swap 较容易，对 geometry perturbation 仍较难。

也就是说模型可能已经学到：

```text
哪里是路
```

但还不够精确地区分：

```text
这条 candidate trajectory 是否真的匹配这段 future。
```

### 风险 4：depth 代码当前是 OpenScene/EFG 体系

`net_depth.py` 依赖：

```text
OpenSceneEFGDataset
MetricAnything_Teacher.openscene
MoGe third_party
sparse LiDAR fields
f_px_gt
gt_inverse_depth_matrix
```

所以不能直接假设它能无缝跑在当前 NAVSIM IAC index 上。

## 9. 下一步总目标

下一步不是简单继续训练，而是：

```text
把 path-grounding 从 2D heuristic 升级成 depth-aware / geometry-aware causal proof。
```

更具体地说：

```text
证明 consistency score 对 candidate trajectory 对应的未来 3D path evidence 敏感，
而不是对天空、背景、道路中心、图像下半部分、trajectory geometry 本身敏感。
```

## 10. Phase 0：深度环境审计

先确认 `net_depth.py + config_depth.yaml` 在服务器上能跑。

需要检查：

```text
EFG_PATH 是否存在
WORK_MA 是否存在
MetricAnything_Teacher.openscene 是否存在
MoGe third_party 路径是否存在
MoGe-2 checkpoint 是否存在
OpenScene parquet 是否存在
```

当前配置依赖：

```text
${oc.env:EFG_PATH}/efg/config/default.yaml
${oc.env:WORK_MA}/public_datasets/data_annotation
/mnt/volumes/base-pi-lx-my/bjzhu/models/moge-2/moge-2-vitl/model.pt
```

Phase 0 产物：

```text
depth_env_audit.md
```

判断标准：

```text
能 import 关键包
能加载 MoGe-2
能读取 parquet
能跑 1 张图
能输出 pred_depth.npy 和 depth_metrics.json
```

## 11. Phase 1：最小 depth sanity check

先跑小样本。

建议配置：

```yaml
dataset.max_eval_samples: 16
dataset.save_npy: true
dataset.save_vis: true
dataset.vis_every_n: 1
```

目的不是刷深度指标，而是确认：

```text
depth 是否非空
depth 是否尺度合理
LiDAR alignment 是否稳定
visualization 是否合理
```

关键检查：

```text
pred_depth 是否大量 inf/nan
align_mask.sum 是否足够
affine scale/shift 是否异常
近处路面是否比远处路面深度更小
天空区域是否被 mask 成 inf 或大深度
```

如果这一步失败，不应该进入 IAC 集成。

## 12. Phase 2：定义 depth-aware path mask

当前 2D path mask：

```text
trajectory → image polyline
```

升级目标：

```text
candidate trajectory in ego coordinates
+ camera intrinsics/extrinsics or approximate projection
+ depth map
→ trajectory-specific image corridor
```

如果有完整相机标定，最好做：

```text
trajectory points (x, y, z=ground)
→ camera coordinates
→ image pixels
→ draw corridor
```

如果没有完整外参，可以做弱版本：

```text
2D heuristic path corridor
+ depth consistency filter
```

例如：

```text
保留 path corridor 中 depth 在合理道路深度范围的区域
排除天空/无穷深/不可信区域
按 depth bins 匹配 control mask
```

新的 control masks 至少应包括：

```text
path_mask: candidate path 对应区域
sky_mask: 天空/上方同面积区域
road_control_mask: 非 candidate path 但同为道路/下半区域
depth_matched_control_mask: 深度分布和 path 类似但空间不在 path 上
```

为什么需要 road/depth control？

因为只和 sky 比不够强。

别人可以说：

```text
当然天空不重要，道路区域重要。
```

所以更强的判断应该是：

```text
path_delta > road_control_delta
path_delta > depth_matched_control_delta
path_delta > sky_delta
```

## 13. Phase 3：新增 depth-aware causal benchmark

在 `benchmark_wam.py` 中新增模式：

```bash
--depth-path-causal-metrics
```

推荐先使用离线 depth cache，而不是在线推理。

建议路径结构：

```text
depth_root/
  sample_id_future0.npy
  sample_id_future1.npy
  sample_id_future2.npy
  ...
```

输出字段建议：

```json
{
  "iac_consistency": 0.71,
  "iac_consistency_path_masked": 0.62,
  "iac_consistency_sky_masked": 0.68,
  "iac_consistency_road_control_masked": 0.67,
  "iac_consistency_depth_control_masked": 0.66,
  "path_mask_delta": 0.09,
  "sky_mask_delta": 0.03,
  "road_control_delta": 0.04,
  "depth_control_delta": 0.05,
  "path_minus_sky_delta": 0.06,
  "path_minus_road_control_delta": 0.05,
  "path_minus_depth_control_delta": 0.04,
  "path_mask_fraction": 0.15,
  "sky_mask_fraction": 0.15,
  "road_control_mask_fraction": 0.15,
  "depth_control_mask_fraction": 0.15
}
```

核心指标：

```text
mean_path_minus_sky_delta
mean_path_minus_road_control_delta
mean_path_minus_depth_control_delta
path_delta_gt_all_controls_fraction
```

建议通过标准：

```text
mean_path_minus_sky_delta > 0.02
mean_path_minus_road_control_delta > 0.01
mean_path_minus_depth_control_delta > 0.01
path_delta_gt_all_controls_fraction > 0.60
```

## 14. Phase 4：trajectory-specific counterfactual

这是最关键的实验。

当前我们比较：

```text
mask candidate path
vs
mask sky
```

下一步应该比较：

```text
mask candidate path
vs
mask wrong path
```

构造：

```text
same history
same future image
candidate trajectory A
wrong trajectory B
```

如果模型真的理解 candidate trajectory，那么对于 candidate A：

```text
score_drop(mask path A) > score_drop(mask path B)
```

指标：

```text
candidate_path_delta
wrong_path_delta
candidate_minus_wrong_delta
candidate_delta_gt_wrong_fraction
```

这个实验直接回答：

```text
模型是否知道图片上的行驶路径和当前 candidate trajectory 有关系？
```

这是比 sky control 更强的因果证据。

## 15. Phase 5：训练层面升级

当前 path-grounding training loss：

```text
original_score - path_masked_score >= margin
sky_masked_score ~= original_score
```

升级后可以变成：

```text
original_score - candidate_path_masked_score >= margin
original_score - wrong_path_masked_score <= smaller_margin
original_score - road_control_masked_score <= smaller_margin
original_score - depth_control_masked_score <= smaller_margin
```

也就是：

```text
candidate path sensitivity
background invariance
wrong-path invariance
road-control invariance
depth-control invariance
```

这比当前 path-grounded loss 更强，因为它不只是证明“道路重要”，而是证明：

```text
candidate trajectory 对应的那条路径重要。
```

## 16. Phase 6：最终实验矩阵

至少比较四组模型：

```text
A. source-aware DINOv2
   无 path-grounding loss

B. 2D path-grounded DINOv2
   当前版本

C. depth-aware path-grounded DINOv2
   加 depth-aware controls

D. trajectory-specific depth-aware path-grounded DINOv2
   加 candidate path vs wrong path
```

每组报告普通 IAC 指标：

```text
balanced accuracy
precision
recall
TNR
F1
score gap
```

报告 shortcut 指标：

```text
image_swap TNR
time_shift_future TNR
trajectory_family TNR
geometry_family TNR
perturb_heading TNR
perturb_lateral TNR
perturb_speed TNR
```

报告 path causal 指标：

```text
mean_path_delta
mean_sky_delta
mean_road_control_delta
mean_depth_control_delta
mean_path_minus_sky_delta
mean_path_minus_road_control_delta
mean_path_minus_depth_control_delta
path_delta_gt_all_controls_fraction
```

报告 trajectory-specific 指标：

```text
candidate_path_delta
wrong_path_delta
candidate_minus_wrong_delta
candidate_delta_gt_wrong_fraction
```

## 17. 推荐立即执行顺序

不要直接大训练。

最短合理路径：

1. 服务器检查 depth 环境。

```text
确认 net_depth.py 依赖能不能跑
```

2. 用 `config_depth.yaml` 跑 16 张 OpenScene depth sanity check。

```text
输出 depth_metrics.json / npy / vis
```

3. 判断能否迁移到当前 IAC/NAVSIM。

```text
如果 NAVSIM 没有同样 LiDAR/相机字段，就只用 MoGe depth，不做 LiDAR alignment
如果有标定，就做真正 projection
```

4. 先做离线 depth cache。

```text
future image → depth .npy
```

5. 在 `benchmark_wam.py` 新增 depth-aware causal metrics。

```text
先评估，不训练
```

6. 比较旧 source-aware checkpoint。

```text
如果 depth-aware path causal 仍然成立，说明证据更强
```

7. 再把 depth-aware loss 加入训练。

```text
避免先训练后发现 diagnostic 本身站不住
```

## 18. 最终项目闭环

最终项目应形成如下闭环：

```text
1. 构造 strict-future IAC benchmark
2. 训练 future-aware consistency critic
3. 发现 geometry shortcut 风险
4. 用 path causal masking 证明不是背景 shortcut
5. 用 depth-aware controls 证明不是 road-region shortcut
6. 用 candidate path vs wrong path 证明 trajectory-specific grounding
7. 用训练约束让模型真正依赖 candidate-path future evidence
```

这条线比单纯堆模型更有创新性，也更能回答当前最大的科学问题。

## 19. 当前最重要的判断

`net_depth.py` 和 `config_depth.yaml` 的真正作用是：

```text
把“模型是不是看路径区域”
升级成
“模型是不是看 candidate trajectory 对应的几何路径证据”
```

这是当前 IAC 项目最值得突破的方向。

