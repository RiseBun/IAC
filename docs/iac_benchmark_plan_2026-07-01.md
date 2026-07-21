# IAC 方案：只保留有用有意义的部分

## 1. 目标
IAC 不做生成器，也不做 planner。它的任务只有一个：给 WAM 输出做可靠评测。

我们要回答的不是“视频好不好看”，而是：
- 未来视觉证据是否支持 candidate trajectory
- 轨迹本身是否物理/语义合理
- 图像到底有没有提供额外信息
- 不同 WAM / planner 在同一套标准下谁更好

## 2. 主贡献

### 2.1 评测框架
IAC 的核心是一个 simulator-free、label-free、plug-and-play 的 critic。

输入：
- `history_images`
- `future_images`
- `ego_state`
- `candidate_traj`

输出：
- `Consistency`
- `Validity`

其中：
- `Consistency` 评估“未来视觉演化是否支持这条轨迹”
- `Validity` 评估“轨迹本身是否合理”

这部分借鉴两类工作，但不照搬：
- `ACT-Bench`：借它的 action fidelity / controllability 视角
- `DriveCritic`：借它的 pairwise / ranking 判别范式

### 2.2 方法增强
当前方法主线不是“大模型”，而是“可解释的视觉层选择”。

现有 DINOv2 critic 已支持：
- `single`
- `multi`
- `gated`

建议主线保持：
- 默认使用 `ViT-S/14`
- 冻结 backbone
- 多层选择作为 ablation 和增强项

可讲清楚的点是：
- 早层更偏局部物理和运动线索
- 中层更偏时序与几何过渡
- 晚层更偏语义和任务相关判别
- 轨迹和 ego 决定“该看哪层”

### 2.3 诊断分析
IAC 真正有价值的，不只是分数，而是解释：
- 哪些层在编码 physics
- 哪些层在编码 action
- 图像信息到底是否必要
- 模型是不是在偷看捷径

建议做三类诊断：
- physics / action probing
- patch-level attribution
- subspace / layer geometry analysis

## 3. 我们现在不做什么
- 不把 IAC 改成重型 world model
- 不先追 full DINO
- 不把纯 MLP 当主线
- 不把 simulator-based evaluation 拉进主线
- 不把模型规模当主要创新点

## 4. 实施顺序
1. 固化 benchmark 规范
   - index audit
   - negative construction audit
   - nuPlan / NAVSIM 分开跑

2. 先证明 IAC 评测本身有用
   - AUROC
   - Top-1
   - NDCG@3
   - MRR
   - per-type recall
   - calibration

3. 再做多层 DINO 对照
   - `single`
   - `multi`
   - `gated`

4. 最后做诊断
   - physics probe
   - action probe
   - patch attribution

## 5. 现在的判断标准
这条线值不值得继续，取决于两个问题：

1. 多层选择是否真的提升了 ranking / consistency
2. probing 是否证明不同层学到了不同类型的信息

如果答案是否定的，就回到简单、干净的 single-layer DINO baseline；
如果答案是肯定的，才把它写成方法增强点。

## 6. 当前结论
IAC 的真正贡献，不在于“我们用了什么更大的 backbone”。
它的贡献在于：
- 定义了更合理的 WAM 评测问题
- 用未来视觉证据直接检验动作一致性
- 把“图像是否必要”变成可测问题
- 用多层 DINO 和诊断分析增强这个评测器的解释性

