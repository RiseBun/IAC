# IAC 主线评估系统

IAC（Image-Action Consistency）判定 WAM 生成的未来图像序列是否与给定的候选动作轨迹一致。当前可信主线架构经过精心设计，保持小巧且可解释。

## 核心设计思想

**问题本质**：不是重建 BEV 或预测轨迹本身，而是回答"如果这个动作真的发生了，生成的未来图像会是这样吗？"

**多解接受**：轻微的速度、航向或横向偏移在前视视频中可能视觉上无法区分，因此评估器报告置信度和模糊性，而不是假装每组只有一个正确答案。

## 当前结果

在 3 个 split 上各 200 组测试：

| Split | v3 acceptable/hard | v3+gate acceptable/hard | 置信度判决 |
| --- | ---: | ---: | --- |
| regular | 0.970 / 0.030 | **0.990** / **0.010** | 189 match, 11 ambiguous, 0 mismatch |
| low_iou | 0.985 / 0.015 | **1.000** / **0.000** | 200 match |
| holdout | 0.985 / 0.015 | **1.000** / **0.000** | 198 match, 2 ambiguous |

**关键指标**：全 600 组零 `mismatch` 判决（无假阴性）。13 个 `ambiguous` 是视觉上无法判定的边缘案例主动挂起，而非被误算为一致。

融合参数：`beta=0.15`, `threshold=0`  
置信度判决：`match_margin=0.2`, `mismatch_margin=-0.5`, `temperature=0.2`

详细策略参见 [`pipeline/README.md`](pipeline/README.md)，模块地图参见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

# 技术架构详解

## 数据契约

每个 JSONL 行 = 一个候选轨迹，属于同一场景的对照组：

```
  "group_id": "scene_123",
  "sample_id": "scene_123__gt_pos",
  "source_type": "gt_pos",  // 或 perturb_speed, image_swap, traj_swap 等
  "candidate_traj": [[x0,y0,heading0], [x1,y1,heading1], ...],  // BEV 坐标
  "history_images": ["path/to/t-4.jpg", ..., "path/to/t0.jpg"],  // 历史帧
  "future_images": ["path/to/t1.jpg", ..., "path/to/t4.jpg"],    // 未来帧
  "iac_consistency": 0.87,  // 上游标量评分
  "recovered_set_agreement": 0.92,
  "recovered_set_minade": 1.2,
  // ... 其他标量特征
}
```

**对照组构成**：

- **acceptable**（可接受）：`gt_pos`（真值）、`perturb_speed`（速度扰动）、`perturb_lateral`（横向扰动）、`perturb_heading`（航向扰动）—— 在前视视频中视觉上近乎无法区分
- **hard mismatch**（硬不匹配）：`image_swap`（图像交换）、`time_shift_future`（时间平移）、`traj_swap`（轨迹交换）、`reverse_traj`（反转轨迹）、`high_pdm_image_mismatch`（高 PDM 图像不匹配）

---

## 主流水线（5 步）

### 步骤 1：V-JEPA2 视觉特征提取（候选轨迹盲）

**模型**：`facebook/vjepa2-vitl-fpc64-256`（Vision Joint-Embedding Predictive Architecture v2）

**关键特性**：**candidate-blind**（候选轨迹盲） —— 特征提取器**从不看**候选轨迹，只看图像序列。这确保视觉侧证据独立。

**处理流程**：

1. **输入**：`history_images`（历史帧）+ `future_images`（未来帧）
2. **重采样**：统一重采样到 **64 帧**（V-JEPA2 的固定输入长度）
3. **V-JEPA2 推理**：
   - 使用 **冻结权重**（`eval()` + `inference_mode()`），不进行微调
   - ViT-Large 架构，64 帧上下文
   - 输出两种表示：
     - **`x`**（全局池化）：整个视频序列的全局表示
     - **`x_tokens`** `[batch, 16, 1024]`：将 64 帧的 token 按时间均匀分 16 块，每块取平均 → 16 个时间感知的局部 token

**为什么是 16 块？**  
平衡时间粒度和计算：16 个 token 能捕捉视频中的时序变化（每块约 4 帧），同时保持交叉注意力机制的计算可行。

**输出**：`work/eval_vjepa.pt` 包含 `{x_tokens, x, sample_id, group_id, ...}`

**脚本**：`pipeline/extract_vjepa_video_features.py`

---

### 步骤 2：v3 可接受性校准器（标量侧证据）

**任务目标**：学习任务度量 —— GT 和视觉上合理的轻微动作扰动是可接受的，而图像/时间/轨迹交换是硬不匹配。

**输入特征**（31 维手工标量特征）：

1. **上游一致性评分**（11 维）：
   - `iac_consistency` + 其 logit
   - 与辅助评分的差值（`iac_consistency` - 各种 recovered_set_* 指标）

2. **轨迹质量指标**（10 维）：
   - `recovered_set_agreement`（恢复集一致性）
   - `minade`（最小平均位移误差）
   - `topmode_ade`、`best_mode_fde`（模式 ADE/FDE）
   - `heading_error`（航向误差）
   - `progress_error`（进程误差）
   - `path_iou`（路径 IoU）
   - `supported`（ordered-motion 支持度）

3. **相对比较特征**（10 维）：
   - `path_minus_sky_delta`（路径 vs 天空基线的增量）
   - `candidate_minus_wrong_*_delta`（候选 vs 错误样本的增量）
   - 轨迹几何特征：终点距离、路径长度、直线性、步长统计、航向变化等

**模型架构**（故意保持极小）：

```
Calibrator(
  Linear(31 → 16)
  LayerNorm → ReLU → Dropout(0.1)
  Linear(16 → 1) → Sigmoid
)
```

**为什么故意小？**  
防止学到 `source_type` 标签泄漏。31 维输入中包含上游评分，这些评分在训练时可能无意中编码了 source 信息。通过限制隐藏层为 **16 维瓶颈**，强制模型只学习真正的视觉-动作一致性模式，而非记忆标签关联。

**损失函数**：

```python
loss = BCE(pred, target) + 0.35 × pairwise_margin_loss
```

- **BCE**（二元交叉熵）：基础分类损失
- **Pairwise margin**（组内成对边际损失）：惩罚同组内 acceptable 样本得分低于 hard 样本的情况，强制模型在组内建立清晰的排序

**样本权重**（类平衡）：
- `gt_pos`（真值）：1.0
- 其他 `acceptable`（扰动）：0.85（略降权，避免过拟合扰动）
- `traj_swap` / `time_shift`：1.2（强化这些难区分的 hard 样本）
- 其他 `hard`：1.0

**训练配置**：
- 优化器：AdamW，学习率 5e-3，权重衰减 1e-3
- 步数：2000 步（小规模数据，短训练防止过拟合）

**输出**：`iac_acceptability_calibrated` ∈ (0, 1)

**脚本**：`pipeline/score_acceptability_calibrator.py`  
**模型**：`models/iac_acceptability_calibrator.pt`

---

### 步骤 3：Clean V-JEPA 轨迹门控（视觉×轨迹证据）

**核心设计**：独立的第二证据源 —— 使用冻结的 V-JEPA2 token + 候选轨迹，**标量侧特征强制清零**，确保与 v3 校准器完全独立。

**输入**：

1. **visual** = `x_tokens` [16, 1024]（步骤 1 产出的时序 token）
2. **traj** = `candidate_traj` 前 8 个点，每点 5 维：
   - `[x, y, sin(heading), cos(heading), cumulative_distance]`
   - 使用 sin/cos 编码航向，避免角度环绕问题
   - 累积距离捕捉速度信息
3. **scalar** = `[0.0]`（**强制清零**） —— gate 无法看到 v3 侧的标量特征

**模型架构**（MismatchGate，轨迹-视觉交叉注意力）：

```
MismatchGate(
  visual_proj: Linear(1024 → 32)  # 压缩视觉 token
  traj_proj:   Linear(5 → 32)     # 投影轨迹特征
  
  MultiHeadAttention(
    query = traj_proj(traj),       # 8 个轨迹点作为 query
    key   = visual_proj(x_tokens), # 16 个视觉 token 作为 key
    value = visual_proj(x_tokens), # 16 个视觉 token 作为 value
    num_heads = 4,
    dim_per_head = 8
  )
  
  # 融合多种交互模式
  fused = [
    attn.mean(dim=1),              # 注意力加权的视觉聚合
    query.mean(dim=1),             # 轨迹全局表示
    attn * query,                  # 逐元素交互
    |attn - query|,                # 差异幅度
    scalar                         # [0.0]（占位，保持接口一致）
  ]
  
  Linear(fused_dim → hidden) → ReLU → Linear(hidden → 1)
)
```

**为什么用交叉注意力？**  
轨迹的每个点（query）可以动态关注视频中最相关的时间段（key/value）。例如：轨迹第 3 秒的点会自动关注视频第 3 秒附近的 token，捕捉时空对齐关系。

**损失函数**（Margin loss，非 BCE）：

```python
loss = ReLU(m⁺ - pos_logits).mean()          # 正样本要 > m⁺
     + ReLU(neg_logits + m⁻).mean()          # 负样本要 < -m⁻
     + softplus_pairwise_margin(group)       # 组内排序
     + w × ReLU(|unknown_logits| - m_u)     # 不确定样本惩罚
```

**为什么用 Margin loss 而非 BCE？**  
我们需要 logit 保留**可用的量级信息**供步骤 4 的融合使用。BCE 会将输出压缩到 (0,1)，丢失相对强度；margin loss 保持 logit 的原始量级，让我们能在融合时用 `penalty = max(0, group_max - current_logit)` 精确衡量"这个候选比组内最佳视觉匹配差多少"。

**输出**：
- `visual_non_mismatch_logit`（原始 logit，供融合用）
- `visual_non_mismatch`（sigmoid 后的概率，供独立查看）

**脚本**：`pipeline/score_visual_mismatch_gate.py`  
**模型**：`models/clean_vjepa_traj_gate.pt`

---

### 步骤 4：保守组内融合

**策略**：v3 保持主排序器地位，gate 只作为**否决旋钮** —— 只能降低、永远不能提升 v3 的评分。

**融合公式**（逐组独立计算）：

```python
# 1. 找到组内视觉最佳者
group_max_gate = max(gate_logit) over this group's rows

# 2. 计算每个候选的惩罚（max(0, ...) 保证 gate 只降不升）
penalty = max(0, group_max_gate - gate_logit - threshold)

# 3. 应用惩罚到 v3 评分
fused_score = v3_score - beta × penalty
```

**冻结参数**：
- `beta = 0.15`（惩罚强度）
- `threshold = 0`（gate 差异容忍度）

**为什么 max(0, ...)?**  
确保 gate 只能**减分**（对视觉较差的候选）或**不改分**（对视觉最佳的候选），永远不会越权成为主排序器。组内视觉最佳者的 `penalty = 0`，完整保留 v3 评分；每个视觉较差的候选被按其"差多少"比例降分。

**输出**：`v3_clean_gate_fused_rank_score`

**脚本**：`pipeline/fuse_v3_clean_gate.py`

---

### 步骤 5：多解接受 + 非对称边际判决

**判决逻辑**（逐组）：

```python
# 1. 找到最佳可接受样本和最佳坏样本
best_accept = argmax(fused_score) where source ∈ acceptable
best_bad    = argmax(fused_score) where source ∈ hard

# 2. 计算原始边际
margin = best_accept.score - best_bad.score

# 3. 非对称阈值判决
if margin >= +0.20:
    verdict = "match"          # 可接受样本明显更好
elif margin <= -0.50:
    verdict = "mismatch"       # 坏样本明显更好（异常）
else:
    verdict = "ambiguous"      # 边界模糊，主动弃权

# 4. 置信度（用于下游加权）
decision_confidence = sigmoid(|margin| / 0.20)
match_confidence    = sigmoid( margin  / 0.20)
```

**为什么非对称阈值？**

- **+0.20（宽松）**：承认"match"相对便宜 —— 下游仍会有额外门控，假阳性不致命
- **-0.50（保守）**：宣判"mismatch"代价高 —— 假阴性会导致错误拒绝合格生成，要求更强证据

**多解接受**：GT 和同场景扰动（`perturb_speed`、`perturb_lateral`、`perturb_heading`）**全部算 acceptable** —— 我们不强制假装每组只有一个唯一赢家，而是承认视觉上可能多个轨迹都合理。

**输出**：
- `work/confidence_groups.jsonl`（逐组判决）
- `work/confidence_summary.json`（汇总：判决计数、top-1 率、边际四分位数等）

**脚本**：`pipeline/score_iac_confidence.py`

---

## 并行支路：Ordered-Motion 支持度

**独立通道**（不属于 v3+gate 主线，报告时合并）：

**输入**：`ordered_motion_segment_ledger`（分段视觉运动残差，候选轨迹盲）

**处理**：
1. `ordered_motion/calibrate_ordered_motion_support.py`（验证集冻结阈值）
2. `ordered_motion/score_ordered_motion_support.py`（测试集推理）
3. 使用 `ordered_motion_support.py` 库函数

**输出**（三态）：
- `supported`（支持）：视觉运动证据与轨迹方向一致
- `unsupported`（不支持）：视觉运动证据与轨迹方向冲突
- `insufficient_evidence`（证据不足）：可见度不够或不确定性过高 → **主动弃权**

**关键特性**：
- **可见度 + 不确定性门控**：缺失可见度或过高不确定性直接弃权，而非被误算为同意
- **正式 split 弃权率约 85%**：故意保守，只在高置信区域发声

**4s 正式运行指标**（20260805 冻结模型，105 组评估集）：
- `unsupported` 决策：105 个，精度 **0.9619**
- `supported` 决策：7 个，经验精度 **1.0**（验证集仅 6 个正例支撑阈值，95% Wilson 下界 0.610，探索性尾部）
- 总决策覆盖率：**0.1524**（623/735 行弃权）

---

# 关键技术决策

## 1. 为什么用 V-JEPA2？

**V-JEPA2**（Vision Joint-Embedding Predictive Architecture v2）是 Meta 的自监督视频表示学习模型：

- **时空联合建模**：同时捕捉空间（物体、场景）和时序（运动、变化）信息
- **预测性架构**：通过预测未来帧的表示（而非像素）学习，天然适合"未来图像是否合理"的判断任务
- **冻结使用**：我们使用预训练权重不微调，保持泛化性，避免过拟合到特定数据集的伪相关
- **ViT-Large + 64 帧**：足够大的模型容量 + 足够长的时间窗口（覆盖历史+未来）

**为什么不用其他模型？**
- 传统 CNN（如 ResNet）：缺乏长程时序建模
- 纯时序模型（如 TimeSformer）：可能丢失细粒度空间细节
- 扩散模型特征：计算代价高，且为生成优化而非判别

## 2. 是否有掩码处理？

**关键回答：否，主流水线不做显式掩码。**

代码审查确认：
- `extract_vjepa_video_features.py`：直接处理完整图像，无掩码逻辑
- `train_visual_mismatch_gate_scorer.py`：无 `mask` / `segmentation` 相关代码
- V-JEPA2 本体：在预训练时使用掩码（masked prediction），但推理时输入完整帧

**为什么不掩码？**
1. **任务需求**：判断"整体视觉-动作一致性"，需要完整场景上下文（路面、车辆、天空、建筑等共同构成可信度线索）
2. **V-JEPA2 设计**：预训练已通过掩码学会鲁棒表示，推理时不需要人工再掩
3. **避免偏差**：强制只看路面可能漏掉关键不一致（如天空时间戳错误、周围车辆运动冲突）

**有掩码的上游？**  
虽然主线不掩码，但上游特征（如 `path_iou`、`path_minus_sky_delta`）可能来自分割模型 —— 这些作为 31 维标量特征的一部分输入 v3 校准器，但 V-JEPA2 和 gate 从不使用。

## 3. 候选轨迹盲（Candidate-Blind）设计

**V-JEPA2 特征提取完全不看候选轨迹**，这是架构的核心约束：

**好处**：
- **证据独立性**：视觉特征不会被候选轨迹"污染"，gate 阶段的视觉-轨迹交互是首次接触
- **计算复用**：同一场景的所有候选共享一次 V-JEPA2 推理（8 个候选 = 1 次特征提取，而非 8 次）
- **防止快捷方式**：如果特征提取时就看轨迹，模型可能学到"直接比较输入轨迹和某个隐式先验"而非真正理解视觉

**代价**：
- 无法直接从视频"读出"BEV 轨迹（但这不是我们的任务 —— 我们只判断一致性，不重建轨迹）

## 4. 为什么 v3 用 MLP，gate 用交叉注意力？

**v3 校准器（MLP）**：
- 输入是 **31 维标量** —— 已经是高度聚合的统计量
- 任务是 **学习任务度量** —— "什么样的数值组合算好"
- MLP 足够：简单的非线性加权组合即可，无需复杂的序列建模

**gate（交叉注意力）**：
- 输入是 **16 个视觉 token × 8 个轨迹点** —— 时空结构化数据
- 任务是 **对齐匹配** —— "轨迹第 t 秒的点是否与视频第 t 秒的视觉一致"
- 交叉注意力天然适合：query（轨迹点）动态查询 key/value（视觉 token），自动学习时空对应关系

## 5. 训练数据如何生成对照组？

**Ground Truth**（真值）：
- `gt_pos`：WAM 生成时使用的实际输入轨迹

**Acceptable 扰动**（视觉近似无区别）：
- `perturb_speed`：速度 ±10%
- `perturb_lateral`：横向偏移 ±0.3m
- `perturb_heading`：航向 ±3°
- 约束：扰动后轨迹仍在合理驾驶范围内（不穿墙、不出路面）

**Hard Mismatch**（明显错误）：
- `image_swap`：替换成不同场景的图像
- `time_shift_future`：未来帧时间戳平移（如 t+2s 的图放在 t+4s）
- `traj_swap`：替换成同数据集不同场景的轨迹
- `reverse_traj`：轨迹时间反转
- `high_pdm_image_mismatch`：通过 PDM（Perceptual Distance Metric）筛选的高不匹配图像对

**分组策略**：每组 8 个候选 = 1 个 GT + 3 个 acceptable + 4 个 hard，保证类平衡。

---

# 完整流水线架构图

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  输入  每行 JSONL = 一个候选轨迹（属于同场景对照组）                           ║
║  ────────────────────────────────────────────────────────────────────────  ║
║  { group_id, sample_id, source_type, candidate_traj,                       ║
║    history_images, future_images, iac_consistency, ... }                   ║
║                                                                            ║
║  每组包含: acceptable = {gt_pos, perturb_*} + hard = {*_swap, reverse_*}   ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                     │
                                     ▼

┌────────────────────────────────────────────────────────────────────────────┐
│ 步骤 1  V-JEPA2 视觉特征提取 [候选轨迹盲: 从不看 candidate_traj]            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  history_images + future_images                                            │
│         │                                                                  │
│         ├──► 重采样到 64 帧                                                 │
│         │                                                                  │
│         └──► facebook/vjepa2-vitl-fpc64-256 (冻结, eval, inference_mode)   │
│                     │                                                      │
│                     ├──► 全局池化 → x [1024]                               │
│                     │                                                      │
│                     └──► 16-chunk 时序平均 → x_tokens [16, 1024]           │
│                                                                            │
│  输出: work/eval_vjepa.pt                                                   │
│  脚本: pipeline/extract_vjepa_video_features.py                            │
└────────────────────────────────────────────────────────────────────────────┘
                            │
                            │ x_tokens (仅步骤 3 用)
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌──────────────────────────┐  ┌───────────────────────────────┐
│ 步骤 2  v3 校准器        │  │ 步骤 3  Clean Gate           │
│ (标量侧 31 维特征)       │  │ (视觉×轨迹交叉注意力)          │
├──────────────────────────┤  ├───────────────────────────────┤
│                          │  │                               │
│ 输入: 31 维标量特征       │  │ 输入:                          │
│  • iac_consistency 等    │  │  visual = x_tokens [16,1024]  │
│  • 轨迹质量指标          │  │  traj   = 前 8 点 × 5 维      │
│  • 相对比较特征          │  │           [x,y,sinθ,cosθ,d]   │
│                          │  │  scalar = [0.0] (强制清零)     │
│ 模型: Calibrator         │  │                               │
│  Linear(31→16)           │  │ 模型: MismatchGate            │
│  LN→ReLU→Dropout         │  │  visual_proj: 1024→32         │
│  Linear(16→1)→Sigmoid    │  │  traj_proj:   5→32            │
│                          │  │  MHA(q=traj, k/v=visual)      │
│ 损失: BCE + 0.35×margin  │  │    4 heads, dim=8             │
│                          │  │  融合[attn,q,attn×q,|差|,0]    │
│ 训练: AdamW lr=5e-3      │  │  Linear→ReLU→Linear(1)        │
│       2000 步            │  │                               │
│                          │  │ 损失: Margin (非 BCE)          │
│ 输出: v3_score           │  │                               │
│ 脚本: score_acceptability│  │ 输出: gate_logit              │
│      _calibrator.py      │  │ 脚本: score_visual_mismatch   │
│                          │  │      _gate.py                 │
└──────────────────────────┘  └───────────────────────────────┘
              │                           │
              └─────────┬─────────────────┘
                        ▼
        ┌───────────────────────────────────────────────┐
        │ 步骤 4  保守组内融合                            │
        ├───────────────────────────────────────────────┤
        │                                               │
        │  group_max_gate = max(gate_logit) in group    │
        │  penalty = max(0, group_max - gate - thresh)  │
        │  fused = v3_score - 0.15 × penalty            │
        │                                               │
        │  max(0,...)保证 gate 只降不升                  │
        │  组内视觉最佳者 penalty=0, 保留 v3 原分         │
        │                                               │
        │  输出: fused_score                             │
        │  脚本: fuse_v3_clean_gate.py                   │
        └───────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────────────┐
        │ 步骤 5  多解接受 + 非对称边际判决                │
        ├───────────────────────────────────────────────┤
        │                                               │
        │  best_accept = argmax(fused) in acceptable    │
        │  best_bad    = argmax(fused) in hard          │
        │  margin      = best_accept - best_bad         │
        │                                               │
        │  margin ≥ +0.20  → match                      │
        │  margin ≤ -0.50  → mismatch                   │
        │  else            → ambiguous (主动弃权)        │
        │                                               │
        │  输出: verdict + confidence                    │
        │  脚本: score_iac_confidence.py                 │
        └───────────────────────────────────────────────┘
                        │
                        ▼
              最终输出: confidence_groups.jsonl
                       confidence_summary.json

┌────────────────────────────────────────────────────────────────────────────┐
│ 并行支路: Ordered-Motion 支持度 (不属于主线，报告时合并)                      │
├────────────────────────────────────────────────────────────────────────────┤
│  ordered_motion_segment_ledger                                             │
│         ↓                                                                  │
│  ordered_motion/calibrate_..._support.py (验证集冻结阈值)                   │
│         ↓                                                                  │
│  ordered_motion/score_..._support.py (测试集推理)                           │
│         ↓                                                                  │
│  输出: supported / unsupported / insufficient_evidence (三态弃权式)          │
└────────────────────────────────────────────────────────────────────────────┘
```

---

# 指标详解

## 主线指标（200 组/split，冻结于 2026-07-27）

| Split | acceptable_top1 | hard_mismatch_top1 | match | ambiguous | mismatch |
| :-- | --: | --: | --: | --: | --: |
| **regular** | **0.990** | **0.010** | 189 | 11 | 0 |
| **low_iou** | **1.000** | **0.000** | 200 | 0 | 0 |
| **holdout** | **1.000** | **0.000** | 198 | 2 | 0 |

**解读**：

- **acceptable_top1**：在每组中，融合评分最高的候选来自 acceptable 类（GT 或扰动）的比例
- **hard_mismatch_top1**：最高分来自 hard mismatch 的比例（越低越好）
- **verdict 分布**：
  - `match` (189+200+198=587/600)：最佳 acceptable 明显优于最佳 bad，margin ≥ 0.20
  - `ambiguous` (13/600)：边际模糊（0.20 > margin > -0.50），主动弃权不强行二分
  - `mismatch` (0/600)：**零假阴性** —— 没有一组被错误判为不匹配

**v3 单独 vs v3+gate 对比**：

| Split | v3 单独 | v3+gate | gate 提升 |
| :-- | :-- | :-- | :-- |
| regular | 0.970/0.030 | 0.990/0.010 | +2.0% acceptable_top1 |
| low_iou | 0.985/0.015 | 1.000/0.000 | +1.5% acceptable_top1 |
| holdout | 0.985/0.015 | 1.000/0.000 | +1.5% acceptable_top1 |

gate 恰好压下了 v3 拿不准的 hard mismatch（regular 从 11 ambiguous 降到仍是 11，但 hard_top1 从 0.030 降到 0.010，说明 gate 成功在 ambiguous 之前的 ranking 阶段就拉低了坏样本）。

---

# 文件结构

```
IAC/
├── models/                                    # 训练好的模型权重
│   ├── iac_acceptability_calibrator.pt        v3 校准器（31d → 16 → 1）
│   └── clean_vjepa_traj_gate.pt               clean gate（视觉×轨迹交叉注意力）
│
├── pipeline/                                  # 主流水线（5 步推理）
│   ├── extract_vjepa_video_features.py        [1] V-JEPA2 特征提取
│   ├── score_acceptability_calibrator.py      [2] v3 校准器评分
│   ├── score_visual_mismatch_gate.py          [3] clean gate 评分
│   ├── fuse_v3_clean_gate.py                  [4] 保守融合
│   ├── score_iac_confidence.py                [5] 多解接受判决
│   └── README.md                              主线策略详解 + 冻结参数表
│
├── training/                                  # 训练脚本（重训模型用）
│   ├── train_iac_acceptability_calibrator.py  v3 校准器训练
│   └── train_visual_mismatch_gate_scorer.py   clean gate 训练
│
├── ordered_motion/                            # 并行支路（三态支持度判决）
│   ├── ordered_motion_support.py              可见度感知聚合库
│   ├── calibrate_ordered_motion_support.py    验证集冻结阈值
│   └── score_ordered_motion_support.py        测试集推理
│
├── audit/                                     # 数据完整性审计
│   ├── audit_formal_splits.py                 fail-closed 分组/场景/图像/时长审计
│   └── audit_ordered_motion_support.py        决策尾精度报告
│
├── repair/                                    # 数据修复工具
│   ├── repartition_formal_splits.py           修复 split 泄漏
│   └── repair_ordered_motion_eval_labels.py   标签修复
│
├── scripts/                                   # 一键运行脚本
│   ├── run_ordered_motion_support_formal.sh   audited 4s 评分入口
│   └── sbatch_*.sh                            SLURM 集群提交脚本
│
├── tests/                                     # 单元测试
│   ├── test_iac_acceptability_calibrator.py
│   ├── test_visual_mismatch_gate_scorer.py
│   ├── test_fuse_v3_clean_gate.py
│   ├── test_iac_confidence.py
│   └── test_ordered_motion_support.py
│
├── multi_horizon_protocol.py                  多时长（2s/4s/6s/8s）强自洽协议
├── ARCHITECTURE.md                            顶层模块地图
└── README.md                                  本文档
```

---

# 快速开始

## 最小运行示例

```bash
# 步骤 1: V-JEPA2 特征提取
python pipeline/extract_vjepa_video_features.py \
  --index work/eval_rows.jsonl \
  --image-root /path/to/images \
  --output work/eval_vjepa.pt \
  --token-summary-size 16

# 步骤 2: v3 校准器评分
python pipeline/score_acceptability_calibrator.py \
  --model models/iac_acceptability_calibrator.pt \
  --primary-scores work/base_scores.jsonl \
  --aux work/aux_scores.jsonl \
  --output-scores work/v3_scores.jsonl \
  --output-summary work/v3_summary.json

# 步骤 3: clean gate 评分
python pipeline/score_visual_mismatch_gate.py \
  --model models/clean_vjepa_traj_gate.pt \
  --rows work/v3_scores.jsonl \
  --visual-cache work/eval_vjepa.pt \
  --visual-cache-key x_tokens \
  --output-scores work/clean_gate_scores.jsonl

# 步骤 4: 保守融合
python pipeline/fuse_v3_clean_gate.py \
  --v3-scores work/v3_scores.jsonl \
  --gate-scores work/clean_gate_scores.jsonl \
  --output-scores work/fused_scores.jsonl \
  --output-summary work/fused_summary.json \
  --beta 0.15 \
  --threshold 0

# 步骤 5: 置信度判决
python pipeline/score_iac_confidence.py \
  --primary-scores work/fused_scores.jsonl \
  --score-key v3_clean_gate_fused_rank_score \
  --margin-space raw \
  --match-margin 0.2 \
  --mismatch-margin -0.5 \
  --confidence-temperature 0.2 \
  --output-groups work/confidence_groups.jsonl \
  --output-summary work/confidence_summary.json
```

**输入要求**：

- JSONL 行必须包含：`group_id`、`sample_id`、`source_type`、`candidate_traj`
- V-JEPA 提取还需要：`history_images`、`future_images`（图像路径列表）
- v3 校准器需要上游标量评分字段（`iac_consistency`、`recovered_set_*` 等）

---

## Ordered-Motion 正式运行

上游 ordered-motion 评分器必须使用 `--include-segment-ledger` 写入 `ordered_motion_segment_ledger`。正式运行分三个有序阶段：

```bash
# 阶段 1: 审计 split 完整性（fail-closed）
python audit/audit_formal_splits.py \
  --split train=work/train_rows.jsonl \
  --split val=work/val_rows.jsonl \
  --split eval=work/eval_rows.jsonl \
  --horizon 4s \
  --require-formal-ready \
  --output-summary work/formal_split_audit.json

# 阶段 2: 验证集冻结阈值（只读标签一次）
python ordered_motion/calibrate_ordered_motion_support.py \
  --scores work/val_segment_scores.jsonl \
  --output-config work/ordered_motion_support_config.json \
  --min-supported-precision 0.95 \
  --min-unsupported-precision 0.95 \
  --min-unsupported-precision-lower-bound 0.95

# 阶段 3: 测试集推理（不读标签）
python ordered_motion/score_ordered_motion_support.py \
  --scores work/eval_segment_scores.jsonl \
  --config work/ordered_motion_support_config.json \
  --output-scores work/eval_support.jsonl \
  --output-summary work/eval_support_summary.json
```

**一键运行**（SLURM 集群）：

```bash
WORK=/path/to/ordered_motion_4s_pilot \
PKG=/path/to/ordered_motion_package \
bash scripts/run_ordered_motion_support_formal.sh
```

**注意**：
- `source_type` 仅用于验证集冻结阈值，推理不读
- 缺失 `scene_id` 会导致审计失败（不会被误认为 split 无关）

---

# 设计哲学

## 为什么这条主线？

原始需求**不是** BEV 重建或轨迹预测本身，而是判断："如果这个动作真的发生了,生成的未来图像会是这样吗？"

**关键洞察**：

1. **视觉模糊性**：轻微的速度、航向、横向差异在前视视频中可能无法区分
2. **多解现实**：GT 和同场景扰动可能都视觉合理，强制唯一答案是假象
3. **假阴性代价高**：错误拒绝合格生成比错误接受边缘样本更致命
4. **主动弃权 > 伪精确**：模糊案例报 `ambiguous` 比强行二分更诚实

**因此架构选择**：

- **保守融合**（gate 只降不升）而非激进重排序
- **非对称阈值**（match 容易，mismatch 困难）
- **置信度报告**（不假装 100% 确定）
- **候选轨迹盲**（防止快捷方式）

---

# 技术栈总结

| 组件 | 技术 | 版本/规格 |
| :-- | :-- | :-- |
| **视觉基座** | V-JEPA2 (Meta) | ViT-Large, 64 frames, 预训练冻结 |
| **v3 校准器** | 浅层 MLP | 31→16→1, AdamW, 2000 步 |
| **Clean Gate** | 交叉注意力 | 4-head MHA, traj×visual, margin loss |
| **融合策略** | 保守惩罚 | max(0, group_max - logit), beta=0.15 |
| **判决逻辑** | 非对称 margin | +0.2 match, -0.5 mismatch, else ambiguous |
| **Ordered-Motion** | 可见度门控聚合 | Wilson 区间, 三态弃权式 |
| **训练框架** | PyTorch | + transformers (V-JEPA2) |
| **数据格式** | JSONL | 逐行候选, 组内对照 |

---

# FAQ

**Q: 为什么不直接用大模型（如 GPT-4V）判断一致性？**

A: 
1. **成本**：每个候选需要完整视频序列，600 组 × 8 候选 = 4800 次调用
2. **可控性**：无法精确控制"什么是 acceptable 扰动"、margin 阈值等
3. **可解释性**：黑盒判断，无法审计哪部分证据起作用
4. **延迟**：API 调用 vs 本地 GPU 批推理

**Q: 能否用这套系统评估其他视频生成模型（非 WAM）？**

A: 可以，只要：
1. 提供 `history_images` + `future_images` + `candidate_traj`（BEV 坐标）
2. 构造对照组（GT + acceptable 扰动 + hard mismatch）
3. 准备上游标量特征（或用占位符，但会降低 v3 性能）

**Q: 为什么 holdout split 的 ambiguous 只有 2 个？**

A: holdout 样本经过额外人工筛选，剔除了明显边界模糊的案例，保留"典型清晰"样本用于最终验证。regular split 保留更多自然分布的边缘案例（11 ambiguous）。

**Q: gate 的交叉注意力能可视化吗？**

A: 可以。`MHA` 输出的 attention map `[8_traj_points, 16_visual_tokens]` 可绘制热力图，显示"轨迹第 t 秒关注视频哪些时间段"。但当前版本未默认导出（可修改 `score_visual_mismatch_gate.py` 添加 `--save-attention` 选项）。

**Q: 如何重训模型？**

A: 
```bash
# v3 校准器
python training/train_iac_acceptability_calibrator.py \
  --train-scores work/train_scores.jsonl \
  --val-scores work/val_scores.jsonl \
  --output-model models/my_v3_calibrator.pt

# clean gate
python training/train_visual_mismatch_gate_scorer.py \
  --train-rows work/train_rows.jsonl \
  --val-rows work/val_rows.jsonl \
  --visual-cache work/train_vjepa.pt \
  --output-model models/my_gate.pt
```

需要准备：
- 训练/验证 split 的 JSONL（带 `source_type` 标签）
- V-JEPA2 特征缓存（训练前运行 `extract_vjepa_video_features.py`）

---

# 引用

如果使用本系统，请引用：

```bibtex
@misc{iac2026,
  title={IAC: Image-Action Consistency Evaluation for WAM-Generated Video},
  author={[Your Team]},
  year={2026},
  url={https://github.com/RiseBun/IAC}
}
```

---

# 许可证

[待补充]

---

# 联系方式

问题或建议请提交 GitHub Issue: https://github.com/RiseBun/IAC/issues

