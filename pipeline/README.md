# IAC 主线（Mainline）：策略与指标

> 冻结版本：`v3_acceptability_plus_clean_vjepa_traj_gate`，日期 `2026-07-27`（见 `models/mainline_manifest.json`）。本文只解释**当前发布**这一版，不涉及历史尝试。

## 0. 一句话总览

判决 "候选轨迹是否与 WAM 生成的未来图像视觉相容"，一次评估内组内比较，产 `match / ambiguous / mismatch` 三态。

主线由**两条独立信号 + 一次组内保守融合 + 一次 margin 判决**组成：

```
标量分特征   ─► v3 校准器 ─►┐
                            ├─► 组内保守融合 ─► margin 判决 ─► verdict
V-JEPA2 视觉 + 候选轨迹 ─► clean gate ─►┘
```

每组同时包含 GT 和多种扰动/交换，见 §1。

## 1. 数据契约

一组（`group_id` 相同）由**同一场景**下的多个候选轨迹组成，每条候选来自以下 `source_type` 之一：

| 类别 | source_type | 语义 |
| --- | --- | --- |
| **acceptable**（可接受） | `gt_pos` | GT 轨迹 |
| | `perturb_speed` / `perturb_lateral` / `perturb_heading` | 对 GT 做小幅速度/横向/朝向扰动，视觉上难以区分 |
| **hard mismatch**（硬不匹配） | `image_swap` | 换成别的场景的图像 |
| | `time_shift_future` | 未来图像整体时间平移 |
| | `traj_swap` / `reverse_traj` | 拿别人轨迹或把 GT 反向 |
| | `high_pdm_image_mismatch` | 独立 PDM 认定的图像-动作严重不匹配 |

**核心设计判断**：不假设"组内唯一正确"。GT 和三种小扰动**都算可接受**——它们在前视图上视觉几乎等价，硬要区分是伪自信。

每行 JSONL 至少包含：
```
group_id, sample_id, source_type, candidate_traj,
history_images, future_images,
iac_consistency, recovered_set_*, path_minus_sky_delta, ...   # 上游给的标量分
```

## 2. 主线 5 步（每步：策略 / 输入 / 做法 / 输出）

### 步 1：V-JEPA2 视觉特征抽取

`pipeline/extract_vjepa_video_features.py`

- **策略**：视觉侧对候选轨迹完全盲（candidate-blind），只看历史+未来图像序列。这样下游 gate 里的"视觉证据"永远不可能偷看轨迹标签。
- **输入**：`history_images + future_images`（默认 4+4 帧，跨 horizon 时按 `multi_horizon_protocol.py` 调）。
- **做法**：
  1. 加载图像 → 时间维度重采样到 `--num-frames`（默认 64）。
  2. 送入 `facebook/vjepa2-vitl-fpc64-256` 冻结编码器，`eval()` + `inference_mode()`，全程无梯度。
  3. 对最后一层隐藏态做**两路压缩**：
     - `x`：pooled 全局向量（默认 `mean_std_diff` = mean ‖ std ‖ (last_chunk_mean − first_chunk_mean)）。
     - `x_tokens`：token 序列在时间维等分成 `--token-summary-size` 段，每段取均值，得到 `[16, 1024]` 的 token 摘要。**主线 gate 读的就是这个**。
- **输出**：`.pt` 缓存，键包括 `x`（pooled）、`x_tokens`（token 摘要）、`sample_id`、`group_id`、`source_type`、`y`（原始轨迹，仅方便审计，不给模型）。

### 步 2：v3 acceptability 校准器打分

`pipeline/score_acceptability_calibrator.py` + `training/train_iac_acceptability_calibrator.py`

- **策略**：把"我们真正想要的度量"直接学出来——不是复刻 GT 排名，而是"acceptable 全部要高，hard 全部要低"，组内相对关系比绝对值重要。**source label 只做训练监督**，不进入特征。
- **输入**：一条 primary JSONL（上游 IAC 分）+ 可选 aux JSONL（其它上游打分），行行对齐。
- **做法**：
  1. **特征工程**（31 维，见 `_row_features`）：
     - 原始分 + 其 logit + 与 aux 的差/绝对差；
     - 上游给的可解释标量：`recovered_set_agreement / minade / topmode_ade / best_mode_fde / heading_error / progress_error / path_iou / supported`、`path_minus_sky_delta`、`candidate_minus_wrong_path_delta` 等；
     - 从 `candidate_traj` 抽 10 维几何：终点、路径长度、直度、平均/最大步长、朝向变化。
  2. **模型**：`Linear(31→16) → LayerNorm → ReLU → Dropout → Linear(16→1)`。**这么小是刻意的**——加复杂度就学到 source label 泄漏，反而变差。
  3. **损失**：`BCE + 0.35 × 组内 pairwise margin`。pairwise 项对每组的 acceptable/hard 交叉对，用 `ReLU(1.0 − (pos_logit − neg_logit))` 强制组内至少 1.0 的分差。样本权重：`gt_pos = 1.0`，其它 acceptable = 0.85，硬项里 `traj_swap / time_shift_future = 1.2`（更该压下去），其余 hard = 1.0。
  4. 优化：AdamW，`lr=5e-3`、`weight_decay=1e-3`、2000 步。
- **输出**：每行加 `iac_acceptability_calibrated` ∈ (0,1)，同时覆盖 `iac_consistency`；原分保留在 `base_iac_consistency`。

### 步 3：Clean V-JEPA 轨迹门控（gate）打分

`pipeline/score_visual_mismatch_gate.py` + `training/train_visual_mismatch_gate_scorer.py`

- **策略**：只走"视觉 × 轨迹"这一路，独立于 v3 已用的标量特征。**scalar 侧被清零（`scalar_feature_mode=zero`）**，逼模型只能从 V-JEPA token 和 8×5 轨迹 token 的交叉注意力里找证据。这样 gate 才有资格作为"独立的第二证据"。
- **输入**：v3 打完分的 JSONL + 步 1 的 `x_tokens`（`[16, 1024]`）。
- **做法**：
  1. **候选轨迹 token 化**：`candidate_traj` 的前 8 个点，每点 5 维：`[x, y, sin(heading), cos(heading), cum_distance]`。
  2. **模型**（`traj_cross_attention`）：
     - Visual token `[B, 16, 1024]` → `Linear(1024 → 32) → LN → ReLU → Dropout`；
     - Traj token `[B, 8, 5]` → 同样 proj 到 32；
     - `MultiheadAttention(query=traj, key=visual, value=visual)`，4 头；
     - 拼接 `[attended.mean, q.mean, attended*q, |attended-q|, scalar=0]` → `Linear(hidden) → ReLU → Linear(1)`。
  3. **训练目标（三分位标签）**：
     - `supported`：`gt_pos` 或（同场景扰动且独立 PDM 分 ≥ 阈值）；
     - `hard`：`image_swap / time_shift_future / high_pdm_image_mismatch`；
     - `unknown`：中等质量的同场景扰动——**不当监督**，但用一个"|logit| 别太大"的正则约束到中性带。
  4. **损失**：margin 版（默认）——`ReLU(supported_margin − pos_logits)` + `ReLU(hard_logits + hard_margin)` + 组内 softplus pairwise + `unknown_weight × ReLU(|unknown_logit| − unknown_margin)` + 可选的 `logit_l2`。**用 margin 而不是 BCE，是为了让 logit 保留有序的、有意义的量级**（后续融合按 logit 差算惩罚），BCE 会把它推到饱和。
- **输出**：每行加两个字段：
  - `visual_non_mismatch_logit`（**这个供融合用**）
  - `visual_non_mismatch = sigmoid(logit)`（可读性用）

### 步 4：组内保守融合

`pipeline/fuse_v3_clean_gate.py`

- **策略**：**gate 只降不升，绝不越权当主 ranker**。v3 仍然是主分；gate 只在组内挑出视觉最合理的那个当参照，把视觉更差的候选往下压。
- **做法**（组内独立算）：

  ```
  group_max_gate = max(gate_logit over this group)
  penalty        = max(0, group_max_gate − gate_logit − threshold)   # threshold=0
  fused_score    = v3_score − beta × penalty                          # beta=0.15
  ```

  - 组内视觉最好的候选：`penalty = 0`，`fused = v3` 原样。
  - 视觉越差 → `penalty` 越大 → 从 v3 分里扣得越多。
  - 因为 `max(0, …)` 有下截断，gate**永远不能把某个候选抬到 v3 之上**——最多不扣，扣多少上限也是 v3 与视觉最差候选的差。
- **输出**：新字段 `v3_clean_gate_fused_rank_score`（供步 5 排序），并覆盖 `iac_consistency`。

### 步 5：多解接受 + margin 三态判决

`pipeline/score_iac_confidence.py`

- **策略**：一组内**多个 acceptable 候选都算对**（GT 和三种同场景扰动同时可接受）。用 acceptable 的**最高分**与 hard 的**最高分**之差决定判决，差不够大就承认"不决定"，输出 `ambiguous` 而不是硬选。
- **做法**（每组独立）：

  ```
  best_accept = argmax(score) over source ∈ acceptable
  best_bad    = argmax(score) over source ∈ hard       # 若无 hard 则 fallback 到 non-acceptable
  margin = best_accept.score − best_bad.score           # margin_space=raw

  if margin ≥ +0.20:  verdict = match
  if margin ≤ −0.50:  verdict = mismatch
  else:               verdict = ambiguous

  decision_confidence = sigmoid(|margin| / 0.20)         # temperature=0.2
  match_confidence    = sigmoid( margin  / 0.20)
  ```

  - `match_margin=0.2`、`mismatch_margin=−0.5` **不对称**——对"承认匹配"更宽松、对"宣判不匹配"更保守。因为承认匹配的风险是漏检硬错误（下游还有兜底），宣判不匹配的风险是给出错误的否定判决（后果更重）。
- **输出**：`confidence_groups.jsonl` 每组一条，字段含 `verdict`、`decision_confidence`、`match_confidence`、`accept_margin_logit`、`top_source`、`best_accept/best_bad` 各自的 sample/score/source。汇总 `confidence_summary.json` 里含 `verdict_counts`、`verdict_fractions`、`accept_margin_logit` 的四分位、`strict_gt_top1 / acceptable_top1 / hard_mismatch_top1`。

## 3. 主线冻结参数

| 位置 | 参数 | 值 | 为什么是这个值 |
| --- | --- | --- | --- |
| v3 校准器 | 特征维 / hidden | 31 / 16 | 小到不足以拟合 source-label 泄漏路径 |
| gate | visual_shape | `[16, 1024]` | 16 段时间 token × V-JEPA2 隐藏维 |
| gate | scalar_dim / scalar_feature_mode | 1 / `zero` | 强制 gate 独立于 v3 的标量证据 |
| gate | traj_shape / interaction_kind | `[8, 5]` / `traj_cross_attention` | 轨迹 token 作 query，注视觉 |
| 融合 | beta / threshold | 0.15 / 0 | gate 是**惩罚项**不是排序器 |
| 融合 | score_key | `v3_clean_gate_fused_rank_score` | 步 5 排序用 |
| 判决 | margin_space | `raw` | 直接在概率空间上算，可解释 |
| 判决 | match_margin / mismatch_margin | +0.20 / −0.50 | 承认匹配宽松、宣判不匹配保守 |
| 判决 | confidence_temperature | 0.20 | 与 `match_margin` 同尺度，边缘 group 的置信度接近 0.5 |

## 4. 主线当前指标

三个 split，每个 200 组（详见 `models/mainline_manifest.json → metrics`）：

| Split | acceptable_top1 | hard_mismatch_top1 | verdict: match | ambiguous | mismatch |
| :-- | --: | --: | --: | --: | --: |
| **regular** | **0.99** | **0.01** | 189 | 11 | 0 |
| **low_iou** | **1.00** | **0.00** | 200 | 0 | 0 |
| **holdout** | **1.00** | **0.00** | 198 | 2 | 0 |

含义：
- **`acceptable_top1`**：top-1 候选落在 acceptable 家族（`gt_pos` 或三种小扰动）的组占比。三个 split 全部 ≥ 0.99。
- **`hard_mismatch_top1`**：top-1 竟然是硬不匹配的组占比。**regular 仅 1%，其它 0%**。
- **verdict**：主线**没有一组被判 mismatch**（下游可以放心相信没有假否定）；`ambiguous` 出现在 regular 的 11 组和 holdout 的 2 组——这些是**真正视觉不可分**的边缘案例，被主动挂起而不是强判。

单看 v3 校准器（不加 gate）时 regular 是 0.970 / 0.030、11 组 ambiguous、0 组 mismatch；**加了 gate 后 regular 升到 0.990 / 0.010**。gate 的惩罚项恰好把那 6 组 v3 拿不准的硬不匹配压下去了，符合它作为"独立第二证据 + 保守否决"的角色定位。

### 复现

```bash
bash scripts/run_mainline_example.sh
```

或按 5 步单独跑（见项目根 `README.md → Minimal Run`）。产物路径：

```
work/mainline/
├── vjepa.pt                    # 步 1
├── v3_scores.jsonl             # 步 2
├── clean_gate_scores.jsonl     # 步 3
├── fused_scores.jsonl          # 步 4
├── confidence_groups.jsonl     # 步 5 (per group)
└── confidence_summary.json     # 步 5 (aggregate)
```

## 5. 为什么这样切分而不是"训一个大模型端到端"

- **v3 用标量证据，gate 用视觉+轨迹 token**，两条信号**输入不相交**。融合是加法而不是共训，任何一路失效都能追溯——审计脚本（`audit/audit_clean_gate_counterfactual.py`、`audit/run_clean_gate_lofo_audit.py`）就是靠这个隔离能定位问题的。
- **candidate-blind 视觉 + scalar_feature_mode=zero**：整条主线不存在"视觉侧偷看轨迹标签"这条路径。
- **多解接受 + 三态判决**：不给"视觉不可分"的组硬造二分类答案。真正判 `mismatch` 时才有意义，`ambiguous` 是主动承认知识边界。
- **gate 只做保守惩罚**：`max(0, …) × beta=0.15` 保证 gate 只可能改变组内相对顺序、且改动幅度受限。
