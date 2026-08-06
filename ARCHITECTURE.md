# IAC 架构说明

> 目的：让新读者在 10 分钟内建立正确的模块划分和数据流心智模型。本文只解释 `IAC/` 子目录，不涵盖仓库根目录的其它工程（如 `iWorld-Bench/`、`rk3588/`、`configs/` 等）。

## 1. IAC 在解决什么问题

给定一段"历史图像 + 候选动作轨迹 + 由 WAM 生成的未来图像"，判断这条候选轨迹是否与生成的未来视频**在视觉上相容**。

这不是回归任务，不是"预测 GT 轨迹"。核心难点：
- 小幅速度/朝向/横向扰动在前视图上**视觉几乎不可分辨**。硬要在一组内选出唯一正确解会引入伪自信。
- 图像换、时间平移、轨迹交换（image_swap / time_shift_future / traj_swap 等）才是**真硬不匹配**。

所以主线不做二分类，而是产出**三态判决**：`match` / `ambiguous` / `mismatch`。

## 2. 顶层组件

```
                                   ┌─ v3 校准器 (models/iac_acceptability_calibrator.pt)
                                   │      标量得分特征 (31 维) → 单一 rank 分
   已存在的基础分 (base_scores)    │
   V-JEPA2 冻结视觉特征            │
             │                     │
             ▼                     ▼
    ┌────────────────┐   ┌────────────────────────────────────┐
    │ 特征提取 &     │   │ v3 acceptability                   │
    │ 打分产出       │──▶│  + clean V-JEPA traj gate 融合     │──▶ 三态置信度
    └────────────────┘   └────────────────────────────────────┘        (verdict)
             │
             └─▶ 有序运动支路（Ordered-motion support）
                 从 segment ledger → supported / unsupported / insufficient_evidence
```

两条独立评估通道：

| 通道 | 输出 | 语义 |
| --- | --- | --- |
| **主线（v3 + clean gate + confidence）** | `match / ambiguous / mismatch` | 候选轨迹与生成视频是否相容 |
| **有序运动支路（ordered-motion support）** | `supported / unsupported / insufficient_evidence` | 视觉证据是否支持该 segment 上的运动量级 |

两者是**并联**关系，不共享推理网络。支路作为独立的可弃权判决，可与主线结论一起报告。

## 3. 模型与产物清单（`models/`）

| 文件 | 类型 | 输入形状 | 用途 |
| --- | --- | --- | --- |
| `iac_acceptability_calibrator.pt` | v3 校准器 | scalar feature 31d → hidden 16 | 学"任务真正想要的度量"：把 GT + 视觉可信扰动打成"可接受"，把 image/time/traj swap 打成"硬不匹配" |
| `clean_vjepa_traj_gate.pt` | 视觉-轨迹交叉注意力门控 | visual `[16, 1024]` + traj `[8, 5]`，scalar 侧被清零 | 只从**视觉+轨迹配对**这一路提供硬不匹配的否决证据；scalar_feature_mode=`zero` 保证不泄漏 v3 已用的标量特征 |
| `mainline_manifest.json` | 冻结的主线配置 | — | 记录当前发布的 v3 + gate 融合参数、置信度阈值、三个 split 的评估结果，作为"这一版是哪一版"的唯一事实源 |

## 4. 主线数据流（Train / Score / Fuse / Confidence）

主线是纯脚本 pipeline，输入输出全部走 JSONL，天生易审计。

### 4.1 训练（离线，一次性）

| 脚本 | 训练目标 |
| --- | --- |
| `training/train_iac_acceptability_calibrator.py` | v3 校准器。**不用 source label 作为输入特征**，只用作监督。学出 acceptable vs hard-mismatch 的排序 |
| `training/train_visual_mismatch_gate_scorer.py` | clean gate。用 margin loss 保持成校准的门控信号，而不是被 BCE 推到 0/1 饱和 |
| `training/train_ordered_motion_speed_rank.py` | 候选无关的视觉速度排序微调（不改变推理时的 scorer 结构，仅解决速度歧义失败模式） |

### 4.2 推理（对每次评估执行）

```
history_images + future_images
    │
    ▼
pipeline/extract_vjepa_video_features.py     ⇒ work/eval_vjepa.pt (candidate-blind)
    │
    ▼
pipeline/score_acceptability_calibrator.py   ⇒ work/v3_scores.jsonl
    │   （吃 primary + aux 标量特征）
    │
    ▼
pipeline/score_visual_mismatch_gate.py       ⇒ work/clean_gate_scores.jsonl
    │   （吃 v3_scores + eval_vjepa 视觉 token）
    │
    ▼
pipeline/fuse_v3_clean_gate.py               ⇒ work/fused_scores.jsonl
    │   penalty = max(0, group_max_gate - gate_logit - threshold)
    │   fused   = v3_score - beta * penalty
    │   —— gate 只在组内做**保守惩罚**，永远不当主 ranker
    ▼
pipeline/score_iac_confidence.py             ⇒ work/confidence_groups.jsonl
        raw margin → match / ambiguous / mismatch
```

关键设计约束（读代码前先读这些）：
- **candidate-blind 视觉**：`extract_vjepa_video_features.py` 只看图像序列，不看候选轨迹。防止视觉侧偷看标签。
- **gate 只做否决，不做主 ranker**：`fuse_v3_clean_gate.py` 里 `beta=0.15` 的惩罚项，且 `penalty` 有 max(0, …) 下截断——gate 只能压低组内视觉更差的候选，不能把它抬到 v3 之上。
- **不假设唯一正确解**：`score_iac_confidence.py` 在 margin 小的时候返回 `ambiguous`，而不是硬选 top1。

### 4.3 输入 JSONL 契约

每行至少包含：`group_id`、`sample_id`、`source_type`、`candidate_traj`，加上 v3 需要的标量分字段。V-JEPA 抽取额外需要 `history_images`、`future_images`。

多 horizon 由 `multi_horizon_protocol.py` 强约束：`future_num_frames`、`trajectory_steps`、`step_time_s` 三者必须自洽。老 checkpoint 在超长 horizon 上评是"非正式训练"，协议会明确拒绝把它当成同一实验。

## 5. 有序运动支路（Ordered-motion support）

独立于主线的三态可弃权判决。核心文件：`ordered_motion_support.py`。

```
上游 ordered-motion scorer（不在本仓库）
    │  --include-segment-ledger
    ▼
segment ledger JSONL
    │
    ├─ 校准（只读 val）
    │   ordered_motion/calibrate_ordered_motion_support.py
    │   ⇒ ordered_motion_support_config.json (frozen thresholds)
    │
    ├─ 推理（不读 label）
    │   ordered_motion/score_ordered_motion_support.py
    │   ⇒ eval_support.jsonl（supported / unsupported / insufficient_evidence）
    │
    └─ 事后审计（可选）
        audit/audit_ordered_motion_support.py
        audit/audit_ordered_motion_physical_labels.py（对 NAVSIM PDM 做集合值对照）
```

设计意图：
- `SupportDecisionConfig` 里 `support_energy_max < unsupported_energy_min` 是硬约束——三态之间必须留出**弃权带**，不允许 supported/unsupported 相互侵蚀。
- **能见度不足或不确定性过大就弃权**，不算做"同意"。当前审计的正式 split 上弃权率约 85%（735 行中 623 弃权），这是有意的保守。
- `source_type` **只在校准阶段用于打标签**，推理阶段完全不读。

一键入口：`scripts/run_ordered_motion_support_formal.sh`。

## 6. 审计工具（`audit/`）

不是运行时依赖，是保证"发布的数字站得住"的离线检查。

| 脚本 | 检查什么 |
| --- | --- |
| `audit_formal_splits.py` | train/val/eval 是否满足"组、场景、图像、horizon"四层不相交，缺一项就 fail-closed |
| `audit_multi_horizon.py` | 各 horizon manifest 是否与协议一致 |
| `audit_clean_gate_counterfactual.py` | 打乱视觉-轨迹配对后 gate 是否还能排名——用来检验 gate 是否依赖真实的配对模态而不是走了捷径 |
| `run_clean_gate_lofo_audit.py` | LOFO（leave-one-hard-family-out）：留出一族硬不匹配训练，测跨族迁移能力 |
| `audit_ordered_motion_support.py` | 冻结阈值后统计三态的 precision 是否符合校准时的承诺 |
| `audit_ordered_motion_physical_labels.py` | 用独立的 NAVSIM PDM 标签做集合值对照，避免把"记录轨迹是唯一正确"当成假设 |

`repair/` 里配套的两件套：`repartition_image_disjoint_splits.py` 当 split 有图像/组重叠时原子重新分配，`repartition_feature_cache.py` 让缓存跟着重排。

## 7. Scripts 层

`scripts/` 只是把上述 pipeline 打包为可提交的 shell / sbatch：

- `run_mainline_example.sh` — 主线最小复现
- `run_ordered_motion_support_formal.sh` — 有序运动支路正式跑
- `run_ordered_motion_4s_sharded.sh`、`run_ordered_motion_raw_frame_controls_sharded.sh` — 分片跑 4s horizon
- `run_clean_gate_lofo_audit.sbatch`、`run_full_clean_gate_counterfactual_audit.sbatch`、`run_ordered_motion_speed_rank_multiseed.sbatch` — 集群作业

## 8. 目录速查

```
IAC/
├── README.md                       发布信息、最小可跑命令、当前数字
├── ARCHITECTURE.md                 本文
├── requirements.txt
├── multi_horizon_protocol.py       horizon 硬协议：2s/4s/6s/8s frame ↔ traj 自洽
├── models/                         冻结的 .pt + mainline_manifest.json
│
├── pipeline/                       主线推理（extract → score → fuse → confidence）
│   ├── extract_vjepa_video_features.py
│   ├── score_acceptability_calibrator.py
│   ├── score_visual_mismatch_gate.py
│   ├── fuse_v3_clean_gate.py
│   ├── score_iac_confidence.py
│   └── _pathfix.py                 sibling 目录 import 兼容
│
├── training/                       离线训练（发布模型的来源，运行时不依赖）
│   ├── train_iac_acceptability_calibrator.py
│   ├── train_visual_mismatch_gate_scorer.py
│   ├── train_ordered_motion_speed_rank.py   （需外部 iac_extensions 包）
│   └── _pathfix.py
│
├── ordered_motion/                 有序运动支路（推理时独立于主线，并联判决）
│   ├── ordered_motion_support.py   支路核心库（三态判决 + 校准）
│   ├── calibrate_ordered_motion_support.py  仅读 val 冻结阈值
│   ├── score_ordered_motion_support.py      不读 label 的推理
│   └── _pathfix.py
│
├── audit/                          事后审计（非运行时依赖，保证"数字站得住"）
│   ├── audit_formal_splits.py
│   ├── audit_multi_horizon.py
│   ├── audit_clean_gate_counterfactual.py
│   ├── audit_ordered_motion_support.py
│   ├── audit_ordered_motion_physical_labels.py
│   ├── run_clean_gate_lofo_audit.py
│   └── _pathfix.py
│
├── repair/                         数据修复（split 泄漏 / 缓存重排）
│   ├── repartition_image_disjoint_splits.py
│   ├── repartition_feature_cache.py
│   └── _pathfix.py
│
├── scripts/                        shell / sbatch 打包入口
└── tests/                          pytest：协议、audit、支路、缓存重排
```

`_pathfix.py` 让脚本原地保留 `from foo import bar` 的扁平 import 写法，同时能跨子目录解析。头部 `import _pathfix  # noqa: F401` 一行触发。

## 9. 读代码建议顺序

1. `multi_horizon_protocol.py` — 先看数据契约，30 行就够。
2. `models/mainline_manifest.json` — 看当前发布这版到底冻结了什么参数。
3. `pipeline/fuse_v3_clean_gate.py` 的 `main()` — 40 行看完融合逻辑，就理解了"gate 为什么只是惩罚项"。
4. `pipeline/score_iac_confidence.py` 顶部 docstring + margin 判决段 — 理解三态。
5. `ordered_motion/ordered_motion_support.py` 的 `SupportDecisionConfig` — 理解支路的弃权带。
6. 主线不清楚的边缘情况再回看 `train_*.py`。

审计脚本、repartition、sbatch 都是"需要时再看"，不影响理解主线。
