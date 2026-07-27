# SCOPE：面向 IAC 的可解释一致性拆分

## 1. 要解决的不是“再换一个更强 backbone”

当前端到端一致性 critic 把图像、视频、轨迹和场景特征压成一个分数。
当结果不好时，很难判断：

1. 视频侧没有读出运动；
2. 读出了运动，但与候选轨迹比较错了；
3. 某一类证据有效，但融合时被其他证据抵消；
4. 模型没有使用视觉，而是利用候选类型、首步或数据来源捷径；
5. 图像本来就支持多条近邻轨迹，不应该强迫唯一 Top-1。

V-JEPA、DINO、RAFT 都可以改善视觉表征，但更强的黑箱表征本身不能回答以上问题。
本扩展的目标是给 IAC 增加一条可审计的证据路径，而不是替换 IAC 的主 scorer。

## 2. 架构拆分

```text
连续图像
   │
   ├─ 视频侧运动估计（不能读取 candidate trajectory）
   │      ├─ 纵向进度 / 平均步长 / 速度变化
   │      ├─ 横向位移
   │      ├─ 航向变化 / 曲率
   │      └─ 路径形状 / 转向时刻
   │
候选轨迹
   │
   └─ 确定性计算同名运动目标
              │
              ▼
      逐属性标准化残差
              │
              ▼
       非负、可加的证据贡献
              │
              ▼
  longitudinal + lateral + heading + path_shape
              │
              ▼
          总运动能量
```

每个候选都可以导出一份证据账本：

- 视频估计值；
- 候选目标值；
- 视频估计的不确定性；
- 标准化残差；
- 非负属性权重；
- 每个属性对总能量的贡献；
- 四个命名 family 对总能量的贡献。

四个 family contribution 的和必须严格等于总运动能量。这是代码测试的不变量，
而不是事后用另一个模型生成的“解释”。

## 3. 两种使用方式

### 3.1 结构内拆分

`train_scope_interpretable_motion_head.py` 继承已有
`ScopeDinoMotionCritic`。它不增加新的融合 MLP，只把已有 36 维运动比较拆成：

- `scope_motion_normalized_residual`
- `scope_motion_weighted_component_contribution`
- `scope_motion_family_contribution`
- `scope_motion_energy`

已有 SCOPE motion checkpoint 的参数名保持兼容。加载时默认要求 missing/unexpected
keys 均为空，避免 `strict=False` 静默漏掉关键 head。

训练入口：

```bash
python train_scope_interpretable_motion_head.py \
  --config configs/train_navsim_future_dinov2_scope_interpretable_motion.py
```

证据账本导出：

```bash
python tools/eval_interpretable_motion_evidence.py \
  --checkpoint work_dirs/iac_navsim_future_dinov2_scope_motion_head/checkpoints/best.pth \
  --config configs/train_navsim_future_dinov2_scope_interpretable_motion.py \
  --index indices_navsim_future/diagnostics/consistency_val_low_iou_g200.jsonl \
  --output-summary work_dirs/interpretable_motion/low_iou_summary.json \
  --output-rows work_dirs/interpretable_motion/low_iou_ledger.jsonl
```

同一个命令应分别运行：

```text
normal
reverse_future
shuffle_future
roll_future
zero_future
```

如果完整视频与这些控制没有稳定差异，就不能说运动 head 真正在使用时序证据。

### 3.2 冻结 IAC 后的可解释纵向残差

`tools/apply_longitudinal_motion_residual.py` 用于不改 checkpoint 的离线验证：

1. 以当前 IAC 组内 winner 为参考；
2. 把其他候选相对参考轨迹的差异分解为纵向、横向和航向；
3. 只有差异主要为纵向时，才启用 speed/multi-interval evidence；
4. 在 logit 空间写出唯一的 signed contribution；
5. 保留原分数，并把新分数写入 `iac_consistency_interpretable`。

```bash
python tools/apply_longitudinal_motion_residual.py \
  --primary-scores work_dirs/iac_base/holdout/wam_iac_scores.jsonl \
  --evidence-rows work_dirs/scope_motion/holdout/flow_rows.jsonl \
  --evidence-key flow_speed_energy \
  --weight 1.0 \
  --share-threshold 0.5 \
  --minimum-longitudinal-m 1.0 \
  --output-scores work_dirs/interpretable_motion/holdout_scores.jsonl \
  --output-summary work_dirs/interpretable_motion/holdout_summary.json
```

这个变换不读取 `source_type`。source label 只在完成评分后用于报告每类
pairwise、激活率和 winner transition。代码测试会直接改写全部 source label，
并要求输出分数完全不变。

权重和阈值必须只在独立 calibration drives 上选择，然后冻结到 low-IoU/holdout。
该工具故意不提供 test sweep。

## 4. 已有独立实验告诉了我们什么

精简、带原始文件 SHA256 的结果见
`docs/assets/scope_interpretable_evidence_verified_summary.json`。

这些是本项目在独立 NAVSIM 协议上的结果，不是当前 IAC checkpoint 或
IAC-PathBench 正式指标，不能直接横向比较。

### 4.1 分项视觉运动确实可学

drive-disjoint test 上：

| 视觉属性 | PLCC |
|---|---:|
| longitudinal | 0.9114 |
| lateral | 0.9307 |
| heading | 0.9183 |

时序连续性 AUC 为 0.9565。完整分项能量在 3,728 个八候选测试组上达到
Top-1 44.66%、MRR 0.6867。

这说明“先估计运动属性，再比较候选”不是纯概念；但 speed pairwise 仍只有
约 68%，所以视觉速度仍是最难的一项。

### 4.2 光流残差有物理意义，但不是独立主模型

在更严格的十二候选 speed-v2 协议中，corrected RAFT-global 使：

- Top-1：49.79% → 52.63%；
- MRR：0.7120 → 0.7327；
- heading pairwise：90.21% → 96.22%；
- speed pairwise：74.92% → 75.33%。

它主要改善 heading 和整体排序，speed 增益很小。因此“用了 RAFT”本身不是
足够的 novelty。

### 4.3 为什么要按 failure 条件触发

把 multi-interval motion evidence 全局融合时：

- speed pairwise 提高约 0.48 pp；
- 总体 Top-1 下降约 1.96 pp；
- 预设主门槛失败。

只对几何上主要呈纵向差异的候选启用同一证据后：

- Top-1 提高 0.33 pp；
- speed pairwise 提高 0.53 pp；
- 三个随机种子 speed 增益均为正；
- 正常候选身份相对 Sattolo candidate derangement 的 Top-1 差为 1.79 pp。

四个预设门槛通过三个，但“speed 至少 +1 pp”没有通过。正确结论是：

> failure-conditioned 使用避免了全局退化，并证明证据依赖正确的
> candidate-video identity；它仍未解决 speed failure，也未单独构成 novelty。

## 5. 怎样判断“哪里不够好”

建议每个 failure group 同时导出以下诊断：

| 现象 | 更可能的问题 |
|---|---|
| visual estimate 与 GT 轨迹目标都不相关 | 视频属性 head 不够好 |
| visual estimate 准，但所有候选 residual 接近 | 候选视觉不可区分或协议太近 |
| longitudinal residual 能分开，最终 scorer 仍排错 | 融合/校准问题 |
| shuffle/reverse 后证据不变 | 没有使用时序信息 |
| candidate derangement 后增益不变 | 可能是通用分数偏置或身份捷径 |
| 某 family contribution 长期压倒其他项 | 权重或尺度失配 |
| 多个近邻候选 residual 都低 | 应输出支持集，不应强迫唯一 Top-1 |

## 6. 下一步建议

### A. 先做冻结 checkpoint 的 failure manifest

对同一批 regular、low-IoU、holdout groups 同时保存：

- IAC base score；
- V-JEPA gate score；
- 四类 motion family contribution；
- RAFT/TIRF longitudinal residual；
- recovered-set support；
- 最终 winner transition。

先回答各模块分别解决了哪些错误，再决定训练什么。

### B. 正样本训练视觉属性，候选只进入比较器

视频侧属性 head 只用 exact/strong positive 学习“图像实际发生了什么”；
候选轨迹不进入视觉 encoder。随后所有候选只通过确定性目标和残差比较。
这比直接学习 image+trajectory→consistent 更容易排除轨迹捷径。

### C. 加入可拒绝机制

当某一视觉属性不确定性高，模型应降低该 family 的影响或 abstain，而不是让
另一个黑箱 gate 猜是否可信。最终输出可以是：

```text
supported / contradicted / visually-underdetermined
```

这与 IAC 的多解支持集比唯一 hard Top-1 更一致。

### D. 单元级反事实

训练和评估都应加入结构化单元测试：

- 只改 speed：主要改变 longitudinal contribution；
- 只改 lateral：主要改变 lateral contribution；
- 只改 heading：主要改变 heading contribution；
- 打乱帧序：降低 temporal reliability，但不改变候选几何目标；
- candidate derangement：破坏正确身份增益；
- source label 全部改名：评分必须完全不变。

## 7. 当前主张边界

可以主张：

- 提供了一个候选盲视觉估计、确定性轨迹目标、非负加法残差和逐候选证据账本；
- 提供了 source-blind failure-conditioned 纵向残差和身份可证伪协议；
- 已有独立实验支持“分项信号存在”和“条件触发比全局融合合理”。

暂时不能主张：

- 已经解决 speed counterfactual；
- 已在最新 IAC checkpoint 上稳定提升；
- 仅凭可解释拆分已经构成论文 novelty。

只有在 RiseBun 最新 checkpoint 和冻结三 split 上同时通过排名、支持集和
identity/temporal controls，才能把它提升为正式论文贡献。
