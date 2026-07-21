# IAC 服务器项目完整总结，2026-07-01

## 1. 这个项目到底在解决什么问题

IAC 不是 planner，也不是 world model。

它的角色是评测一个 World Action Model（WAM）。

核心问题是：

- 给定 `history images`
- 给定一条 `candidate trajectory` 或动作
- 给定一组来自 WAM 的 `future images`

我们能不能判断：

- 这些未来图像是否真的支持这个动作或轨迹，也就是 `consistency`
- 这条动作或轨迹本身是否物理上、语义上合理，也就是 `validity`

这件事重要，是因为当前 WAM 的评测通常是割裂的：

- 视觉预测常常用像素或感知指标评估
- 动作质量常常只看下游任务成功率

这种割裂忽略了 WAM 最核心的前提：

- 动作应该建立在视觉前瞻之上
- 视觉前瞻应该约束动作选择

所以 IAC 的目标，是做一个不依赖 simulator、不依赖人工标注、可外挂任意 WAM 的耦合式评测 benchmark。

## 2. 当前项目的整体定位

服务器上的这个项目，已经逐渐收敛到下面这个定位：

- 主线：把 IAC 做成 WAM benchmark
- 方法线：把 critic 做得更强，让它更好地区分“被未来支持的轨迹”和“没有被未来支持的轨迹”
- 分析线：通过 probing 和层级诊断，弄清楚模型到底在用什么信号做判断

这个项目已经不再被当成“做一个更大的 critic”。

它现在更像是在做三件事：

- 定义一个合理的 WAM 评测问题
- 训练一个足够可用的 evaluator
- 解释 evaluator 到底在依赖什么

## 3. 数据与 benchmark 设定

### 3.1 数据集方向

项目里讨论过两条数据线：

- `NAVSIM`
- `nuPlan`

当前服务器上的主要有效实验，核心还是建立在严格 future 约束下的 `NAVSIM` 设置上。

关键不只是“用哪个数据集”，而是“这个 index 是不是构建对了”。

### 3.2 Strict-future 是前提，不是细节

项目早期发现了一个很关键的失败模式：

- 有些旧设置可能会把 history 尾帧重放，或者把时间错位的帧，误当成 future evidence

这样一来，benchmark 从根上就是错的。

所以项目早期做的一件非常重要的事是：

- 构建 strict-future 的 NAVSIM index
- 对 index 做 audit
- 保证正样本的 future frame 真的是“真实未来帧”

这是整个项目最重要的前提之一，因为如果这个环节错了，后面的训练和报告都会失真。

相关文件：

- [README.md](/C:/Users/LPN19/Desktop/iac/IAC/README.md)
- [index_audit.md](/C:/Users/LPN19/Desktop/iac/IAC/docs/index_audit.md)
- [evaluation_protocol_2026-07-01.md](/C:/Users/LPN19/Desktop/iac/IAC/docs/evaluation_protocol_2026-07-01.md)

### 3.3 样本和标签是如何构造的

IAC 不依赖人工去标“图像和动作是否一致”。

它是自动构造正负样本的。

正样本：

- 真实历史帧
- 真实未来帧
- 真实 ego future trajectory

负样本主要通过扰动和交换构造，例如：

- `traj_swap`
- `image_swap`
- `time_shift_future`
- `perturb_lateral`
- `perturb_heading`
- `perturb_speed`
- 有些设置里还会有 reverse 类负样本

这件事很重要，因为 benchmark 想成立，必须：

- 可规模化
- 不依赖主观人工标注
- 对不同 WAM 接口统一

## 4. 当前最强模型的 pipeline 到底是什么

当前最强这条线，不是 full multilayer DINO 方法。

它本质上是一个更强的 single-layer evaluator，加上更合理的训练结构。

当前主要 pipeline 是：

- 冻结的 `DINOv2 ViT-S/14`
- 使用单层视觉特征，目前是 layer 11
- 分别编码 `history images` 和 `future images`
- 用 MLP 编码 `candidate trajectory`
- 用 MLP 编码 `ego state`
- 融合后形成：
  - 用于 consistency 推断的 `z_shared`
  - 用于 validity 推断的 `z_validity`
- 最终输出两个主分数：
  - `consistency`
  - `validity`

除此之外，模型里还包含：

- speed / steering / progress / temporal 的辅助头
- future evidence 分支
- hierarchical consistency 分支

核心实现文件：

- [train.py](/C:/Users/LPN19/Desktop/iac/IAC/train.py)
- [train_dinov2_v5_minimal.py](/C:/Users/LPN19/Desktop/iac/IAC/train_dinov2_v5_minimal.py)
- [train_navsim_future_dinov2_evidence.py](/C:/Users/LPN19/Desktop/iac/IAC/configs/train_navsim_future_dinov2_evidence.py)
- [train_navsim_future_dinov2_evidence_recallboost.py](/C:/Users/LPN19/Desktop/iac/IAC/configs/train_navsim_future_dinov2_evidence_recallboost.py)

## 5. 为什么引入 DINOv2

早期视觉 baseline 不够强。

项目切到 DINOv2，主要是因为：

- 它的视觉语义能力明显强于之前的轻量视觉编码器
- 它适合冻结后直接拿来做 evaluator backbone
- 它让我们能更认真地问：未来视觉证据到底有没有真正帮助 consistency 判断

当前选择是偏保守的：

- `DINOv2 ViT-S/14`
- 冻结 backbone
- 单层使用

这是有意为之。

目标不是先靠更大 backbone 赢，而是先验证：

- future visual evidence 到底能不能对评测有帮助

## 6. 在 plain critic 基础上做了哪些增强

### 6.1 Future evidence 注入

项目加了一个 `future_consistency_evidence` 分支，让 future frame 不再只是被动拼接进去。

目的：

- 让 future image 的证据显式影响 consistency 判断
- 测试模型是否真的对 future frame 更敏感

这条分支非常关键，因为 benchmark 真正想回答的是：

- 这条轨迹是不是被未来图像支持

而不是：

- 这条轨迹是不是一般意义上看起来还算合理

### 6.2 Hierarchical consistency

项目还把 consistency 拆成了更细的中间结构：

- `physics_support`
- `action_support`
- `future_support`
- 最后融合成 `consistency_fuse`

目的：

- 把物理合理性和未来图像支持分开
- 让 consistency 判别不再是一个黑箱单头
- 为后面的 probing 提供更有信息量的中间层

### 6.3 Ranking 式训练

项目后面不再满足于纯点式 BCE。

当前训练包含：

- consistency / validity 的 BCE 风格监督
- 分组 candidate 上的 `group ranking loss`
- future evidence 的辅助监督
- hierarchical consistency 的辅助监督

Ranking 这一块非常重要，因为 IAC 不是只想做绝对分类。

它还要回答：

- 真正被支持的 candidate，能不能排在那些不被支持的 candidate 前面

## 7. 训练里最重要的观念变化

项目里一个非常重要的发现是：

- 用 `val_loss` 选 best checkpoint，和 benchmark 目标并不一致

原因是：

- 一个模型可能 loss 不难看
- 但 consistency 分数会保守塌缩
- 在糟糕阈值下看起来 accuracy 很高
- 实际 recall 可能是 0

这种模型对 benchmark 来说几乎是没用的。

所以项目后来把选模标准从：

- `val_loss`

切到了更贴近 benchmark 的指标上，例如：

- `c_score_gap`
- `c_balanced_acc`
- 后来又尝试加入更偏 `precision / tnr` 的版本

这件事和模型结构改动一样重要。

它改变了训练过程中“什么算 best model”。

## 8. Probing 和诊断：我们到底学到了什么

项目没有停在最终 benchmark 分数上。

还做了批量特征抽取和线性 probe。

抽取和 probe 的层包括：

- `hist_seq`
- `fut_seq`
- `z_traj_cons`
- `z_traj_val`
- `z_shared`
- `z_validity`
- future-evidence 相关中间层

相关工具：

- [extract_probe_features.py](/C:/Users/LPN19/Desktop/iac/IAC/tools/extract_probe_features.py)
- [train_layer_probes.py](/C:/Users/LPN19/Desktop/iac/IAC/tools/train_layer_probes.py)

### 8.1 最重要的 probe 结论

最强的 physics / geometry 信号，并不是先出现在图像序列层里。

它先出现在轨迹分支里。

对物理量最可读的层主要是：

- `z_traj_val`
- `z_traj_cons`

特别强的量包括：

- `mean_speed`
- `heading_change`
- `final_disp`
- `lateral_abs`

### 8.2 Consistency 信号主要在哪

Consistency 最可读的地方主要是：

- `z_shared`

而不是：

- `hist_seq`
- `fut_seq`

这意味着：

- 当前 consistency 推理主要还是发生在融合层
- future image 的证据还没有被真正前移到早期视觉层

### 8.3 对“多层”概念的影响

这个 probe 结果直接改变了项目对“multilayer”这件事的理解。

更合理的结论是：

- 当前对 IAC 更有意义的“层”，更像是 critic / WAM 内部的推理层级
- 而不只是 DINO backbone 的深度层

所以，单纯做 backbone multi-layer fusion，并不是最自然的主叙事。

更重要的是：

- physics 在哪里出现
- consistency 在哪里变得可读
- future evidence 是往前走了，还是一直停留在晚期融合层

## 9. 关键实验线和结果

下面是目前真正重要的实验结果。

### 9.1 Stable evidence baseline

目录：

- `work_dirs/iac_navsim_future_dinov2_evidence_ddp4_resume`

2048 benchmark 最好结果：

- consistency balanced accuracy: `0.6089`
- consistency precision: `0.1722`
- consistency recall: `0.8352`
- consistency F1: `0.2855`
- consistency TNR: `0.3825`
- validity accuracy: `0.9868`

它说明：

- future evidence 这条线是能跑通的
- 但模型校准仍然很差
- 默认阈值下很容易塌成全负类判断

### 9.2 Hierarchical smoke

目录：

- `work_dirs/iac_navsim_future_dinov2_hierarchical_smoke`

1024 benchmark：

- consistency balanced accuracy: `0.5961`
- precision: `0.1600`
- recall: `0.9481`
- F1: `0.2738`
- TNR: `0.2441`

它说明：

- hierarchical 分支能阻止 total all-negative collapse
- 但也非常容易制造大量 false positive

### 9.3 Metric-aware checkpoint line

目录：

- `work_dirs/iac_navsim_future_dinov2_metric_ddp4_continue`

2048 benchmark：

- consistency balanced accuracy: `0.6517`
- precision: `0.1953`
- recall: `0.8278`
- F1: `0.3161`
- TNR: `0.4755`
- validity accuracy: `0.9888`

这是第一次非常明确的实质性进步。

它之所以变强，主要不是因为 backbone 变大了，而是因为：

- 训练和选模目标终于更贴近 benchmark 的真正用途

这次实验证明：

- 只要选模逻辑正确，evaluator 的质量是可以明显上升的

### 9.4 Hard-negative continuation

目录：

- `work_dirs/iac_navsim_future_dinov2_hardneg_ddp4_continue`

2048 benchmark：

- consistency balanced accuracy: `0.6604`
- precision: `0.1921`
- recall: `0.9084`
- F1: `0.3171`
- TNR: `0.4124`
- validity accuracy: `0.9893`

这是服务器上当前最强的正式 2048 结果。

解释：

- 它在 balanced accuracy 和 F1 上超过了上一版
- 但这种超过不是最理想的方式
- 它的提升，更多来自 recall 被继续拉高
- precision 并没有真正显著改善

所以这是一个真实进步，但不是“最想要的进步”。

### 9.5 Precision-first smoke

目录：

- `work_dirs/iac_navsim_future_dinov2_precision_smoke`

128 benchmark：

- consistency balanced accuracy: `0.6624`
- precision: `0.1892`
- recall: `0.5833`
- F1: `0.2857`
- TNR: `0.7414`

这条线来自：

- 更强的 hard-negative 约束
- 更偏 precision/TNR 的 checkpoint 选择

它说明：

- 模型确实被推向了更保守的工作点
- TNR 上去了
- 但小样本 smoke 的整体质量反而不如前一版 hard-negative smoke

这说明方向不是错的，但当前强度设置太激进了。

## 10. 当前最强模型到底是什么

服务器上当前最强正式结果对应：

- checkpoint:
  `~/IAC/work_dirs/iac_navsim_future_dinov2_hardneg_ddp4_continue/checkpoints/latest.pth`
- benchmark:
  `~/IAC/work_dirs/iac_navsim_future_dinov2_hardneg_ddp4_continue/benchmark_val_2048/score_analysis.json`

这个模型的本质是：

- 单层冻结 DINOv2 视觉骨干
- future evidence 注入
- hierarchical consistency 分支
- ranking-based training
- hard-negative 辅助约束
- consistency-aware checkpoint selection

它不是什么：

- 不是 full multilayer DINO 方法
- 不是 learned world model
- 不是 planner
- 也还不是一个高 precision 的成熟 evaluator

## 11. 项目过程中暴露出来的问题

项目里已经暴露出几个非常关键的失败模式。

### 11.1 Future-image 设置可能无效

如果正样本没有真实 future frame，整个 benchmark 就从根上不成立。

这个问题现在已经通过 strict-future index 和 audit 修掉了。

### 11.2 默认阈值会制造“假好看”

有些模型 naive accuracy 很高，只是因为负样本多。

但 consistency recall 可能直接是 0。

这是项目中最重要的教训之一。

### 11.3 Shortcut 风险仍然存在

Probe 结果提示：

- geometry priors 很强
- trajectory branch 很可读
- future image branch 相对弱

这意味着 evaluator 仍然有可能部分依赖轨迹合理性，而不是真正依赖 future evidence。

### 11.4 长训练不总是稳定

有一轮 continuation 在训练中途被外部 `SIGKILL` 杀掉。

虽然没有毁掉实验，因为更早的有效 best checkpoint 已经保存了，但这说明：

- 服务器上的长时间 uninterrupted run 并不总是可靠

## 12. 当前这些数值应该怎么理解

当前最强正式结果大约是：

- consistency balanced accuracy: `0.66`
- consistency precision: `0.19`
- consistency recall: `0.91`
- consistency F1: `0.317`

这些数值应该这样理解。

好的一面：

- 模型明显比早期 baseline 强
- benchmark setup 已经不是随便凑出来的
- evaluator 的确具备了真实区分能力

不够好的地方：

- precision 还是低
- false positives 还是多
- future evidence 还没有成为早期主导表征

所以，当前 evaluator 是有用的，但离“很强”还差得很明显。

## 13. 当前最可靠的项目判断

到现在为止，这个项目最稳妥的判断是下面这些。

### 13.1 我们可以有把握说的

- IAC 作为 WAM benchmark 的方向是成立的
- strict-future setup 是必要条件，而且现在已经建立起来了
- 单层冻结 DINOv2 是一个可行的 evaluator backbone
- ranking-aware、benchmark-aware 的训练和选模，确实能带来实质提升
- 轨迹分支里有强 physics 信息
- consistency 主要还是在融合 latent 中形成

### 13.2 现在还不能诚实声称的

- 未来图像证据已经成为 consistency 判断的主导来源
- multilayer DINO fusion 是这个项目的核心创新
- 当前 evaluator 已经足够高精度
- 我们已经完全回答了“图像到底是否必要”这个问题

## 14. 现在最值得和其他人讨论什么

这一部分最适合你拿去和别人一起思考。

### 14.1 这个项目的贡献到底更偏 benchmark，还是更偏 method

现在有两种可能的讲法：

- benchmark-first：
  定义了一个 simulator-free、label-free、plug-and-play 的 WAM evaluator
- method-first：
  提出一个更强的 consistency critic，带 future evidence 和 hierarchical constraints

以目前代码和结果来看，第一种叙事更强。

### 14.2 “多层”在这个项目里到底该指什么

当前证据更支持：

- meaningful multilayer 应该优先指 critic / WAM 内部的层级
- 而不是简单指 DINO backbone 的深度层

这会直接改变什么才算真正有意义的创新点。

### 14.3 我们真正缺的是什么

当前最核心的未解问题不是 backbone 大小。

而是：

- 怎么降低 false positives，又不把 recall 全压没
- 怎么让 future evidence 更直接、更早地影响 consistency
- 怎么让 evaluator 对 future perturbation 真正具有因果敏感性

### 14.4 接下来该继续优化 critic，还是加强诊断分析

一个很合理的策略是：

- 保留一条稳定的 best benchmark 主线
- 保留一条受控的小步训练改造线
- 更多投入到 diagnostic 和 causal analysis 上

这很可能比无止境调一个 critic 更能形成好的论文叙事。

## 15. 当前值得推进的几个具体方向

如果团队还想沿现在这条线继续推进，最合理的方向是：

### 方向 A：source-specific negative shaping

不要再用一个全局 hard-negative 惩罚去管所有负样本。

可以按 source type 分开做更细的约束，例如：

- `time_shift_future`
- `traj_swap`
- `perturb_speed`

原因是：

- 当前 false positives 很可能不是均匀分布的
- 全局压负样本太粗糙

### 方向 B：更好的 consistency calibration

- 尽量保留当前最强 latent 表征
- 单独学习更好的 consistency 决策边界
- 可以考虑 source-aware calibration 或 held-out calibration

原因是：

- 当前分数分离已经有了
- 但还没有被稳定映射成更高 precision

### 方向 C：更强的 future-evidence 因果测试

- 显式报告 future perturbation sensitivity
- 比较 image perturbation 和 trajectory perturbation 对分数的影响
- 检查模型决策是否真的由 future evidence 驱动

原因是：

- 这会直接加强 benchmark 叙事

### 方向 D：继续做更干净的 WAM-level probe

- 扩大 probe 样本量
- 比较图像层、轨迹层、融合层的可读性差异
- 给每次 benchmark 附解释性 artifact

原因是：

- 这与项目目前真正学到的东西是对齐的

## 16. 当前建议保留的 baseline

如果团队现在想保留一个“当前最强主结果”，建议保留：

- `work_dirs/iac_navsim_future_dinov2_hardneg_ddp4_continue`

理由：

- 它有当前最强的正式 2048 balanced accuracy
- 它是完整跑完的
- 它包含了后续训练改造的收益

如果团队还想保留一个“概念最干净的对照点”，建议同时保留：

- `work_dirs/iac_navsim_future_dinov2_metric_ddp4_continue`

理由：

- 它最清楚地体现了 checkpoint-selection 改造的收益
- 比 hard-negative continuation 更容易讲清楚

## 17. 最后的总判断

这个项目其实已经做出了一件有意义的事：

- 它把 IAC 从一个比较松散的 critic 想法，推进成了一个有审计数据、有可复现实验、有正式 benchmark 输出、还有诊断工具的评测系统
- 它证明了 benchmark-aware 的训练和选模，的确能提升 evaluator 质量
- 它说明当前最有信息量的信号，主要还在轨迹和融合层
- 它也拿到了比早期 baseline 更强的正式结果

但这个项目还没有解决最难的那个问题：

- 如何让 future-image evidence 形成一个真正高 precision、低 shortcut 的 consistency 判断

所以最诚实的总结应该是：

- benchmark 方向：强，而且有意义
- critic 质量：在持续变强，但还不够强
- 科学洞察：已经有用了
- 最终方法结论：还没定型

这正是最适合停下来总结、拿给别人一起反思和共创的阶段。
