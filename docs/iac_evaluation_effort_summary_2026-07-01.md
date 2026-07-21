# IAC 评测努力总结, 2026-07-01

## 目标

我们不是在做一个更大的 critic，而是在做一个更可信的 WAM 评测器。
核心问题是：
- future image 是否真的在支撑动作判断
- 轨迹是否真的被未来视觉证据约束
- 哪一层开始出现可解释的 physics / action 信号

## 已做的努力

### 1. 先把评测前提做对

- 建了 strict-future NAVSIM index。
- 做了 index audit，避免 history replay 被误当 future evidence。
- 把 benchmark 和 ranking 的输入组织回 anchor/candidate 级别，避免评测对象错位。

### 2. 把 IAC 作为统一 benchmark

- IAC 输出两个主信号：
  - `consistency`
  - `validity`
- 不把它做成 simulator，不把它做成 planner。
- 目标是统一评价不同 WAM / planner，而不是替代它们。

### 3. 做了 WAM 层级 probe

我们抽了批量特征，并在这些层上做线性 probe：
- `hist_seq`
- `fut_seq`
- `z_traj_cons`
- `z_traj_val`
- `z_shared`
- `z_validity`
- 以及 future evidence 相关中间层

这一步不是为了提分，而是为了回答：
- physics 先出现在哪
- action 先出现在哪
- consistency 先沉到哪一层

### 4. 做了 future evidence 注入

我们把 future image 证据从“被动输入”变成“显式注入”：
- 通过 `future_consistency_evidence` 分支进入 consistency
- 再加单独的辅助监督项

目的很直接：
- 验证 future image 是否能更早、更强地影响一致性判断

## 已经证实的结论

### 1. 轨迹分支先出现物理几何信息

在 512 / 2048 级 probe 上，`z_traj_val` 和 `z_traj_cons` 对物理量最可读，尤其是：
- `mean_speed`
- `heading_change`
- `final_disp`
- `lateral_abs`

这说明：
- physics / geometry 主要先沉到轨迹分支
- 不是先出现在图像序列层

### 2. 一致性主要在 `z_shared`

`consistency` 的可读性主要集中在 `z_shared`，而不是 `hist_seq / fut_seq`。
说明：
- 当前模型里的一致性信号仍主要依赖融合层
- future image 还没有稳定前移到早期层

### 3. future evidence 能跑通，但没显著前移信号

我们做了更强的 future evidence 注入和单独监督。
结果是：
- 训练稳定
- 但 smoke 上没有明显优于 single

这说明：
- 注入链路是通的
- 但还没有形成更强的早期可读信号

## 目前的判断

- IAC 作为 benchmark 是成立的。
- 当前最有信息量的层不是 DINO backbone，而是 WAM 内部的轨迹分支和 shared fusion。
- 所谓“多层”，应该优先理解为 WAM 内部层级，而不是 backbone 层融合。
- future evidence 这条路值得保留，但还需要更干净的约束设计。

## 现在的重点

1. 更干净的 WAM 层级 probe
2. 更明确的 future evidence 注入
3. 继续保持 strict-future benchmark 不变

## 不再投入的方向

- 不把 DINO single / multi / gated 当主叙事
- 不继续用 backbone 层融合冒充 WAM 多层
- 不把更大的 backbone 当作当前问题的主解

## 当前结论

我们已经把 IAC 从“一个 critic”推进成了“一个可诊断 WAM 的 benchmark”。
下一步的关键不是再堆 backbone，而是：
- 让 future evidence 真正前移
- 让 WAM 层级信号更可解释
- 让评测能回答“为什么这个 WAM 更好”
