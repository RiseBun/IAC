# FCS / future-to-action mediation 审计（2026-09-16）

## 结论

当前 WorldDrive 的 25 组 pilot 不能作为 FCS mediation 结果发布。它已经证明了一个较弱但有价值的事实：模型内部确实存在“未来 latent permutation → native action head”的可干预路径，并且动作输出是模型自己的 action head 输出；但它还没有证明动作变化是通过该未来路径传递的。

审计结果保存在：

- `reports/future_to_action_mediation_readiness_20260916.json`
- `reports/future_to_action_runner_inventory_20260916.json`
- 原始输入：`reports/worlddrive_future_latent_native_actions_all_20260916.jsonl`、`reports/worlddrive_future_latent_level2_20260916.jsonl`、`reports/worlddrive_future_latent_level2_swap_20260916.jsonl`

## 当前证据

| 项目 | 当前值 | 含义 |
|---|---:|---|
| counterfactual groups | 25 | 达到 pilot 规模，但未达到确认所需的 30 个独立 source |
| native action rows | 50 | 25 个 `future_native` + 25 个 `future_reverse` |
| native action head recorded | 50/50 | 动作来自模型内部 action head，而非 evaluator 轨迹 |
| formal foresight mediation input | 50/50 | 运行入口标记为 mediation-oriented |
| future perturbation | 有 | `internal_future_latent_permutation` |
| pathway-blocked branch | 无 | 无法计算 blocked effect |
| fixed-action pathway control | 无 | 无法计算 specificity control |
| frozen normalization / future fingerprint | 无 | 无法进行契约化距离比较 |

`level2_records_swap.jsonl` 里的 image-swap control 不能替代 pathway block：它改变输入图像/未来内容的配对方式，却没有把未来信息从 action head 的路径中阻断，也没有保持“同一被扰动 future、只改变 pathway state”的关系。

因此当前状态是 `blocked_missing_controls`，而不是分数为 0，也不是 mediation 失败。

## FCS 的正式四条件

每个独立 source 必须同时产生以下四条记录：

1. `baseline`：原始 future，action pathway normal；
2. `future_perturbed`：只扰动 future，action pathway normal；
3. `future_perturbed_pathway_blocked`：使用与第 2 条完全相同的 perturbed future，但在送入 action head 前阻断该 future pathway；
4. `future_fixed_action_pathway_control`：使用与 baseline 完全相同的 future，但 action pathway 置为 fixed-action control。

四条记录必须保持 source/history/command/nuisance seed/model revision/model id/native action source 相同。动作距离使用仅在 calibration sources 上拟合并冻结的 normalization。确认 sources 必须与 calibration source manifest 不相交，至少 30 个独立 source。

## 下一步执行规格

服务器结果目录的全树扫描也没有找到 WorldDrive 的 Python/Shell runner 源码；当前只保留了 checkpoint 和结果 artifact。因此下一轮不再扩大现有 25 组 pilot，而是先取得 inference source 或实现一个能访问模型内部 future latent/action head 的 adapter，再运行正式四条件：

- 在同一 forward pass 中保存 baseline future 表示；
- 生成 future perturbation，并保存其 fingerprint；
- 对同一 perturbed future 运行 normal 与 pathway-blocked 两个 action-head 分支；
- 对 baseline future 运行 normal 与 fixed-action control 两个分支；
- 每个分支记录 `condition`、`future_fingerprint`、`pathway_state`、`native_action`、`action_normalization_fingerprint/scale` 以及完整 invariance metadata；
- 先用 calibration source 拟合 normalization，再冻结并跑 source-disjoint confirmation；
- 使用 `tools/score_future_to_action_mediation.py` 计算 future effect、blocked effect、pathway suppression 和 specificity control。

正式 promotion gates 已固定在 `configs/future_to_action_mediation_v1.json`：future effect CI 下界至少 0.05，pathway suppression CI 下界至少 0.50，specificity control CI 上界不超过 0.25，且至少 30 个 source、确认集与 calibration 集不相交。

## 与 FCS 独立 rollout 的关系

FCS 独立 rollout 可以回答“给定未来和动作是否共同变化”；本 mediation 实验回答“未来变化是否经由指定的 future-to-action pathway 改变 native action”。后者是因果路径证据，不能由 MAS、RCS、GS 或普通 future/action 相关性替代。
