# DriveVA future-to-action mediation smoke（2026-09-16）

## 结论

DriveVA 是目前服务器上第一个具备足够源码、checkpoint 和 native trajectory head、可以真正执行四条件 mediation runner 的 WAM。四个条件均已成功运行，记录也通过现有 fail-closed scorer；但这只是 1 个 source 的 runner smoke，不是正式 mediation 结果。

结果：

- [四条件 JSONL](../reports/driveva_future_to_action_mediation_smoke5_20260916.jsonl)
- [scorer 输出](../reports/driveva_future_to_action_mediation_smoke5_score_20260916.json)

随后使用 10 条 calibration source 冻结 action scale，并在另外 30 条 source 上完成了 confirmation：

- [calibration scale](../reports/driveva_future_to_action_calibration_20260916.json)
- [confirmation JSONL](../reports/driveva_future_to_action_mediation_confirmation30_20260916.jsonl)
- [confirmation scorer](../reports/driveva_future_to_action_mediation_confirmation30_score_20260916.json)

## Smoke 数值

| 量 | 结果 |
|---|---:|
| source 数 | 1 |
| 四条件覆盖率 | 1.0 |
| future effect | 0.7813 |
| blocked effect | 0.0000 |
| pathway suppression | 1.0000 |
| specificity control | 0.0000 |
| promotion | `insufficient_evidence` |

这组数值证明 runner 的四条件关系已经闭合，但仍只是 1 个 source 的 smoke，不是正式 promotion。future perturbation 改变了 native action（0.7813）；用 baseline future K/V 做 action-query 的 activation patch 后，blocked effect 和 specificity control 都为 0，因此 smoke 层面的 suppression 为 1.0。

## 30-source confirmation

正式 confirmation 的覆盖率为 30/30，calibration 与 confirmation source 完全不相交：

| 量 | 结果 |
|---|---:|
| future effect median / CI95 | 0.4661 / [0.4063, 1.5574] |
| blocked effect median / CI95 | 0.0947 / [0.1447, 0.4040] |
| suppression median / CI95 | 0.5480 / [-0.4522, 0.5664] |
| specificity control median / CI95 | 0 / [0.0177, 0.1147] |
| source-disjoint | verified |
| promotion | `failed` |

因此 DriveVA confirmation 支持“future perturbation 能改变 native action，且多数 source 的 activation patch 有抑制趋势”，但 suppression 的 source-level bootstrap 下界仍为负，未达到 0.50 晋级门。不能把 DriveVA 宣布为已验证的 future-to-action mediation。

## 根因判断

DriveVA 的视频 token 和 trajectory token 在同一个 DiT self-attention 序列中联合去噪。最初的硬 mask/null replacement 会造成 action trajectory 分布外发散，不能作为正式 control。最终采用了 activation/KV patching：

1. baseline branch 缓存每个 DiT block 的 future K/V；
2. blocked branch 仍生成 perturbed future，但只在 action query 的 future K/V 位置替换成 baseline cache；
3. fixed-action control 在 baseline future 上使用同一个 baseline cache。

这样保持了序列长度、action noise 和 K/V 统计分布，fixed-action control 回到 baseline。该结果支持“代码控制有效”，但不能从单个 source 推广到模型级 mediation。

## 代码变更

服务器 DriveVA checkout 中已加入：

- `run_future_to_action_mediation.py`
- `WanVideoPipeline` 的独立 `trajectory_seed`
- future-video 到 action-token 的 directional control
- 四条件 metadata：future fingerprint、pathway state、native action provenance、normalization fingerprint

本地可复用 runner：

- `tools/run_driveva_future_to_action_mediation.py`

## DriveWAM / WorldDrive / Epona

- DriveWAM 服务器目录目前只有 manifest、log 和结果，没有可修改的模型 runner；因此不能可靠地重建 pathway-blocked 条件。
- WorldDrive 同样只有 checkpoint 和结果 artifact，之前的 25 组 pilot 缺少 blocked/fixed-action 分支。
- Epona 当前也只有外部 rollout/评测产物，没有 native future latent → action head 的可访问接口。

因此当前最短路径是先对 DriveVA 解决 control validity，再决定是否把同一接口移植到其他 WAM。不能用外部轨迹注入或图像 swap 代替 mediation control。
