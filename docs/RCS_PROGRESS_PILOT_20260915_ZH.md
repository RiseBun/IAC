# RCS-progress 纵向响应试验（2026-09-15）

本轮不是正式冻结指标，而是用已经生成的 `pure_speed` 同源干预检查纵向响应是否可测。

## 固定协议

- 每个比较对共享同一历史帧、source 和随机种子；只改变 `slow / fast` 纵向速度档位。
- 动作标签使用轨迹终点的自车纵向位置；只保留 `fast - slow >= 0.5 m` 的对。
- 视觉量使用冻结视觉层（Reloc3r-512 旋转 + Metric3Dv2-v2-S 深度 + SIFT/PnP 平移）。
- 只使用未来帧之间的三个区间（索引 1、2、3），不把历史最后帧到第一张生成帧的边界区间计入。
- 每个分支至少有两个区间的 SIFT 内点率达到门槛，才输出纵向累计量；没有则 abstain。

实现：`src/iac_new/progress_response.py`，命令：`tools/score_progress_response.py`。

## 结果

| 数据 | 声明组 | 可比较对 | 得到视觉量 | 覆盖率 | 正确排序 | 单调排序准确率 |
|---|---:|---:|---:|---:|---:|---:|
| Epona pure-speed | 59 | 56 | 39 | 69.6% | 23/39 | 59.0% |
| DriveWAM pure-speed | 86 | 81 | 81 | 100% | 38/81 | 46.9% |

Epona 的质量门槛敏感性：

- 内点率 ≥ 0.10：41/56 有效，29/41 正确，70.7%；
- 内点率 ≥ 0.20（默认）：39/56 有效，23/39 正确，59.0%；
- 内点率 ≥ 0.30：27/56 有效，18/27 正确，66.7%。

## 解释与结论

1. 小规模 sanity check 中，fast 分支确实能读出更大的未来纵向响应；因此视觉层不是完全没有纵向信息。
2. 全量 Epona 结果只达到中等、且随几何质量门槛明显波动，不能冻结为正式 RCS-progress。
3. DriveWAM 结果接近随机（46.9%）。此外该批 manifest 的 `future_images_source` 仍写成 `epona_generated_pure_speed_action`，模型 lineage 元数据不干净，因此只能作为诊断，不能宣布 DriveWAM 的正式失败分数。
4. 这不是再换运动模型的问题：当前主要瓶颈是生成视频对速度干预的响应弱，以及纵向估计质量不稳定。下一步应先修复 pure-speed 的模型来源/同源生成记录，并增加 `stop / slow / normal / fast` 四档真实重生成；在此之前不冻结 RCS-progress。

## 2026-09-16 已推进的修复

- `scripts/annotate_pure_speed_output.py` 不再强制写入 Epona 来源；可以显式传入 `--future-images-source` 和 `--model-id`。
- `tools/build_multilevel_progress_manifest.py` 生成四档动作根（默认纵向尺度 `0 / 0.5 / 1.0 / 1.5`），只改纵向位移，保持横向路径和 yaw 不变，并明确标记 `images_pending`。
- `tools/audit_progress_lineage.py` 检查四档角色、历史指纹、随机种子、source sample、图像来源和模型 ID。

现有旧 DriveWAM pure-speed 清单的 lineage 审计为失败（258 个问题），因此已被阻断，不能再进入正式 RCS-progress。

已在服务器生成新的四档 DriveWAM action roots：255 个 NavSim source × 4 个速度档 = 1020 个输入条件，未来图像仍明确标记为 pending：

`/mnt/slurmfs-4090node3/user_data/zchen897/benchmark_v3_runs/pure_speed_multilevel_action_roots_20260916_v2`

本地登记：[pure_speed_multilevel_action_roots_20260916.json](../reports/pure_speed_multilevel_action_roots_20260916.json)。

该新清单已通过本地 lineage 审计：255/255 组四档完整、历史/随机种子/source sample 一致、来源为 `drivewam_generated_pure_speed_multilevel_pending`、模型为 `drivewam_navsim`。

## 2026-09-16 DriveWAM 原生生成 pilot

从上述 roots 取 8 个 source，每个 source 原生生成 `stop / slow / normal / fast` 四档，共 32 条未来视频。四档均满足：

- `action_injection_verified=true`；
- `future_images_source=drivewam_generated`；
- 同一 source 使用同一 nuisance seed。

冻结视觉层评分结果：

| 指标 | 结果 |
|---|---:|
| 四档完整组覆盖率 | 8/8 = 100% |
| 相邻排序命中（3×8） | 12/24 = 50.0% |
| 完整 stop→slow→normal→fast 单调组 | 0/8 = 0% |

因此当前根因不是视觉层覆盖率，而是 DriveWAM 的纵向 action condition 没有稳定改变生成视频中的纵向运动。四档图像文件虽然不同，但纵向估计不随 action 单调变化。基于此，暂不启动 255 组全量生成；应先修复动作条件进入生成网络的路径，再重跑相同 pilot。

从 adapter 实现看，外部轨迹被写入 transformer cache 的 action chunk，并且 `action_injection_verified` 只能证明张量写入成功；它不能证明训练时的视频 token 对该 action channel 学到了因果响应。这正是当前“输入钩子成功、视觉运动不响应”的边界。

pilot 结果：[rcs_progress_drivewam_multilevel_pilot8_20260916.json](../reports/rcs_progress_drivewam_multilevel_pilot8_20260916.json)。

## 2026-09-16 动作—视频耦合审计

进一步对照 DriveWAM 的训练和推理代码后，当前结果不能解释为“视觉层把所有速度都读成静止”，而应解释为“外部速度没有进入一个经过训练的视频条件路径”：

1. 官方 `predict` 先缓存首帧和全零 action condition，再单独生成未来视频；未来 action chunk 在视频生成完成后才采样。
2. 训练的 joint forward 确实同时看到 future noisy action 和 future noisy video，但官方 rollout 没有复现这个 joint token layout。
3. `condition_chunk=0` 把外部动作写入训练中恒为零的条件槽，是 OOD 干预；`condition_chunk=1` 虽写入 future action cache，却仍调用 video-only rollout，也不等价于训练时的 joint input。

因此 `action_injection_verified=true` 不能升级为 `action_to_video_verified=true`。审计记录见：[drivewam_action_coupling_audit_20260916.json](../reports/drivewam_action_coupling_audit_20260916.json)。在实现 joint action-conditioned sampler 并通过同一 8 组 pilot 之前，不再生成全部 1020 个条件，也不冻结 DriveWAM 的 `RCS-progress`。

## 2026-09-16 joint sampler 探针

已实现一个独立的 joint forward probe：把外部未来 action token 与未来视频 latent 一起送入训练阶段的 `forward_train` 结构，并在同一 8 个 source、同一随机种子上重跑四档速度。

结果：

- 四档覆盖率：`8/8 = 100%`；
- 相邻排序准确率：`13/24 = 54.2%`；
- 完整四档单调组：`0/8 = 0%`。

这比原始 cache 注入略高，但仍接近随机，不能视为 RCS-progress 通过。该探针还必须绕过当前环境无法编译的 FlexAttention mask，并且推理时不存在训练所需的未来干净视觉条件，因此它只能说明“简单复现 joint token 排布也没有得到稳定响应”，不能作为正式分数。

结果：[rcs_progress_joint_sampler_pilot8_20260916.json](../reports/rcs_progress_joint_sampler_pilot8_20260916.json)。

## 2026-09-16 Epona 原生四档速度 pilot

为区分“WAM 没有响应动作”和“视觉层测不出来”，在同一 NAVSIM 原生日志窗口上切换到 Epona。每个 source 固定 10 帧历史和随机种子，只把原生未来纵向 pose 分别缩放为 `stop / slow(0.5) / normal(1.0) / fast(1.5)`；每个未来帧都通过 Epona 的 `generate_gt_pose_gt_yaw` 生成，因此这里的动作注入路径是可验证的。

本轮使用 8 个 source、32 条视频，冻结视觉层为 Reloc3r-512 + Metric3Dv2-v2-S + SIFT/PnP，结果：

| 指标 | 结果 |
|---|---:|
| 四档完整组覆盖率 | 8/8 = 100% |
| 有效动作比较组 | 6 |
| `slow < normal < fast` 相邻排序 | 12/18 = 66.7% |
| 两档 `fast > slow` 排序 | 3/7 = 42.9% |
| 完整四档单调组 | 0/6 = 0% |

结果：[rcs_progress_epona_multilevel_pilot8_20260916.json](../reports/rcs_progress_epona_multilevel_pilot8_20260916.json)。

解释：Epona 的生成调用确实接收了不同速度控制，且图像中的纵向量在 `slow → normal` 多数情况下增加；但 `normal → fast` 在后半段经常回落，导致完整单调响应失败。这不是“所有分支都被读成静止”，而是当前 Epona 在递归生成和较大速度干预下的视觉响应不稳定。因而 Epona 比 DriveWAM 更适合作为后续 RCS 候选，但本 pilot 仍不足以冻结正式 RCS-progress，也不应直接扩大到全量数据。

下一步应先做两个小控制：

1. 在同一 source 上加入 `zero / 0.25 / 0.5 / 0.75 / 1.0` 的较小干预，检查失败是否主要发生在 fast 饱和区；
2. 记录每个生成步的图像质量和几何内点率，区分“动作响应非单调”和“递归视频后段退化”。

## 2026-09-16 Epona 细粒度速度控制

在相同 8 个 source 上进一步使用五档控制 `stop / quarter(0.25) / half(0.5) / threequarter(0.75) / normal(1.0)`，共 40 条视频。所有 40 条均通过视觉探针，6 个动作排序有效组均有足够几何覆盖，但结果仍未形成稳定单调响应：

| 指标 | 结果 |
|---|---:|
| 完整组覆盖率 | 6/6 = 100% |
| 五档相邻排序 | 14/24 = 58.3% |
| 五档完整单调组 | 0/6 |

这说明问题不是只由 `fast=1.5` 过大造成。细粒度控制下，`half → threequarter` 通常有响应，但 `stop → quarter` 和 `threequarter → normal` 经常反向或回落。当前 Epona 可以作为“动作进入视频生成器”的验证模型，却还不能作为稳定的纵向 RCS 生成器。

几何质量没有随速度档位崩溃：五档未来区间的平均 SIFT 内点率约为 `0.670 / 0.672 / 0.648 / 0.626 / 0.643`（从 stop 到 normal）。因此这次失败主要是生成视频的动作响应非单调，而不是视觉层在高速分支覆盖率下降。

结果：[rcs_progress_epona_multiscale_pilot8_20260916.json](../reports/rcs_progress_epona_multiscale_pilot8_20260916.json)。

## 2026-09-16 逐区间响应复核

为检查“把视觉层压缩为一个总 progress”是否制造了假失败，直接使用上述 40 条视频的原始未来区间输出，不再先求和。对 6 个动作排序有效 source、3 个未来区间、4 个相邻速度对统计：

| 指标 | 结果 |
|---|---:|
| 局部相邻排序 | 39/72 = 54.2% |
| `stop → normal` 端点方向 | 16/18 = 88.9% |
| 每区间动作—视觉 Spearman 中位数 | 0.60 |
| 所有速度档同时单调 | 0/6 |

结果：[rcs_progress_epona_multiscale_pilot8_vector_20260916.json](../reports/rcs_progress_epona_multiscale_pilot8_vector_20260916.json)。

这排除了“总量求和是唯一问题”：逐区间细粒度排序仍接近随机；但大幅 `stop → normal` 干预多数能被识别。因此 Epona 当前更像是一个二元/粗粒度运动响应器，而不是连续速度响应器。AS 可以继续使用纵向视觉量，RCS 则应分别报告粗粒度端点响应和细粒度 dose-response，不能把两者混成一个分数。

## 2026-09-16 相对纵向进度 RCS 重算

重新明确 RCS 的目标不是恢复绝对米制距离或显式速度，而是比较同一历史下不同动作分支的相对纵向进度排序。对同一组内的视觉 progress 做归一化/排序，不使用绝对距离标签；动作端只要求分支进度顺序有效。

在上述 8 个 source 中，6 组动作间隔达到门槛并进入评分：

| 指标 | 结果 |
|---|---:|
| 评分覆盖率 | 6/6 = 100% |
| 相邻档位排序 | 14/24 = 58.3% |
| 全部两两分支排序 | 49/60 = 81.7% |
| 最低到最高端点排序 | 6/6 = 100% |
| 组内 Spearman 中位数 | 0.80 |
| 完整五档单调组 | 0/6 |

结果：[rcs_relative_progress_epona_multiscale_pilot8_20260916.json](../reports/rcs_relative_progress_epona_multiscale_pilot8_20260916.json)。

这个结果比“连续速度是否严格单调”更符合任务本质：Epona 能较稳定地表达粗粒度相对进度（最低档到最高档、全部两两排序），但不能稳定表达每一个相邻细档。因此后续 RCS 应将 `relative-progress endpoint / pairwise` 与 `fine-dose-response` 分开报告，而不是用一个绝对距离或速度分数概括。

## 2026-09-16 RAFT 光流对照

同一批 40 条 Epona 视频的 RAFT-Large 结果没有超过几何视觉层。最佳光流描述子的有效组覆盖率为 `4/6 = 66.7%`，全部两两排序为 `72.5%`；Metric3D + Reloc3r + static matching 的对应结果为覆盖率 `6/6`、两两排序 `81.7%`。因此光流保留为 `motion_present`、identity control 和异常检测辅助，不作为 progress 主估计。

结果：[rcs_relative_flow_epona_multiscale_pilot8_20260916.json](../reports/rcs_relative_flow_epona_multiscale_pilot8_20260916.json)。

## 2026-09-16 RCS v2 冻结与独立日志确认

根据任务允许较大距离误差的边界，冻结新的粗粒度 `RCS-relative-progress`：

- 同一历史、同一随机种子、真实重生成；
- 每组自动选择动作纵向进度最低和最高的两个分支；
- 动作端点差 `>= 0.5 m` 才进入评分；
- 视觉侧使用冻结视觉层的未来区间相对 progress；
- 只判断 `visual(high) > visual(low)`，不要求绝对米制距离；
- 相邻档位、完整多档单调性和细粒度 dose-response 均为诊断项；
- 正式报告至少需要 30 个独立日志/场景的动作合格组；同一 source 的重叠窗口不能重复计数。覆盖率需 `>= 0.9`、点估计 `>= 0.75`、Wilson 95% CI 下界 `> 0.5`。

最初的 40 个窗口全部来自同一日志且互相重叠。虽然得到 `27/35 = 0.771`，但独立 source 数仅为 1，因此只保留为相关窗口 pilot，不能作为正式通过证据。评分器已将其自动降级为 `pilot`。

随后对 64 个独立 NAVSIM mini 日志各取一个窗口，原生生成 `stop / normal-progress` 两个端点，共 128 条视频：

| 指标 | 结果 |
|---|---:|
| 声明组 | 64 |
| 动作差合格的独立 source | 36 |
| 视觉评分组 | 35 |
| 覆盖率 | 35/36 = 97.2% |
| 端点方向命中 | 26/35 |
| `RCS-relative-progress` | **0.743** |
| Wilson 95% CI | **[0.579, 0.858]** |
| 晋级结果 | **未通过** |

该结果显著高于随机，且覆盖率足够；但点估计比冻结门槛 `0.75` 低 `0.007`，因此 progress 通道不能宣布晋级。结论限定为：Epona 在大幅纵向干预下多数会生成方向正确的相对 progress 响应，但稳定性尚未达到冻结标准。它不证明精确米制距离，也不证明细粒度速度控制。

失败审计中只有 1/36 组是视觉层无结果；动作差 `0.5–1.5 m`、`1.5–3 m`、`>=3 m` 的命中率依次为 `57.1%`、`72.7%`、`82.4%`。因此主要问题是 WAM 对中小干预的视觉响应不稳定，不是视觉层覆盖率不足。该事后分层只用于定位根因，不改变正式评分口径。

与既有正式 `RCS-yaw = 0.875` 等权宏平均后的数值为 `RCS v2 = 0.809`（80.9/100），bootstrap 95% CI 为 `[0.718, 0.893]`。由于 progress 未晋级，该值状态为 `not_promoted`，只作完整性诊断，不是正式 RCS 成绩。

冻结配置：[response_consistency_v2.json](../configs/response_consistency_v2.json)。独立日志结果：[rcs_v2_relative_progress_epona_multilog64_20260916.json](../reports/rcs_v2_relative_progress_epona_multilog64_20260916.json)。相关窗口 pilot：[rcs_v2_relative_progress_epona_n40_20260916.json](../reports/rcs_v2_relative_progress_epona_n40_20260916.json)。

原始视觉输出：

- `reports/rcs_progress_epona_full_visual.json`
- `reports/rcs_progress_drivewam_full_visual.json`

汇总输出：

- `reports/rcs_progress_epona_pilot_20260915.json`
- `reports/rcs_progress_drivewam_pilot_20260915.json`
