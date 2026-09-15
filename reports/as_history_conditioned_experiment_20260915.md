# IAC History-conditioned AS 实验记录

日期：2026-09-15
协议：`iac-history-conditioned-as-v1.1`

## 结论

benchmark v3 的字段结构适合作为 AS 输入，但必须先通过“历史—未来同源”审计。当前结果是：

- NAVSIM logged-realized 参考集通过，覆盖率感知 `AS=89.1/100`。
- benchmark v3 真实未来帧配 WAM action 通过输入审计，`AS=76.5/100`。
- 修复 lineage 后的原生 DriveWAM benchmark v3 通过输入审计，正式 `AS=46.8/100`。
- 保留的 DriveWAM controlled-pair 通过输入审计，`AS=73.8/100`。
- WorldDrive 10 条探索集通过输入审计，`AS=68.9/100`，但样本量太小，不能作正式结论。

## 输入与独立性

统一输入为：

```text
history frames:    t = -1.5, -1.0, -0.5, 0 s
future frames:     t =  1,    2,    3,   4 s
trajectory states: t =  1,    2,    3,   4 s，相对 t=0 自车坐标
```

因此有四个真实可观测区间：

```text
history[-1]→future[0]  对应 0→1 s
future[0] →future[1]   对应 1→2 s
future[1] →future[2]   对应 2→3 s
future[2] →future[3]   对应 3→4 s
```

视觉分支只接收图像，使用冻结的 `Reloc3r-512 + Metric3Dv2-v2-S + known-R PnP`；轨迹只在视觉运动提取完成后加入评分，不存在轨迹向视觉后端泄漏。

## AS 定义

- `AS-yaw`：四段视觉 yaw 累加后，与轨迹末端 yaw 比较方向；仅在 `|action yaw| ≥ 0.0128 rad` 的明确转弯样本上计分。
- `AS-progress`：四段纵向位移中，几何质量合格区间的轨迹对齐率。容差沿用冻结输出层：绝对/相对误差阈值为 `max(3 m, 50%)`，或距离档位至多相邻一档。
- `AS-progress-effective`：通过纵向区间数除以每条样本固定四个应测区间；不可测区间不获得分数。
- `AS-yaw-effective`：方向正确数除以全部明确转弯样本；视觉 yaw 缺失不获得分数。
- `AS-overall`：`sqrt(AS-progress-effective × AS-yaw-effective)`，是正式单一 AS。
- `AS-composite`：旧的可观测样本诊断均值；缺失不补零，因此不得再作为主分数。
- `coverage`：与分数并列报告。`uncertain` 只降低覆盖率，不会被当成“不一致”，质量合格但超出容差的结果必须计为失败。

## benchmark v3 输入审计

仅检查“4+4+4”的数组长度不够。AS 首先检查 `t=0→1` 边界：

- 历史—未来绝对 yaw 超过 `30°` 的比例不得超过 `10%`；
- 边界几何可测覆盖率至少 `20%`；
- 阈值由真实 NAVSIM 和已知同源控制集冻结，真实集边界 yaw 中位数约 `0.56°`，超过 `30°` 的比例为 0。

| 输入 | 条目 | 边界 yaw 中位数 | `>30°` | 边界几何覆盖 | 审计 |
|---|---:|---:|---:|---:|---|
| NAVSIM logged-realized | 95 | 0.558° | 0% | 90.5% | 通过 |
| benchmark v3 real counterpart | 255 | 0.557° | 0% | 90.2% | 通过 |
| DriveWAM controlled-pair | 348 | 0.494° | 0% | 79.6% | 通过 |
| WorldDrive v3 adapter | 10 | 10.639° | 0% | 60.0% | 通过，小样本 |
| benchmark v3 native DriveWAM（旧清单） | 1490 | **99.050°** | **90.4%** | **0.13%** | **失败** |
| benchmark v3 native DriveWAM（lineage 修复） | 1490 | 0.586° | 0% | 66.4% | **通过** |

### 原生 DriveWAM 清单错配证据

旧清单失败不是 WAM 真实产生 99° 转向。人工抽查确认，同一清单记录中的历史与
生成未来来自不同场景。根因是四个生成 shard 使用轮转分配，旧构建器却按连续
offset 恢复全局索引。

修复方式是直接读取 WAM 实际消费的输入 pkl 中 `metadata.source_key`，再据此连接历史、生成未来和 action，完全取消 ordinal join。修复后边界 yaw 中位数从 99.050° 降至 0.586°，大于 30° 的比例从 90.4% 降至 0%，说明错配已解决。旧清单保留为回归测试，禁止复用。

## 最终可用结果

`progress coverage` 是至少有一个质量合格纵向区间的样本比例；`interval coverage` 是全部四段中质量合格的比例。

| 条件 | N | interval coverage | 条件 progress | yaw | effective progress | 正式 AS /100 |
|---|---:|---:|---:|---:|---:|---:|
| NAVSIM logged-realized 参考/上限 | 95 | 88.2% | 0.906 | 1.000 | 0.795 | **89.1** |
| benchmark v3 真实未来 + WAM action | 255 | 89.4% | 0.856 | 0.775 | 0.756 | **76.5** |
| 原生 DriveWAM benchmark v3（修复后） | 1490 | 48.4% | 0.422 | 0.935 | 0.235 | **46.8** |
| DriveWAM controlled-pair | 348 | 68.2% | 0.834 | 0.970 | 0.562 | **73.8** |
| WorldDrive exact-time v3 adapter | 10 | 52.5% | 0.927 | 1.000 | 0.475 | **68.9** |

解释边界：

- `NAVSIM logged-realized` 才是视觉读取能力的真实参考；95 条是已有完整 probe 的子集。
- `real counterpart` 的图像是真实未来，但轨迹字段是 `wam_action_head`，所以 0.828 表示“WAM 动作与真实发生未来”的观察性一致性，不是纯视觉准确率。
- `controlled-pair` 表示保留 WAM 输出与其 action 的自洽性，不证明轨迹因果控制了生成。
- WorldDrive 只有 10 条，当前数字只证明流程可运行，不能比较模型优劣。
- `AS` 与 `GS` 分离：AS 测轨迹—图像运动一致性，GS 测生成未来—真实未来的现实接近程度。

## 历史无效清单的审计数字（禁止作为模型分数）

旧版原生 DriveWAM 1490 条错配清单得到：`AS-progress=0.523`、`AS-yaw=0.498`、`AS-composite=0.523`、纵向区间覆盖率 `31.8%`。这些数字只描述被错配输入污染后的结果，正式表格不采用；正式结果改用 lineage-fixed 清单。

## 决策与最短下一步

当前 AS v1.1 的输入、视觉后端、容差、coverage、弃权语义、覆盖率感知聚合和 lineage gate 均已冻结。原生 DriveWAM manifest 已按不可变 `source_key` 重建并通过审计。

今后所有 WAM 必须按 ID join 历史、未来和 action，禁止按目录序号或过滤后的行号对齐；先运行 `t=0→1` 同源审计，通过后再运行完整四段 AS。当前仍待完成的是 Epona 标准化 AS，以及扩充 WorldDrive 样本量。

## 现有 WAM 覆盖状态

| 模型 | 轨迹来源 | 当前状态 | 可比性 |
|---|---|---|---|
| DriveWAM | 原生 action head | lineage-fixed 1490 条已完成 | 正式 joint-output AS |
| Epona | 输入的匹配物理 action；没有同口径原生 joint action head | 已有 348 条旧产物，待按 AS v1 重跑 | 只能标为 conditional AS |
| DriveVA | 原生 action head | 已有 100 条、8 future/0.5 Hz，待精确抽取 1/2/3/4 s 后重跑 | 可形成 joint-output AS |
| WorldDrive | 原生 action head | 10 条 pilot 已完成 | 样本量不足，待扩大 |

只有使用相同 NAVSIM source 池、相同时间点、相同相机适配并通过 lineage gate 后，四个模型才可横向比较；当前结果不能直接作为模型排行榜。

## 产物

- 公开协议与实现：`configs/as_history_conditioned_v1.json`、
  `src/iac_new/history_conditioned_as.py`、`tools/run_history_conditioned_as.py`；
- 公开紧凑结果：`reports/as_release_20260915.json`、
  `reports/as_history_conditioned_summary_20260915.json`；
- 逐样本视觉 probe、私有图像路径和 benchmark manifest 不进入公开仓库，保存在
  本地归档和评测服务器。

验证：视觉层、纵向几何、输入契约与 AS 聚合共 22 项相关单元测试全部通过。
