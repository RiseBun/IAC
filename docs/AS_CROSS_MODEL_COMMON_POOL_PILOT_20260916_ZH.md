# AS 跨模型共同池 pilot（2026-09-16）

## 问题

旧结果不能直接比较：DriveWAM、Epona 和 WorldDrive 没有使用同一 source pool；
Epona 的现有 AS 是给定动作条件生成，不是自身 action head；WorldDrive 的旧样本还
经过动作幅度预筛选。因而旧表只能说明视觉接口可运行，不能形成模型排名。

## 修复

从 DriveWAM 已覆盖的 Benchmark-v3 source 中，仅依据 NAVSIM stratum 和固定哈希
选择 10 个 source：lateral-turn 4、straight-cruise 2、acceleration 2、braking 1、
stop 1。选择过程不读取任何模型动作、生成图像或分数。

两个模型均使用同一历史 source、冻结的 `4 history + 4 future + 4 action` 时间轴、
自身原生 action head 和同一视觉层。DriveWAM 每个 source 有两个命令分支，
WorldDrive 每个 source 有一个原生分支；正式比较单位因此固定为 source，而不是行。

## 结果

| 模型 | source / 行 | AS-conditional | progress coverage | yaw | AS-overall |
|---|---:|---:|---:|---:|---:|
| DriveWAM | 10 / 20 | 75.7 | 42.5% | 92.9% | 49.4 |
| WorldDrive | 10 / 10 | 75.6 | 52.5% | 100.0% | 54.8 |

两模型的条件一致性几乎相同。WorldDrive 的 coverage 较高，但 `n=10` 不足以支持
排名；正式跨模型门槛固定为至少 30 个共同 source，并要求 source-cluster 置信区间。

作为偏差对照，WorldDrive 的 10 个动作幅度预筛选 source（20 分支）在修正时间轴后
得到 `AS-conditional=100`、`AS-overall=79.8`。这不是模型总体质量，而是强动作信号
条件下的可读性，验证了预筛选会显著抬高结果。

## 决策

- 当前共同池结果状态为 `pilot`，禁止用于排行榜；
- Epona 的给定动作条件结果单列，不与原生联合输出模型排名；
- 正式比较必须满足相同 source pool、相同视觉协议、原生动作 provenance、lineage
  通过、模型输出盲选和至少 30 个共同 source；
- 扩大共同池时沿用当前固定选择器，不再改变视觉层或 AS 阈值。

机器可读产物：

- `reports/as_common10_source_pool_20260916.jsonl`
- `reports/as_common10_drivewam_20260916.json`
- `reports/as_common10_worlddrive_20260916.json`
- `reports/as_cross_model_common10_audit_20260916.json`
