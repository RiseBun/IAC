# GS 当前格式审计与重算（2026-09-16）

## 目的

将旧版 generated/logged descriptor 对照转换为当前 GS evidence contract，重新执行冻结的 structural grounding 聚合器，并检查 source identity、reference identity、缺失值和 source-cluster 不确定性。

本轮不重新生成 WAM 视频，也不改变 GS 定义、descriptor、真实域 calibration scale 或阈值。因此这是格式审计与数值重算，不是新的模型实验。

## 当前冻结口径

- 比较同一 source、同一 branch、同一 interval 的 generated flow structure 与 logged future flow structure；
- descriptor：`median_flow_magnitude_px`、`horizontal_flow_center`、`vertical_flow_center`、`divergence`、`curl`；
- magnitude 使用 `log1p`；每个 descriptor 的相似度为 `exp(-abs(delta)/scale)`；
- interval 内对 descriptor 取 median，branch 内对有效 interval 取 median；
- 至少 3 个有效 interval、每个 interval 至少 3 个 descriptor；
- 缺失值为 `unavailable`，不补零；
- 不使用 native action，因此 GS 不声称 future-to-action 因果。

## Evidence contract 审计

现有 current-format evidence packet 验证结果：

| 模型 | packet 行数 | valid scored | unavailable | invalid |
|---|---:|---:|---:|---:|
| Epona | 118 | 112 | 6 | 0 |
| DriveWAM | 172 | 162 | 10 | 0 |

两个 packet 均通过 `GS` evidence contract。旧 raw report 中完全缺失的 branch 已依据 packet 的 `unavailable` 状态补成显式占位；因此它们进入 coverage 分母，但不会进入 score 分子。

## 冻结聚合器重算

### Branch-level

| 模型 | branch 数 | scored | coverage | GS median |
|---|---:|---:|---:|---:|
| Epona | 118 | 112 | 94.92% | **0.552898** |
| DriveWAM | 172 | 162 | 94.19% | **0.190754** |

### Source-cluster bootstrap

bootstrap 单位是 source，左右 branch 保持在同一个 source cluster 内：

| 模型 | source 数 | GS median | bootstrap 95% CI |
|---|---:|---:|---:|
| Epona | 56 | **0.549630** | `[0.486577, 0.635340]` |
| DriveWAM | 81 | **0.191361** | `[0.158404, 0.215432]` |

共同 source 的 Epona−DriveWAM 中位差为 `0.313573`，source-cluster bootstrap 95% CI 为 `[0.251202, 0.339997]`。

## 审计结论

1. GS 数值重算通过：branch score、coverage、source-cluster bootstrap 均与冻结结果一致；
2. 两个模型的 GS evidence packet 均通过当前 metric contract；
3. missing/unavailable 处理正确，没有把缺失样本当作零分；
4. 现有 raw JSON 本身没有显式携带 `candidate_blind` lineage 字段，这是 provenance warning，不改变已经由 packet 明确记录的 candidate-blind measurement 声明；后续正式生成的新报告应把该字段直接写入样本级数据。

当前结论仍限定为：GS 衡量生成未来与同源外部真实未来的视觉结构 grounding，不证明动作轨迹导致了生成视频，也不证明 future-to-action mediation。

## 产物

- 审计与重算汇总：[gs_current_audit_rescore_20260916.json](../reports/gs_current_audit_rescore_20260916.json)
- 当前格式 Epona generated：[gs_current_epona_generated.jsonl](../reports/gs_current_epona_generated.jsonl)
- 当前格式 Epona reference：[gs_current_epona_reference.jsonl](../reports/gs_current_epona_reference.jsonl)
- 当前格式 DriveWAM generated：[gs_current_drivewam_generated.jsonl](../reports/gs_current_drivewam_generated.jsonl)
- 当前格式 DriveWAM reference：[gs_current_drivewam_reference.jsonl](../reports/gs_current_drivewam_reference.jsonl)
- 冻结 descriptor scales：[gs_frozen_descriptor_scales_20260916.json](../reports/gs_frozen_descriptor_scales_20260916.json)
- 可复现脚本：[audit_and_rescore_gs_current.py](../tools/audit_and_rescore_gs_current.py)
