# Documentation index

## Start here

| Doc | Language | Content |
|---|---|---|
| [`VISUAL_LAYER_AND_AS_ZH.md`](VISUAL_LAYER_AND_AS_ZH.md) | ZH | **Current canonical visual layer and coverage-aware AS** |
| [`AS_CROSS_MODEL_COMMON_POOL_PILOT_20260916_ZH.md`](AS_CROSS_MODEL_COMMON_POOL_PILOT_20260916_ZH.md) | ZH | Cross-model AS common-pool pilot, comparability gates and current evidence boundary |
| [`../README.md`](../README.md) | EN | Protocol overview, install, submit |
| [`../README_zh.md`](../README_zh.md) | ZH | Same overview in Chinese |
| [`WAM_SCOPE_AND_UNIFIED_PROTOCOL_ZH.md`](WAM_SCOPE_AND_UNIFIED_PROTOCOL_ZH.md) | ZH | Admission + three-step contract |
| [`WAM_SUBMISSION_ZH.md`](WAM_SUBMISSION_ZH.md) | ZH | Author JSONL fields and scoreboard cells |
| [`RELEASE_MANIFEST_ZH.md`](RELEASE_MANIFEST_ZH.md) | ZH | What is public vs private |
| [`WAM_JOINT_EVALUATION_FRAMEWORK_ZH.md`](WAM_JOINT_EVALUATION_FRAMEWORK_ZH.md) | ZH | Current joint framework, claims and validation gates |
| [`STEP1_S1_3_AUDIT_AND_METRIC_ROADMAP_ZH.md`](STEP1_S1_3_AUDIT_AND_METRIC_ROADMAP_ZH.md) | ZH | S1.3 audit, pure-speed preparation and MAS-yaw boundary |
| [`CCFC_S_VALIDATION_PLAN_ZH.md`](CCFC_S_VALIDATION_PLAN_ZH.md) | ZH | RCS/CCFC-S controls and cross-model validation matrix |
| [`../configs/cfac_structure_calibration_v1.json`](../configs/cfac_structure_calibration_v1.json) | JSON | Legacy CFAC-S diagnostic calibration contract |
| [`../reports/pure_speed_confirmation_20260911.json`](../reports/pure_speed_confirmation_20260911.json) | JSON | Cross-model pure-speed confirmation result and promotion decision |
| [`../reports/progress_structure_pure_speed_audit_20260912.json`](../reports/progress_structure_pure_speed_audit_20260912.json) | JSON | Non-yaw progress descriptor audit and promotion boundary |
| [`../tools/score_independent_execution.py`](../tools/score_independent_execution.py) | CLI | Fail-closed independent rollout scorer for external task validation |
| [`FUTURE_TO_ACTION_MEDIATION_AUDIT_20260916_ZH.md`](FUTURE_TO_ACTION_MEDIATION_AUDIT_20260916_ZH.md) | ZH | WorldDrive future-to-action mediation 四条件审计与下一轮执行规格 |
| [`../reports/future_to_action_runner_inventory_20260916.json`](../reports/future_to_action_runner_inventory_20260916.json) | JSON | 服务器 WorldDrive runner/source inventory；当前仅有结果 artifact |
| [`FUTURE_TO_ACTION_DRIVEVA_SMOKE_20260916_ZH.md`](FUTURE_TO_ACTION_DRIVEVA_SMOKE_20260916_ZH.md) | ZH | DriveVA 四条件 runner smoke 与 pathway-control 失败审计 |
| [`../tools/audit_future_to_action_readiness.py`](../tools/audit_future_to_action_readiness.py) | CLI | 对 raw WAM intervention 产物做 mediation readiness 审计，不把缺失对照填成 0 |
| [`../tools/score_future_to_action_mediation.py`](../tools/score_future_to_action_mediation.py) | CLI | 四条件 future-to-action pathway mediation scorer |
| [`../tools/score_conditional_foresight.py`](../tools/score_conditional_foresight.py) | CLI | Paired future-perturbation contrast on task success (foresight-conditioned, not marginal) |
| [`../tools/analyse_joint_source_table.py`](../tools/analyse_joint_source_table.py) | CLI | Source-level join of consistency scores with execution outcomes |
| [`JOINT_SOURCE_ANALYSIS_20260916_ZH.md`](JOINT_SOURCE_ANALYSIS_20260916_ZH.md) | ZH | Current source-level AS/RCS versus independent execution association audit |
| [`../tools/audit_action_alignment.py`](../tools/audit_action_alignment.py) | CLI | Fail-closed action fingerprint audit before joint consistency/execution analysis |
| [`../tools/score_structural_grounding.py`](../tools/score_structural_grounding.py) | CLI | Recompute GS from generated and user-supplied reference flow-structure JSONL |
| [`../tools/export_gs_reference_release.py`](../tools/export_gs_reference_release.py) | CLI | HMAC-pseudonymized descriptor-only reference export for public GS replay |
| [`../tools/verify_gs_reference_release.py`](../tools/verify_gs_reference_release.py) | CLI | Privacy and integrity checks for a descriptor-only GS reference bundle |
| [`../tools/validate_metric_comparability.py`](../tools/validate_metric_comparability.py) | CLI | Fail-closed MAS/RCS adapter comparability contract check |
| [`../tools/build_metric_evidence_packets.py`](../tools/build_metric_evidence_packets.py) | CLI | Build source-level MAS/RCS evidence packets with explicit provenance and fail-closed status |
| [`../reports/release_readiness_20260912.json`](../reports/release_readiness_20260912.json) | JSON | Machine-readable release claims, boundaries and pending evidence |
| [`../datasets/public_gs_fixture_generated.jsonl`](../datasets/public_gs_fixture_generated.jsonl) | JSONL | Public GS scorer replay input (synthetic, not benchmark data) |
| [`../datasets/public_gs_fixture_reference.jsonl`](../datasets/public_gs_fixture_reference.jsonl) | JSONL | Public GS scorer replay reference (synthetic, not logged GT) |
| [`../reports/gs_public_fixture_expected_20260912.json`](../reports/gs_public_fixture_expected_20260912.json) | JSON | Expected public replay result and privacy boundary |
| [`../reports/release_validation_20260912.json`](../reports/release_validation_20260912.json) | JSON | Output of the fail-closed release boundary validator |
| [`../reports/model_capability_matrix_20260912.json`](../reports/model_capability_matrix_20260912.json) | JSON | Formal vs diagnostic model evidence; prevents cross-architecture overclaim |
| [`../tools/validate_release_readiness.py`](../tools/validate_release_readiness.py) | CLI | Fail-closed check separating conditional release from complete causal release |
| [`RELEASE_BLOCKERS_AND_NEXT_EXPERIMENTS_ZH.md`](RELEASE_BLOCKERS_AND_NEXT_EXPERIMENTS_ZH.md) | ZH | Remaining evidence required for causal/cross-model claims |
| [`STEP1_IMPROVEMENT_PLAN_ZH.md`](STEP1_IMPROVEMENT_PLAN_ZH.md) | ZH | Step1.3 改进优先级、深度/光流 A/B 与验收门 |
| [`STEP1_UNIVERSAL_RESPONSE_DESIGN_ZH.md`](STEP1_UNIVERSAL_RESPONSE_DESIGN_ZH.md) | ZH | 面向多数 WAM 的反事实视觉响应层实验契约 |
| [`../tools/prove_step1_universal_response.py`](../tools/prove_step1_universal_response.py) | CLI | 合成 normal/reversed/identity/zero 最小证明（非 benchmark 证据） |
| [`../tools/audit_universal_response_controls.py`](../tools/audit_universal_response_controls.py) | CLI | 四模型真实 twin-control 桥接审计与晋级门 |
| [`../reports/step1_universal_response_real_control_audit_20260912.json`](../reports/step1_universal_response_real_control_audit_20260912.json) | JSON | 新 Step1 真实 control 审计（当前未晋级） |
| [`../tools/run_raw_universal_response_controls.py`](../tools/run_raw_universal_response_controls.py) | CLI | 从归档原始 flow 重算 normal/reversed/identity/zero |
| [`../tools/audit_raw_universal_response_controls.py`](../tools/audit_raw_universal_response_controls.py) | CLI | 原始四控制的 coverage、方向 CI 和负控制门 |
| [`../reports/step1_universal_response_raw_control_audit_20260912.json`](../reports/step1_universal_response_raw_control_audit_20260912.json) | JSON | 四架构原始流最小实验（仅 WorldDrive 通过 pilot 门） |
| [`../tools/run_step1_evidence_from_raw_flows.py`](../tools/run_step1_evidence_from_raw_flows.py) | CLI | 将原始 flow 转为 temporal/response/reliability evidence |
| [`../tools/audit_step1_evidence_channels.py`](../tools/audit_step1_evidence_channels.py) | CLI | source/twin 原子聚合多通道 evidence |
| [`../reports/step1_evidence_channel_audit_20260912.json`](../reports/step1_evidence_channel_audit_20260912.json) | JSON | 四模型多通道 evidence 首轮结果 |
| [`VISUAL_EVIDENCE_V2.md`](VISUAL_EVIDENCE_V2.md) | EN | v2 视觉证据 schema、后端接口和指标适配器 |
| [`../configs/visual_evidence_v2.json`](../configs/visual_evidence_v2.json) | JSON | v2 后端、支持交集和 promotion gates |
| [`../tools/build_visual_evidence_v2.py`](../tools/build_visual_evidence_v2.py) | CLI | 从 raw flow 或既有 flow-structure archive 构建 v2 evidence |
| [`../tools/run_visual_evidence_v2_pilot.py`](../tools/run_visual_evidence_v2_pilot.py) | CLI | 在现有归档上做 MAS/RCS/GS v2 兼容性 pilot |
| [`../tools/run_visual_evidence_v2_backend_ab.py`](../tools/run_visual_evidence_v2_backend_ab.py) | CLI | 在同一帧序列上比较 flow-only、flow+tracking、flow+tracking+depth |
| [`../tools/score_visual_evidence_v2_backend_ab.py`](../tools/score_visual_evidence_v2_backend_ab.py) | CLI | 对后端 A/B evidence 做小规模 RCS 控制审计 |
| [`../src/iac_new/motion_tokens_v3.py`](../src/iac_new/motion_tokens_v3.py) | Python | 不拟合 SE(2) 的结构运动 token 与成对差分 pilot |
| [`../tools/run_motion_token_pilot.py`](../tools/run_motion_token_pilot.py) | CLI | 在同一批归档上比较结构 token 的 normal/reversed 控制 |
| [`../reports/motion_tokens_v3_epona_pilot_20260913.json`](../reports/motion_tokens_v3_epona_pilot_20260913.json) | JSON | Epona 结构 token pilot（未晋级） |
| [`../reports/motion_tokens_v3_drivewam_pilot_20260913.json`](../reports/motion_tokens_v3_drivewam_pilot_20260913.json) | JSON | DriveWAM 结构 token pilot（未晋级） |
| [`../src/iac_new/appearance_evidence.py`](../src/iac_new/appearance_evidence.py) | Python | 冻结外观/身份/时间证据后端与控制接口 |
| [`../tools/run_appearance_evidence_pilot.py`](../tools/run_appearance_evidence_pilot.py) | CLI | DINOv2/测试 embedding 的 identity 与 temporal controls |
| [`../reports/appearance_dinov2_pilot_summary_20260913.json`](../reports/appearance_dinov2_pilot_summary_20260913.json) | JSON | DINOv2 外观 pilot：覆盖高但全局池化无反事实特异性 |
| [`../src/iac_new/multimodal_evidence_v3.py`](../src/iac_new/multimodal_evidence_v3.py) | Python | flow、track、DINOv2 patch 的固定网格证据融合（不做硬交集） |
| [`../tools/run_multimodal_evidence_v3_pilot.py`](../tools/run_multimodal_evidence_v3_pilot.py) | CLI | 多模态固定网格 evidence 真实小样本 pilot |
| [`../reports/multimodal_evidence_v3_pilot_summary_20260913.json`](../reports/multimodal_evidence_v3_pilot_summary_20260913.json) | JSON | flow + DINOv2 patch 对齐 pilot |
| [`../configs/visual_evidence_v3_promotion.json`](../configs/visual_evidence_v3_promotion.json) | JSON | MAS/RCS/GS 正式校准、控制和晋级门槛 |
| [`../src/iac_new/promotion_v3.py`](../src/iac_new/promotion_v3.py) | Python | v3 缺失即阻断、source-level CI 与控制验收逻辑 |
| [`../tools/validate_visual_evidence_v3_promotion.py`](../tools/validate_visual_evidence_v3_promotion.py) | CLI | 运行 v3 MAS/RCS/GS 正式验收，不把缺失证据填成 0 |
| [`../tools/audit_v3_controls.py`](../tools/audit_v3_controls.py) | CLI | v3 source/twin 原子 normal/reversed/zero 控制审计 |
| [`../tools/build_visual_evidence_v3_status.py`](../tools/build_visual_evidence_v3_status.py) | CLI | 汇总 MAS/RCS/GS 当前正式状态 |
| [`../tools/fit_visual_evidence_v3_adapter_pilot.py`](../tools/fit_visual_evidence_v3_adapter_pilot.py) | CLI | 透明线性校准 adapter pilot（不冻结、不作正式确认） |
| [`../tools/crossval_visual_evidence_v3_adapter.py`](../tools/crossval_visual_evidence_v3_adapter.py) | CLI | 按 source_key 的校准/确认拆分与 bootstrap |
| [`../src/iac_new/visual_evidence_v4.py`](../src/iac_new/visual_evidence_v4.py) | Python | action-aligned MAS、control-ready RCS、mediation pathway evidence 输出层 |
| [`../configs/visual_evidence_v4.json`](../configs/visual_evidence_v4.json) | JSON | v4 指标需求反推的视觉输出契约 |
| [`../tools/summarize_visual_evidence_v3_calibration.py`](../tools/summarize_visual_evidence_v3_calibration.py) | CLI | 真实视频校准 pilot 的候选映射诊断（不冻结） |
| [`../reports/visual_evidence_v3_promotion_audit_20260913.json`](../reports/visual_evidence_v3_promotion_audit_20260913.json) | JSON | 当前 v3 正式验收结果：三项均 blocked |
| [`../reports/real_calibration_v3_summary_20260913.json`](../reports/real_calibration_v3_summary_20260913.json) | JSON | 16 条 logged real 分支的校准 pilot（不作确认集准确率） |
| [`../reports/real_calibration_v3_255_summary_20260913.json`](../reports/real_calibration_v3_255_summary_20260913.json) | JSON | 255 条 logged real 分支的校准 pilot（映射仍未冻结） |
| [`../reports/multimodal_evidence_v3_epona_control_audit_20260913.json`](../reports/multimodal_evidence_v3_epona_control_audit_20260913.json) | JSON | Epona v3 source-level controls（当前 blocked） |
| [`../reports/multimodal_evidence_v3_drivewam_control_audit_20260913.json`](../reports/multimodal_evidence_v3_drivewam_control_audit_20260913.json) | JSON | DriveWAM v3 source-level controls（当前 blocked） |
| [`../reports/visual_evidence_v3_formal_status_20260913.json`](../reports/visual_evidence_v3_formal_status_20260913.json) | JSON | v3 三指标正式状态汇总（当前 blocked） |
| [`../reports/visual_evidence_v3_adapter_epona_pilot_20260913.json`](../reports/visual_evidence_v3_adapter_epona_pilot_20260913.json) | JSON | 255 real 校准映射在 Epona 生成对上的诊断结果 |
| [`../reports/visual_evidence_v3_adapter_drivewam_pilot_20260913.json`](../reports/visual_evidence_v3_adapter_drivewam_pilot_20260913.json) | JSON | 255 real 校准映射在 DriveWAM 生成对上的诊断结果 |
| [`../reports/real_calibration_v3_source_disjoint_cv_20260913.json`](../reports/real_calibration_v3_source_disjoint_cv_20260913.json) | JSON | 204/51 source-disjoint adapter confirmation |
| [`../reports/visual_evidence_v3_fullstruct_adapter_epona_20260913.json`](../reports/visual_evidence_v3_fullstruct_adapter_epona_20260913.json) | JSON | 加入 magnitude/expansion 后的 Epona adapter pilot |
| [`../reports/visual_evidence_v3_fullstruct_adapter_drivewam_20260913.json`](../reports/visual_evidence_v3_fullstruct_adapter_drivewam_20260913.json) | JSON | 加入 magnitude/expansion 后的 DriveWAM adapter pilot |
| [`../reports/visual_evidence_v2_pilot_epona_20260913.json`](../reports/visual_evidence_v2_pilot_epona_20260913.json) | JSON | Epona v2 pilot（未晋级） |
| [`../reports/visual_evidence_v2_pilot_drivewam_20260913.json`](../reports/visual_evidence_v2_pilot_drivewam_20260913.json) | JSON | DriveWAM v2 pilot（未晋级） |
| [`../reports/visual_evidence_v2_pilot_driveva_20260913.json`](../reports/visual_evidence_v2_pilot_driveva_20260913.json) | JSON | DriveVA v2 pilot（未晋级） |

The formal two-model gate is intentionally not an architecture-universal
claim: Epona and DriveWAM establish protocol validation only. A third distinct
architecture family, with the same source-disjoint controls, is required before
that broader claim can be enabled. The release validator enforces this scope.

## Benchmark

| Doc | Content |
|---|---|
| [`BENCHMARK_PROTOCOL_AUDIT_ZH.md`](BENCHMARK_PROTOCOL_AUDIT_ZH.md) | 1000-row selection / leakage audit |
| [`DRIVEWAM_BENCHMARK_RESULTS_ZH.md`](DRIVEWAM_BENCHMARK_RESULTS_ZH.md) | Historical DriveWAM consistency, grounding, and external execution numbers |
| [`IAC_FROZEN_PIPELINE.mmd`](IAC_FROZEN_PIPELINE.mmd) | Frozen pipeline diagram source |

## English summary of the reference pilot

DriveWAM on `benchmark` (1,000 NAVSIM windows):

- **CFAC** (shape): 0.7638 on 823/1000
- **CCFC** (arc-relative): 0.2178 on 453/1000 pairs
- **FAU**: 0.5169 (`FAU_F` 0.5449, `FAU_A` 0.4904)
- **External execution success (not an IAC metric)**: 0.5143 (503/978 executable)

The frozen Step 1 primary is the candidate-blind S1.3 yaw structural response.
Metric SE(2) reconstruction, lateral/curvature/distance/speed fields and
progress descriptors remain diagnostic until their independent validation
gates pass. Unsupported capabilities stay `unavailable`, never zero-filled.
