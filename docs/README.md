# Documentation index

## Start here

| Doc | Language | Content |
|---|---|---|
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
| [`../tools/score_fcs_rollout.py`](../tools/score_fcs_rollout.py) | CLI | Fail-closed independent rollout scorer for FCS |
| [`../tools/score_structural_grounding.py`](../tools/score_structural_grounding.py) | CLI | Recompute GS from generated and user-supplied reference flow-structure JSONL |
| [`../tools/export_gs_reference_release.py`](../tools/export_gs_reference_release.py) | CLI | HMAC-pseudonymized descriptor-only reference export for public GS replay |
| [`../tools/verify_gs_reference_release.py`](../tools/verify_gs_reference_release.py) | CLI | Privacy and integrity checks for a descriptor-only GS reference bundle |
| [`../tools/validate_metric_comparability.py`](../tools/validate_metric_comparability.py) | CLI | Fail-closed MAS/RCS adapter comparability contract check |
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

The formal two-model gate is intentionally not an architecture-universal
claim: Epona and DriveWAM establish protocol validation only. A third distinct
architecture family, with the same source-disjoint controls, is required before
that broader claim can be enabled. The release validator enforces this scope.

## Benchmark

| Doc | Content |
|---|---|
| [`BENCHMARK_PROTOCOL_AUDIT_ZH.md`](BENCHMARK_PROTOCOL_AUDIT_ZH.md) | 1000-row selection / leakage audit |
| [`DRIVEWAM_BENCHMARK_RESULTS_ZH.md`](DRIVEWAM_BENCHMARK_RESULTS_ZH.md) | Reference DriveWAM CFAC/CCFC/FAU/FCS numbers |
| [`IAC_FROZEN_PIPELINE.mmd`](IAC_FROZEN_PIPELINE.mmd) | Frozen pipeline diagram source |

## English summary of the reference pilot

DriveWAM on `benchmark` (1,000 NAVSIM windows):

- **CFAC** (shape): 0.7638 on 823/1000
- **CCFC** (arc-relative): 0.2178 on 453/1000 pairs
- **FAU**: 0.5169 (`FAU_F` 0.5449, `FAU_A` 0.4904)
- **FCS**: 0.5143 (503/978 executable)

The frozen Step 1 primary is the candidate-blind S1.3 yaw structural response.
Metric SE(2) reconstruction, lateral/curvature/distance/speed fields and
progress descriptors remain diagnostic until their independent validation
gates pass. Unsupported capabilities stay `unavailable`, never zero-filled.
