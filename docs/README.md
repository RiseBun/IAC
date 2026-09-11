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
| [`STEP1_S1_3_AUDIT_AND_METRIC_ROADMAP_ZH.md`](STEP1_S1_3_AUDIT_AND_METRIC_ROADMAP_ZH.md) | ZH | S1.3 audit, pure-speed preparation and CFAC-S roadmap |
| [`CCFC_S_VALIDATION_PLAN_ZH.md`](CCFC_S_VALIDATION_PLAN_ZH.md) | ZH | CCFC-S controls and cross-model validation matrix |
| [`../configs/cfac_structure_calibration_v1.json`](../configs/cfac_structure_calibration_v1.json) | JSON | CFAC-S calibration contract and claim boundary |

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
