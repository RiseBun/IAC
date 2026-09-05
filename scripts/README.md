# Script map

## Public evaluation entrypoints

These are the scripts needed to validate a submission and run the frozen
protocol once a private evaluation server has joined images and ground truth:

| Script | Role |
|---|---|
| `validate_wam_submission.py` | Fail-closed submission / leakage audit |
| `audit_wam_level1_outputs.py` | Accept 4- or 8-frame WAM futures before join |
| `build_wam_level1_continuous_manifest.py` | Join public identity with private frames |
| `evaluate_continuous_decoder.py` | Step 1 candidate-blind image probe |
| `evaluate_continuous_motion_alignment.py` | Align probed motion with native action / GT |
| `evaluate_cfac_fau.py` | Compute CFAC and FAU after the private GT join |
| `evaluate_counterfactual_alignment.py` | Compute paired-intervention CCFC |
| `score_iac_submission.py` | Capability-stratified scorecard |
| `audit_benchmark_manifest.py` | Public (`--public`) or paired private-manifest audit |

Future-frame policy: history is usually 4 frames; generated futures may be
**4 or 8** points covering about 4.0 s. Pin with `--expected-future-count` when
needed.

## Reproduction and dataset tools

`reproduction/drivewam/` contains the adapter used for the reference DriveWAM
run. `reproduction/navsim/` contains the independent PDM rollout used for FCS.
They are not required to score a WAM that already emits the submission JSONL
contract. `tools/dataset/` rebuilds the public manifest from licensed NAVSIM
data; raw frames are never shipped.

A script that needs a private mount must fail with a missing-input error rather
than silently substituting a logged action or oracle state.
