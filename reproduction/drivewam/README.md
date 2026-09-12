# DriveWAM reference adapter

This directory contains the model-specific bridge used for the DriveWAM
reference row. Run the files in this order:

1. `build_inputs.py`: create compact model inputs from the private benchmark.
2. Run DriveWAM with its native checkpoint and retain both generated future
   frames and native action-head output.
3. `merge_outputs.py` and `build_measurement_input.py`: join generated outputs
   to the frozen image probe without exposing action candidates to the probe.
4. `build_ccfc_manifest.py`: assemble two regenerated command branches for
   paired CCFC.
5. `build_fcs_staging.py`: prepare native actions for independent NAVSIM/PDM
   execution.

`prepare_reuse.py` and `build_missing_partition.py` are deterministic helpers
for resuming an interrupted reference run. External DriveWAM and LingBot-VA
code and weights are intentionally not redistributed.

## Future-to-action mediation boundary

`adapter.py::rollout_external_action` is an **action-to-video** intervention:
it writes an external action condition before the video rollout. It must not be
reported as future-to-action mediation. The native DriveWAM execution order is
the useful hook for a real mediation probe: future frames are generated first,
their transformer cache is retained, and the native action chunk is sampled
afterwards. A valid probe must therefore (a) keep history, command, nuisance
seed and model revision fixed, (b) perturb only the retained future pathway,
(c) run the same perturbation with that pathway blocked before the action
chunk, and (d) record future fingerprints, pathway state, and the frozen action
normalization fields required by
`configs/future_to_action_mediation_v1.json`.

Changing `condition_chunk`, denoise seed, or native action input alone is not a
future-only intervention. Until a DriveWAM adapter exposes the four required
conditions and a source-disjoint confirmation set, mediation remains
`unavailable` even though the execution graph has a plausible pathway hook.

## Temporal input contract

DriveWAM's `NavSimEpisodeDataset` requires the serialized `images` array to be
`[current, future_0.5s, ..., future_4.0s]`. It then selects indices
`[0, 2, 4, 6, 8]`, corresponding to `0, 1, 2, 3, 4s`. Historical ego poses
belong in `history_poses`; they must not be prepended to `images`.

Inputs produced by `build_inputs.py` carry
`input_image_contract=drivewam_current_plus_8_future_v1`. Legacy 12-frame
pickles must first be copied through `repair_temporal_input_contract.py`; the
repair never overwrites the source archive. Results generated from the legacy
`4 history + 8 future` image layout are temporally misaligned and must not be
compared with actions or measurements anchored at the current frame.

`build_ccfc_manifest.py` reopens every generated branch's immutable source
pickle and verifies the contract, nine-frame shape, and timestamps before it
will assemble an evaluation manifest. The source coordinate size must likewise
come from each private record or an explicit model-adapter CLI argument; it is
never inferred from resized image files.
