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
