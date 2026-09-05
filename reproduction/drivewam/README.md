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
