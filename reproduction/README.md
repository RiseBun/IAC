# Reference reproduction

These helpers reproduce the published reference run after the user supplies
licensed datasets and model dependencies:

- `drivewam/`: convert the frozen benchmark into DriveWAM inputs, join native
  actions and generated frames, and stage CFAC/FAU/CCFC/FCS inputs.
- `navsim/`: execute native actions with NAVSIM/PDM and export simulator-derived
  state and FCS labels.

The generic benchmark does not depend on DriveWAM. A new WAM only needs to
follow `docs/WAM_SUBMISSION_ZH.md` and use the entrypoints in `scripts/`.

No command in this directory embeds a server path, account name, checkpoint,
or dataset. Missing external inputs fail closed.
