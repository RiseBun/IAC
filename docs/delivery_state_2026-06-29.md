# Delivery State, 2026-06-29

This workspace is not a byte-for-byte superset of the local delivery package:

`C:\Users\LPN19\Desktop\iac\deliverables\iac_delivery_package_2026-06-17`

That package is a 2026-06-17 archive from `/root/autodl-tmp/IAC` at git head
`a9be5514c4c824fa822d8c7bc3de355818c6d71c`. It contains the old DINOv2
current-best checkpoint, progress reports, summary tables, failure exports, and
fine-tune configs. It does not contain raw nuPlan data or full JSONL indexes.

The current server workspace is the newer NAVSIM strict-future line at git head
`fb6b21932c0d87210e6e1cb400a5e695f8e3ece2`, plus local working-tree updates.
It contains the 2026-06-29 `indices_navsim_future` index built with
`future_image_policy=future`, audit tooling, a NAVSIM future config, and a
reproducible baseline script.

Do not overwrite this workspace with the 2026-06-17 package. If that package is
needed on the server, copy it as an archive under a separate directory, for
example `archives/iac_delivery_package_2026-06-17`, and keep the current NAVSIM
future work as the active project state.
