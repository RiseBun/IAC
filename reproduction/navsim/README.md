# NAVSIM/PDM execution

`run_rollouts.py` sends each submitted native action through the NAVSIM PDM
kinematic-bicycle simulation and derives realized state, PDM score, and the FCS
success label. It never reads generated future images and never treats WAM
waypoints as realized state.

```bash
python reproduction/navsim/run_rollouts.py \
  --branches <fcs_branches.jsonl> \
  --metric-cache <navsim_metric_cache> \
  --output <rollout_records.jsonl> \
  --summary <fcs_summary.json>
```

NAVSIM and its metric cache must be installed separately under their licenses.
