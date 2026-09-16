# NAVSIM/PDM execution

`run_rollouts.py` sends each submitted native action through the NAVSIM PDM
kinematic-bicycle simulation and derives realized state, PDM score, and external
success label. It never reads generated future images and never treats WAM
waypoints as realized state.

```bash
python reproduction/navsim/run_rollouts.py \
  --branches <execution_branches.jsonl> \
  --metric-cache <navsim_metric_cache> \
  --output <rollout_records.jsonl> \
  --summary <execution_summary.json>
```

The runner expects the official NAVSIM/nuplan-devkit checkout used by the
benchmark, including `navsim.evaluate.pdm_score` and the PDM simulator APIs.
Do **not** satisfy this requirement with the unrelated package named `navsim`
from PyPI; that package has a different module layout. NAVSIM and its metric
cache must be installed separately under their licenses.
If the runtime is absent, the command fails with an explicit dependency error;
that condition is `unavailable`, not a failure or zero IAC score.
