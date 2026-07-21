# IAC Evaluation Protocol, 2026-07-01

## 1. Goal

IAC is a benchmark for WAM evaluation, not a new planner.
The benchmark should answer two questions:
- Does the WAM output support the candidate trajectory?
- Where does the support come from?

## 2. Primary scores

Keep the main benchmark on a fixed strict-future index.

Report:
- `consistency`
- `validity`
- ranking metrics over grouped candidates

Use these as the official benchmark numbers.

## 3. Diagnostic scores

Add a second layer of scores for explanation.

Probe these internal states:
- `hist_seq`
- `fut_seq`
- `z_traj_cons`
- `z_traj_val`
- `z_shared`
- `z_validity`
- `future_consistency_evidence`
- `future_traj_geometry_pred`

For each layer, fit linear probes for:
- consistency label
- validity label
- path length
- mean speed
- lateral displacement
- heading change
- curvature

## 4. What the diagnostics are for

The probe layer is not a separate competition score.
It is a mechanism check.

It should answer:
- Is physics already visible in the trajectory branch?
- Is consistency only readable after fusion?
- Does future image evidence move earlier in the stack?
- Is the model relying on shortcut geometry instead of future evidence?

## 5. Layer-gap report

For every benchmark run, include a short layer-gap summary:
- strongest consistency layer
- strongest physics layer
- gap between image layers and trajectory layers
- gap between `z_shared` and early layers

This is the main explanation artifact.

## 6. Future evidence test

Run an explicit future-evidence ablation:
- evidence off
- evidence on
- evidence with future-only context

The key question is not whether the score changes.
The key question is whether consistency becomes more sensitive to future frames.

## 7. Shortcut detection

Flag a model as shortcut-prone if:
- trajectory probes are strong
- future image probes remain weak
- consistency is only visible at `z_shared`
- future perturbations barely move the final score

That means the model may be using geometry priors rather than future evidence.

## 8. Reporting rule

Each report should contain:
- main benchmark metrics
- layer-gap summary
- future-evidence sensitivity
- shortcut warning if applicable

The benchmark is useful only if it can explain why a WAM is good.
