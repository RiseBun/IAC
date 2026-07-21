# Depth Environment Audit, 2026-07-02

Server:

```text
ssh zchen897@10.120.17.131
host = 4090node1
project = /mnt/slurmfs-4090node1/homes/zchen897/IAC
```

## Result

The full `net_depth.py` + `config_depth.yaml` path cannot run directly on this
server yet.

Missing runtime variables:

```text
EFG_PATH =
WORK_MA =
PYTHONPATH =
```

Missing project/model dependencies:

```text
MetricAnything_Teacher.openscene
MoGe third_party checkout
/mnt/volumes/base-pi-lx-my/bjzhu/models/moge-2/moge-2-vitl/model.pt
```

Python import audit:

```text
drivingworld python: torch/numpy/PIL/matplotlib OK; efg/moge/omegaconf/pyarrow missing
navsim python: torch/numpy/PIL/matplotlib/omegaconf/pyarrow OK; efg/moge missing
```

## Decision

Do not block the IAC work on MoGe/EFG installation.

The shortest useful next step is a dependency-free trajectory-specific causal
test using the existing strict-future IAC rows:

```text
mask current candidate path
vs
mask same-group wrong candidate path
```

This directly tests whether the score is tied to the current trajectory's path,
not just to generic road/path pixels.

## GPU Constraint

Available project runs should use at most 3 GPUs.

Recommended training launch pattern:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 ...
```

Small benchmark/debug runs can use a single GPU.

