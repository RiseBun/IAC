#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${IAC_ROOT:-}" ]]; then
  ROOT="$IAC_ROOT"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

WORK_DIR="${WORK_DIR:-work_dirs/iac_navsim_future_dinov2_multisolution_official_pdms_vnext}"
CHECKPOINT="${RESUME_FROM:-$WORK_DIR/checkpoints/latest.pth}"
TARGET_EPOCH="${TARGET_EPOCH:-80}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$HOME/miniforge3/envs/drivingworld/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniforge3/envs/drivingworld/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

JOB_NAME="${JOB_NAME:-iac_epchain}"
SBATCH_PARTITION="${SBATCH_PARTITION:-GPU}"
SBATCH_GRES="${SBATCH_GRES:-gpu:4}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-16}"
SBATCH_NODELIST="${SBATCH_NODELIST:-4090node1}"

mkdir -p "$WORK_DIR"

submit_self() {
  local args=(
    --partition="$SBATCH_PARTITION" \
    --gres="$SBATCH_GRES" \
    --cpus-per-task="$SBATCH_CPUS_PER_TASK" \
    --job-name="$JOB_NAME" \
    --output="$ROOT/$WORK_DIR/slurm_epochchain_%j.log" \
    --chdir="$ROOT"
    --export="ALL,IAC_ROOT=$ROOT"
  )
  if [[ -n "$SBATCH_NODELIST" ]]; then
    args+=(--nodelist="$SBATCH_NODELIST")
  fi
  if [[ -n "${1:-}" ]]; then
    args+=(--dependency="$1")
  fi
  sbatch "${args[@]}" "$ROOT/scripts/run_official_pdms_epoch_chain_step.sh"
}

if [[ "${1:-}" == "--submit" ]]; then
  existing="$(squeue -u "$USER" -h -n "$JOB_NAME" -o '%i' 2>/dev/null | wc -l || true)"
  if [[ "$existing" -gt 0 ]]; then
    echo "[IAC] Existing $JOB_NAME jobs found; not submitting duplicates."
    squeue -u "$USER" -n "$JOB_NAME" || true
    exit 0
  fi
  submit_self "" | tee "$WORK_DIR/slurm_epoch_chain_submit.log"
  exit 0
fi

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "[IAC] Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

current_epoch="$("$PYTHON_BIN" - <<'PY'
import torch
ck = torch.load(
    "work_dirs/iac_navsim_future_dinov2_multisolution_official_pdms_vnext/checkpoints/latest.pth",
    map_location="cpu",
)
print(int(ck.get("epoch", 0)))
PY
)"

if [[ "$current_epoch" -ge "$TARGET_EPOCH" ]]; then
  echo "[IAC] Target reached: current_epoch=$current_epoch target=$TARGET_EPOCH"
  exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" && "${IAC_CHAIN_SUBMIT_NEXT:-1}" == "1" ]]; then
  submit_self "afterany:$SLURM_JOB_ID" | tee -a "$WORK_DIR/slurm_epoch_chain_submit.log"
fi

next_epoch=$((current_epoch + 1))
echo "[IAC] Chunk start: current_epoch=$current_epoch target_epoch=$next_epoch final_target=$TARGET_EPOCH"

env \
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
  EPOCHS="$next_epoch" \
  BATCH_SIZE="${BATCH_SIZE:-8}" \
  NUM_WORKERS="${NUM_WORKERS:-1}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}" \
  MAX_VAL_STEPS="${MAX_VAL_STEPS:-150}" \
  PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-32}" \
  RESUME_FROM="$CHECKPOINT" \
  bash scripts/run_multisolution_official_pdms_vnext.sh
