#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_SH="${CONDA_SH:-/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Missing conda setup: $CONDA_SH. Set CONDA_SH explicitly." >&2
  exit 2
fi
source "$CONDA_SH"
conda activate "${CONDA_ENV:-lewm}"
export STABLEWM_HOME="${STABLEWM_HOME:-/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Short, node-local path; never delete any pre-existing user directory.
local_tmp="$(mktemp -d "/tmp/pq_${SLURM_JOB_ID:-$$}_XXXXXX")"
export TMPDIR="$local_tmp" TMP="$local_tmp" TEMP="$local_tmp"
trap 'rm -rf -- "$local_tmp"' EXIT

cd "$REPO"
python -m unittest discover -s pcqr_tta_v1/tests -v
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

OUT_DIR="${OUT_DIR:-$STABLEWM_HOME/pcqr_tta_v1_runs/smoke_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)}"
POLICY_DIR="${POLICY_DIR:-$STABLEWM_HOME/pusht_pc_mh_h5_r01565_seed3072_ep10_ddp4}"
read -r -a modes <<< "${MODES:-frozen obs response}"
source_args=(--policy-dir "$POLICY_DIR")
if [[ -n "${POLICY:-}" ]]; then
  source_args=(--policy "$POLICY")
fi
case_args=()
if [[ -n "${CASES:-}" ]]; then
  case_args=(--cases "$CASES")
fi
python -m pcqr_tta_v1.run_pusht \
  --repo "$REPO" "${source_args[@]}" --out "$OUT_DIR" \
  --modes "${modes[@]}" --num-cases "${NUM_CASES:-3}" \
  --budget "${BUDGET:-1000}" --seed "${SEED:-1729}" \
  --eval-budget "${EVAL_BUDGET:-50}" --device cuda "${case_args[@]}"

echo "Results: $OUT_DIR"
