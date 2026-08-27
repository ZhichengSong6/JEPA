#!/usr/bin/env bash
# Replay-only correction for M1/M2 using already-saved formal CEM traces.
# No CEM rerun, no training, no M3 rerun.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_official_replay_diagnostics.sh
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node2}"

LEWM_POLICY="${LEWM_POLICY:-lewm_epoch_10}"
ALD_POLICY="${ALD_POLICY:-pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10}"

LEWM_TRACE="$REPO/outputs/pusht_cem_population_trace_lewm_formal"
ALD_TRACE="$REPO/outputs/pusht_cem_population_trace_ald_formal"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_official_replay"
mkdir -p "$LOG_DIR" "$GEN_DIR"

for d in "$LEWM_TRACE" "$ALD_TRACE"; do
  if [[ ! -f "$d/solve_000001.npz" ]]; then
    echo "ERROR: missing saved formal CEM traces under $d" >&2
    exit 2
  fi
done

SLURM_FILE="$GEN_DIR/official_replay.slurm"

cat > "$SLURM_FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=ms_replay
#SBATCH --output=$LOG_DIR/official_replay_%j.out
#SBATCH --error=$LOG_DIR/official_replay_%j.err

set -euo pipefail

source "$CONDA_SH"
conda activate lewm
cd "$REPO"

export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python -m py_compile \
  pusht_trace_eval_utils.py \
  eval_pusht_cem_population_fidelity.py \
  eval_pusht_center_value_trajectory.py

echo "============================================================"
echo "Official-execution replay diagnostic"
echo "node=\$(hostname)"
echo "No CEM rerun; reusing saved formal populations."
echo "============================================================"

echo
echo "=== R1: M1 on LeWM-CEM trace, official unclipped replay ==="
rm -rf "$LEWM_TRACE/cem_population_fidelity_official"
python -u eval_pusht_cem_population_fidelity.py \
  --trace-dir "$LEWM_TRACE" \
  --trace-label lewm_cem \
  --policies "$LEWM_POLICY" "$ALD_POLICY" \
  --labels lewm ald \
  --iterations 0 1 3 5 10 20 29 \
  --max-solves 20 \
  --max-candidates 0

echo
echo "=== R2: M1 on ALD-CEM trace, official unclipped replay ==="
rm -rf "$ALD_TRACE/cem_population_fidelity_official"
python -u eval_pusht_cem_population_fidelity.py \
  --trace-dir "$ALD_TRACE" \
  --trace-label ald_cem \
  --policies "$LEWM_POLICY" "$ALD_POLICY" \
  --labels lewm ald \
  --iterations 0 1 3 5 10 20 29 \
  --max-solves 20 \
  --max-candidates 0

echo
echo "=== R3: M2 on LeWM-CEM trace, official unclipped replay ==="
rm -rf "$LEWM_TRACE/center_value_trajectory_official"
python -u eval_pusht_center_value_trajectory.py \
  --trace-dir "$LEWM_TRACE" \
  --trace-label lewm_cem \
  --policies "$LEWM_POLICY" "$ALD_POLICY" \
  --labels lewm ald \
  --max-solves 20

echo
echo "=== R4: M2 on ALD-CEM trace, official unclipped replay ==="
rm -rf "$ALD_TRACE/center_value_trajectory_official"
python -u eval_pusht_center_value_trajectory.py \
  --trace-dir "$ALD_TRACE" \
  --trace-label ald_cem \
  --policies "$LEWM_POLICY" "$ALD_POLICY" \
  --labels lewm ald \
  --max-solves 20

echo
echo "=== DONE ==="
echo "LeWM M1: $LEWM_TRACE/cem_population_fidelity_official/"
echo "ALD  M1: $ALD_TRACE/cem_population_fidelity_official/"
echo "LeWM M2: $LEWM_TRACE/center_value_trajectory_official/"
echo "ALD  M2: $ALD_TRACE/center_value_trajectory_official/"
EOF

echo "Submitting official replay diagnostics to $NODE"
echo "Slurm file: $SLURM_FILE"
sbatch "$SLURM_FILE"
