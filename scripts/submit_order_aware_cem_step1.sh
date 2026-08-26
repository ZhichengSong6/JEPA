#!/usr/bin/env bash
# Step 1: test whether a planner-side landscape-order rule can avoid harmful
# near-goal CEM over-optimization without retraining the world model.
#
# Runs TWO matched evaluations on the frozen official LeWM epoch-10 model:
#   1) paired_raw : symmetric Monte Carlo sampling, ordinary raw-cost CEM.
#                   This controls for the change in sampling pattern.
#   2) order_stop : identical paired sampling/raw updates, but stops the CEM
#                   mean once odd/even RMS <= 1 after at least 5 iterations.
#
# Both use the official PushT planning budget:
#   num_samples=300, topk=30, n_steps=30, H=5, action_block=5,
#   receding_horizon=5, goal_offset=25, eval_budget=50, num_eval=50.

set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="4090node2"
GEN_DIR="$REPO/slurm/generated_order_aware_cem_step1"
LOG_DIR="$REPO/logs"

mkdir -p "$GEN_DIR" "$LOG_DIR"
cd "$REPO"

source "$CONDA_SH"
conda activate lewm

python -m py_compile order_aware_cem.py eval.py

SLURM_FILE="$GEN_DIR/order_aware_cem_step1.slurm"

cat > "$SLURM_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=oa_cem1
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/order_aware_cem_step1_%j.out
#SBATCH --error=$LOG_DIR/order_aware_cem_step1_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"

export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

POLICY="lewm_epoch_10"

rm -f \
  "$STABLEWM_HOME/pusht_orderaware_paired_raw.txt" \
  "$STABLEWM_HOME/pusht_orderaware_order_stop.txt"

printf '\n============================================================\n'
printf 'STEP 1A/2: PAIRED-SAMPLING RAW-COST CONTROL\n'
printf '============================================================\n'
python -u eval.py \
  policy=\$POLICY \
  solver=order_aware_cem \
  solver.mode=paired_raw \
  solver.verbose=false \
  output.filename=pusht_orderaware_paired_raw.txt

printf '\n============================================================\n'
printf 'STEP 1B/2: LANDSCAPE-ORDER-AWARE CEM STOP\n'
printf '============================================================\n'
python -u eval.py \
  policy=\$POLICY \
  solver=order_aware_cem \
  solver.mode=order_stop \
  solver.transition_ratio=1.0 \
  solver.min_steps=5 \
  solver.verbose=true \
  output.filename=pusht_orderaware_order_stop.txt

printf '\n============================================================\n'
printf 'STEP-1 FINISHED\n'
printf '============================================================\n'
printf 'Paired raw result:  %s\n' "$STABLEWM_HOME/pusht_orderaware_paired_raw.txt"
printf 'Order-stop result:  %s\n' "$STABLEWM_HOME/pusht_orderaware_order_stop.txt"
EOF

bash -n "$SLURM_FILE"
JID=$(sbatch --parsable "$SLURM_FILE")

echo "Submitted Step-1 order-aware CEM evaluation: $JID"
echo "node: $NODE | GPU: 1"
echo
echo "Queue:"
echo "  squeue -j $JID -o '%.18i %.12j %.2t %.12M %.12l %.20R'"
echo
echo "Log:"
echo "  tail -f $LOG_DIR/order_aware_cem_step1_${JID}.out"
echo
echo "When finished, send me the .out log or the two txt result files."
