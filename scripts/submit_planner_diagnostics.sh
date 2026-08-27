#!/usr/bin/env bash
# Submit the three PushT planner diagnostics to a 4090 compute node.
#
# Usage:
#   bash scripts/submit_planner_diagnostics.sh smoke
#   bash scripts/submit_planner_diagnostics.sh formal
#   NODE=4090node3 bash scripts/submit_planner_diagnostics.sh formal
#
# smoke:
#   NUM_EVAL=3, MAX_SOLVES=5, 16 perturbation directions
# formal:
#   NUM_EVAL=10, MAX_SOLVES=20, 64 perturbation directions
#
# The job runs:
#   D1 traced CEM -> D2 distance-binned order transition -> D3 physical oracle
# planner-center diagnostic.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_planner_diagnostics.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node2}"

LEWM_POLICY="${LEWM_POLICY:-lewm_epoch_10}"
ALD_POLICY="${ALD_POLICY:-pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10}"
GOAL_PROX_DIR="${GOAL_PROX_DIR:-$STABLEWM_HOME/fixed_horizon_goal_proximity}"

if [[ "$MODE" == "smoke" ]]; then
  NUM_EVAL=3
  MAX_SOLVES=5
  NUM_DIRECTIONS=16
  TAG="smoke"
else
  NUM_EVAL=10
  MAX_SOLVES=20
  NUM_DIRECTIONS=64
  TAG="formal"
fi

TRACE_DIR="$REPO/outputs/pusht_cem_trace_lewm_$TAG"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_planner_diagnostics"
mkdir -p "$LOG_DIR" "$GEN_DIR"

SLURM_FILE="$GEN_DIR/planner_diag_${TAG}.slurm"

cat > "$SLURM_FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=pd_$TAG
#SBATCH --output=$LOG_DIR/planner_diag_${TAG}_%j.out
#SBATCH --error=$LOG_DIR/planner_diag_${TAG}_%j.err

set -euo pipefail

source "$CONDA_SH"
conda activate lewm
cd "$REPO"

export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python -m py_compile \
  traced_cem.py \
  analyze_pusht_order_transition_bins.py \
  eval_pusht_planner_center_landscape.py

TRACE_DIR="$TRACE_DIR"
GOAL_PROX_DIR="$GOAL_PROX_DIR"
LEWM_POLICY="$LEWM_POLICY"
ALD_POLICY="$ALD_POLICY"

echo "============================================================"
echo "Planner diagnostics: $TAG"
echo "node=$(hostname)"
echo "trace_dir=\$TRACE_DIR"
echo "goal_prox_dir=\$GOAL_PROX_DIR"
echo "============================================================"

rm -rf "\$TRACE_DIR"
mkdir -p "\$TRACE_DIR"

echo
echo "=== D1: trace official CEM refinement ==="
JEPA_CEM_TRACE_DIR="\$TRACE_DIR" \
python -u eval.py \
  --config-name=pusht.yaml \
  solver=traced_cem \
  policy="\$LEWM_POLICY" \
  eval.num_eval=$NUM_EVAL

echo
echo "=== D2: latent-distance-binned order transition ==="
if [[ ! -f "\$GOAL_PROX_DIR/anchor_goal_metrics.csv" ]]; then
  echo "ERROR: missing \$GOAL_PROX_DIR/anchor_goal_metrics.csv" >&2
  echo "Set GOAL_PROX_DIR to the existing fixed-horizon goal-proximity result." >&2
  exit 3
fi

python -u analyze_pusht_order_transition_bins.py \
  --input "\$GOAL_PROX_DIR/anchor_goal_metrics.csv" \
  --num-bins 5

echo
echo "=== D3: oracle landscape at planner-visited centers ==="
python -u eval_pusht_planner_center_landscape.py \
  --trace-dir "\$TRACE_DIR" \
  --policies "\$LEWM_POLICY" "\$ALD_POLICY" \
  --labels lewm ald \
  --iterations 0 1 3 5 10 30 \
  --max-solves $MAX_SOLVES \
  --radius 0.1565 \
  --num-directions $NUM_DIRECTIONS

echo
echo "=== DONE ==="
echo "D1 traces: \$TRACE_DIR/solve_*.npz"
echo "D2: \$GOAL_PROX_DIR/distance_binned_order_transition/"
echo "D3: \$TRACE_DIR/planner_center_landscape/"
EOF

echo "Submitting $MODE planner diagnostics to $NODE"
echo "Slurm file: $SLURM_FILE"
sbatch "$SLURM_FILE"
