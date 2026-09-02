#!/usr/bin/env bash
# Paired B=3000 LeWM vs ALD+TF failure/rescue analysis.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_b3000_paired_failure_analysis.sh smoke
#   NODE=4090node3 bash scripts/submit_b3000_paired_failure_analysis.sh formal
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_b3000_paired_failure_analysis.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
NODE="${NODE:-4090node3}"

LEWM_POLICY="lewm_epoch_10"
ALD_TF_POLICY="pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10"

LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_b3000_paired"
OUT_ROOT="$REPO/outputs/b3000_paired_failure_analysis"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$REPO/outputs"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile eval_b3000_paired_failure_analysis.py eval_lowbudget_failure_autopsy.py

if [[ "$MODE" == "smoke" ]]; then
  FILE="$GEN_DIR/b3000_paired_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=b3k_pair_sm
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/b3000_paired_smoke_%j.out
#SBATCH --error=$LOG_DIR/b3000_paired_smoke_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

rm -rf "$OUT_ROOT/smoke"

CUDA_VISIBLE_DEVICES=0 python -u eval_b3000_paired_failure_analysis.py \
  --config-name=pusht.yaml \
  seed=42 \
  solver.num_samples=30 \
  solver.n_steps=2 \
  solver.topk=3 \
  solver.batch_size=1 \
  eval.num_eval=6 \
  eval.eval_budget=50 \
  +paired.output_dir="$OUT_ROOT/smoke" \
  +paired.lewm_policy="$LEWM_POLICY" \
  +paired.ald_policy="$ALD_TF_POLICY" \
  +paired.replay_iterations='[0,1]' \
  +paired.max_success_controls=2 \
  2>&1

echo "=== SLURM DONE ==="
EOF
else
  FILE="$GEN_DIR/b3000_paired_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=b3k_pair
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/b3000_paired_formal_%j.out
#SBATCH --error=$LOG_DIR/b3000_paired_formal_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

rm -rf "$OUT_ROOT/formal"

echo "============================================================"
echo "STEP 1: Paired B=3000 LeWM vs ALD+TF failure analysis"
echo "Frozen CEM: N=300 I=10 K=30 B=3000"
echo "100 identical official PushT starts"
echo "Cross-eval iterations: 0,3,9"
echo "All non-both-success cases + matched both-success controls"
echo "No planner modification. Oracle is diagnosis-only."
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u eval_b3000_paired_failure_analysis.py \
  --config-name=pusht.yaml \
  seed=42 \
  solver.num_samples=300 \
  solver.n_steps=10 \
  solver.topk=30 \
  solver.batch_size=1 \
  eval.num_eval=100 \
  eval.eval_budget=50 \
  +paired.output_dir="$OUT_ROOT/formal" \
  +paired.lewm_policy="$LEWM_POLICY" \
  +paired.ald_policy="$ALD_TF_POLICY" \
  +paired.replay_iterations='[0,3,9]' \
  +paired.max_success_controls=12 \
  +paired.expected_lewm_success=90.0 \
  +paired.expected_ald_success=97.0 \
  2>&1

echo "=== SLURM DONE ==="
echo "Results: $OUT_ROOT/formal"
echo "Collect: bash scripts/package_b3000_paired_failure_analysis.sh"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted $MODE paired B3000 analysis: $JID"
echo "node: $NODE"
squeue -j "$JID" -o "%.18i %.14j %.2t %.12M %.12l %.20R" || true
echo
if [[ "$MODE" == "smoke" ]]; then
  echo "tail -f $LOG_DIR/b3000_paired_smoke_${JID}.out"
else
  echo "tail -f $LOG_DIR/b3000_paired_formal_${JID}.out"
fi
