#!/usr/bin/env bash
# Low-budget CEM failure autopsy.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_lowbudget_failure_autopsy.sh smoke
#   NODE=4090node3 bash scripts/submit_lowbudget_failure_autopsy.sh formal
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_lowbudget_failure_autopsy.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
NODE="${NODE:-4090node3}"
POLICY="pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10"

LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_failure_autopsy"
OUT_DIR="$REPO/outputs/lowbudget_failure_autopsy"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$REPO/outputs"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile eval_lowbudget_failure_autopsy.py

if [[ "$MODE" == "smoke" ]]; then
  FILE="$GEN_DIR/failure_autopsy_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=fa_smoke
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/failure_autopsy_smoke_%j.out
#SBATCH --error=$LOG_DIR/failure_autopsy_smoke_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

rm -rf "$OUT_DIR/smoke"

CUDA_VISIBLE_DEVICES=0 python -u eval_lowbudget_failure_autopsy.py \
  --config-name=pusht.yaml \
  policy="$POLICY" \
  seed=42 \
  solver.num_samples=10 \
  solver.n_steps=2 \
  solver.topk=2 \
  solver.batch_size=1 \
  eval.num_eval=4 \
  eval.eval_budget=10 \
  +autopsy.output_dir="$OUT_DIR/smoke" \
  +autopsy.replay_iterations='[0,1]' \
  +autopsy.num_success_controls=2

echo "=== SLURM DONE ==="
EOF
else
  FILE="$GEN_DIR/failure_autopsy_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=fa_n30i10
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/failure_autopsy_formal_%j.out
#SBATCH --error=$LOG_DIR/failure_autopsy_formal_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

rm -rf "$OUT_DIR/formal"

echo "============================================================"
echo "Low-budget failure autopsy"
echo "policy=$POLICY"
echo "CEM N=30 I=10 K=3, 100 official PushT episodes"
echo "Replay iterations: 0,1,3,5,9"
echo "All failed cases + up to 12 matched success controls"
echo "Physical replay is diagnosis-only."
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u eval_lowbudget_failure_autopsy.py \
  --config-name=pusht.yaml \
  policy="$POLICY" \
  seed=42 \
  solver.num_samples=30 \
  solver.n_steps=10 \
  solver.topk=3 \
  solver.batch_size=1 \
  eval.num_eval=100 \
  eval.eval_budget=50 \
  +autopsy.output_dir="$OUT_DIR/formal" \
  +autopsy.replay_iterations='[0,1,3,5,9]' \
  +autopsy.num_success_controls=12 \
  +autopsy.expected_success=88.0

echo "=== SLURM DONE ==="
echo "Results: $OUT_DIR/formal"
echo "Package: bash scripts/package_lowbudget_failure_autopsy.sh"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted $MODE failure autopsy: $JID"
echo "node: $NODE"
squeue -j "$JID" -o "%.18i %.14j %.2t %.12M %.12l %.20R" || true
echo
if [[ "$MODE" == "smoke" ]]; then
  echo "tail -f $LOG_DIR/failure_autopsy_smoke_${JID}.out"
else
  echo "tail -f $LOG_DIR/failure_autopsy_formal_${JID}.out"
fi
