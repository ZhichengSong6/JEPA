#!/usr/bin/env bash
# STEP 2 critical-case JEPA mechanism diagnostic.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_b3000_critical_jepa_mechanism.sh smoke
#   NODE=4090node3 bash scripts/submit_b3000_critical_jepa_mechanism.sh formal
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_b3000_critical_jepa_mechanism.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
NODE="${NODE:-4090node3}"

LEWM_POLICY="lewm_epoch_10"
ALD_TF_POLICY="pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10"

STEP1_MANIFEST="$REPO/outputs/b3000_paired_failure_analysis/formal/paired_manifest.csv"
OUT_ROOT="$REPO/outputs/b3000_critical_jepa_mechanism"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_b3000_mechanism"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$REPO/outputs"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile   eval_b3000_critical_jepa_mechanism.py   eval_b3000_paired_failure_analysis.py   eval_lowbudget_failure_autopsy.py

if [[ "$MODE" == "smoke" ]]; then
  FILE="$GEN_DIR/b3000_mechanism_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=b3k_mech_sm
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/b3000_mechanism_smoke_%j.out
#SBATCH --error=$LOG_DIR/b3000_mechanism_smoke_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

rm -rf "$OUT_ROOT/smoke"

echo "============================================================"
echo "STEP 2 smoke: JEPA mechanism path"
echo "Small CEM only for code-path validation"
echo "indices=[0,1], contexts=1,2,3"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u eval_b3000_critical_jepa_mechanism.py \
  --config-name=pusht.yaml \
  seed=42 \
  solver.num_samples=30 \
  solver.n_steps=2 \
  solver.topk=3 \
  solver.batch_size=1 \
  eval.num_eval=6 \
  eval.eval_budget=50 \
  +mechanism.output_dir="$OUT_ROOT/smoke" \
  +mechanism.lewm_policy="$LEWM_POLICY" \
  +mechanism.ald_policy="$ALD_TF_POLICY" \
  +mechanism.contexts='[1,2,3]' \
  +mechanism.replay_iterations='[0,1]' \
  +mechanism.eval_indices='[0,1]' \
  +mechanism.model_batch_size=64 \
  +mechanism.replay_state_tol_px=1.0 \
  +mechanism.replay_state_tol_deg=1.0 \
  2>&1

echo "=== SLURM DONE ==="
EOF
else
  if [[ ! -s "$STEP1_MANIFEST" ]]; then
    echo "Missing Step-1 manifest: $STEP1_MANIFEST" >&2
    exit 3
  fi

  FILE="$GEN_DIR/b3000_mechanism_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=b3k_mech
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/b3000_mechanism_formal_%j.out
#SBATCH --error=$LOG_DIR/b3000_mechanism_formal_%j.err

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
echo "STEP 2: Critical-case JEPA mechanism diagnostic"
echo "Frozen CEM: N=300 I=10 K=30 B=3000"
echo "Formal critical cases come from Step-1 paired manifest"
echo "Expected partition: 89 both success / 8 rescue / 2 both fail / 1 regression"
echo "Mechanisms: encoder ceiling + endpoint fidelity + C=1,2,3 causal prefix"
echo "CEM iterations: 0,3,9"
echo "No training. No planner modification."
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u eval_b3000_critical_jepa_mechanism.py \
  --config-name=pusht.yaml \
  seed=42 \
  solver.num_samples=300 \
  solver.n_steps=10 \
  solver.topk=30 \
  solver.batch_size=1 \
  eval.num_eval=100 \
  eval.eval_budget=50 \
  +mechanism.output_dir="$OUT_ROOT/formal" \
  +mechanism.lewm_policy="$LEWM_POLICY" \
  +mechanism.ald_policy="$ALD_TF_POLICY" \
  +mechanism.contexts='[1,2,3]' \
  +mechanism.replay_iterations='[0,3,9]' \
  +mechanism.reference_manifest="$STEP1_MANIFEST" \
  +mechanism.model_batch_size=64 \
  2>&1

echo "=== SLURM DONE ==="
echo "Results: $OUT_ROOT/formal"
echo "Collect: bash scripts/package_b3000_critical_jepa_mechanism.sh"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted $MODE B3000 JEPA mechanism diagnostic: $JID"
echo "node: $NODE"
squeue -j "$JID" -o "%.18i %.14j %.2t %.12M %.12l %.20R" || true
echo
if [[ "$MODE" == "smoke" ]]; then
  echo "tail -f $LOG_DIR/b3000_mechanism_smoke_${JID}.out"
else
  echo "tail -f $LOG_DIR/b3000_mechanism_formal_${JID}.out"
fi
