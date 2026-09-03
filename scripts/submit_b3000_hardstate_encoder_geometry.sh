#!/usr/bin/env bash
# Hard-state encoder geometry autopsy.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_b3000_hardstate_encoder_geometry.sh smoke
#   NODE=4090node3 bash scripts/submit_b3000_hardstate_encoder_geometry.sh formal
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_b3000_hardstate_encoder_geometry.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
NODE="${NODE:-4090node3}"

ALD_TF_POLICY="pusht_ald_tf_h5_seed3072_ep10_ddp4/lewm_ald_tf_h5_ddp4_epoch_10"
STEP1_MANIFEST="$REPO/outputs/b3000_paired_failure_analysis/formal/paired_manifest.csv"
STEP2_MEAN="$REPO/outputs/b3000_critical_jepa_mechanism/formal/mean_plan_causal_chain.csv"

OUT_ROOT="$REPO/outputs/b3000_hardstate_encoder_geometry"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_b3000_encoder_geometry"
mkdir -p "$LOG_DIR" "$GEN_DIR" "$REPO/outputs"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile   eval_b3000_hardstate_encoder_geometry.py   eval_b3000_critical_jepa_mechanism.py

if [[ "$MODE" == "smoke" ]]; then
  FILE="$GEN_DIR/b3000_encoder_geometry_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=b3k_encgeo_sm
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/b3000_encoder_geometry_smoke_%j.out
#SBATCH --error=$LOG_DIR/b3000_encoder_geometry_smoke_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

rm -rf "$OUT_ROOT/smoke"

CUDA_VISIBLE_DEVICES=0 python -u eval_b3000_hardstate_encoder_geometry.py \
  --config-name=pusht.yaml \
  seed=42 \
  solver.num_samples=30 \
  solver.n_steps=2 \
  solver.topk=3 \
  solver.batch_size=1 \
  eval.num_eval=6 \
  eval.eval_budget=50 \
  +geometry.output_dir="$OUT_ROOT/smoke" \
  +geometry.ald_policy="$ALD_TF_POLICY" \
  +geometry.manual_cases='[0,1]' \
  +geometry.solve_index=1 \
  +geometry.replay_iterations='[0,1]' \
  +geometry.model_batch_size=64 \
  2>&1

echo "=== SLURM DONE ==="
EOF
else
  if [[ ! -s "$STEP1_MANIFEST" ]]; then
    echo "Missing Step-1 manifest: $STEP1_MANIFEST" >&2
    exit 3
  fi
  if [[ ! -s "$STEP2_MEAN" ]]; then
    echo "Missing Step-2 mean chain: $STEP2_MEAN" >&2
    exit 3
  fi

  FILE="$GEN_DIR/b3000_encoder_geometry_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=b3k_encgeo
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/b3000_encoder_geometry_formal_%j.out
#SBATCH --error=$LOG_DIR/b3000_encoder_geometry_formal_%j.err

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
echo "Hard-state encoder geometry autopsy"
echo "Frozen ALD+TF CEM: N=300 I=10 K=30 B=3000"
echo "Primary hard cases: 27,53"
echo "Regression control: 23"
echo "Rescue controls: top-3 hardest by previous ALD solve-1 physical cost"
echo "Analyze solve1 populations at CEM iters 0,3,9"
echo "Factors: pusher XY / block XY / theta / object-task / official joint"
echo "No planner modification."
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u eval_b3000_hardstate_encoder_geometry.py \
  --config-name=pusht.yaml \
  seed=42 \
  solver.num_samples=300 \
  solver.n_steps=10 \
  solver.topk=30 \
  solver.batch_size=1 \
  eval.num_eval=100 \
  eval.eval_budget=50 \
  +geometry.output_dir="$OUT_ROOT/formal" \
  +geometry.ald_policy="$ALD_TF_POLICY" \
  +geometry.reference_manifest="$STEP1_MANIFEST" \
  +geometry.reference_mean_chain="$STEP2_MEAN" \
  +geometry.hard_cases='[27,53]' \
  +geometry.regression_cases='[23]' \
  +geometry.num_rescue_controls=3 \
  +geometry.solve_index=1 \
  +geometry.replay_iterations='[0,3,9]' \
  +geometry.model_batch_size=64 \
  2>&1

echo "=== SLURM DONE ==="
echo "Results: $OUT_ROOT/formal"
echo "Collect: bash scripts/package_b3000_hardstate_encoder_geometry.sh"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted $MODE hard-state encoder geometry: $JID"
echo "node: $NODE"
squeue -j "$JID" -o "%.18i %.16j %.2t %.12M %.12l %.20R" || true
echo
if [[ "$MODE" == "smoke" ]]; then
  echo "tail -f $LOG_DIR/b3000_encoder_geometry_smoke_${JID}.out"
else
  echo "tail -f $LOG_DIR/b3000_encoder_geometry_formal_${JID}.out"
fi
