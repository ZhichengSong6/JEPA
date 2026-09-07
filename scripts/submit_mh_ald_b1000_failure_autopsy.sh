#!/usr/bin/env bash
# Focused MH-ALD B=1000 failure autopsy.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_mh_ald_b1000_failure_autopsy.sh smoke
#   NODE=4090node3 bash scripts/submit_mh_ald_b1000_failure_autopsy.sh formal

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/submit_mh_ald_b1000_failure_autopsy.sh {smoke|formal}" >&2
  exit 2
fi
MODE="$1"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_mh_ald_b1000_failure_autopsy.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO/outputs/mh_ald_b1000_failure_autopsy/$RUN_TAG"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_mh_ald_b1000_failure_autopsy"
CEILING_MANIFEST="$REPO/outputs/mh_ald_b1000_ceiling_latest/ceiling_case_manifest.csv"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$GEN_DIR"
ln -sfn "$OUT_DIR" "$REPO/outputs/mh_ald_b1000_failure_autopsy_latest"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm
python -m py_compile eval_mh_ald_b1000_failure_autopsy.py pusht_exact_replay.py

if [[ "$MODE" == "smoke" ]]; then
  NUM_EVAL=4
  NUM_SAMPLES=12
  ITERS=2
  TOPK=3
  REPLAY_ITERS="[0,1]"
  MAX_SOLVES=1
  CEILING_ARG="+failure_autopsy.ceiling_manifest="
  EXPECTED=""
else
  [[ -s "$CEILING_MANIFEST" ]] || {
    echo "ERROR: missing ceiling manifest: $CEILING_MANIFEST" >&2
    exit 3
  }
  NUM_EVAL=100
  NUM_SAMPLES=100
  ITERS=10
  TOPK=10
  REPLAY_ITERS="[0,1,3,5,9]"
  MAX_SOLVES=0
  CEILING_ARG="+failure_autopsy.ceiling_manifest=$CEILING_MANIFEST"
  EXPECTED=""
fi

FILE="$GEN_DIR/b1000_failure_autopsy_${MODE}_${RUN_TAG}.slurm"
cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=b1k_fail
#SBATCH --output=$LOG_DIR/b1000_failure_autopsy_${RUN_TAG}_%j.out
#SBATCH --error=$LOG_DIR/b1000_failure_autopsy_${RUN_TAG}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "============================================================"
echo "MH-ALD B=1000 failure autopsy ($MODE)"
echo "node=\$(hostname)"
echo "output=$OUT_DIR"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python -u eval_mh_ald_b1000_failure_autopsy.py \
  --config-name=pusht.yaml \
  seed=42 \
  solver.num_samples=$NUM_SAMPLES \
  solver.n_steps=$ITERS \
  solver.topk=$TOPK \
  solver.batch_size=1 \
  eval.num_eval=$NUM_EVAL \
  eval.eval_budget=50 \
  +failure_autopsy.output_dir=$OUT_DIR \
  +failure_autopsy.replay_iterations='$REPLAY_ITERS' \
  +failure_autopsy.max_solves_per_case=$MAX_SOLVES \
  $CEILING_ARG \
  $EXPECTED

echo
echo "=== FAILURE AUTOPSY ARTIFACT VALIDATION ==="
REQ=(
  "$OUT_DIR/population_autopsy.csv"
  "$OUT_DIR/candidate_autopsy.npz"
  "$OUT_DIR/case_summary.csv"
  "$OUT_DIR/iteration_group_summary.csv"
  "$OUT_DIR/failure_autopsy_summary.json"
)
for f in "\${REQ[@]}"; do
  [[ -s "\$f" ]] || { echo "ERROR: missing/empty \$f" >&2; exit 4; }
done

if [[ "$MODE" == "smoke" ]]; then
  echo
  echo "=== FAILURE AUTOPSY SMOKE PACKAGE VALIDATION ==="
  bash scripts/package_mh_ald_b1000_failure_autopsy.sh
fi

echo
echo "=== FAILURE AUTOPSY JOB DONE ==="
echo "Results: $OUT_DIR"
echo "Package: bash scripts/package_mh_ald_b1000_failure_autopsy.sh"
EOF

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")
echo "Submitted B=1000 failure autopsy ($MODE): $JID"
echo "node: $NODE"
echo "run_root: $OUT_DIR"
echo "status: squeue -j $JID"
echo "stdout: tail -f $LOG_DIR/b1000_failure_autopsy_${RUN_TAG}_${JID}.out"
echo "stderr: tail -f $LOG_DIR/b1000_failure_autopsy_${RUN_TAG}_${JID}.err"
