#!/usr/bin/env bash
# B=1000 MH-ALD ceiling experiment.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_mh_ald_b1000_ceiling.sh smoke
#   NODE=4090node3 bash scripts/submit_mh_ald_b1000_ceiling.sh formal

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/submit_mh_ald_b1000_ceiling.sh {smoke|formal}" >&2
  exit 2
fi
MODE="$1"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_mh_ald_b1000_ceiling.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO/outputs/mh_ald_b1000_ceiling/$RUN_TAG"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_mh_ald_b1000_ceiling"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$GEN_DIR"
ln -sfn "$OUT_DIR" "$REPO/outputs/mh_ald_b1000_ceiling_latest"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm
python -m py_compile eval_pusht_b1000_ceiling.py pusht_exact_replay.py

if [[ "$MODE" == "smoke" ]]; then
  NUM_EVAL=6
  MAX_CONTROLS=2
  EXPECTED=""
else
  NUM_EVAL=100
  MAX_CONTROLS=4
  EXPECTED="+ceiling.expected_baseline_success=92.0"
fi

FILE="$GEN_DIR/b1000_ceiling_${MODE}_${RUN_TAG}.slurm"
cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --job-name=b1k_ceil
#SBATCH --output=$LOG_DIR/b1000_ceiling_${RUN_TAG}_%j.out
#SBATCH --error=$LOG_DIR/b1000_ceiling_${RUN_TAG}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

OUT_DIR="$OUT_DIR"
NUM_EVAL="$NUM_EVAL"
MAX_CONTROLS="$MAX_CONTROLS"
EXPECTED="$EXPECTED"

COMMON_ARGS=(
  --config-name=pusht.yaml
  seed=42
  solver.num_samples=100
  solver.n_steps=10
  solver.topk=10
  solver.batch_size=1
  eval.num_eval="$NUM_EVAL"
  eval.eval_budget=50
  +ceiling.output_dir="$OUT_DIR"
  +ceiling.max_success_controls="$MAX_CONTROLS"
  +ceiling.target_success=95.6
)

echo "============================================================"
echo "B=1000 ceiling experiment ($MODE)"
echo "node=\$(hostname)"
echo "output=$OUT_DIR"
echo "============================================================"

echo
echo "=== PHASE 1: current MH-ALD baseline ==="
CUDA_VISIBLE_DEVICES=0 python -u eval_pusht_b1000_ceiling.py   "\${COMMON_ARGS[@]}"   +ceiling.phase=baseline   $EXPECTED

echo
echo "=== PHASE 2A: encoder oracle on baseline failures + controls ==="
CUDA_VISIBLE_DEVICES=1 python -u eval_pusht_b1000_ceiling.py   "\${COMMON_ARGS[@]}"   +ceiling.phase=encoder &
P1=\$!

echo
echo "=== PHASE 2B: physical oracle on same cases ==="
CUDA_VISIBLE_DEVICES=2 python -u eval_pusht_b1000_ceiling.py   "\${COMMON_ARGS[@]}"   +ceiling.phase=physical &
P2=\$!

wait "\$P1"
wait "\$P2"

echo
echo "=== PHASE 3: summarize ceilings ==="
CUDA_VISIBLE_DEVICES=0 python -u eval_pusht_b1000_ceiling.py   "\${COMMON_ARGS[@]}"   +ceiling.phase=summary

echo
echo "=== CEILING JOB DONE ==="
echo "Results: $OUT_DIR"
echo "Package: bash scripts/package_mh_ald_b1000_ceiling.sh"
EOF

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")
echo "Submitted B=1000 ceiling ($MODE): $JID"
echo "node: $NODE"
echo "run_root: $OUT_DIR"
echo "status: squeue -j $JID"
echo "stdout: tail -f $LOG_DIR/b1000_ceiling_${RUN_TAG}_${JID}.out"
echo "stderr: tail -f $LOG_DIR/b1000_ceiling_${RUN_TAG}_${JID}.err"
