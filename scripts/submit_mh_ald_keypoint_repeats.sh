#!/usr/bin/env bash
# Paired multi-seed repeats for the five key MH-ALD planning points.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_mh_ald_keypoint_repeats.sh smoke
#   NODE=4090node3 bash scripts/submit_mh_ald_keypoint_repeats.sh formal

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/submit_mh_ald_keypoint_repeats.sh {smoke|formal}" >&2
  exit 2
fi
MODE="$1"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_mh_ald_keypoint_repeats.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"

LEWM_POLICY="lewm_epoch_10"
ALD_POLICY="pusht_ald_h5_seed3072_ep10_ddp4/lewm_ald_h5_ddp4_epoch_10"
MHALD_POLICY="pusht_mh_ald_h5_seed3072_ep10_ddp4/lewm_mh_ald_h5_ddp4_epoch_10"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$REPO/outputs/mh_ald_keypoint_repeats/$RUN_TAG"
RAW_DIR="$RUN_ROOT/raw"
SUM_DIR="$RUN_ROOT/summary"
LOG_DIR="$REPO/logs"
GEN_DIR="$REPO/slurm/generated_mh_ald_keypoint_repeats"
mkdir -p "$RAW_DIR" "$SUM_DIR" "$LOG_DIR" "$GEN_DIR"
ln -sfn "$RUN_ROOT" "$REPO/outputs/mh_ald_keypoint_repeats_latest"

cd "$REPO"
source "$CONDA_SH"
conda activate lewm
python -m py_compile scripts/summarize_mh_ald_keypoint_repeats.py

FILE="$GEN_DIR/keypoint_${MODE}_${RUN_TAG}.slurm"
cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --job-name=mhald_rep
#SBATCH --output=$LOG_DIR/mh_ald_keypoint_${RUN_TAG}_%j.out
#SBATCH --error=$LOG_DIR/mh_ald_keypoint_${RUN_TAG}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

MODE="$MODE"
RAW_DIR="$RAW_DIR"
SUM_DIR="$SUM_DIR"
LEWM_POLICY="$LEWM_POLICY"
ALD_POLICY="$ALD_POLICY"
MHALD_POLICY="$MHALD_POLICY"
EOF

cat >> "$FILE" <<'EOF'
run_model () {
  local GPU="$1"
  local LABEL="$2"
  local POLICY="$3"

  if [[ "$MODE" == "smoke" ]]; then
    SEEDS="42"
    POINTS="100,3,10"
    EP=4
  else
    SEEDS="42 43 44 45 46"
    POINTS="30,10,3 100,5,10 100,10,10 300,10,30 300,30,30"
    EP=100
  fi

  for SEED in $SEEDS; do
    for SPEC in $POINTS; do
      IFS=',' read -r N I K <<< "$SPEC"
      OUT="$RAW_DIR/${LABEL}_n${N}_i${I}_k${K}_seed${SEED}_ep${EP}.txt"
      echo "--- $LABEL seed=$SEED N=$N I=$I K=$K B=$((N*I)) ---"
      CUDA_VISIBLE_DEVICES="$GPU" python -u eval.py         --config-name=pusht.yaml         policy="$POLICY"         seed="$SEED"         solver.num_samples="$N"         solver.n_steps="$I"         solver.topk="$K"         eval.num_eval="$EP"         eval.eval_budget=50         output.filename="$OUT"
    done
  done
}

run_model 0 lewm "$LEWM_POLICY" &
P0=$!
run_model 1 ald "$ALD_POLICY" &
P1=$!
run_model 2 mh_ald "$MHALD_POLICY" &
P2=$!

wait "$P0"
wait "$P1"
wait "$P2"

python -u scripts/summarize_mh_ald_keypoint_repeats.py   --input-dir "$RAW_DIR"   --output-dir "$SUM_DIR"

echo "=== KEYPOINT REPEATS DONE ==="
echo "raw=$RAW_DIR"
echo "summary=$SUM_DIR"
EOF

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")
echo "Submitted MH-ALD keypoint repeats ($MODE): $JID"
echo "node: $NODE"
echo "run_root: $RUN_ROOT"
echo "status: squeue -j $JID"
echo "stdout: tail -f $LOG_DIR/mh_ald_keypoint_${RUN_TAG}_${JID}.out"
echo "stderr: tail -f $LOG_DIR/mh_ald_keypoint_${RUN_TAG}_${JID}.err"
