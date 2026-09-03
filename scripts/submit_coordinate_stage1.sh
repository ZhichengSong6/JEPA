#!/usr/bin/env bash
# Coordinate-geometry Stage-I launcher.
#
# Usage:
#   MODE=smoke  NODE=4090node3 bash scripts/submit_coordinate_stage1.sh
#   MODE=formal NODE=4090node3 bash scripts/submit_coordinate_stage1.sh
#
# Smoke and formal are intentionally separate: inspect reliability/conditioning
# diagnostics before spending the full Stage-I budget.
set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="lewm"
NODE="${NODE:-4090node3}"
MODE="${MODE:-smoke}"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "NODE must be 4090node2 or 4090node3" >&2
  exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "MODE must be smoke or formal" >&2
  exit 2
fi

GEN_DIR="$REPO/slurm/generated_coordinate_stage1"
LOG_DIR="$REPO/logs"
mkdir -p "$GEN_DIR" "$LOG_DIR"

if [[ "$MODE" == "smoke" ]]; then
  RUN_DIR="pusht_coordinate_stage1_seed3072_smoke"
  MODEL_NAME="lewm_coordinate_stage1_smoke"
  JOB_NAME="coord_s1_sm"
  MAX_STEPS=20
  VAL_INTERVAL=10
  LIMIT_VAL=5
else
  RUN_DIR="pusht_coordinate_stage1_seed3072"
  MODEL_NAME="lewm_coordinate_stage1"
  JOB_NAME="coord_s1"
  MAX_STEPS=2000
  VAL_INTERVAL=500
  LIMIT_VAL=20
fi

OUT_DIR="$STABLEWM_HOME/$RUN_DIR"
SLURM_FILE="$GEN_DIR/${MODE}.slurm"

cd "$REPO"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python -m py_compile \
  coordinate_geometry.py \
  jepa.py \
  train_coordinate_stage1.py

python -m pytest -q tests/test_coordinate_geometry.py

if [[ "$MODE" == "smoke" ]]; then
  rm -rf "$OUT_DIR"
elif [[ -e "$OUT_DIR" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "ERROR: formal Stage-I output already exists: $OUT_DIR" >&2
  echo "Delete it intentionally before retraining." >&2
  exit 2
fi

cat > "$SLURM_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=$LOG_DIR/coordinate_stage1_${MODE}_%j.out
#SBATCH --error=$LOG_DIR/coordinate_stage1_${MODE}_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"

export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

echo "node=$(hostname)"
echo "python=$(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train_coordinate_stage1.py \
  subdir=$RUN_DIR \
  output_model_name=$MODEL_NAME \
  trainer.max_steps=$MAX_STEPS \
  trainer.val_check_interval=$VAL_INTERVAL \
  trainer.limit_val_batches=$LIMIT_VAL

echo
echo "===== SUMMARY ====="
cat "$OUT_DIR/summary.json"
EOF

bash -n "$SLURM_FILE"
JID="$(sbatch --parsable "$SLURM_FILE")"

cat <<EOF
Submitted coordinate Stage-I $MODE.

job     : $JID
node    : $NODE
output  : $OUT_DIR
stdout  : $LOG_DIR/coordinate_stage1_${MODE}_${JID}.out
stderr  : $LOG_DIR/coordinate_stage1_${MODE}_${JID}.err

Queue:
  squeue -u zsong469 -o '%.18i %.12j %.2t %.10M %.24R'

Status:
  sacct -j $JID --format=JobID,JobName,State,Elapsed,ExitCode,NodeList

Log:
  tail -f $LOG_DIR/coordinate_stage1_${MODE}_${JID}.out
EOF
