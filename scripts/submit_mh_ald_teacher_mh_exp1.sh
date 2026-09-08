#!/usr/bin/env bash
# Exp1: replacement-teacher MH-ALD.
#
# Scientific control:
#   student init = official LeWM epoch-10
#   teacher      = current formal MH-ALD epoch-10
# Everything else is identical to the formal MH-ALD training.
#
# Usage:
#   NODE=4090node3 bash scripts/submit_mh_ald_teacher_mh_exp1.sh smoke
#   NODE=4090node3 bash scripts/submit_mh_ald_teacher_mh_exp1.sh formal

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_mh_ald_teacher_mh_exp1.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node3}"

if [[ "$NODE" != "4090node2" && "$NODE" != "4090node3" ]]; then
  echo "ERROR: Exp1 is restricted to 4090node2 or 4090node3." >&2
  exit 4
fi

WORLD_SIZE=4
LOCAL_BATCH=16
GLOBAL_BATCH=$((WORLD_SIZE * LOCAL_BATCH))
NUM_WORKERS_PER_RANK=4

TEACHER_POLICY="pusht_mh_ald_h5_seed3072_ep10_ddp4/lewm_mh_ald_h5_ddp4_epoch_10"
TEACHER_CKPT="$STABLEWM_HOME/pusht_mh_ald_h5_seed3072_ep10_ddp4/lewm_mh_ald_h5_ddp4_epoch_10_object.ckpt"

SMOKE_DIR="pusht_mh_ald_teacher_mh_exp1_smoke_ddp4"
SMOKE_MODEL="lewm_mh_ald_teacher_mh_exp1_smoke_ddp4"
FORMAL_DIR="pusht_mh_ald_teacher_mh_h5_seed3072_ep10_ddp4"
FORMAL_MODEL="lewm_mh_ald_teacher_mh_h5_ddp4"

GEN_DIR="$REPO/slurm/generated_mh_ald_teacher_mh_exp1"
LOG_DIR="$REPO/logs"
mkdir -p "$GEN_DIR" "$LOG_DIR"

[[ -s "$TEACHER_CKPT" ]] || {
  echo "ERROR: formal MH-ALD teacher checkpoint missing:" >&2
  echo "  $TEACHER_CKPT" >&2
  exit 5
}

cd "$REPO"
source "$CONDA_SH"
conda activate lewm

python -m py_compile \
  stage1_bias_calibration.py \
  anchored_local_dynamics.py \
  train_mh_ald.py

COMMON_HEADER=$(cat <<EOF
#SBATCH --partition=GPU
#SBATCH --nodelist=$NODE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$WORLD_SIZE
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:$WORLD_SIZE
EOF
)

COMMON_SETUP=$(cat <<EOF
set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
EOF
)

COMMON_HYDRA="seed=3072 num_workers=$NUM_WORKERS_PER_RANK trainer.devices=$WORLD_SIZE trainer.num_nodes=1 trainer.strategy=ddp trainer.sync_batchnorm=false loader.batch_size=$LOCAL_BATCH"

if [[ "$MODE" == "smoke" ]]; then
  rm -rf "$STABLEWM_HOME/$SMOKE_DIR"
  FILE="$GEN_DIR/00_smoke.slurm"

  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=mht2_sm4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/mh_teacher_mh_exp1_smoke_%j.out
#SBATCH --error=$LOG_DIR/mh_teacher_mh_exp1_smoke_%j.err

$COMMON_SETUP

echo "============================================================"
echo "EXP1 SMOKE: LeWM student + MH-ALD teacher"
echo "student_init=lewm_epoch_10"
echo "teacher=$TEACHER_POLICY"
echo "node=\$(hostname)"
echo "============================================================"

srun --kill-on-bad-exit=1 python -u train_mh_ald.py \
  --config-name=lewm_mh_ald_teacher_mh \
  $COMMON_HYDRA \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  output_model_name=$SMOKE_MODEL \
  subdir=$SMOKE_DIR

echo
echo "=== EXP1 SMOKE ARTIFACT VALIDATION ==="
[[ -s "$STABLEWM_HOME/$SMOKE_DIR/${SMOKE_MODEL}_epoch_1_object.ckpt" || \
   -s "$STABLEWM_HOME/$SMOKE_DIR/${SMOKE_MODEL}_epoch_0_object.ckpt" ]] || {
  echo "ERROR: Exp1 smoke object checkpoint not found." >&2
  find "$STABLEWM_HOME/$SMOKE_DIR" -maxdepth 1 -type f -printf '%f\n' || true
  exit 6
}

echo "=== EXP1 SMOKE DONE ==="
EOF
else
  if [[ -e "$STABLEWM_HOME/$FORMAL_DIR" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "ERROR: formal Exp1 output already exists: $STABLEWM_HOME/$FORMAL_DIR" >&2
    echo "Remove/rename it before a fresh formal run." >&2
    exit 3
  fi

  FILE="$GEN_DIR/mh_teacher_mh_exp1_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=mht2_h5_4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/mh_teacher_mh_exp1_formal_%j.out
#SBATCH --error=$LOG_DIR/mh_teacher_mh_exp1_formal_%j.err

$COMMON_SETUP

echo "============================================================"
echo "EXP1 FORMAL: LeWM student + MH-ALD teacher"
echo "student_init=lewm_epoch_10"
echo "teacher=$TEACHER_POLICY"
echo "node=\$(hostname)"
echo "============================================================"

srun --kill-on-bad-exit=1 python -u train_mh_ald.py \
  --config-name=lewm_mh_ald_teacher_mh \
  $COMMON_HYDRA \
  trainer.max_epochs=10 \
  output_model_name=$FORMAL_MODEL \
  subdir=$FORMAL_DIR

echo
echo "=== EXP1 FORMAL ARTIFACT VALIDATION ==="
[[ -s "$STABLEWM_HOME/$FORMAL_DIR/${FORMAL_MODEL}_epoch_10_object.ckpt" ]] || {
  echo "ERROR: Exp1 epoch-10 object checkpoint missing." >&2
  find "$STABLEWM_HOME/$FORMAL_DIR" -maxdepth 1 -type f -printf '%f\n' || true
  exit 7
}

echo "=== EXP1 FORMAL DONE ==="
echo "policy=$FORMAL_DIR/${FORMAL_MODEL}_epoch_10"
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted MH teacher Exp1 ($MODE): $JID"
echo "node         : $NODE"
echo "GPUs         : $WORLD_SIZE"
echo "local batch  : $LOCAL_BATCH"
echo "global batch : $GLOBAL_BATCH"
echo "teacher      : $TEACHER_POLICY"
echo
squeue -j "$JID" -o "%.18i %.12j %.2t %.12M %.12l %.20R" || true

echo
if [[ "$MODE" == "smoke" ]]; then
  echo "stdout: tail -f $LOG_DIR/mh_teacher_mh_exp1_smoke_${JID}.out"
  echo "stderr: tail -f $LOG_DIR/mh_teacher_mh_exp1_smoke_${JID}.err"
else
  echo "stdout: tail -f $LOG_DIR/mh_teacher_mh_exp1_formal_${JID}.out"
  echo "stderr: tail -f $LOG_DIR/mh_teacher_mh_exp1_formal_${JID}.err"
  echo "Final policy:"
  echo "  $FORMAL_DIR/${FORMAL_MODEL}_epoch_10"
fi
