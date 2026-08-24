#!/usr/bin/env bash
# Submit exactly ONE Stage-II-v2 A job on all 4 GPUs of 4090node2.
#
# Usage:
#   bash scripts/submit_stage2_v2_A.sh smoke
#   bash scripts/submit_stage2_v2_A.sh formal
#
# The launcher generates *.slurm files only on the server.  They remain under
# slurm/generated_stage2_v2_A_ddp4/ and are not intended for Git tracking.

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_stage2_v2_A.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="4090node2"
WORLD_SIZE=4
LOCAL_BATCH=16
GLOBAL_BATCH=$((WORLD_SIZE * LOCAL_BATCH))
NUM_WORKERS_PER_RANK=4

SMOKE_DIR="pusht_stage2_v2_A_smoke_ddp4"
SMOKE_MODEL="lewm_stage2_v2_A_smoke_ddp4"
FORMAL_DIR="pusht_stage2_v2_A_apbjvp_h5_seed3072_ep10_ddp4"
FORMAL_MODEL="lewm_stage2_v2_A_apbjvp_h5_ddp4"

GEN_DIR="$REPO/slurm/generated_stage2_v2_A_ddp4"
LOG_DIR="$REPO/logs"
mkdir -p "$GEN_DIR" "$LOG_DIR"
cd "$REPO"

source "$CONDA_SH"
conda activate lewm

python -m py_compile \
  distributed_training.py \
  stage1_bias_calibration.py \
  stage2_v2_apb_jvp.py \
  train_stage2_v2.py

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

COMMON_HYDRA="seed=3072 num_workers=$NUM_WORKERS_PER_RANK trainer.devices=$WORLD_SIZE trainer.num_nodes=1 trainer.strategy=ddp trainer.sync_batchnorm=true loader.batch_size=$LOCAL_BATCH loss.sigreg.global_batch_ddp=true"

if [[ "$MODE" == "smoke" ]]; then
  rm -rf "$STABLEWM_HOME/$SMOKE_DIR"
  FILE="$GEN_DIR/00_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=s2v2_sm4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/stage2_v2_A_smoke_%j.out
#SBATCH --error=$LOG_DIR/stage2_v2_A_smoke_%j.err

$COMMON_SETUP

srun --kill-on-bad-exit=1 python -u train_stage2_v2.py \
  $COMMON_HYDRA \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  output_model_name=$SMOKE_MODEL \
  subdir=$SMOKE_DIR
EOF
else
  if [[ -e "$STABLEWM_HOME/$FORMAL_DIR" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "ERROR: formal output already exists: $STABLEWM_HOME/$FORMAL_DIR" >&2
    echo "Remove/rename it before a fresh formal run." >&2
    exit 3
  fi
  FILE="$GEN_DIR/A_v2_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=s2v2_A4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/stage2_v2_A_formal_%j.out
#SBATCH --error=$LOG_DIR/stage2_v2_A_formal_%j.err

$COMMON_SETUP

srun --kill-on-bad-exit=1 python -u train_stage2_v2.py \
  $COMMON_HYDRA \
  trainer.max_epochs=10 \
  output_model_name=$FORMAL_MODEL \
  subdir=$FORMAL_DIR
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted Stage-II-v2 A ($MODE): $JID"
echo "node         : $NODE"
echo "GPUs         : $WORLD_SIZE"
echo "local batch  : $LOCAL_BATCH"
echo "global batch : $GLOBAL_BATCH"
echo
squeue -j "$JID" -o "%.18i %.12j %.2t %.12M %.12l %.20R" || true

echo
if [[ "$MODE" == "smoke" ]]; then
  echo "Log: tail -f $LOG_DIR/stage2_v2_A_smoke_${JID}.out"
  echo "After it COMPLETES successfully, submit formal with:"
  echo "  bash scripts/submit_stage2_v2_A.sh formal"
else
  echo "Log: tail -f $LOG_DIR/stage2_v2_A_formal_${JID}.out"
  echo "Final model:"
  echo "  $STABLEWM_HOME/$FORMAL_DIR/${FORMAL_MODEL}_epoch_10_object.ckpt"
fi
