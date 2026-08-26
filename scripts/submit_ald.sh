#!/usr/bin/env bash
# Submit Anchored Local Dynamics (ALD) calibration on all four GPUs of a chosen 4090 node.
#
# Usage:
#   bash scripts/submit_ald.sh smoke
#   bash scripts/submit_ald.sh formal
#   NODE=4090node3 bash scripts/submit_ald.sh formal
#
# NODE defaults to 4090node2 for backward compatibility, but can be overridden
# from the shell without editing the script.
#
# The smoke run is two optimization steps on the exact same 4-GPU DDP path.
# Its initial validation should satisfy the ALD construction invariants:
#   response_cosine ~= 1
#   response_gain   ~= 1
#   half_response_loss ~= 0
#   ald_init_equivalence_ratio ~= 1

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: bash scripts/submit_ald.sh {smoke|formal}" >&2
  exit 2
fi

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="${NODE:-4090node2}"
WORLD_SIZE=4
LOCAL_BATCH=16
GLOBAL_BATCH=$((WORLD_SIZE * LOCAL_BATCH))
NUM_WORKERS_PER_RANK=4

SMOKE_DIR="pusht_ald_smoke_ddp4"
SMOKE_MODEL="lewm_ald_smoke_ddp4"
FORMAL_DIR="pusht_ald_h5_seed3072_ep10_ddp4"
FORMAL_MODEL="lewm_ald_h5_ddp4"

GEN_DIR="$REPO/slurm/generated_ald_ddp4"
LOG_DIR="$REPO/logs"
mkdir -p "$GEN_DIR" "$LOG_DIR"
cd "$REPO"

source "$CONDA_SH"
conda activate lewm

python -m py_compile \
  stage1_bias_calibration.py \
  anchored_local_dynamics.py \
  train_ald.py

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

# No SyncBatchNorm: ALD deliberately calibrates under inference semantics with
# frozen BN running statistics.  local batch 16 x 4 ranks preserves global 64.
COMMON_HYDRA="seed=3072 num_workers=$NUM_WORKERS_PER_RANK trainer.devices=$WORLD_SIZE trainer.num_nodes=1 trainer.strategy=ddp trainer.sync_batchnorm=false loader.batch_size=$LOCAL_BATCH"

if [[ "$MODE" == "smoke" ]]; then
  rm -rf "$STABLEWM_HOME/$SMOKE_DIR"
  FILE="$GEN_DIR/00_smoke.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=ald_sm4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/ald_smoke_%j.out
#SBATCH --error=$LOG_DIR/ald_smoke_%j.err

$COMMON_SETUP

srun --kill-on-bad-exit=1 python -u train_ald.py \
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

  FILE="$GEN_DIR/ald_formal.slurm"
  cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=ald_h5_4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/ald_formal_%j.out
#SBATCH --error=$LOG_DIR/ald_formal_%j.err

$COMMON_SETUP

srun --kill-on-bad-exit=1 python -u train_ald.py \
  $COMMON_HYDRA \
  trainer.max_epochs=10 \
  output_model_name=$FORMAL_MODEL \
  subdir=$FORMAL_DIR
EOF
fi

bash -n "$FILE"
JID=$(sbatch --parsable "$FILE")

echo "Submitted ALD ($MODE): $JID"
echo "node         : $NODE"
echo "GPUs         : $WORLD_SIZE"
echo "local batch  : $LOCAL_BATCH"
echo "global batch : $GLOBAL_BATCH"
echo
squeue -j "$JID" -o "%.18i %.12j %.2t %.12M %.12l %.20R" || true

echo
if [[ "$MODE" == "smoke" ]]; then
  echo "Log: tail -f $LOG_DIR/ald_smoke_${JID}.out"
  echo "After the smoke completes cleanly, submit:"
  echo "  NODE=$NODE bash scripts/submit_ald.sh formal"
else
  echo "Log: tail -f $LOG_DIR/ald_formal_${JID}.out"
  echo "Final model:"
  echo "  $STABLEWM_HOME/$FORMAL_DIR/${FORMAL_MODEL}_epoch_10_object.ckpt"
fi
