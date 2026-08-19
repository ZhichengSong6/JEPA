#!/bin/bash
#SBATCH --job-name=stage1_sig
#SBATCH --partition=GPU
#SBATCH --nodelist=4090node2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage1_biascal_signal_%j.out
#SBATCH --error=logs/stage1_biascal_signal_%j.err

set -euo pipefail

source /mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh
conda activate lewm

cd /mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official
export STABLEWM_HOME=/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

mkdir -p logs

echo "===== Stage I signal smoke ====="
echo "date: $(date)"
echo "host: $(hostname)"
echo "git:  $(git rev-parse HEAD)"
echo "branch: $(git branch --show-current)"
echo "STABLEWM_HOME: ${STABLEWM_HOME}"
nvidia-smi

# -----------------------------------------------------------------------------
# Remove ONLY outputs from the failed Stage-I smoke attempts on 2026-08-19.
# Do not touch baseline LeWM / Factor checkpoints or their logs.
# -----------------------------------------------------------------------------
echo "===== Cleaning failed Stage-I smoke outputs ====="
rm -rf \
  "${STABLEWM_HOME}/pusht_stage1_biascal_smoke" \
  "${STABLEWM_HOME}/pusht_stage1_biascal_smoke_v2"

rm -rf \
  lightning_logs/version_2 \
  lightning_logs/version_3

rm -rf wandb/offline-run-*-pusht_stage1_biascal_smoke*

rm -f \
  environment_v3.json \
  environment_v4.json \
  requirements_frozen_v4.txt \
  requirements_frozen_v5.txt

echo "cleanup complete"

# -----------------------------------------------------------------------------
# Signal smoke, not a full epoch.
# The previous node1 run measured ~3.4 it/s at batch_size=16, so 2000 training
# batches are enough to verify that AdaLN-zero action conditioning wakes up and
# Stage-I response/APB metrics become non-zero without spending a full epoch.
# -----------------------------------------------------------------------------
python -u train_stage1.py \
  data=pusht_stage1 \
  stage1.enabled=true \
  factor.enabled=false \
  wandb.enabled=false \
  trainer.devices=1 \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=2000 \
  +trainer.limit_val_batches=20 \
  loader.batch_size=16 \
  output_model_name=lewm_stage1_biascal_signal \
  subdir=pusht_stage1_biascal_signal_smoke_node2

echo "===== Stage I signal smoke finished ====="
echo "date: $(date)"
