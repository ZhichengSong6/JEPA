#!/usr/bin/env bash
# Submit a 4-GPU Stage-II smoke followed by formal A -> B -> C experiments.
#
# Resource policy requested for this experiment:
#   * ONLY 4090node2
#   * all four RTX 4090 GPUs on that node
#   * global training batch = 64 = 4 ranks * local batch 16
#   * A -> B -> C strictly sequential via afterok dependencies
#
# The repository stores only this launcher. Actual *.slurm files are generated
# locally on the server under slurm/generated_stage2_abc_ddp4/ and should not be
# committed.

set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
NODE="4090node2"
WORLD_SIZE=4
LOCAL_BATCH=16
GLOBAL_BATCH=$((WORLD_SIZE * LOCAL_BATCH))
NUM_WORKERS_PER_RANK=4

# Leave TIME_LIMIT unset by default so Slurm uses the GPU partition's configured
# default/maximum-compatible time. If your site requires an explicit value, run
# e.g. TIME_LIMIT=3-00:00:00 bash scripts/submit_stage2_abc.sh
TIME_LIMIT="${TIME_LIMIT:-}"

SMOKE_DIR="pusht_stage2_ddp4_smoke"
A_DIR="pusht_stage2_A_finetune_oddcurv_h5_seed3072_ep10_ddp4"
B_DIR="pusht_stage2_B_lewm_continue_seed3072_ep10_ddp4"
C_DIR="pusht_stage2_C_scratch_oddcurv_h5_seed3072_ep10_ddp4"

SMOKE_MODEL="lewm_stage2_ddp4_smoke"
A_MODEL="lewm_stage2_A_oddcurv_h5_ddp4"
B_MODEL="lewm_stage2_B_lewm_continue_ddp4"
C_MODEL="lewm_stage2_C_scratch_oddcurv_h5_ddp4"

GEN_DIR="$REPO/slurm/generated_stage2_abc_ddp4"
LOG_DIR="$REPO/logs"

cd "$REPO"
mkdir -p "$GEN_DIR" "$LOG_DIR"

# Smoke is disposable; formal outputs are guarded against accidental overwrite.
rm -rf "$STABLEWM_HOME/$SMOKE_DIR"
if [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
  for d in "$A_DIR" "$B_DIR" "$C_DIR"; do
    if [[ -e "$STABLEWM_HOME/$d" ]]; then
      echo "ERROR: formal output directory already exists: $STABLEWM_HOME/$d" >&2
      echo "Remove/rename it, or rerun with ALLOW_EXISTING=1 only if intentional." >&2
      exit 2
    fi
  done
fi

source "$CONDA_SH"
conda activate lewm
python -m py_compile \
  distributed_training.py \
  stage2_landscape_faithful.py \
  train_stage2.py \
  train_lewm_continuation.py

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

cat > "$GEN_DIR/00_smoke.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=s2_sm4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/stage2_ddp4_smoke_%j.out
#SBATCH --error=$LOG_DIR/stage2_ddp4_smoke_%j.err

$COMMON_SETUP

srun --kill-on-bad-exit=1 python -u train_stage2.py \
  data=pusht_stage2 \
  \$COMMON_HYDRA \
  stage2.enabled=true \
  stage2.init_mode=pretrained \
  stage2.init_policy=lewm_epoch_10 \
  stage2.teacher_policy=lewm_epoch_10 \
  stage1.enabled=false \
  factor.enabled=false \
  trainer.max_epochs=1 \
  +trainer.max_steps=2 \
  output_model_name=$SMOKE_MODEL \
  subdir=$SMOKE_DIR
EOF

cat > "$GEN_DIR/A_stage2_finetune.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=s2_A4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/stage2_A_ddp4_%j.out
#SBATCH --error=$LOG_DIR/stage2_A_ddp4_%j.err

$COMMON_SETUP

srun --kill-on-bad-exit=1 python -u train_stage2.py \
  data=pusht_stage2 \
  \$COMMON_HYDRA \
  stage2.enabled=true \
  stage2.init_mode=pretrained \
  stage2.init_policy=lewm_epoch_10 \
  stage2.teacher_policy=lewm_epoch_10 \
  stage1.enabled=false \
  factor.enabled=false \
  trainer.max_epochs=10 \
  output_model_name=$A_MODEL \
  subdir=$A_DIR
EOF

cat > "$GEN_DIR/B_lewm_continue.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=s2_B4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/stage2_B_ddp4_%j.out
#SBATCH --error=$LOG_DIR/stage2_B_ddp4_%j.err

$COMMON_SETUP

srun --kill-on-bad-exit=1 python -u train_lewm_continuation.py \
  data=pusht_stage2 \
  \$COMMON_HYDRA \
  continuation.enabled=true \
  continuation.init_policy=lewm_epoch_10 \
  stage1.enabled=false \
  stage2.enabled=false \
  factor.enabled=false \
  trainer.max_epochs=10 \
  output_model_name=$B_MODEL \
  subdir=$B_DIR
EOF

cat > "$GEN_DIR/C_stage2_scratch.slurm" <<EOF
#!/bin/bash
#SBATCH --job-name=s2_C4
$COMMON_HEADER
#SBATCH --output=$LOG_DIR/stage2_C_ddp4_%j.out
#SBATCH --error=$LOG_DIR/stage2_C_ddp4_%j.err

$COMMON_SETUP

srun --kill-on-bad-exit=1 python -u train_stage2.py \
  data=pusht_stage2 \
  \$COMMON_HYDRA \
  stage2.enabled=true \
  stage2.init_mode=random \
  stage2.teacher_policy=lewm_epoch_10 \
  stage1.enabled=false \
  factor.enabled=false \
  trainer.max_epochs=10 \
  output_model_name=$C_MODEL \
  subdir=$C_DIR
EOF

submit_job() {
  local dependency="$1"
  local file="$2"
  local args=(--parsable)
  if [[ -n "$TIME_LIMIT" ]]; then
    args+=(--time="$TIME_LIMIT")
  fi
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:${dependency}")
  fi
  sbatch "${args[@]}" "$file"
}

jid_smoke=$(submit_job "" "$GEN_DIR/00_smoke.slurm")
jid_a=$(submit_job "$jid_smoke" "$GEN_DIR/A_stage2_finetune.slurm")
jid_b=$(submit_job "$jid_a" "$GEN_DIR/B_lewm_continue.slurm")
jid_c=$(submit_job "$jid_b" "$GEN_DIR/C_stage2_scratch.slurm")

cat <<EOF
Submitted 4-GPU Stage-II chain on $NODE successfully.

Resources per job:
  node          : $NODE
  GPUs          : $WORLD_SIZE x RTX 4090
  DDP ranks     : $WORLD_SIZE
  local batch   : $LOCAL_BATCH
  global batch  : $GLOBAL_BATCH
  SyncBatchNorm : enabled
  SIGReg        : differentiable global-batch gather (B=$GLOBAL_BATCH)

Dependency chain:
  smoke : $jid_smoke
    -> A : $jid_a   (afterok:$jid_smoke)
    -> B : $jid_b   (afterok:$jid_a)
    -> C : $jid_c   (afterok:$jid_b)

Formal outputs:
A: $STABLEWM_HOME/$A_DIR/${A_MODEL}_epoch_10_object.ckpt
B: $STABLEWM_HOME/$B_DIR/${B_MODEL}_epoch_10_object.ckpt
C: $STABLEWM_HOME/$C_DIR/${C_MODEL}_epoch_10_object.ckpt

Queue:
  squeue -u zsong469 -o '%.18i %.12j %.2t %.10M %.10l %.24R'

Smoke log:
  tail -f $LOG_DIR/stage2_ddp4_smoke_${jid_smoke}.out

A log after smoke succeeds:
  tail -f $LOG_DIR/stage2_A_ddp4_${jid_a}.out
EOF
