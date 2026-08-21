#!/usr/bin/env bash
# Submit the three formal Stage-II experiments as an unattended SLURM chain.
# This file is safe to keep in git; it creates the actual *.slurm files only on
# the server at runtime (the repository's local exclude keeps them untracked).

set -euo pipefail

REPO="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_official"
STABLEWM_HOME="/mnt/slurmfs-3090node3/user_data/zsong469/LeWM_data"
CONDA_SH="/mnt/slurmfs-4090node1/homes/zsong469/miniforge3/etc/profile.d/conda.sh"
TIME_LIMIT="${TIME_LIMIT:-7-00:00:00}"

A_DIR="pusht_stage2_A_finetune_oddcurv_h5_seed3072_ep10"
B_DIR="pusht_stage2_B_lewm_continue_seed3072_ep10"
C_DIR="pusht_stage2_C_scratch_oddcurv_h5_seed3072_ep10"

A_MODEL="lewm_stage2_A_oddcurv_h5"
B_MODEL="lewm_stage2_B_lewm_continue"
C_MODEL="lewm_stage2_C_scratch_oddcurv_h5"

cd "$REPO"
mkdir -p slurm/generated_stage2_abc logs

# Refuse accidental mixing with earlier formal runs. Set ALLOW_EXISTING=1 only
# when intentionally resubmitting a run whose Manager checkpoint should resume.
if [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
  for d in "$A_DIR" "$B_DIR" "$C_DIR"; do
    if [[ -e "$STABLEWM_HOME/$d" ]]; then
      echo "ERROR: output directory already exists: $STABLEWM_HOME/$d" >&2
      echo "Remove/rename it, or rerun with ALLOW_EXISTING=1 to resume intentionally." >&2
      exit 2
    fi
  done
fi

source "$CONDA_SH"
conda activate lewm
python -m py_compile stage2_landscape_faithful.py train_stage2.py train_lewm_continuation.py

cat > slurm/generated_stage2_abc/A_stage2_finetune.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=s2_A_ft
#SBATCH --partition=GPU
#SBATCH --nodelist=4090node[2-3]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=$REPO/logs/stage2_A_finetune_%j.out
#SBATCH --error=$REPO/logs/stage2_A_finetune_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

python -u train_stage2.py \
  data=pusht_stage2 \
  seed=3072 \
  stage2.enabled=true \
  stage2.init_mode=pretrained \
  stage2.init_policy=lewm_epoch_10 \
  stage2.teacher_policy=lewm_epoch_10 \
  stage1.enabled=false \
  factor.enabled=false \
  trainer.max_epochs=10 \
  loader.batch_size=64 \
  output_model_name=$A_MODEL \
  subdir=$A_DIR
EOF

cat > slurm/generated_stage2_abc/B_lewm_continue.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=s2_B_ctl
#SBATCH --partition=GPU
#SBATCH --nodelist=4090node[2-3]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=$REPO/logs/stage2_B_continue_%j.out
#SBATCH --error=$REPO/logs/stage2_B_continue_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

python -u train_lewm_continuation.py \
  data=pusht_stage2 \
  seed=3072 \
  continuation.enabled=true \
  continuation.init_policy=lewm_epoch_10 \
  stage1.enabled=false \
  stage2.enabled=false \
  factor.enabled=false \
  trainer.max_epochs=10 \
  loader.batch_size=64 \
  output_model_name=$B_MODEL \
  subdir=$B_DIR
EOF

cat > slurm/generated_stage2_abc/C_stage2_scratch.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=s2_C_scr
#SBATCH --partition=GPU
#SBATCH --nodelist=4090node[2-3]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=$REPO/logs/stage2_C_scratch_%j.out
#SBATCH --error=$REPO/logs/stage2_C_scratch_%j.err

set -euo pipefail
source "$CONDA_SH"
conda activate lewm
cd "$REPO"
export STABLEWM_HOME="$STABLEWM_HOME"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

python -u train_stage2.py \
  data=pusht_stage2 \
  seed=3072 \
  stage2.enabled=true \
  stage2.init_mode=random \
  stage2.teacher_policy=lewm_epoch_10 \
  stage1.enabled=false \
  factor.enabled=false \
  trainer.max_epochs=10 \
  loader.batch_size=64 \
  output_model_name=$C_MODEL \
  subdir=$C_DIR
EOF

jid_a=$(sbatch --parsable --time="$TIME_LIMIT" slurm/generated_stage2_abc/A_stage2_finetune.slurm)
jid_b=$(sbatch --parsable --time="$TIME_LIMIT" --dependency="afterok:${jid_a}" slurm/generated_stage2_abc/B_lewm_continue.slurm)
jid_c=$(sbatch --parsable --time="$TIME_LIMIT" --dependency="afterok:${jid_b}" slurm/generated_stage2_abc/C_stage2_scratch.slurm)

cat <<EOF
Submitted Stage-II ABC chain successfully.

A  fine-tune Stage II : job $jid_a
B  LeWM continuation  : job $jid_b   (afterok:$jid_a)
C  Stage II scratch   : job $jid_c   (afterok:$jid_b)

Time limit per job: $TIME_LIMIT
Allowed nodes: 4090node2, 4090node3 only

Outputs:
A: $STABLEWM_HOME/$A_DIR/${A_MODEL}_epoch_10_object.ckpt
B: $STABLEWM_HOME/$B_DIR/${B_MODEL}_epoch_10_object.ckpt
C: $STABLEWM_HOME/$C_DIR/${C_MODEL}_epoch_10_object.ckpt

Check queue:
  squeue -u zsong469 -o '%.18i %.12j %.2t %.10M %.10l %.20R'

Check logs:
  tail -f $REPO/logs/stage2_A_finetune_${jid_a}.out
EOF
